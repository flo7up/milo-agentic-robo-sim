import asyncio
from collections import OrderedDict
from concurrent.futures import Future
from io import BytesIO
from queue import Empty, Queue
import threading
import time

import numpy as np
from PIL import Image
from pydantic import ValidationError

from backend.contracts import AgentObservation, NavigationFeedback, SkillFeedback, ToolResult
from backend.navigation import NAVIGATION_TOOLS, NavigationRuntime
from backend.policy import SKILL_TOOLS, SkillRuntime
from backend.simulation import BulletSimulation, MotionError, TIMESTEP


class PolicyPacer:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.reset()

    def reset(self):
        self.deadline = None
        self.revision = None

    def wait(self, cancel, revision, started):
        now = self.clock()
        if self.deadline is None or revision != self.revision or now - self.deadline > .05:
            self.deadline = started + TIMESTEP
        else:
            self.deadline += TIMESTEP
        self.revision = revision
        remaining = self.deadline - self.clock()
        if remaining > 0:
            cancel.wait(remaining)


class CameraActivity:
    def __init__(self):
        self.reference = None
        self.changed_at = time.monotonic()
        self.revision = 0

    def sample(self, image, now=None):
        now = time.monotonic() if now is None else now
        with Image.open(BytesIO(image)) as decoded:
            pixels = np.asarray(decoded.convert("RGB").resize((64, 48)), dtype=np.int16)
        changed = self.reference is None or np.mean(np.max(np.abs(pixels - self.reference), axis=2) > 12) >= .01
        if changed:
            self.reference = pixels
            self.changed_at = now
            self.revision += 1
        return {"revision": self.revision, "unchanged_s": max(0, now - self.changed_at)}


class SimulationWorker:
    camera_interval_s = .1
    camera_capacity = 16

    def __init__(self, epoch=0, scene=None, pace=True, challenge=None):
        self.queue = Queue()
        self.ready = Future()
        self.latest = {}
        self.sim = None
        self.closed = False
        self.pace = pace
        self.epoch = epoch
        self.scene = scene
        self.challenge = challenge
        self.camera_frames = OrderedDict()
        self.camera_seq = 0
        self.camera_published_at = 0
        self.camera_activity = CameraActivity()
        self.stop_revision = 0
        self.navigation = None
        self.skill = None
        self.renderer = None
        self.render_frame = None
        self.render_requested_at = 0
        self.policy_pacer = PolicyPacer()
        self.thread = threading.Thread(target=self._run, name=f"physics-{epoch}", daemon=True)
        self.thread.start()

    def _publish(self):
        self.latest = {**self.latest, "snapshot": self.sim.snapshot(), "challenge": self.sim.challenge_status(),
                   "proximity": self.sim.proximity_sensors().model_dump(),
                   "navigation": self.navigation.state() if self.navigation else None,
                   "skill": self.skill.state() if self.skill else None}
        if self.renderer:
            self._render_update()
        elif time.monotonic() - self.camera_published_at >= self.camera_interval_s:
            self._publish_camera()
        if self.pace and not self.renderer:
            remaining = self.wall_start + (self.sim.ticks - self.tick_start) / 240 - time.monotonic()
            if remaining > 0:
                self.sim.cancel.wait(remaining)

    def _publish_camera(self, image=None, simulated_time_s=None):
        if image is None:
            image = self.sim.capture()
        self.camera_activity.sample(image)
        self.camera_seq += 1
        reference = f"live-{self.camera_seq}.png"
        self.camera_frames[reference] = image
        while len(self.camera_frames) > self.camera_capacity:
            self.camera_frames.popitem(last=False)
        self.camera_published_at = time.monotonic()
        self.latest = {**self.latest, "camera": {
            "seq": self.camera_seq, "frame_ref": reference,
            "simulated_time_s": self.sim.ticks / 240 if simulated_time_s is None else simulated_time_s,
            "url": f"/api/camera/{self.sim.run_id}/{reference}"}}

    def _render_update(self):
        import pybullet as bullet
        frame = self.renderer.latest()
        if frame:
            data, image, captured_at = frame
            observation = AgentObservation.model_validate(data)
            self.render_frame = observation, image, captured_at
            self._publish_camera(image, observation.simulated_time_s)
        if time.monotonic() - self.render_requested_at >= .05:
            self.render_requested_at = time.monotonic()
            observation = self.sim.observe(render=False)
            self.renderer.submit({"observation": observation.model_dump(), "captured_at": self.render_requested_at,
                "bodies": [(body, *bullet.getBasePositionAndOrientation(body, physicsClientId=self.sim.client))
                           for body in [item["id"] for item in self.sim.objects] + [self.sim.robot]],
                "joints": [(name, bullet.getJointState(self.sim.robot, index, physicsClientId=self.sim.client)[0])
                           for name, index in self.sim.joints.items()]})

    def _run(self):
        try:
            self.sim = BulletSimulation(epoch=self.epoch, scene=self.scene, challenge=self.challenge)
            observation = self.sim.observe()
            self.latest = {"run_id": self.sim.run_id, "episode_epoch": self.epoch, "observation": observation.model_dump(),
                           "snapshot": self.sim.snapshot(), "geometry": self.sim.geometry(), "busy": False, "stopped": False,
                           "assisted": False, "result": None, "challenge": self.sim.challenge_status(),
                           "navigation": None,
                           "robot_body_id": self.sim.robot, "manual_placements": 0, "proximity": observation.proximity.model_dump()}
            self._publish_camera(self.sim.frame(observation.frame_ref))
            self.sim.on_tick = self._publish
            self.ready.set_result(True)
            while True:
                if self.skill and self.sim.cancel.is_set() and self.skill.active:
                    self.skill.cancel(self.sim, "Interrupted by Stop")
                    self._publish_skill()
                if self.navigation and self.sim.cancel.is_set() and self.navigation.status in {"running", "awaiting_feedback", "idle"}:
                    self.navigation.cancel(self.sim)
                    self._publish_navigation()
                try:
                    active = self.navigation and self.navigation.status in {"running", "awaiting_feedback"}
                    skill_active = self.skill and self.skill.active
                    buffered = (active and self.navigation.buffer) or (skill_active and self.skill.buffer)
                    rendering = self.renderer and not self.sim.cancel.is_set()
                    work = self.queue.get(timeout=0 if buffered else .01 if skill_active or rendering else .05 if active else None)
                except Empty:
                    if rendering and not skill_active:
                        self._render_update()
                        continue
                    if skill_active:
                        started = time.perf_counter()
                        if self.renderer and not self.skill.buffer:
                            self._render_update()
                        try:
                            self.skill.tick(self.sim)
                        except Exception:
                            self.skill.fail(self.sim, "EXECUTION_FAILED: Policy executor stopped")
                        self._publish_skill()
                        if self.pace and self.skill.buffer:
                            self.policy_pacer.wait(self.sim.cancel, self.skill.motion_revision, started)
                        else:
                            self.policy_pacer.reset()
                        continue
                    try:
                        self.navigation.tick(self.sim)
                    except Exception:
                        self.navigation.fail(self.sim, "EXECUTION_FAILED: Navigation worker stopped.")
                    self._publish_navigation()
                    continue
                if work is None:
                    break
                future, operation = work
                if not future.set_running_or_notify_cancel():
                    continue
                self.wall_start, self.tick_start = time.monotonic(), self.sim.ticks
                try:
                    future.set_result(operation(self.sim))
                except Exception as error:
                    future.set_exception(error)
        except Exception as error:
            if not self.ready.done():
                self.ready.set_exception(error)
        finally:
            if self.renderer:
                self.renderer.close()
            if self.sim:
                self.sim.close()
            self.camera_frames.clear()

    async def call(self, operation):
        await asyncio.wrap_future(self.ready)
        if self.closed:
            raise RuntimeError("Episode closed")
        future = Future()
        self.queue.put((future, operation))
        return await asyncio.wrap_future(future)

    async def execute(self, command, assisted=True):
        def operation(sim):
            if self.skill and self.skill.active:
                if command.tool not in {"stop", "observe"}:
                    raise MotionError("CONTROL_CONFLICT", "Cancel the policy skill before manual motion")
                if command.tool == "stop":
                    self.skill.cancel(sim, "Stopped by supervisor")
                    self._publish_skill()
            if self.navigation and self.navigation.status in {"running", "awaiting_feedback"}:
                if command.tool not in {"stop", "observe"}:
                    raise MotionError("CONTROL_CONFLICT", "Cancel navigation before manual motion.")
                if command.tool == "stop":
                    self.navigation.cancel(sim)
                    self._publish_navigation()
            self.latest = {**self.latest, "busy": True, "assisted": self.latest["assisted"] or assisted}
            try:
                result = sim.execute(command)
                self.latest = {**self.latest, "busy": False, "observation": result.observation.model_dump(), "result": result.model_dump(),
                               "snapshot": sim.snapshot(), "stopped": sim.cancel.is_set(), "challenge": sim.challenge_status(),
                               "proximity": result.observation.proximity.model_dump()}
                image = sim.frame(result.observation.frame_ref) if result.observation.seq == sim.seq else None
                self._publish_camera(image)
                return result
            finally:
                self.latest = {**self.latest, "busy": False}

        return await self.call(operation)

    def _publish_navigation(self):
        self.latest = {**self.latest, "navigation": self.navigation.state(), "busy": bool(self.navigation.buffer),
                       "snapshot": self.sim.snapshot(), "stopped": self.sim.cancel.is_set()}

    def _feedback(self, sim):
        if self.renderer:
            self._render_update()
            cached, image, captured_at = self.render_frame
            if time.monotonic() - captured_at > 1:
                raise MotionError("CAMERA_STALE", "Snapshot camera feedback expired")
            observation = cached.model_copy(deep=True)
        else:
            observation = sim.observe()
            image = sim.frame(observation.frame_ref)
        if self.navigation:
            observation.navigation = NavigationFeedback.model_validate(self.navigation.observe(sim, observation))
        if self.skill:
            observation.skill = SkillFeedback.model_validate(self.skill.state())
        self.latest = {**self.latest, "observation": observation.model_dump()}
        return observation, image

    async def feedback(self):
        return await self.call(self._feedback)

    async def begin_navigation(self, expected_stop_revision, enabled=True):
        def operation(sim):
            if sim.cancel.is_set() or expected_stop_revision != self.stop_revision:
                raise MotionError("CANCELLED", "Navigation start was invalidated by Stop.")
            if self.navigation:
                self.navigation.cancel(sim, "New navigation session")
            if self.skill:
                self.skill.cancel(sim, "Execution mode changed")
                self.skill = None
            if self.renderer:
                self.renderer.close()
                self.renderer = None
                self.render_frame = None
            self.navigation = NavigationRuntime() if enabled else None
            self.latest = {**self.latest, "navigation": self.navigation.state() if self.navigation else None, "skill": None}
        await self.call(operation)

    def _publish_skill(self):
        self.latest = {**self.latest, "skill": self.skill.state(), "busy": bool(self.skill.buffer),
                       "stopped": self.sim.cancel.is_set()}

    async def begin_skill_mode(self, metadata, expected_stop_revision):
        def operation(sim):
            from backend.camera import SnapshotRenderer
            if sim.cancel.is_set() or self.stop_revision != expected_stop_revision:
                raise MotionError("CANCELLED", "Skill mode start was invalidated")
            if self.skill:
                self.skill.cancel(sim)
            if self.navigation:
                self.navigation.cancel(sim)
                self.navigation = None
            if not self.renderer:
                self.renderer = SnapshotRenderer(sim.scene)
            captured_at = time.monotonic()
            observation = sim.observe()
            self.render_frame = observation, sim.frame(observation.frame_ref), captured_at
            self.skill = SkillRuntime(metadata)
            self._publish_skill()
        await self.call(operation)

    async def end_skill_mode(self):
        def operation(sim):
            if self.skill and self.skill.active:
                self.skill.cancel(sim, "Supervisor session ended")
            if self.renderer:
                self.renderer.close()
                self.renderer = None
                self.render_frame = None
            if self.skill:
                self._publish_skill()
                observation, image = self._feedback(sim)
                self.latest = {**self.latest, "snapshot": sim.snapshot(), "observation": observation.model_dump()}
                self._publish_camera(image)
        await self.call(operation)

    async def execute_skill(self, command):
        def operation(sim):
            if command.run_id != sim.run_id or command.episode_epoch != sim.epoch or not self.skill:
                raise MotionError("CANCELLED", "Skill episode changed")
            if command.action_id in sim.results:
                return sim.results[command.action_id]
            status, code, message = "ok", None, "Skill request accepted; task completion is not implied"
            try:
                arguments = SKILL_TOOLS[command.tool].model_validate(command.arguments)
                self.skill.apply(sim, command.tool, arguments)
            except (ValidationError, MotionError) as error:
                status, code = "error", error.code if isinstance(error, MotionError) else "INVALID_ARGUMENT"
                message = str(error)
            observation, image = self._feedback(sim)
            result = ToolResult(action_id=command.action_id, status=status, error=code, message=message, observation=observation)
            sim.results[command.action_id] = result
            self.latest = {**self.latest, "result": result.model_dump()}
            self._publish_skill()
            self._publish_camera(image)
            return result
        return await self.call(operation)

    async def policy_feedback(self):
        def operation(sim):
            observation, image = self._feedback(sim)
            return self.skill.ticket(sim, observation, self.render_frame[2] if self.render_frame else None), observation, image
        return await self.call(operation)

    async def accept_policy(self, chunk):
        def operation(sim):
            if not self.skill:
                raise MotionError("CANCELLED", "Policy mode ended")
            try:
                self.skill.accept(sim, chunk)
            except MotionError:
                self.skill.rejected_chunks += 1
                raise
            self._publish_skill()
        await self.call(operation)

    async def fail_policy(self, reason):
        def operation(sim):
            if self.skill and self.skill.active:
                self.skill.fail(sim, reason)
                self._publish_skill()
        await self.call(operation)

    async def pause_navigation(self, reason):
        def operation(sim):
            if self.navigation and self.navigation.status not in {"failed", "cancelled", "completed"}:
                self.navigation.brake(sim, "awaiting_feedback", reason)
                self._publish_navigation()
        await self.call(operation)

    async def execute_navigation(self, command):
        def operation(sim):
            if command.run_id != sim.run_id or command.episode_epoch != sim.epoch:
                raise MotionError("CANCELLED", "Episode changed before navigation update.")
            if command.action_id in sim.results:
                return sim.results[command.action_id]
            if not self.navigation:
                raise MotionError("NO_ACTIVE_PLAN", "Navigation mode is not active.")
            status, code, message = "ok", None, "Navigation update accepted; buffered motion is not yet complete."
            try:
                arguments = NAVIGATION_TOOLS[command.tool].model_validate(command.arguments)
                self.navigation.apply(sim, command.tool, arguments, command.observation_seq)
            except (ValidationError, MotionError) as error:
                status = "error"
                code = error.code if isinstance(error, MotionError) else "INVALID_ARGUMENT"
                message = str(error) if isinstance(error, MotionError) else "Navigation arguments violate the skill/buffer limits."
                if sim.cancel.is_set():
                    if self.navigation.status != "failed":
                        self.navigation.cancel(sim, message)
                else:
                    self.navigation.brake(sim, "awaiting_feedback", message)
            observation, image = self._feedback(sim)
            result = ToolResult(action_id=command.action_id, status=status, error=code, message=message, observation=observation)
            sim.results[command.action_id] = result
            self.latest = {**self.latest, "result": result.model_dump()}
            self._publish_navigation()
            self._publish_camera(image)
            return result
        return await self.call(operation)

    async def reposition(self, placement):
        def operation(sim):
            if self.skill and self.skill.active:
                raise MotionError("CONTROL_CONFLICT", "Cancel the policy skill before placement")
            if self.navigation and self.navigation.status in {"running", "awaiting_feedback"}:
                raise MotionError("CONTROL_CONFLICT", "Cancel navigation before placement.")
            observation = sim.reposition(placement)
            self.latest = {**self.latest, "observation": observation.model_dump(), "snapshot": sim.snapshot(),
                           "result": None, "challenge": sim.challenge_status(), "proximity": observation.proximity.model_dump(),
                           "manual_placements": self.latest["manual_placements"] + 1}
            self._publish_camera(sim.frame(observation.frame_ref))
            return self.latest
        return await self.call(operation)

    async def sample_camera_activity(self):
        def operation(sim):
            image = sim.capture()
            prior = self.camera_activity.revision
            activity = self.camera_activity.sample(image)
            if activity["revision"] != prior:
                self._publish_camera(image)
            return activity
        return await self.call(operation)

    def stop(self):
        self.stop_revision += 1
        if self.sim:
            self.sim.stop()
        self.latest = {**self.latest, "stopped": True}

    async def resume_manual(self, expected_stop_revision=None):
        def operation(sim):
            if expected_stop_revision is not None and self.stop_revision != expected_stop_revision:
                return False
            sim.cancel.clear()
            self.latest = {**self.latest, "stopped": False}
            return True
        return await self.call(operation)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop()
        self.queue.put(None)
        await asyncio.to_thread(self.thread.join)