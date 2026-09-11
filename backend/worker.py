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

from backend.contracts import NavigationFeedback, ToolResult
from backend.navigation import NAVIGATION_TOOLS, NavigationRuntime
from backend.simulation import BulletSimulation, MotionError


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
        self.thread = threading.Thread(target=self._run, name=f"physics-{epoch}", daemon=True)
        self.thread.start()

    def _publish(self):
        self.latest = {**self.latest, "snapshot": self.sim.snapshot(), "challenge": self.sim.challenge_status(),
                   "proximity": self.sim.proximity_sensors().model_dump(),
                   "navigation": self.navigation.state() if self.navigation else None}
        if time.monotonic() - self.camera_published_at >= self.camera_interval_s:
            self._publish_camera()
        if self.pace:
            remaining = self.wall_start + (self.sim.ticks - self.tick_start) / 240 - time.monotonic()
            if remaining > 0:
                self.sim.cancel.wait(remaining)

    def _publish_camera(self, image=None):
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
            "simulated_time_s": self.sim.ticks / 240,
            "url": f"/api/camera/{self.sim.run_id}/{reference}"}}

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
                if self.navigation and self.sim.cancel.is_set() and self.navigation.status in {"running", "awaiting_feedback", "idle"}:
                    self.navigation.cancel(self.sim)
                    self._publish_navigation()
                try:
                    active = self.navigation and self.navigation.status in {"running", "awaiting_feedback"}
                    work = self.queue.get(timeout=0 if active and self.navigation.buffer else .05 if active else None)
                except Empty:
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
        observation = sim.observe()
        if self.navigation:
            observation.navigation = NavigationFeedback.model_validate(self.navigation.observe(sim, observation))
        self.latest = {**self.latest, "observation": observation.model_dump()}
        return observation, sim.frame(observation.frame_ref)

    async def feedback(self):
        return await self.call(self._feedback)

    async def begin_navigation(self, expected_stop_revision, enabled=True):
        def operation(sim):
            if sim.cancel.is_set() or expected_stop_revision != self.stop_revision:
                raise MotionError("CANCELLED", "Navigation start was invalidated by Stop.")
            if self.navigation:
                self.navigation.cancel(sim, "New navigation session")
            self.navigation = NavigationRuntime() if enabled else None
            self.latest = {**self.latest, "navigation": self.navigation.state() if self.navigation else None}
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