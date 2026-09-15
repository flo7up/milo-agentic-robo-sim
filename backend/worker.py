import asyncio
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
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


class MotionThroughput:
    def __init__(self):
        self.windows = deque(maxlen=8)
        self.anchor = None

    def observe(self, simulated_s, wall_s, active):
        if not active:
            self.anchor = None
            return
        if self.anchor is None or simulated_s < self.anchor[0] or wall_s < self.anchor[1]:
            self.anchor = (simulated_s, wall_s)
            return
        elapsed_sim_s = simulated_s - self.anchor[0]
        if elapsed_sim_s >= .5:
            self.windows.append(max(1., (wall_s - self.anchor[1]) / elapsed_sim_s))
            self.anchor = (simulated_s, wall_s)

    def state(self):
        return {"wall_seconds_per_sim_second": max(2., 1.2 * max(self.windows, default=0.)),
            "sample_windows": len(self.windows), "fallback_wall_ratio": 2., "measurement_margin": 1.2,
            "scope": "Recent active continuous-motion windows only; idle inference excluded"}


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

    def __init__(self, epoch=0, scene=None, pace=True, challenge=None, rendering=None, render_resources=None):
        self.queue = Queue()
        self.ready = Future()
        self.latest = {}
        self.sim = None
        self.closed = False
        self.pace = pace
        self.epoch = epoch
        self.scene = scene
        self.challenge = challenge
        self.rendering = rendering
        self.render_resources = render_resources
        self.owns_render_resources = render_resources is None
        self.camera_frames = OrderedDict()
        self.camera_seq = 0
        self.camera_published_at = 0
        self.camera_activity = CameraActivity()
        self.stop_revision = 0
        self.task_revision = 0
        self.navigation = None
        self.continuous = None
        self.proposed_goal_id = None
        self.ros_navigation = None
        self.ros_bridge = None
        self.ros_bridge_at = 0.
        self.ros_sensor_sequence = 0
        self.ros_tick_at = 0.
        self.skill = None
        self.renderer = None
        self.preview_renderer = None
        self.preview_requested_at = 0.
        self.render_frame = None
        self.render_requested_at = 0
        self.policy_pacer = PolicyPacer()
        self.motion_throughput = MotionThroughput()
        self.spatial_enabled = False
        self.spatial_map = None
        self.map_history = None
        self.camera_history = None
        self.spatial_frames = OrderedDict()
        self.spatial_sequence = 0
        self.spatial_sampled_at = 0.
        self.spatial_error = None
        self.spatial_processor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spatial-processing")
        self.spatial_pending = None
        self.spatial_pending_map = None
        self.spatial_pending_stop_revision = None
        self.spatial_capture_executor = None
        self.spatial_capture_pending = None
        self.spatial_capture_stop_revision = None
        self.spatial_capture_generation = 0
        self.spatial_generation = 0
        self.mapped_depth_sensor = None
        self.spatial_processing_max_s = 0.
        self.spatial_submitted_at = 0.
        self.spatial_timing = {"capture_max_s": 0., "post_capture_max_s": 0.,
            "accepted_results": 0, "expired_results": 0, "processing_timeouts": 0}
        self.recorder = None
        self.home_mission = None
        self.thread = threading.Thread(target=self._run, name=f"physics-{epoch}", daemon=True)
        self.thread.start()

    def _publish(self):
        self._record_map_pose()
        if self.home_mission:
            self.home_mission.sample()
        ros_active = self.ros_navigation and self.ros_navigation.active
        self.latest = {**self.latest, "snapshot": self.sim.snapshot(), "challenge": self.sim.challenge_status(),
                   "proximity": self.sim.proximity_sensors().model_dump(),
                   "navigation": self.navigation.state() if self.navigation else None,
                   "skill": self.skill.state() if self.skill else None}
        if self.renderer:
            self._render_update()
        elif ros_active:
            pass
        elif self.sim.rendering == "enhanced" and self.spatial_enabled:
            pass
        elif self.preview_renderer:
            self._preview_update()
        elif not (self.ros_navigation and self.ros_navigation.active) and time.monotonic() - self.camera_published_at >= self.camera_interval_s:
            self._publish_camera()
        if not ros_active:
            self._sample_spatial()
        if self.recorder:
            self.recorder.capture(self)
        if self.pace and not self.renderer and not ros_active:
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

    def _preview_update(self):
        import pybullet as bullet
        from backend.camera import scene_snapshot
        if self.latest.get("rendering_error"):
            return
        try:
            frame = self.preview_renderer.latest()
        except Exception as error:
            self._camera_failed(error)
            return
        if frame:
            metadata, image, captured_at = frame
            if (metadata["run_id"] == self.sim.run_id and 0 <= time.monotonic() - captured_at < .5
                    and metadata.get("stop_revision", self.stop_revision) == self.stop_revision
                    and metadata["simulated_s"] >= self.latest.get("camera", {}).get("simulated_time_s", 0)):
                self._publish_camera(image, metadata["simulated_s"])
        if time.monotonic() - self.preview_requested_at >= self.camera_interval_s:
            self.preview_requested_at = time.monotonic()
            if self.sim.rendering == "enhanced":
                self.preview_renderer.submit({"observation": {"run_id": self.sim.run_id, "simulated_s": self.sim.ticks / 240,
                    "stop_revision": self.stop_revision},
                    "captured_at": self.preview_requested_at, "packet": scene_snapshot(self.sim)})
                return
            self.preview_renderer.submit({"observation": {"run_id": self.sim.run_id, "simulated_s": self.sim.ticks / 240},
                "captured_at": self.preview_requested_at, "dimensions": (self.sim.width, self.sim.height),
                "bodies": [(body, *bullet.getBasePositionAndOrientation(body, physicsClientId=self.sim.client))
                    for body in [item["id"] for item in self.sim.objects] + [self.sim.robot]],
                "joints": [(name, bullet.getJointState(self.sim.robot, index, physicsClientId=self.sim.client)[0])
                    for name, index in self.sim.joints.items()]})

    def _render_update(self):
        import pybullet as bullet
        from backend.camera import scene_snapshot
        if self.latest.get("rendering_error"):
            return
        try:
            frame = self.renderer.latest()
        except Exception as error:
            self._camera_failed(error)
            return
        if frame:
            data, image, captured_at = frame
            observation = AgentObservation.model_validate(data)
            if (observation.run_id == self.sim.run_id and observation.episode_epoch == self.sim.epoch
                    and 0 <= time.monotonic() - captured_at <= 1.
                    and observation.simulated_time_s >= self.latest.get("camera", {}).get("simulated_time_s", 0)):
                self.render_frame = observation, image, captured_at
                self._publish_camera(image, observation.simulated_time_s)
        if time.monotonic() - self.render_requested_at >= .05:
            self.render_requested_at = time.monotonic()
            observation = self.sim.observe(render=False)
            if self.sim.rendering == "enhanced":
                self.renderer.submit({"observation": observation.model_dump(), "captured_at": self.render_requested_at,
                    "packet": scene_snapshot(self.sim, 320, 240)})
                return
            self.renderer.submit({"observation": observation.model_dump(), "captured_at": self.render_requested_at,
                "bodies": [(body, *bullet.getBasePositionAndOrientation(body, physicsClientId=self.sim.client))
                           for body in [item["id"] for item in self.sim.objects] + [self.sim.robot]],
                "joints": [(name, bullet.getJointState(self.sim.robot, index, physicsClientId=self.sim.client)[0])
                           for name, index in self.sim.joints.items()]})

    def _camera_failed(self, error):
        self.stop()
        reason = f"RENDERER_FAILED: {error}"
        self.latest = {**self.latest, "rendering_error": reason}
        if self.skill and self.skill.active:
            self.skill.fail(self.sim, reason)
            self._publish_skill()
        if self.navigation and self.navigation.status in {"running", "awaiting_feedback"}:
            self.navigation.fail(self.sim, reason)
            self._publish_navigation()

    def _run(self):
        try:
            self.sim = BulletSimulation(epoch=self.epoch, scene=self.scene, challenge=self.challenge, rendering=self.rendering)
            if self.sim.rendering == "enhanced":
                from backend.camera import EnhancedResources, snapshot_renderer
                if self.render_resources is None:
                    self.render_resources = EnhancedResources()
                self.spatial_processor.shutdown(wait=True)
                self.spatial_processor = self.render_resources.processor
                self.sim.camera_renderer = self.render_resources.renderer
                self.sim.owns_camera_renderer = False
                self.preview_renderer = snapshot_renderer(self.sim)
            observation = self.sim.observe()
            self.latest = {"run_id": self.sim.run_id, "episode_epoch": self.epoch, "observation": observation.model_dump(),
                           "snapshot": self.sim.snapshot(), "geometry": self.sim.geometry(), "busy": False, "stopped": False,
                           "assisted": False, "result": None, "challenge": self.sim.challenge_status(),
                           "navigation": None, "rendering": self.sim.rendering,
                           "robot_body_id": self.sim.robot, "manual_placements": 0, "proximity": observation.proximity.model_dump()}
            self._publish_camera(self.sim.frame(observation.frame_ref))
            self.sim.on_tick = self._publish
            self.ready.set_result(True)
            while True:
                if self.home_mission:
                    self.home_mission.tick()
                self.motion_throughput.observe(self.sim.ticks * TIMESTEP, time.monotonic(),
                    bool(self.continuous and self.continuous.active and self.navigation and self.navigation.buffer))
                if (self.ros_navigation and self.ros_navigation.active
                    and (self.sim.cancel.is_set() or not self.pace or time.monotonic() >= self.ros_tick_at)):
                    self.ros_tick_at = time.monotonic() + .05
                    try:
                        self.ros_navigation.tick(self.sim, self.stop_revision, task_revision=self.task_revision)
                    except Exception:
                        self.ros_navigation.finish(self.sim, "blocked", "ROS executor failed")
                        self.sim.stop()
                    self._publish_navigation()
                if self.continuous and self.continuous.active and self.sim.cancel.is_set():
                    self.continuous.finish(self.sim, self.navigation, "cancelled", "Stopped by operator")
                    self._publish_navigation()
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
                    if self.ros_navigation and self.ros_navigation.active:
                        queue_timeout = max(0., self.ros_tick_at - time.monotonic()) if self.pace else .01
                    else:
                        queue_timeout = 0 if buffered else .01 if skill_active or rendering else .05 if active or (self.home_mission and self.home_mission.active) else .25 if self.spatial_enabled else None
                    work = self.queue.get(timeout=queue_timeout)
                except Empty:
                    if self.ros_navigation and self.ros_navigation.active:
                        continue
                    if not active and not skill_active and not rendering:
                        if self.preview_renderer and not (self.sim.rendering == "enhanced" and self.spatial_enabled):
                            self._preview_update()
                        self._sample_spatial()
                        if self.home_mission:
                            self.home_mission.sample()
                        if self.recorder:
                            self.recorder.capture(self)
                        continue
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
                        if self.continuous and self.continuous.active:
                            self._update_continuous()
                        self.navigation.tick(self.sim)
                        if self.continuous and self.continuous.active and self.navigation.status in {"failed", "cancelled"}:
                            self.continuous.status, self.continuous.reason = "blocked", self.navigation.reason
                    except Exception as error:
                        self.navigation.fail(self.sim, "EXECUTION_FAILED: Navigation worker stopped.")
                        if self.continuous and self.continuous.active:
                            self.continuous.status, self.continuous.reason = "blocked", f"Execution stopped: {type(error).__name__}"
                    self._publish_navigation()
                    continue
                if work is None:
                    break
                if self.ros_navigation and self.ros_navigation.active and self.sim.cancel.is_set():
                    self.ros_navigation.finish(self.sim, "cancelled", "Stopped by operator")
                    self._publish_navigation()
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
            if self.sim and self.ros_navigation and self.ros_navigation.active:
                self.ros_navigation.finish(self.sim, "cancelled", "Physics worker closed")
            if self.renderer:
                self.renderer.close()
            if self.preview_renderer:
                self.preview_renderer.close()
            if self.spatial_capture_executor:
                self.spatial_capture_executor.shutdown(wait=True, cancel_futures=True)
            if self.spatial_pending:
                try:
                    self.spatial_pending.result(timeout=5)
                except Exception:
                    self.spatial_pending.cancel()
            if self.render_resources:
                if self.owns_render_resources:
                    self.render_resources.close()
            else:
                self.spatial_processor.shutdown(wait=True, cancel_futures=True)
            if self.sim:
                self.sim.close()
            self.camera_frames.clear()
            self.spatial_frames.clear()
            self.camera_history = None

    def _receive_spatial(self, wait=False):
        if self.spatial_pending is None or (not wait and not self.spatial_pending.done()):
            return
        pending, source = self.spatial_pending, self.spatial_pending_map
        self.spatial_pending = None
        self.spatial_pending_map = None
        try:
            mapped, observation, image, depth = pending.result(timeout=1.)
            if source is not self.spatial_map or not self.spatial_enabled:
                return
            if self.spatial_pending_stop_revision is not None and self.spatial_pending_stop_revision != self.stop_revision:
                return
            received_at = time.monotonic()
            age = received_at - observation.captured_at
            self.spatial_processing_max_s = max(self.spatial_processing_max_s, age)
            self.spatial_timing["post_capture_max_s"] = max(self.spatial_timing["post_capture_max_s"],
                received_at - self.spatial_submitted_at)
            if not 0 <= age <= mapped.max_frame_age_s:
                self.spatial_timing["expired_results"] += 1
                raise ValueError("Spatial processing result expired")
            self.spatial_timing["accepted_results"] += 1
            self.spatial_map = mapped
            self.spatial_frames[observation.sequence] = (observation, image, depth)
            while len(self.spatial_frames) > 8:
                self.spatial_frames.popitem(last=False)
            self._camera_history().record(observation, image)
            if self.sim.rendering == "enhanced" and self.home_mission and self.home_mission.home:
                self._publish_camera(image, observation.simulated_time_s)
            self.spatial_error = None
        except TimeoutError:
            self.spatial_pending, self.spatial_pending_map = pending, source
            self.spatial_timing["processing_timeouts"] += 1
            self.spatial_error = "Spatial processor has not returned a fresh frame"
        except Exception as error:
            self.spatial_error = str(error)

    def _receive_mapped_capture(self, wait=False):
        from backend.camera import process_spatial
        pending = self.spatial_capture_pending
        if pending is None or (not wait and not pending.done()):
            return
        try:
            observation, image = pending.result(timeout=1.)
        except TimeoutError:
            return
        except Exception as error:
            self.spatial_capture_pending = None
            self.spatial_error = f"RENDERER_FAILED: {error}"
            self._camera_failed(error)
            return
        self.spatial_capture_pending = None
        if (not self.spatial_enabled or self.spatial_capture_generation != self.spatial_generation
                or self.spatial_capture_stop_revision != self.stop_revision
                or observation.run_id != self.sim.run_id or observation.episode_epoch != self.sim.epoch):
            return
        age = time.monotonic() - observation.captured_at
        if not 0 <= age <= 1.:
            self.spatial_error = "Spatial capture expired"
            self.spatial_timing["expired_results"] += 1
            return
        self.mapped_depth_sensor = observation
        self._publish_camera(image, observation.simulated_time_s)
        if self.home_mission:
            self.home_mission.observe_depth(observation)
        if self.spatial_pending is not None:
            return
        self.spatial_pending_map = self.spatial_map
        self.spatial_pending_stop_revision = self.stop_revision
        self.spatial_submitted_at = time.monotonic()
        self.spatial_pending = self.spatial_processor.submit(process_spatial, self.spatial_map, observation, image)

    def _sample_spatial(self, force=False):
        from backend.camera import process_spatial
        self._receive_spatial(wait=force)
        self._receive_mapped_capture(wait=force)
        if force:
            self._receive_spatial(wait=True)
        mapped_capture = not force and self.sim.rendering == "enhanced" and self.home_mission and self.home_mission.home
        if not self.spatial_enabled or self.renderer or self.spatial_capture_pending is not None or (self.spatial_pending is not None and not mapped_capture):
            return
        if not force and time.monotonic() - self.spatial_sampled_at < .2:
            return
        self.spatial_sampled_at = time.monotonic()
        self.spatial_sequence += 1
        if mapped_capture:
            import pybullet as bullet
            from backend.camera import capture_spatial_snapshot, scene_snapshot
            from backend.spatial import calibration
            if self.spatial_capture_executor is None:
                self.spatial_capture_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mapped-rgbd")
            intrinsics = calibration(160, 120)
            metadata = {"run_id": self.sim.run_id, "episode_epoch": self.sim.epoch, "sequence": self.spatial_sequence,
                "captured_at": self.spatial_sampled_at, "simulated_time_s": self.sim.ticks * TIMESTEP,
                "calibration": intrinsics.model_dump(), "odometry_m_rad": self.sim.odometry.tolist(),
                "head_rad": [bullet.getJointState(self.sim.robot, self.sim.joints[name], physicsClientId=self.sim.client)[0]
                    for name in ("head_yaw", "head_pitch")]}
            packet = scene_snapshot(self.sim, intrinsics.width, intrinsics.height)
            self.spatial_capture_stop_revision = self.stop_revision
            self.spatial_capture_generation = self.spatial_generation
            self.spatial_submitted_at = time.monotonic()
            self.spatial_capture_pending = self.spatial_capture_executor.submit(capture_spatial_snapshot,
                self.sim.camera_renderer, packet, metadata)
            return
        self.spatial_pending_stop_revision = None
        try:
            observation, image = self.sim.capture_spatial(self.spatial_sequence)
        except Exception as error:
            if self.sim.rendering != "enhanced":
                raise
            self.spatial_error = f"RENDERER_FAILED: {error}"
            self._camera_failed(error)
            return
        self.spatial_submitted_at = time.monotonic()
        self.spatial_timing["capture_max_s"] = max(self.spatial_timing["capture_max_s"],
            self.spatial_submitted_at - observation.captured_at)
        if self.sim.rendering == "enhanced":
            self._publish_camera(image, observation.simulated_time_s)
            if self.home_mission and self.home_mission.home:
                self.mapped_depth_sensor = observation
                self.home_mission.observe_depth(observation)
        self.spatial_pending_map = self.spatial_map
        self.spatial_pending = self.spatial_processor.submit(process_spatial, self.spatial_map, observation, image)
        if force:
            self._receive_spatial(wait=True)

    def _camera_history(self):
        from backend.visual_history import CameraHistory
        if self.camera_history is None:
            self.camera_history = CameraHistory(self.sim.run_id, self.sim.epoch)
        return self.camera_history

    def _camera_history_guard(self, sim, stop_revision):
        if self.stop_revision != stop_revision or sim.cancel.is_set():
            raise MotionError("CANCELLED", "Camera history request expired or was stopped")

    async def camera_history_review(self, sensor, stop_revision):
        def operation(sim):
            self._camera_history_guard(sim, stop_revision)
            if sensor.run_id != sim.run_id or sensor.episode_epoch != sim.epoch:
                raise MotionError("STALE_STATE", "Camera history belongs to another episode")
            return self._camera_history().review(sensor.sequence)
        return await self.call(operation)

    async def acknowledge_camera_history(self, review, stop_revision):
        def operation(sim):
            self._camera_history_guard(sim, stop_revision)
            self._camera_history().acknowledge(review["generation"], review["through_sequence"])
        return await self.call(operation)

    async def inspect_camera_history(self, frame_id, stop_revision):
        def operation(sim):
            self._camera_history_guard(sim, stop_revision)
            return self._camera_history().original(frame_id)
        return await self.call(operation)

    def _history(self):
        from backend.spatial import MapHistory
        if self.map_history is None:
            self.map_history = MapHistory(self.sim.run_id, self.sim.epoch)
        return self.map_history

    def _record_map_pose(self):
        history = self._history()
        if not history.needs_pose(self.sim.odometry):
            return
        bounds = self.sim.robot_footprint()
        lower, upper = bounds["lower_xy_m"], bounds["upper_xy_m"]
        history.record_pose(self.sim.odometry, [[lower[0], lower[1]], [upper[0], lower[1]],
            [upper[0], upper[1]], [lower[0], upper[1]]])

    async def place_sightings(self):
        return await self.call(lambda sim: self._history().public()["labels"])

    async def record_place_sighting(self, observation, sighting, stop_revision):
        def operation(sim):
            if self.stop_revision != stop_revision or sim.cancel.is_set():
                raise MotionError("CANCELLED", "Place sighting reply was stopped")
            age = time.time() - observation.wall_timestamp
            if not 0 <= age <= 45:
                raise ValueError("Place sighting observation expired")
            return self._history().record_label(observation, sighting.label, sighting.evidence, time.monotonic() - age)
        return await self.call(operation)

    def spatial_state(self):
        current = next(reversed(self.spatial_frames.values()), None) if self.spatial_frames else None
        metadata = current[0].model_dump(exclude={"depth_m"}) if current else None
        if metadata:
            prefix = f"/api/spatial/{self.sim.run_id}/{metadata['sequence']}"
            metadata.update(rgb_url=prefix + "/rgb.png", depth_url=prefix + "/depth.png", data_url=prefix + "/depth.json")
        return {"enabled": self.spatial_enabled, "paused": self.renderer is not None, "error": self.spatial_error,
            "processing_pending": self.spatial_pending is not None, "maximum_processing_age_s": self.spatial_processing_max_s,
            "timing": dict(self.spatial_timing),
            "frame": metadata, "map": self.spatial_map.public() if self.spatial_map else None,
            "history": self._history().public(),
            "continuous": self.continuous.state() if self.continuous else None,
            "footprint": self.sim.robot_footprint() if self.spatial_enabled else None}

    def _continuous_guard(self, sim, request, stop_revision):
        if self.home_mission and self.home_mission.active:
            raise MotionError("CONTROL_CONFLICT", "Cancel the home mission before selecting another controller")
        if request.run_id != sim.run_id or request.episode_epoch != sim.epoch or self.stop_revision != stop_revision or sim.cancel.is_set():
            raise MotionError("CANCELLED", "Continuous navigation request expired or was stopped")
        if self.renderer or (self.skill and self.skill.active) or (self.navigation and self.navigation.status in {"running", "awaiting_feedback"}):
            raise MotionError("CONTROL_CONFLICT", "Take manual control before continuous navigation")

    async def scan_continuous(self, request):
        from backend.contracts import Command
        from backend.robot import TRAVEL
        from backend.spatial import ObservedMap
        from uuid import uuid4
        stop_revision = self.stop_revision
        def operation(sim):
            self._continuous_guard(sim, request, stop_revision)
            if sim.held or sim.proximity_sensors().collisions:
                raise MotionError("CONTROL_CONFLICT", "Release objects and clear contacts before scanning")
            if self.preview_renderer is None:
                from backend.camera import snapshot_renderer
                self.preview_renderer = snapshot_renderer(sim)
            self.latest = {**self.latest, "busy": True}
            self.spatial_enabled = True
            if self.spatial_map is None:
                self.spatial_map = ObservedMap(sim.run_id, sim.epoch)
            try:
                if request.compact_arms:
                    if sim.held or any(any(sensor.contact) for sensor in sim.observe(render=False).grippers.values()):
                        raise MotionError("OBJECT_HELD", "Release objects before folding the arms")
                    for side in ("left", "right"):
                        result = sim.execute(Command(run_id=sim.run_id, episode_epoch=sim.epoch, observation_seq=sim.seq,
                            action_id=str(uuid4()), tool="set_arm_joints", arguments={"arm": side, "joint_positions_rad": TRAVEL, "duration_s": 2.}))
                        if result.status != "ok":
                            raise MotionError(result.error or "POSTURE_FAILED", result.message)
                for yaw, pitch in [(-.9, .85), (0., .85), (.9, .85), (0., .45)]:
                    result = sim.execute(Command(run_id=sim.run_id, episode_epoch=sim.epoch, observation_seq=sim.seq,
                        action_id=str(uuid4()), tool="set_head", arguments={"yaw_rad": yaw, "pitch_rad": pitch, "duration_s": 1.}))
                    if result.status != "ok":
                        raise MotionError(result.error or "SCAN_FAILED", result.message)
                    self._sample_spatial(force=True)
                observation, image = self._feedback(sim)
                self._publish_camera(image)
                self.spatial_error = None
                return self.spatial_state()
            finally:
                self.latest = {**self.latest, "busy": False, "snapshot": sim.snapshot(), "stopped": sim.cancel.is_set()}
        return await self.call(operation)

    async def start_continuous(self, request, selected_sensor=None, selected_target=None, exploration=False):
        from backend.continuous_navigation import target_from_depth
        from backend.navigation_backends import ConventionalBackend, NavigationFrame, NavigationGoal
        stop_revision = self.stop_revision
        def operation(sim):
            self._continuous_guard(sim, request, stop_revision)
            retained = self.spatial_frames.get(request.spatial_sequence)
            if not self.spatial_enabled or self.spatial_map is None or (selected_sensor is None and retained is None):
                raise MotionError("SPATIAL_REQUIRED", "Scan the floor and select a current depth image")
            sensor = selected_sensor if selected_sensor is not None else retained[0]
            max_age = 45. if selected_sensor is not None else 1.
            if sensor.run_id != sim.run_id or sensor.episode_epoch != sim.epoch or not 0 <= time.monotonic() - sensor.captured_at <= max_age or np.linalg.norm(np.array(sensor.odometry_m_rad) - sim.odometry) > .05:
                raise MotionError("SPATIAL_STALE", "Camera selection expired; choose a point in the current image")
            self._sample_spatial(force=True)
            if self.spatial_error:
                raise MotionError("SPATIAL_STALE", self.spatial_error)
            target = target_from_depth(sensor, request.pixel) if selected_target is None else selected_target
            if selected_target is not None and (selected_sensor is None or len(target) != 2 or not np.isfinite(target).all()
                    or not .15 <= np.linalg.norm(np.array(target) - sim.odometry[:2]) <= 1.8):
                raise ValueError("Invalid observed-map destination")
            footprint = sim.robot_footprint()
            goal = NavigationGoal(kind="point", evidence="depth_pixel" if selected_target is None else "observed_map", point_m=list(target))
            frame = NavigationFrame(run_id=sim.run_id, episode_epoch=sim.epoch, sequence=self.spatial_map.sequence,
                captured_at=self.spatial_map.captured_at, stop_revision=stop_revision, goal_id=goal.goal_id,
                odometry_m_rad=sim.odometry.tolist(), head_rad=self.spatial_map.observation.head_rad)
            backend = ConventionalBackend()
            proposal = backend.propose(frame, goal, self.spatial_map, footprint["radius_m"])
            allowed = self.spatial_map.traversable(sim.odometry[:2], footprint["radius_m"])
            path = proposal.checked_path(frame, goal, backend.capabilities, time.monotonic(),
                lambda points: self.spatial_map.contains_path(allowed, np.concatenate([
                    np.linspace(begin, end, max(2, int(np.ceil(np.linalg.norm(end - begin) / .025)) + 1))
                    for begin, end in zip(points, points[1:])], axis=0)))
            heading = float(np.arctan2(target[1] - sim.odometry[1], target[0] - sim.odometry[0])) if exploration else None
            return self._start_navigation_path(path, selected_sensor is not None, exploration_heading=heading)
        return await self.call(operation)

    def _start_navigation_path(self, path, recover_clearance=False, exploration_heading=None, ai_route=False, home_owned=False):
        from backend.continuous_navigation import ContinuousNavigation, ObservedExploration
        sim = self.sim
        self.navigation = NavigationRuntime()
        self.navigation.recover_clearance = recover_clearance
        self.navigation.recover_deadline = recover_clearance
        self.continuous = ContinuousNavigation(path)
        self.continuous.home_owned = home_owned
        self.continuous.ai_route = ai_route
        self.continuous.start_position = sim.odometry[:2].copy()
        self.continuous.radius = sim.robot_footprint()["radius_m"]
        self.continuous.preview_dimensions = (sim.width, sim.height)
        sim.width, sim.height = 320, 240
        self.continuous.start(sim, self.navigation)
        if ai_route:
            self.navigation.authorized_travel_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) + .15
        if exploration_heading is not None:
            self.continuous.exploration = ObservedExploration(exploration_heading)
            self.continuous.exploration_stop_revision = self.stop_revision
            self.continuous.exploration_run_id = sim.run_id
            self.continuous.exploration_epoch = sim.epoch
            self.navigation.skill_deadline = self.continuous.exploration.expires_at
            self.navigation.authorized_travel_m = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) + .15
        self.wall_start, self.tick_start = time.monotonic(), sim.ticks
        self._update_continuous()
        self._publish_navigation()
        return self.spatial_state()

    async def capture_navigation_frame(self, goal):
        from backend.navigation_backends import NavigationFrame
        revision = self.stop_revision
        def operation(sim):
            if sim.cancel.is_set() or self.stop_revision != revision or (self.continuous and self.continuous.active) or self.renderer or (self.skill and self.skill.active) or (self.navigation and self.navigation.status in {"running", "awaiting_feedback"}):
                raise MotionError("CONTROL_CONFLICT", "Policy capture requires a stopped, available base")
            if not self.spatial_enabled or self.spatial_map is None:
                raise MotionError("SPATIAL_REQUIRED", "Scan before capturing a navigation policy frame")
            self._sample_spatial(force=True)
            if self.spatial_error:
                raise MotionError("SPATIAL_STALE", self.spatial_error)
            sensor, image, _ = next(reversed(self.spatial_frames.values()))
            self.proposed_goal_id = goal.goal_id
            ticket = NavigationFrame(run_id=sim.run_id, episode_epoch=sim.epoch, sequence=sensor.sequence,
                captured_at=sensor.captured_at, stop_revision=revision, goal_id=goal.goal_id,
                odometry_m_rad=sensor.odometry_m_rad, head_rad=sensor.head_rad)
            return ticket, sensor, image
        return await self.call(operation)

    async def start_navigation_proposal(self, proposal, goal, capabilities):
        from backend.navigation_backends import NavigationFrame
        def operation(sim):
            self._continuous_guard(sim, proposal.ticket, proposal.ticket.stop_revision)
            if self.proposed_goal_id != goal.goal_id:
                raise MotionError("STALE_PLAN", "Policy goal was superseded")
            observation = sim.observe(render=False)
            current = NavigationFrame(run_id=sim.run_id, episode_epoch=sim.epoch, sequence=self.spatial_sequence,
                captured_at=time.monotonic(), stop_revision=self.stop_revision, goal_id=self.proposed_goal_id,
                odometry_m_rad=sim.odometry.tolist(), head_rad=observation.head_rad)
            if self.spatial_map is None or time.monotonic() - self.spatial_map.captured_at > 1.:
                raise MotionError("SPATIAL_STALE", "A current clearance map is required")
            allowed = self.spatial_map.traversable(sim.odometry[:2], sim.robot_footprint()["radius_m"])
            path = proposal.checked_path(current, goal, capabilities, time.monotonic(),
                lambda points: self.spatial_map.contains_path(allowed, np.concatenate([
                    np.linspace(begin, end, max(2, int(np.ceil(np.linalg.norm(end - begin) / .025)) + 1))
                    for begin, end in zip(points, points[1:])], axis=0)))
            self.proposed_goal_id = None
            result = self._start_navigation_path(path)
            self.navigation.skill_deadline = min(self.navigation.skill_deadline, proposal.ticket.captured_at + goal.timeout_s)
            if capabilities.fixed_head:
                self.continuous.policy_head_rad = proposal.ticket.head_rad
            return result
        return await self.call(operation)

    async def locate_observed_orbit(self, sensor, bounds, label, direction):
        from backend.continuous_navigation import ObservedOrbit
        from backend.robot import TRAVEL
        import pybullet as bullet
        revision = self.stop_revision
        def operation(sim):
            self._continuous_guard(sim, sensor, revision)
            if sim.held or any(any(sensor.contact) for sensor in sim.observe(render=False).grippers.values()):
                raise ValueError("Release held objects before circling furniture")
            if any(abs(bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0] - target) > .1
                   for side in sim.arms for joint, target in zip(sim.arms[side], TRAVEL)):
                raise ValueError("Stow both arms before circling furniture")
            if not 0 <= time.monotonic() - sensor.captured_at <= 45 or np.linalg.norm(np.array(sensor.odometry_m_rad) - sim.odometry) > .05:
                raise ValueError("Object image expired or the base moved; identify it in a fresh view")
            return ObservedOrbit(sensor, bounds, label, direction, sim.robot_footprint()["radius_m"])
        return await self.call(operation)

    async def start_observed_orbit_step(self, orbit, stop_revision):
        def operation(sim):
            self._continuous_guard(sim, orbit, stop_revision)
            if time.monotonic() - orbit.started > 180:
                raise MotionError("SKILL_TIMEOUT", "Observed circuit exceeded its 180-second mission limit")
            self._sample_spatial(force=True)
            self._continuous_guard(sim, orbit, stop_revision)
            if self.spatial_map is None or self.spatial_error or self.spatial_map.public()["stale"]:
                raise MotionError("SPATIAL_STALE", "A fresh observed map is required for each circuit arc")
            path, complete = orbit.path(sim.odometry.tolist())
            progress = {"object": orbit.label, "direction": orbit.direction, "observed_center_m": orbit.center.tolist(),
                "radius_m": orbit.radius, "swept_degrees": orbit.swept * 180 / np.pi, "complete": complete}
            if complete:
                return progress
            radius = sim.robot_footprint()["radius_m"]
            if len(path) == 2:
                path = self.spatial_map.plan(sim.odometry[:2], path[-1], radius)
            allowed = self.spatial_map.traversable(sim.odometry[:2], radius)
            points = np.concatenate([np.linspace(begin, end, max(2, int(np.ceil(np.linalg.norm(np.array(end) - begin) / .025)) + 1))
                                     for begin, end in zip(path, path[1:])])
            length = sum(np.linalg.norm(np.array(end) - begin) for begin, end in zip(path, path[1:]))
            if length > 1.8 or not self.spatial_map.contains_path(allowed, points):
                raise MotionError("OBSERVED_PATH_BLOCKED", "Circuit arc is unknown or obstructed; inspect its next section")
            self._continuous_guard(sim, orbit, stop_revision)
            if time.monotonic() - orbit.started > 180:
                raise MotionError("SKILL_TIMEOUT", "Observed circuit exceeded its 180-second mission limit")
            self._start_navigation_path(path, recover_clearance=True)
            self.navigation.recover_deadline = False
            self.navigation.skill_deadline = min(self.navigation.skill_deadline, orbit.started + 180)
            if len(path) > 2:
                self.continuous.circuit_following = True
                self.continuous.observed_orbit = orbit
                self.continuous.orbit_stop_revision = stop_revision
                self.continuous.orbit_progress = progress
                self.continuous.orbit_final_arc = orbit.swept + .65 >= 2 * np.pi + .08
                self.continuous.orbit_last_attempt = -np.inf
                self.continuous.orbit_travel_start = orbit.travel_m
                self.continuous.orbit_travel_limit = 2 * np.pi * orbit.radius + 2. - orbit.travel_m
                self.navigation.authorized_travel_m = float(length) + .15
            return progress
        return await self.call(operation)

    def _continue_observed_orbit(self):
        control, runtime, sim = self.continuous, self.navigation, self.sim
        orbit = getattr(control, "observed_orbit", None)
        if orbit is None:
            return
        orbit.travel_m = control.orbit_travel_start + runtime.travel
        if orbit.travel_m >= 2 * np.pi * orbit.radius + 2.:
            control.finish(sim, runtime, "blocked", "SKILL_DISTANCE_LIMIT: Observed circuit distance budget exhausted")
            return
        now = time.monotonic()
        if (sim.cancel.is_set() or self.stop_revision != control.orbit_stop_revision
                or sim.run_id != orbit.run_id or sim.epoch != orbit.episode_epoch):
            control.finish(sim, runtime, "cancelled", "Circuit motion authority changed")
            return
        if now >= orbit.started + 180:
            control.finish(sim, runtime, "blocked", "SKILL_TIMEOUT: Observed circuit mission expired")
            return
        if control.orbit_final_arc:
            return
        if (not control.active or runtime.status != "running" or not runtime.buffer or now >= runtime.expires_at
                or now >= runtime.skill_deadline or self.spatial_map is None or self.spatial_map.public()["stale"]
                or self.spatial_error or now - control.orbit_last_attempt < .2):
            return
        if np.linalg.norm(control.path[-1] - sim.odometry[:2]) > .85:
            return
        control.orbit_last_attempt = now
        path, complete = orbit.path(sim.odometry.tolist())
        if complete or len(path) < 3:
            return
        radius = sim.robot_footprint()["radius_m"]
        if radius > control.radius + .01 or sim.held:
            control.finish(sim, runtime, "blocked", "FOOTPRINT_CHANGED: Circuit posture changed")
            return
        allowed = self.spatial_map.traversable(sim.odometry[:2], radius)
        points = np.concatenate([np.linspace(begin, end, max(2, int(np.ceil(np.linalg.norm(np.array(end) - begin) / .025)) + 1))
            for begin, end in zip(path, path[1:])])
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        authorization = runtime.travel + length + .15
        if length > 1.8 or authorization > control.orbit_travel_limit or not self.spatial_map.contains_path(allowed, points):
            return
        if sim.cancel.is_set() or self.stop_revision != control.orbit_stop_revision:
            return
        control.continue_path(path)
        control.start_position = sim.odometry[:2].copy()
        control.orbit_final_arc = orbit.swept + .65 >= 2 * np.pi + .08
        control.orbit_progress = {"object": orbit.label, "direction": orbit.direction, "observed_center_m": orbit.center.tolist(),
            "radius_m": orbit.radius, "swept_degrees": orbit.swept * 180 / np.pi, "complete": False, "handoffs": control.handoffs}
        runtime.authorized_travel_m = authorization
        runtime.skill_deadline = min(now + 60, orbit.started + 180)

    def _continue_exploration(self):
        control, runtime, sim = self.continuous, self.navigation, self.sim
        intent = control.exploration
        if intent is None:
            return
        now = time.monotonic()
        if (sim.cancel.is_set() or self.stop_revision != control.exploration_stop_revision
                or sim.run_id != control.exploration_run_id or sim.epoch != control.exploration_epoch):
            control.finish(sim, runtime, "cancelled", "Exploration motion authority changed")
            return
        if now >= intent.expires_at or runtime.travel >= intent.travel_limit:
            control.finish(sim, runtime, "blocked", "EXPLORATION_REVIEW_EXPIRED: Fresh supervisor review required")
            return
        if (intent.ending or not control.active or runtime.status != "running" or not runtime.buffer
                or now >= runtime.expires_at or now >= runtime.skill_deadline or self.spatial_map is None
                or self.spatial_map.public()["stale"] or self.spatial_error or now - intent.last_attempt < .25
                or np.linalg.norm(control.path[-1] - sim.odometry[:2]) > 1.):
            return
        intent.last_attempt = now
        radius = sim.robot_footprint()["radius_m"]
        if radius > control.radius + .01 or sim.held:
            control.finish(sim, runtime, "blocked", "FOOTPRINT_CHANGED: Exploration posture changed")
            return
        allowed = self.spatial_map.traversable(sim.odometry[:2], radius)
        for path in intent.paths(sim.odometry):
            points = np.concatenate([np.linspace(begin, end, max(2, int(np.ceil(np.linalg.norm(end - begin) / .025)) + 1))
                for begin, end in zip(path, path[1:])])
            authorization = runtime.travel + float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) + .15
            if authorization > intent.travel_limit or not self.spatial_map.contains_path(allowed, points):
                continue
            if sim.cancel.is_set() or self.stop_revision != control.exploration_stop_revision:
                return
            control.continue_path(path)
            control.start_position = sim.odometry[:2].copy()
            runtime.authorized_travel_m = authorization
            return

    async def review_exploration(self, proposal, candidate_id=None, ending=False):
        def operation(sim):
            control = self.continuous
            if (sim.cancel.is_set() or self.stop_revision != proposal["stop_revision"] or sim.run_id != proposal["run_id"]
                    or sim.epoch != proposal["epoch"] or control is None or not control.active
                    or control.identity != proposal["identity"] or control.exploration is None
                    or control.exploration.revision != proposal.get("intent_revision")):
                raise ValueError("Exploration proposal lost its motion authority")
            intent, runtime = control.exploration, self.navigation
            if ending:
                intent.ending = True
                intent.revision += 1
                return control.state()
            now = time.monotonic()
            if (self.spatial_map is None or self.spatial_map.public()["stale"] or self.spatial_error
                    or runtime.status != "running" or not runtime.buffer or now >= runtime.expires_at
                    or now >= runtime.skill_deadline):
                raise ValueError("Fresh sensing and active exploration authorization are required")
            heading = intent.heading
            if candidate_id is not None:
                choice = next((item for item in proposal["candidates"] if item["id"] == candidate_id), None)
                if choice is None:
                    raise ValueError("Unknown exploration candidate")
                delta = np.asarray(choice["target_m"]) - proposal["position"][:2]
                heading = float(np.arctan2(delta[1], delta[0]))
            difference = np.arctan2(np.sin(heading - sim.odometry[2]), np.cos(heading - sim.odometry[2]))
            if abs(difference) > .65:
                raise ValueError("Exploration direction needs a stopped review before a sharp turn")
            intent.renew(heading, proposal["captured_at"], proposal["captured_travel_m"], now, runtime.travel)
            runtime.skill_deadline = intent.expires_at
            self._publish_navigation()
            return control.state()
        return await self.call(operation)

    async def pause_continuous(self, request, stop_revision, task_revision, reason):
        def operation(sim):
            if request.run_id != sim.run_id or request.episode_epoch != sim.epoch or self.stop_revision != stop_revision or sim.cancel.is_set():
                raise MotionError("CANCELLED", "Continuous pause belongs to an expired or stopped request")
            if task_revision != self.task_revision:
                raise MotionError("STALE_PLAN", "Decision belongs to a superseded instruction")
            if self.continuous and self.continuous.active:
                self.continuous.finish(sim, self.navigation, "cancelled", reason)
                self._publish_navigation()
        await self.call(operation)

    async def start_motion_skills(self, sensor, plan, candidates, stop_revision, task_revision):
        from backend.continuous_navigation import ContinuousScan, MotionSkillBudgetError, compile_motion_skills, motion_skill_timing
        request = ContinuousScan(run_id=sensor.run_id, episode_epoch=sensor.episode_epoch)

        def operation(sim):
            self._continuous_guard(sim, request, stop_revision)
            if task_revision != self.task_revision or not 0 <= time.monotonic() - sensor.captured_at <= 15.:
                raise MotionError("STALE_PLAN", "Skill plan belongs to an expired observation or instruction")
            if np.linalg.norm(sim.odometry - sensor.odometry_m_rad) > .05 or sim.held:
                raise MotionError("STALE_PLAN", "Skill plan requires the unchanged observed pose and free grippers")
            path, components = compile_motion_skills(sensor.odometry_m_rad, plan, candidates)
            throughput = self.motion_throughput.state()

            def estimate(phase):
                try:
                    timing = motion_skill_timing(path, sensor.captured_at, time.monotonic(),
                        wall_seconds_per_sim_second=throughput["wall_seconds_per_sim_second"])
                except MotionSkillBudgetError as error:
                    error.timing.update(throughput=throughput, validation_phase=phase,
                        path_length_m=components[-1]["distance_m"])
                    raise
                return {**timing, "throughput": throughput, "validation_phase": phase,
                    "path_length_m": components[-1]["distance_m"]}

            estimate("before_depth")
            self._sample_spatial(force=True)
            self._continuous_guard(sim, request, stop_revision)
            if task_revision != self.task_revision or not 0 <= time.monotonic() - sensor.captured_at <= 15.:
                raise MotionError("STALE_PLAN", "Skill plan expired during depth validation")
            if self.spatial_map is None or self.spatial_error or self.spatial_map.public()["stale"]:
                raise MotionError("SPATIAL_STALE", "Skill plans require fresh observed floor")
            allowed = self.spatial_map.traversable(sim.odometry[:2], sim.robot_footprint()["radius_m"])
            if not self.spatial_map.contains_path(allowed, path):
                raise MotionError("OBSERVED_PATH_BLOCKED", "Skill plan crosses unknown or obstructed floor")
            self._continuous_guard(sim, request, stop_revision)
            timing = estimate("after_depth")
            self._start_navigation_path(path, recover_clearance=True, ai_route=True)
            self.continuous.skill_components = components
            self.continuous.skill_timing = timing
            self.continuous.reason = "Following locally compiled motion skills"
            self.navigation.skill_deadline = min(self.navigation.skill_deadline, sensor.captured_at + 20.)
            self._publish_navigation()
            return self.continuous.state()

        return await self.call(operation)

    async def ai_route_feedback(self):
        import copy
        def operation(sim):
            moving = bool(self.continuous and self.continuous.active)
            self._sample_spatial(force=not moving)
            if self.spatial_map is None or self.spatial_error or self.spatial_map.public()["stale"]:
                raise MotionError("SPATIAL_STALE", "Fresh observed floor is required for AI route feedback")
            self.spatial_sequence += 1
            sensor, image = sim.capture_spatial(self.spatial_sequence)
            self._camera_history().record(sensor, image)
            observation, image = self._feedback(sim, image=image)
            ticket = {"run_id": sim.run_id, "epoch": sim.epoch, "stop_revision": self.stop_revision,
                "task_revision": self.task_revision, "captured_at": sensor.captured_at,
                "position": sim.odometry.copy(), "identity": self.continuous.identity if moving else None,
                "travel": self.navigation.travel if moving else 0.}
            return observation, image, sensor, ticket, copy.deepcopy(self.spatial_map), sim.robot_footprint()["radius_m"]
        observation, image, sensor, ticket, mapped, radius = await self.call(operation)
        def support():
            allowed = mapped.traversable(ticket["position"][:2], radius)
            rows, columns = np.where(allowed[::5, ::5])
            points = (np.column_stack((columns * 5, rows * 5)) + mapped.origin + .5) * mapped.resolution_m
            near = np.linalg.norm(points - ticket["position"][:2], axis=1) <= 3.5
            return np.round(points[near], 3).tolist()
        points = await asyncio.to_thread(support)
        return observation, image, sensor, ticket, points

    async def apply_ai_route(self, ticket, waypoints):
        from backend.continuous_navigation import smooth_observed_route
        def operation(sim):
            now = time.monotonic()
            if (sim.cancel.is_set() or ticket["run_id"] != sim.run_id or ticket["epoch"] != sim.epoch
                    or ticket["stop_revision"] != self.stop_revision or ticket["task_revision"] != self.task_revision):
                raise MotionError("STALE_PLAN", "AI route belongs to a superseded command or episode")
            if not 0 <= now - ticket["captured_at"] <= 15.:
                raise MotionError("STALE_PLAN", "AI route response expired")
            control, runtime = self.continuous, self.navigation
            moving = bool(control and control.active)
            if ticket["identity"] is not None and (not moving or control.identity != ticket["identity"]):
                raise MotionError("STALE_PLAN", "The route stopped or changed during inference; review fresh feedback")
            if moving and (not control.ai_route or not runtime.buffer or runtime.status != "running"
                    or now >= runtime.expires_at or now >= runtime.skill_deadline):
                raise MotionError("STALE_PLAN", "Motion lease cannot be revived by a late AI route")
            if not moving and np.linalg.norm(sim.odometry - ticket["position"]) > .05:
                raise MotionError("STALE_PLAN", "Robot moved since stopped route feedback")
            if not moving:
                self._sample_spatial(force=True)
                if not 0 <= time.monotonic() - ticket["captured_at"] <= 15.:
                    raise MotionError("STALE_PLAN", "AI route expired during fresh depth validation")
            if self.spatial_map is None or self.spatial_error or self.spatial_map.public()["stale"]:
                raise MotionError("SPATIAL_STALE", "AI routes require a fresh observed map")
            radius = sim.robot_footprint()["radius_m"]
            if sim.held or (moving and radius > control.radius + .01):
                raise MotionError("FOOTPRINT_CHANGED", "AI route footprint changed")
            path = smooth_observed_route(ticket["position"], waypoints, moving=moving)
            if moving:
                lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
                travelled = runtime.travel - ticket["travel"]
                eligible = np.where((lengths >= max(0., travelled - .75)) & (lengths <= travelled + .75))[0]
                if not len(eligible):
                    raise ValueError("AI route has no remaining compatible continuation")
                nearest = eligible[np.argmin(np.linalg.norm(path[eligible] - sim.odometry[:2], axis=1))]
                if np.linalg.norm(path[nearest] - sim.odometry[:2]) > .6:
                    raise ValueError("AI route diverged from the current motion; review again")
                ahead = np.where(lengths >= lengths[nearest] + .4)[0]
                if not len(ahead):
                    raise ValueError("AI route endpoint already reached; review again")
                remaining = path[ahead[0]:]
                selected = np.linspace(0, len(remaining) - 1, min(17, max(2, len(remaining) // 6))).astype(int)
                path = smooth_observed_route(sim.odometry, remaining[selected].tolist(), moving=True)
            allowed = self.spatial_map.traversable(sim.odometry[:2], radius)
            if not self.spatial_map.contains_path(allowed, path):
                raise MotionError("OBSERVED_PATH_BLOCKED", "AI route crosses unknown or obstructed floor")
            if sim.cancel.is_set() or ticket["stop_revision"] != self.stop_revision or ticket["task_revision"] != self.task_revision:
                raise MotionError("STALE_PLAN", "AI route authority changed during validation")
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if moving:
                runtime.check_clearance(sim, *runtime.velocity)
                control.path = path
                control.index = 1
                control.revision += 1
                control.handoffs += 1
                control.start_position = sim.odometry[:2].copy()
                control.distance_m = float(np.linalg.norm(path[-1] - sim.odometry[:2]))
            else:
                self._start_navigation_path(path, recover_clearance=True, ai_route=True)
                control, runtime = self.continuous, self.navigation
            control.reason = "Following a validated AI-generated route"
            runtime.authorized_travel_m = runtime.travel + length + .15
            runtime.skill_deadline = ticket["captured_at"] + 20.
            self._publish_navigation()
            return control.state()
        return await self.call(operation)

    async def continuous_candidates(self, visited=(), floor_target=None):
        from scipy.ndimage import label, distance_transform_edt
        from backend.continuous_navigation import target_from_depth
        def operation(sim):
            self._sample_spatial(force=True)
            sensor, image, _ = next(reversed(self.spatial_frames.values()))
            radius = sim.robot_footprint()["radius_m"]
            allowed = self.spatial_map.traversable(sim.odometry[:2], radius)
            components, _ = label(allowed)
            clearance = distance_transform_edt(np.pad(self.spatial_map.cells == 0, 1, constant_values=False))[1:-1, 1:-1] * self.spatial_map.resolution_m
            origin_component = components[self.spatial_map.cell_index(sim.odometry[:2])]
            candidates = []
            if origin_component:
                for column in (.15, .32, .5, .68, .85):
                    points = []
                    for row in np.linspace(.12, .88, 24):
                        try:
                            target = target_from_depth(sensor, [column, float(row)])
                        except ValueError:
                            continue
                        cell = self.spatial_map.cell_index(target)
                        distance = np.linalg.norm(np.array(target) - sim.odometry[:2])
                        if all(0 <= value < self.spatial_map.size for value in cell) and components[cell] == origin_component and .3 < distance < 1.65:
                            points.append((distance, float(row), target))
                    for desired in (.8, 1.5):
                        if not points:
                            continue
                        distance, row, target = min(points, key=lambda point: abs(point[0] - desired))
                        if any(np.linalg.norm(np.array(target) - item["target_m"]) < .2 for item in candidates):
                            continue
                        try:
                            path = self.spatial_map.plan(sim.odometry[:2], target, radius)
                        except ValueError:
                            continue
                        candidates.append({"id": len(candidates), "pixel": [column, row], "distance_m": round(float(distance), 2),
                            "target_m": target, "path_length_m": round(sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:])), 2)})
                rows, columns = np.where((components == origin_component) & (self.spatial_map.cells == 0))
                mapped = (np.column_stack((columns, rows)) + self.spatial_map.origin + .5) * self.spatial_map.resolution_m
                relative = mapped - sim.odometry[:2]
                distances = np.linalg.norm(relative, axis=1)
                bearings = np.arctan2(relative[:, 1], relative[:, 0])
                for sector in range(12):
                    heading = sector * np.pi / 6
                    difference = np.arctan2(np.sin(bearings - heading), np.cos(bearings - heading))
                    selected = mapped[(np.abs(difference) < np.pi / 12) & (distances > .4) & (distances < 1.6)]
                    if not len(selected):
                        continue
                    novelty = np.min(np.linalg.norm(selected[:, None] - np.array(visited or [sim.odometry[:2]]), axis=2), axis=1)
                    for index in np.argsort(-novelty)[:5]:
                        target = selected[index].tolist()
                        if any(np.linalg.norm(np.array(target) - item["target_m"]) < .25 for item in candidates):
                            continue
                        try:
                            path = self.spatial_map.plan(sim.odometry[:2], target, radius)
                        except ValueError:
                            continue
                        candidates.append({"id": len(candidates), "pixel": None, "target_m": target,
                            "distance_m": round(float(np.linalg.norm(np.array(target) - sim.odometry[:2])), 2),
                            "path_length_m": round(sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:])), 2)})
                        break
                central = np.argsort(-clearance[rows, columns])
                added = 0
                for index in central:
                    target = mapped[index].tolist()
                    if not .2 <= distances[index] < 1.6 or any(np.linalg.norm(np.array(target) - item["target_m"]) < .2 for item in candidates):
                        continue
                    try:
                        path = self.spatial_map.plan(sim.odometry[:2], target, radius)
                    except ValueError:
                        continue
                    candidates.append({"id": len(candidates), "pixel": None, "target_m": target,
                        "distance_m": round(float(distances[index]), 2), "path_length_m": round(sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:])), 2)})
                    added += 1
                    if added == 3:
                        break
            goal = None
            if floor_target is not None:
                if floor_target.get("source") != "head_rgb_depth" or not 0 <= time.monotonic() - floor_target["captured_at"] <= 120:
                    raise ValueError("Observed floor target expired")
                goal = np.asarray(floor_target["center_m"], dtype=float)
                if goal.shape != (2,) or not np.isfinite(goal).all():
                    raise ValueError("Observed floor target must contain two finite coordinates")
                delta = goal - sim.odometry[:2]
                distance = float(np.linalg.norm(delta))
                for step in (min(distance, 1.5), min(distance, .9), min(distance, .5)):
                    if step < .15:
                        continue
                    target = (sim.odometry[:2] + delta * step / distance).tolist()
                    if any(np.linalg.norm(np.array(target) - item["target_m"]) < .04 for item in candidates):
                        continue
                    try:
                        path = self.spatial_map.plan(sim.odometry[:2], target, radius)
                    except ValueError:
                        continue
                    candidates.append({"id": len(candidates), "pixel": None, "target_m": target,
                        "distance_m": round(step, 3), "floor_target_id": floor_target["id"],
                        "path_length_m": round(sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:])), 3)})
            for candidate in candidates:
                delta = np.array(candidate["target_m"]) - sim.odometry[:2]
                angle = np.arctan2(delta[1], delta[0]) - sim.odometry[2]
                candidate["bearing_from_base_rad"] = round(float(np.arctan2(np.sin(angle), np.cos(angle))), 2)
                candidate["observed_clearance_m"] = round(float(clearance[self.spatial_map.cell_index(candidate["target_m"])]), 2)
                candidate["floor_rgb"] = self.spatial_map.floor_rgb(candidate["target_m"])
                if goal is not None:
                    candidate["goal_distance_m"] = round(float(np.linalg.norm(np.array(candidate["target_m"]) - goal)), 3)
                    candidate["goal_progress_m"] = round(float(np.linalg.norm(sim.odometry[:2] - goal)) - candidate["goal_distance_m"], 3)
            observation, image = self._feedback(sim)
            return sensor, image, candidates, observation
        return await self.call(operation)

    async def moving_candidates(self):
        from backend.continuous_navigation import target_from_depth
        def capture(sim):
            if not self.continuous or not self.continuous.active or sim.cancel.is_set():
                raise ValueError("No active route for a moving proposal")
            self.spatial_sequence += 1
            sensor, image = sim.capture_spatial(self.spatial_sequence)
            self._camera_history().record(sensor, image)
            observation, image = self._feedback(sim, image=image)
            proposal = {"run_id": sim.run_id, "epoch": sim.epoch, "stop_revision": self.stop_revision,
                "identity": self.continuous.identity, "revision": self.continuous.revision,
                "captured_at": sensor.captured_at, "position": sim.odometry.copy(),
                "remaining_budget_m": 1.8 - self.navigation.travel, "candidates": []}
            intent = self.continuous.exploration
            if intent is not None:
                proposal.update(exploration=True, intent_revision=intent.revision,
                    captured_travel_m=self.navigation.travel, remaining_budget_m=1.8)
            return self.spatial_map, self.continuous.radius, sensor, image, observation, proposal
        mapped, radius, sensor, image, observation, proposal = await self.call(capture)
        def plan():
            candidates = proposal["candidates"]
            for column in (.25, .5, .75):
                for row in (.35, .5, .65):
                    pixel = [column, row]
                    try:
                        target = target_from_depth(sensor, pixel)
                        path = mapped.plan(sensor.odometry_m_rad[:2], target, radius)
                    except ValueError:
                        continue
                    length = sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:]))
                    if length > proposal["remaining_budget_m"] or length < .3:
                        continue
                    candidates.append({"id": len(candidates), "pixel": pixel, "target_m": target,
                        "path_length_m": length, "distance_m": float(np.linalg.norm(np.array(target) - sensor.odometry_m_rad[:2])),
                        "floor_rgb": mapped.floor_rgb(target)})
            return sensor, image, candidates, observation, proposal
        return await asyncio.to_thread(plan)

    async def retarget_continuous(self, proposal, candidate_id):
        import math
        def operation(sim):
            control = self.continuous
            if (sim.cancel.is_set() or self.stop_revision != proposal["stop_revision"] or sim.run_id != proposal["run_id"]
                    or sim.epoch != proposal["epoch"] or control is None or not control.active
                    or control.identity != proposal["identity"] or control.revision != proposal["revision"]):
                raise ValueError("Moving proposal lost its route authority")
            if (not 0 <= time.monotonic() - proposal["captured_at"] <= 15
                    or np.linalg.norm(sim.odometry[:2] - proposal["position"][:2]) > .9):
                raise ValueError("Moving proposal expired")
            if self.spatial_map.public()["stale"] or time.monotonic() >= self.navigation.skill_deadline or not self.navigation.buffer:
                raise ValueError("Fresh sensing and active motion authorization are required")
            choice = next((item for item in proposal["candidates"] if item["id"] == candidate_id), None)
            if choice is None:
                raise ValueError("Unknown moving candidate")
            path = self.spatial_map.plan(sim.odometry[:2], choice["target_m"], control.radius)
            length = sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:]))
            if self.navigation.travel + length > 1.8 or control.handoffs >= 2:
                raise ValueError("Continuation exceeds the original bounded route budget")
            allowed = self.spatial_map.traversable(control.start_position, control.radius)
            for begin, end in zip(path, path[1:]):
                points = np.linspace(begin, end, max(2, math.ceil(np.linalg.norm(np.array(end) - begin) / .02)))
                if not self.spatial_map.contains_path(allowed, points):
                    raise ValueError("Continuation lacks originally anchored observed support")
            delta = np.array(path[1]) - sim.odometry[:2]
            bearing = math.atan2(delta[1], delta[0]) - sim.odometry[2]
            if abs(math.atan2(math.sin(bearing), math.cos(bearing))) > .35:
                raise ValueError("Continuation changes heading too sharply; finish the current route first")
            self.navigation.check_clearance(sim, *self.navigation.velocity)
            control.path = np.array(path)
            control.index = 1
            control.revision += 1
            control.handoffs += 1
            control.reason = "Following a freshly validated moving continuation"
            control.distance_m = float(np.linalg.norm(control.path[-1] - sim.odometry[:2]))
            self._publish_navigation()
            return control.state()
        return await self.call(operation)

    async def parking_clearance(self, floor_target=None, *, stationary_request=None):
        from scipy.ndimage import distance_transform_edt
        stop_revision = self.stop_revision
        def operation(sim):
            if stationary_request is not None:
                self._continuous_guard(sim, stationary_request, stop_revision)
                if not self.spatial_enabled or self.spatial_map is None:
                    raise MotionError("SPATIAL_REQUIRED", "Stationary clearance requires an observed map")
                self._sample_spatial(force=True)
                self._continuous_guard(sim, stationary_request, stop_revision)
                if self.spatial_error or self.spatial_map.public()["stale"]:
                    raise MotionError("SPATIAL_STALE", self.spatial_error or "Stationary clearance sensing expired")
            footprint = sim.robot_footprint()
            radius = footprint["radius_m"]
            mapped = self.spatial_map
            clearance = distance_transform_edt(np.pad(mapped.cells == 0, 1, constant_values=False))[1:-1, 1:-1] * mapped.resolution_m
            measured = float(clearance[mapped.cell_index(sim.odometry[:2])])
            result = {"observed_clearance_m": round(measured, 3), "required_clearance_m": round(radius + .08, 3),
                "parking_margin_ok": measured >= radius + .08, "floor_rgb": mapped.floor_rgb(sim.odometry[:2])}
            if floor_target is not None:
                from backend.spatial import FloorRegionTracker
                lower, upper = footprint["lower_xy_m"], footprint["upper_xy_m"]
                polygon = np.array([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]])
                cosine, sine = np.cos(sim.odometry[2]), np.sin(sim.odometry[2])
                polygon = polygon @ np.array([[cosine, sine], [-sine, cosine]]) + sim.odometry[:2]
                recent = {**floor_target, "age_s": time.monotonic() - floor_target["captured_at"]}
                result.update(floor_target_id=floor_target["id"], inside_floor_region=not mapped.public()["stale"]
                    and recent["age_s"] >= 0 and FloorRegionTracker.contains(recent, polygon))
            return result
        return await self.call(operation)

    def _update_continuous(self):
        import math
        import pybullet as bullet
        if time.monotonic() - self.continuous.last_update < .05 and not self.sim.cancel.is_set():
            return
        components = self.continuous.skill_components
        if components:
            component = next((item for item in components if self.continuous.index <= item["end_index"]), components[-1])
            yaw = {"forward": 0., "left": .6, "right": -.6}[component["view"]]
            targets = {"head_yaw": (yaw, 5), "head_pitch": (.2, 5)}
            if any(self.sim.targets.get(self.sim.joints[name]) != target for name, target in targets.items()):
                for name, target in targets.items():
                    self.sim.targets[self.sim.joints[name]] = target
                self.sim._hold()
        head = getattr(self.continuous, "policy_head_rad", None)
        if head is not None:
            measured = [bullet.getJointState(self.sim.robot, self.sim.joints[name], physicsClientId=self.sim.client)[0]
                        for name in ("head_yaw", "head_pitch")]
            if np.max(np.abs(np.array(measured) - head)) > .02:
                self.continuous.finish(self.sim, self.navigation, "blocked", "CAMERA_CHANGED: Reset policy history before continuing")
                return
        self._sample_spatial()
        if getattr(self.continuous, "home_owned", False):
            self.home_mission.sample()
            self.continuous.update(self.sim, self.navigation, self.home_mission, self.home_mission.path_valid)
            if self.continuous.active:
                self.navigation.expires_at = min(self.navigation.expires_at, time.monotonic() + .5)
            return
        self._continue_observed_orbit()
        self._continue_exploration()
        allowed = self.spatial_map.traversable(self.continuous.start_position, self.continuous.radius)
        def valid(pose, linear, angular):
            points = [pose[:2]]
            for speed, turn in [(linear, angular), self.navigation.velocity]:
                for horizon in np.linspace(0, 1., 16):
                    heading = pose[2] + turn * horizon / 2
                    points.append(pose[:2] + speed * horizon * np.array([math.cos(heading), math.sin(heading)]))
            return self.spatial_map.contains_path(allowed, points)
        def replan(start, target):
            path = self.spatial_map.plan(start, target, self.continuous.radius)
            for begin, end in zip(path, path[1:]):
                if not self.spatial_map.contains_path(allowed, np.linspace(begin, end, max(2, math.ceil(np.linalg.norm(np.array(end) - begin) / .02)))):
                    raise ValueError("Replanned route leaves the original observed support area")
            return path
        self.continuous.update(self.sim, self.navigation, self.spatial_map, valid, None if components else replan)

    async def configure_spatial(self, settings):
        from backend.spatial import ObservedMap
        def operation(sim):
            if settings.run_id != sim.run_id or settings.episode_epoch != sim.epoch:
                raise MotionError("STALE_STATE", "Spatial sensor request belongs to another episode")
            if self.renderer or (self.latest.get("busy") and not settings.enabled):
                raise MotionError("CONTROL_CONFLICT", "Wait for manual control before changing the spatial sensor")
            if not settings.enabled and self.home_mission and self.home_mission.active:
                raise MotionError("CONTROL_CONFLICT", "Cancel the home mission before disabling required sensors")
            if self.spatial_enabled == settings.enabled:
                return self.spatial_state()
            self.spatial_enabled = settings.enabled
            self.spatial_generation += 1
            self.mapped_depth_sensor = None
            self.spatial_map = ObservedMap(sim.run_id, sim.epoch) if settings.enabled else None
            self.spatial_frames.clear()
            self.camera_history = None
            self.spatial_error = None
            self._sample_spatial(force=True)
            return self.spatial_state()
        return await self.call(operation)

    async def call(self, operation):
        await asyncio.wrap_future(self.ready)
        if self.closed:
            raise RuntimeError("Episode closed")
        future = Future()
        self.queue.put((future, operation))
        return await asyncio.wrap_future(future)

    async def home_command(self, request, expected_stop_revision=None, expected_task_revision=None, selected_evidence=None, operator_review=False):
        from backend.home_mission import HomeMission
        stop_revision = self.stop_revision if expected_stop_revision is None else expected_stop_revision
        task_revision = self.task_revision if expected_task_revision is None else expected_task_revision
        def operation(sim):
            if self.home_mission is None:
                self.home_mission = HomeMission(self)
            return self.home_mission.command(request, stop_revision, task_revision, selected_evidence, operator_review)
        return await self.call(operation)

    async def home_state(self, compact=False):
        from backend.home_mission import HomeMission
        def operation(sim):
            if self.home_mission is None:
                self.home_mission = HomeMission(self)
            return self.home_mission.state(compact)
        return await self.call(operation)

    async def ros_odometry(self):
        def operation(sim):
            return {"run_id": sim.run_id, "episode_epoch": sim.epoch, "captured_unix_s": time.time(),
                "simulated_time_s": sim.ticks * TIMESTEP, "odometry_m_rad": sim.odometry.tolist()}
        return await self.call(operation)

    async def ros_sensors(self):
        import base64
        from backend.ros_navigation import capture_sensors
        def operation(sim):
            self.ros_sensor_sequence += 1
            packet = capture_sensors(sim, self.ros_sensor_sequence)
            if self.ros_navigation and self.ros_navigation.active:
                self.ros_navigation.observe(packet["sequence"], packet["captured_at"])
                self._publish_camera(base64.b64decode(packet["head_rgb_png"]), packet["simulated_time_s"])
            packet["navigation"] = self.ros_navigation.state() if self.ros_navigation else None
            return packet
        return await self.call(operation)

    def nav2_status(self):
        online = (self.ros_bridge is not None and self.ros_bridge.run_id == self.latest.get("run_id")
            and self.ros_bridge.episode_epoch == self.epoch and 0 <= time.monotonic() - self.ros_bridge_at <= 2.)
        return {"ready": bool(online and self.ros_bridge.ready), "online": online,
            "message": self.ros_bridge.message if online else "Nav2 bridge is offline; start Nav2 or select Built-in backup",
            "controller": "nav2_external"}

    async def ros_heartbeat(self, status):
        def operation(sim):
            if status.run_id != sim.run_id or status.episode_epoch != sim.epoch:
                raise MotionError("STALE_STATE", "Bridge heartbeat belongs to an old episode")
            if self.ros_bridge and self.ros_bridge.bridge_id != status.bridge_id and self.nav2_status()["online"]:
                raise MotionError("CONTROL_CONFLICT", "Another Nav2 bridge owns this episode")
            self.ros_bridge, self.ros_bridge_at = status, time.monotonic()
            return self.nav2_status()
        return await self.call(operation)

    async def start_nav2_target(self, sensor, target, stop_revision, task_revision):
        from backend.ros_navigation import RosNavigationSession, RosStart
        def operation(sim):
            self._continuous_guard(sim, sensor, stop_revision)
            if task_revision != self.task_revision or not 0 <= time.monotonic() - sensor.captured_at <= 15.:
                raise MotionError("STALE_PLAN", "Nav2 destination belongs to an expired observation or instruction")
            if not self.nav2_status()["ready"]:
                raise MotionError("NAV2_UNAVAILABLE", self.nav2_status()["message"])
            if np.linalg.norm(sim.odometry - sensor.odometry_m_rad) > .05 or sim.held:
                raise MotionError("STALE_PLAN", "Nav2 destination requires the unchanged observed pose and free grippers")
            self._sample_spatial(force=True)
            self._continuous_guard(sim, sensor, stop_revision)
            if task_revision != self.task_revision or not 0 <= time.monotonic() - sensor.captured_at <= 15.:
                raise MotionError("STALE_PLAN", "Nav2 selection expired during depth validation")
            if self.spatial_map is None or self.spatial_error or self.spatial_map.public()["stale"]:
                raise MotionError("SPATIAL_STALE", "Nav2 requires fresh observed destination support")
            allowed = self.spatial_map.traversable(sim.odometry[:2], sim.robot_footprint()["radius_m"])
            if not self.spatial_map.contains_path(allowed, [target]):
                raise MotionError("OBSERVED_PATH_BLOCKED", "Nav2 destination is outside observed traversable floor")
            heading = float(np.arctan2(target[1] - sim.odometry[1], target[0] - sim.odometry[0]))
            request = RosStart(run_id=sim.run_id, episode_epoch=sim.epoch, target_m_rad=[*target, heading])
            session = RosNavigationSession(sim, request, stop_revision, task_revision=task_revision, owner="agent")
            session.runtime.skill_deadline = min(session.runtime.skill_deadline, sensor.captured_at + 20.)
            self.continuous = None
            self.ros_navigation, self.navigation = session, session.runtime
            self.wall_start, self.tick_start = time.monotonic(), sim.ticks
            self._publish_navigation()
            return session.state()
        return await self.call(operation)

    async def ros_result(self, result):
        def operation(sim):
            if self.ros_navigation is None:
                raise MotionError("CANCELLED", "No Nav2 goal is active")
            try:
                self.ros_navigation.complete(sim, result, self.stop_revision, self.task_revision)
            finally:
                self._publish_navigation()
            return self.ros_navigation.state()
        return await self.call(operation)

    async def start_ros(self, request):
        from backend.ros_navigation import RosNavigationSession
        stop_revision = self.stop_revision
        def operation(sim):
            if self.home_mission and self.home_mission.active:
                raise MotionError("CONTROL_CONFLICT", "Cancel the home mission before ROS navigation")
            if self.latest.get("busy") or self.renderer or (self.skill and self.skill.active):
                raise MotionError("CONTROL_CONFLICT", "Take manual control before ROS navigation")
            if stop_revision != self.stop_revision:
                raise MotionError("CANCELLED", "ROS start was invalidated by Stop")
            session = RosNavigationSession(sim, request, stop_revision, task_revision=self.task_revision)
            self.continuous = None
            self.ros_navigation = session
            self.navigation = session.runtime
            self.wall_start, self.tick_start = time.monotonic(), sim.ticks
            self._publish_navigation()
            return session.state()
        return await self.call(operation)

    async def ros_velocity(self, command):
        def operation(sim):
            if not self.ros_navigation:
                raise MotionError("CANCELLED", "No ROS session is active")
            try:
                self.ros_navigation.apply(sim, command, self.stop_revision, task_revision=self.task_revision)
            finally:
                self._publish_navigation()
            return self.ros_navigation.state()
        return await self.call(operation)

    async def execute(self, command, assisted=True):
        def operation(sim):
            if self.home_mission and self.home_mission.active and command.tool not in {"stop", "observe"}:
                raise MotionError("CONTROL_CONFLICT", "Cancel the home mission before manual motion")
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
                if self.home_mission and self.home_mission.stage == "mapping" and command.tool == "drive_base":
                    self._sample_spatial(force=True)
                    self.home_mission.sample(force=True)
                    result = sim.execute(command, drive_guard=self.home_mission.guard_guided_drive)
                else:
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
        if self.continuous and not self.continuous.active and getattr(self.continuous, "preview_dimensions", None):
            self.sim.width, self.sim.height = self.continuous.preview_dimensions
            self.continuous.preview_dimensions = None
            observation, image = self._feedback(self.sim)
            self._publish_camera(image)
        self.latest = {**self.latest, "navigation": self.navigation.state(),
                       "continuous_navigation": self.continuous.state() if self.continuous else None,
                       "ros_navigation": self.ros_navigation.state() if self.ros_navigation else None,
                       "busy": bool(self.navigation.buffer) or bool(self.continuous and self.continuous.active) or bool(self.ros_navigation and self.ros_navigation.active),
                       "snapshot": self.sim.snapshot(), "stopped": self.sim.cancel.is_set()}

    def _feedback(self, sim, image=None):
        if self.renderer:
            self._render_update()
            cached, image, captured_at = self.render_frame
            if time.monotonic() - captured_at > 1:
                raise MotionError("CAMERA_STALE", "Snapshot camera feedback expired")
            observation = cached.model_copy(deep=True)
        else:
            observation = sim.observe(render=image is None)
            if image is None:
                image = sim.frame(observation.frame_ref)
            else:
                sim.frames[observation.frame_ref] = image
                while len(sim.frames) > 16:
                    sim.frames.popitem(last=False)
        if self.navigation:
            observation.navigation = NavigationFeedback.model_validate(self.navigation.observe(sim, observation))
        if self.skill:
            observation.skill = SkillFeedback.model_validate(self.skill.state())
        if self.home_mission and self.home_mission.home:
            observation.spatial = self.home_mission.state(compact=True)
        self.latest = {**self.latest, "observation": observation.model_dump()}
        return observation, image

    async def feedback(self):
        return await self.call(self._feedback)

    async def hold_stopped(self):
        def operation(sim):
            sim.hold_current()
            if self.home_mission:
                self.home_mission.fail("Stopped by operator", "cancelled")
            if self.ros_navigation and self.ros_navigation.active:
                self.ros_navigation.finish(sim, "cancelled", "Stopped by operator")
            if self.continuous and self.continuous.active:
                self.continuous.finish(sim, self.navigation, "cancelled", "Stopped by operator")
            if self.navigation:
                self.navigation.cancel(sim)
                self._publish_navigation()
        await self.call(operation)

    async def begin_navigation(self, expected_stop_revision, enabled=True):
        def operation(sim):
            if self.home_mission:
                self.home_mission.fail("Navigation ownership changed", "cancelled")
            if self.ros_navigation and self.ros_navigation.active:
                raise MotionError("CONTROL_CONFLICT", "Stop ROS navigation before changing controllers")
            if sim.cancel.is_set() or expected_stop_revision != self.stop_revision:
                raise MotionError("CANCELLED", "Navigation start was invalidated by Stop.")
            if self.navigation:
                self.navigation.cancel(sim, "New navigation session")
            self.continuous = None
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
            from backend.camera import snapshot_renderer
            if sim.cancel.is_set() or self.stop_revision != expected_stop_revision:
                raise MotionError("CANCELLED", "Skill mode start was invalidated")
            if self.home_mission:
                self.home_mission.fail("Policy ownership changed", "cancelled")
            if self.skill:
                self.skill.cancel(sim)
            if self.navigation:
                self.navigation.cancel(sim)
                self.navigation = None
            if not self.renderer:
                self.renderer = snapshot_renderer(sim)
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
                if command.tool == "replace_motion_buffer" and self.latest.get("supervised_navigation"):
                    for segment in arguments.segments:
                        if segment.kind == "drive":
                            self.navigation.check_clearance(sim, segment.linear_mps, segment.angular_radps)
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
            self.spatial_generation += 1
            self.mapped_depth_sensor = None
            if self.home_mission:
                self.home_mission.invalidate("LOCALIZATION_LOST: manual relocation requires scan matching")
            self.map_history = None
            self.camera_history = None
            self._record_map_pose()
            if self.spatial_enabled:
                from backend.spatial import ObservedMap
                self.spatial_map = ObservedMap(sim.run_id, sim.epoch)
                self.spatial_frames.clear()
                self._sample_spatial(force=True)
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
            if self.home_mission:
                self.home_mission.fail("Manual control resumed", "cancelled")
                self.home_mission.room_verification = None
                self.home_mission.room_evidence_after = time.monotonic()
                self.home_mission.workflow = None
            if sim.rendering == "enhanced" and sim.camera_renderer and (
                    sim.camera_renderer.closed or sim.camera_renderer.process.poll() is not None):
                raise MotionError("RENDERER_FAILED", "Reset the episode before resuming after a camera failure")
            if self.ros_navigation and self.ros_navigation.active:
                self.ros_navigation.finish(sim, "cancelled", "Manual control resumed")
                self._publish_navigation()
            sim.cancel.clear()
            if self.home_mission:
                self.home_mission.stop_revision = self.stop_revision
                self.home_mission.task_revision = self.task_revision
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