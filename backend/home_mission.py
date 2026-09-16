import math
import os
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import Field

from backend.contracts import StrictModel
from backend.home_mapping import HomeMap, MapStore, inverse_pose, laser_points, transform_pose
from backend.simulation import MotionError
from backend.worker_timing import timed


DEFAULT_MAP_PATH = Path(__file__).resolve().parents[1] / ".runtime" / "maps" / "homes.sqlite3"


class HomeRequest(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    action: Literal["start_mapping", "start_exploration", "continue_mapping", "guided_to", "review", "save_map", "load_map", "localize", "add_place", "navigate_to", "explore", "explore_frontier", "cancel_task", "get_spatial_state", "remember_object", "observe_room", "review_room"]
    evidence_text: str = Field(default="", max_length=500)
    evidence_id: str | None = Field(default=None, max_length=80)
    room_matches: bool | None = None
    frontier_id: str | None = Field(default=None, max_length=80)
    object_label: str = Field(default="", max_length=80)
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    confidence: float = Field(default=.5, ge=0., le=1., allow_inf_nan=False)
    spatial_sequence: int | None = Field(default=None, ge=1)
    map_id: str | None = Field(default=None, max_length=80)
    name: str = Field(default="", max_length=80)
    place_id: str | None = Field(default=None, max_length=80)
    return_place_id: str | None = Field(default=None, max_length=80)
    kind: Literal["room", "doorway", "destination"] = "destination"
    pose_m_rad: list[float] | None = Field(default=None, min_length=3, max_length=3)
    connects: list[str] = Field(default_factory=list, max_length=20)
    region_id: str | None = Field(default=None, max_length=80)
    time_budget: float = Field(default=60., ge=1., le=300., allow_inf_nan=False)


class HomeMission:
    max_frame_age_s = 1.

    @property
    def captured_at(self):
        sensor = getattr(self.worker, "mapped_depth_sensor", None)
        if sensor is not None:
            return sensor.captured_at
        return self.worker.spatial_map.captured_at if self.worker.spatial_map else None

    def __init__(self, worker, store=None):
        self.worker = worker
        self.store = store or MapStore(Path(os.environ.get("MILO_HOME_MAP_STORE", DEFAULT_MAP_PATH)))
        challenge = worker.challenge
        self.environment_id = (challenge.environment if challenge and challenge.environment != "standalone"
            else "standalone:" + (challenge.id if challenge else "bench"))
        self.home = None
        self.stage = "empty"
        self.transform = None
        self.pose = None
        self.localization = {"status": "unlocalized", "method": "scan_to_saved_map", "quality": None}
        self.sampled_at = 0.
        self.validated_at = 0.
        self.last_odometry = None
        self.last_mapped_pose = None
        self.last_depth_sequence = 0
        self.live = {}
        self.task = None
        self.workflow = None
        self.route = []
        self.allowed_cache = None
        self.stop_revision = worker.stop_revision
        self.task_revision = worker.task_revision
        self.recheck_at = 0.
        self.last_laser = None
        self.error = None
        self.object_records = []
        self.room_records = []
        self.room_verification = None
        self.room_evidence_after = time.monotonic()
        self.room_report_deadline = 0.
        self.allow_expansion = True
        self.mission_owner = None

    @property
    def active(self):
        return bool(self.task and self.task["status"] == "running")

    def fail(self, reason, status="failed"):
        worker = self.worker
        if self.workflow and self.workflow["status"] in {"navigating_to_room", "awaiting_room_report", "returning"}:
            self.workflow.update(status=status, reason=reason)
        was_active = self.active
        if was_active:
            self.task.update(status=status, reason=reason)
        if worker.continuous and worker.continuous.active and getattr(worker.continuous, "home_owned", False):
            worker.continuous.finish(worker.sim, worker.navigation, status, reason)
        elif was_active:
            worker.sim.hold_current()
        if was_active:
            worker.latest = {**worker.latest, "busy": False}
        self.route = []

    def invalidate(self, reason):
        self.fail(reason, "cancelled")
        self.room_verification = None
        self.room_evidence_after = time.monotonic()
        self.transform = None
        self.pose = None
        self.localization["status"] = "unlocalized"
        self.live.clear()
        if self.stage == "mapping":
            self.stage = "review"

    @timed("map.sample")
    def sample(self, force=False):
        try:
            self._sample(force)
        except Exception as error:
            self.error = f"MAPPING_SENSOR_FAILED: {type(error).__name__}: {error}"
            self.invalidate(self.error)
            self.worker.sim.stop()

    def _sample(self, force=False):
        from backend.ros_navigation import capture_laser
        if self.home is None or not self.worker.powered:
            return
        now = time.monotonic()
        interval = self.worker.idle_sensor_interval_s if self.worker.power_state()["mode"] == "idle" else .2
        if not force and now - self.sampled_at < interval:
            return
        sim = self.worker.sim
        if self.stop_revision != self.worker.stop_revision or self.task_revision != self.worker.task_revision or sim.cancel.is_set():
            self.fail("CANCELLED: motion authority changed", "cancelled")
            if self.stage == "mapping":
                self.stage = "review"
            return
        laser = capture_laser(sim)
        odometry = sim.odometry.tolist()
        self.last_laser = laser
        self.sampled_at = now
        if self.transform is None:
            return
        if self.last_odometry is not None and (math.dist(odometry[:2], self.last_odometry[:2]) > .5
                or abs(odometry[2] - self.last_odometry[2]) > .5):
            self.invalidate("LOCALIZATION_LOST: discontinuous odometry")
            return
        self.last_odometry = odometry
        self.pose = transform_pose(odometry, self.transform)
        if now - self.validated_at >= 1. and self.home.scan_count >= 3:
            try:
                self.localization["quality"] = self.home.scan_quality(laser, self.pose)
                self.localization["quality"].update(evaluation_order="before_current_laser_integration",
                    reference_geometry="accumulated_sensor_map_not_independent_reference",
                    reference_scan_count=self.home.scan_count)
                self.validated_at = time.monotonic()
            except ValueError as error:
                self.error = str(error)
                self.invalidate(str(error))
                sim.stop()
                return
        elif self.home.scan_count < 3:
            self.validated_at = now
        origin, endpoints, hits = laser_points(laser, self.pose)
        samples = origin + (endpoints[:, None, :] - origin) * np.linspace(0., .99, 160)[None, :, None]
        cleared = self.home.indices(samples.reshape(-1, 2))
        clear_codes = set((cleared[:, 1] * self.home.size + cleared[:, 0]).tolist())
        self.live = {cell: timestamp for cell, timestamp in self.live.items()
            if now - timestamp <= 2. and cell[1] * self.home.size + cell[0] not in clear_codes}
        for column, row in self.home.indices(endpoints[hits]):
            self.live[(int(column), int(row))] = now
        mapped = self.stage == "mapping" or (self.active and self.task["kind"] == "explore") or bool(self.mission_owner and self.allow_expansion)
        moved = self.last_mapped_pose is None or math.dist(self.pose[:2], self.last_mapped_pose[:2]) > .08 or abs(self.pose[2] - self.last_mapped_pose[2]) > .08
        if mapped and (force or moved):
            self.home.observe(laser, self.pose, time.time())
            radius = sim.robot_footprint()["radius_m"]
            rows, columns = np.indices(self.home.evidence.shape)
            positions = self.home.origin + (np.stack((columns, rows), axis=-1) + .5) * self.home.resolution_m
            support = np.linalg.norm(positions - self.pose[:2], axis=-1) <= radius + .15
            self.home.evidence[support & (self.home.evidence == 0)] = -1
            self.last_mapped_pose = list(self.pose)
        spatial = self.worker.spatial_map
        sensor = getattr(self.worker, "mapped_depth_sensor", None)
        if sensor is None and spatial:
            sensor = spatial.observation
        if sensor:
            self.observe_depth(sensor)
        self.localization["status"] = "localized"
        self.allowed_cache = None

    @timed("map.observe_depth")
    def observe_depth(self, sensor):
        from backend.spatial import point_cloud
        now = time.monotonic()
        if self.home is None or self.transform is None:
            return
        if sensor.sequence != self.last_depth_sequence and 0 <= now - sensor.captured_at <= 1.:
            _, points = point_cloud(sensor, stride=4)
            points = points[(points[:, 2] > .04) & (points[:, 2] < 1.3)]
            mapped = self.stage == "mapping" or (self.active and self.task["kind"] == "explore") or bool(self.mission_owner and self.allow_expansion)
            cosine, sine = math.cos(self.transform[2]), math.sin(self.transform[2])
            positions = np.column_stack((cosine * points[:, 0] - sine * points[:, 1] + self.transform[0],
                sine * points[:, 0] + cosine * points[:, 1] + self.transform[1]))
            cells = np.unique(self.home.indices(positions), axis=0)
            for column, row in cells:
                self.live[(int(column), int(row))] = now
            if mapped:
                cells = cells[self.home.inside(cells)]
                columns, rows = cells.T
                self.home.evidence[rows, columns] = np.maximum(4, self.home.evidence[rows, columns])
                if len(cells):
                    self.home.updated_at = time.time()
            self.last_depth_sequence = sensor.sequence
            self.allowed_cache = None

    def obstacles(self):
        now = time.monotonic()
        return [(self.home.origin + (np.array(cell) + .5) * self.home.resolution_m).tolist()
            for cell, timestamp in self.live.items() if now - timestamp <= 2.]

    def require_localized(self):
        if self.pose is None or self.localization["status"] != "localized" or time.monotonic() - self.sampled_at > .75 or time.monotonic() - self.validated_at > 2.:
            raise ValueError("LOCALIZATION_REQUIRED: localize against the saved map with fresh sensors")

    def command(self, request, stop_revision, task_revision, selected_evidence=None, operator_review=False, selected_frontier=None):
        worker, sim = self.worker, self.worker.sim
        if request.run_id != sim.run_id or request.episode_epoch != sim.epoch:
            raise MotionError("STALE_STATE", "Map request belongs to another episode")
        if request.action == "get_spatial_state":
            return self.state()
        if request.action == "cancel_task":
            self.fail("Cancelled by operator", "cancelled")
            self.room_verification = None
            self.room_evidence_after = time.monotonic()
            if self.stage in {"mapping", "expansion"}:
                self.stage = "review"
            return self.state()
        if stop_revision != worker.stop_revision or task_revision != worker.task_revision or sim.cancel.is_set():
            raise MotionError("CANCELLED", "Map operation was invalidated; resume control explicitly")
        if self.active or (worker.continuous and worker.continuous.active) or (worker.ros_navigation and worker.ros_navigation.active) or worker.renderer or (worker.skill and worker.skill.active):
            raise MotionError("CONTROL_CONFLICT", "Cancel the active controller before changing the home map")
        self.stop_revision, self.task_revision = stop_revision, task_revision
        if request.action != "load_map" and request.map_id and (self.home is None or request.map_id != self.home.identity):
            raise ValueError("STALE_MAP: request belongs to another map")
        if request.action == "start_exploration":
            if not self.allow_expansion:
                raise ValueError("MAP_READ_ONLY: exploration cannot change this frozen map")
            if self.home is None:
                self.home = HomeMap(self.environment_id)
                self.transform = inverse_pose(sim.odometry.tolist())
                self.last_odometry = sim.odometry.tolist()
                self.stage = "mapping"
                self.enable_sensing()
                self.sample(force=True)
            self.command(request.model_copy(update={"action": "explore"}), stop_revision, task_revision)
            self.task["local_exploration"] = True
            self.task["model_calls"] = 0
            return self.state()
        if request.action == "start_mapping":
            existing = next((item for item in self.store.catalog() if item["environment_id"] == self.environment_id), None)
            if existing:
                raise ValueError("MAP_EXISTS: load " + existing["map_id"] + "; this home is already mapped")
            if self.home is None:
                self.home = HomeMap(self.environment_id)
                self.transform = inverse_pose(sim.odometry.tolist())
                self.last_odometry = sim.odometry.tolist()
            elif self.transform is None:
                raise ValueError("LOCALIZATION_REQUIRED: localize before resuming this draft")
            self.stage = "mapping"
            self.enable_sensing()
            self.sample(force=True)
        elif request.action == "load_map":
            self.home = self.store.load(request.map_id, self.environment_id)
            self.workflow = None
            self.object_records = self.store.object_observations(self.home.identity)
            self.room_records = self.store.room_observations(self.home.identity)
            self.room_verification = None
            self.invalidate("Saved map loaded; localization required")
            self.stage, self.task = "loaded", None
            self.last_odometry = self.last_mapped_pose = None
            self.enable_sensing()
        elif self.home is None:
            raise ValueError("MAP_REQUIRED: create or load a home map")
        elif request.action == "localize":
            from backend.ros_navigation import capture_laser
            seed = request.pose_m_rad
            if request.place_id:
                seed = self.place(request.place_id)["pose_m_rad"]
            laser = capture_laser(sim)
            pose, quality = self.home.match_scan(laser, seed)
            if worker.stop_revision != stop_revision or worker.task_revision != task_revision or sim.cancel.is_set():
                raise MotionError("CANCELLED", "Localization was cancelled")
            self.transform = transform_pose(inverse_pose(sim.odometry.tolist()), pose)
            self.room_verification = None
            self.room_evidence_after = time.monotonic()
            self.pose, self.last_odometry = pose, sim.odometry.tolist()
            self.localization = {"status": "localized", "method": "scan_to_saved_map", "quality": quality}
            self.validated_at = time.monotonic()
            self.error = None
            self.sample(force=True)
        elif request.action == "review":
            self.stage = "review"
        elif request.action == "continue_mapping":
            if not self.allow_expansion:
                raise ValueError("MAP_READ_ONLY: benchmark map cannot be expanded")
            self.sample(force=True)
            self.require_localized()
            self.home.saved = False
            self.stage = "mapping"
        elif request.action == "save_map":
            self.store.save(self.home, request.name or self.home.name)
            self.stage = "loaded"
        elif request.action == "add_place":
            self.sample(force=True)
            self.require_localized()
            pose = request.pose_m_rad or self.pose
            if not np.isfinite(pose).all():
                raise ValueError("Place coordinates must be finite")
            self.home.add_place(request.name, request.kind, pose, sim.robot_footprint()["radius_m"], request.connects)
        elif request.action == "observe_room":
            self.sample(force=True)
            self.require_localized()
            if not self.home.revision:
                raise ValueError("SAVE_REQUIRED: save the geometric map first")
            paired = selected_evidence or worker.spatial_frames.get(request.spatial_sequence)
            if (paired is None or not request.evidence_text.strip() or paired[0].run_id != sim.run_id
                    or paired[0].episode_epoch != sim.epoch or paired[0].sequence != request.spatial_sequence
                    or not 0 <= time.monotonic() - paired[0].captured_at <= 15.
                    or paired[0].captured_at < self.room_evidence_after
                    or not self.room_view_matches(transform_pose(paired[0].odometry_m_rad, self.transform), paired[0].head_rad)):
                raise ValueError("STALE_OBSERVATION: room evidence needs a fresh paired image at the unchanged pose and head view")
            if len(paired[1]) > 1024 * 1024 or not paired[1].startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Room evidence requires a bounded PNG image")
            pose = transform_pose(paired[0].odometry_m_rad, self.transform)
            places_before, edges_before, dirty_before = len(self.home.places), len(self.home.edges), self.home.annotations_dirty
            place = self.place(request.place_id) if request.place_id else None
            if place is None:
                if request.room_matches is not None:
                    raise ValueError("ARRIVAL_REQUIRED: verification needs an existing destination")
                place = self.home.add_place(request.name, request.kind if request.kind in {"room", "doorway"} else "room",
                    pose, sim.robot_footprint()["radius_m"], request.connects)
                place["identity_status"] = "tentative"
                place["source"] = "camera_room_hypothesis"
            if place["kind"] not in {"room", "doorway"} or math.dist(pose[:2], place["pose_m_rad"][:2]) > .75:
                raise ValueError("ROOM_LOCATION_MISMATCH: observe at the named room or doorway")
            if request.room_matches is not None:
                task = self.task or {}
                if (task.get("place_id") != place["place_id"] or task.get("status") != "completed"
                        or task.get("arrived_simulated_s") is None or paired[0].simulated_time_s < task["arrived_simulated_s"]
                        or paired[0].captured_at <= task.get("arrived_at", math.inf)
                        or math.dist(pose[:2], place["pose_m_rad"][:2]) > .15):
                    raise ValueError("ARRIVAL_REQUIRED: use a fresh post-arrival image at the selected room")
            identity = str(uuid4())
            record = {"observation_id": identity, "map_id": self.home.identity, "place_id": place["place_id"],
                "label": place["name"], "kind": place["kind"], "frame": "map", "pose_m_rad": pose,
                "run_id": sim.run_id, "episode_epoch": sim.epoch, "spatial_sequence": paired[0].sequence,
                "stop_revision": stop_revision, "task_revision": task_revision,
                "head_rad": paired[0].head_rad, "observed_unix_s": time.time() - (time.monotonic() - paired[0].captured_at),
                "evidence": request.evidence_text.strip(), "confidence": request.confidence,
                "review_status": "tentative", "room_matches": request.room_matches, "identity_verified": False,
                "task_id": self.task["task_id"] if request.room_matches is not None else None,
                "image_url": f"/api/home/{self.home.identity}/rooms/{identity}/image.png"}
            try:
                self.store.remember_room(record, paired[1])
            except (OSError, ValueError, RuntimeError):
                del self.home.places[places_before:]
                del self.home.edges[edges_before:]
                self.home.annotations_dirty = dirty_before
                raise
            self.room_records = self.store.room_observations(self.home.identity)
            if request.room_matches is not None:
                self.room_verification = record
                self.room_report_deadline = paired[0].captured_at + 15.
                if self.workflow and self.workflow["status"] == "awaiting_room_report":
                    remaining = self.workflow["deadline"] - time.monotonic()
                    if remaining < 1.:
                        self.fail("TIMEOUT: room workflow budget expired", "limited")
                    elif request.room_matches is False:
                        self.fail("ROOM_MISMATCH: visual report did not match the selected room")
                    else:
                        self.workflow.update(status="returning", room_evidence_id=identity,
                            reason="Visual match reported; returning to selected Home")
                        try:
                            return self.command(HomeRequest(run_id=sim.run_id, episode_epoch=sim.epoch, action="navigate_to",
                                place_id=self.workflow["return_place_id"], time_budget=min(300., remaining)), stop_revision, task_revision)
                        except (ValueError, MotionError) as error:
                            self.fail("RETURN_BLOCKED: " + str(error))
                            raise
        elif request.action == "review_room":
            if not operator_review:
                raise ValueError("OPERATOR_REVIEW_REQUIRED: the model cannot confirm its own room labels")
            record = next((record for record in self.room_records if record["observation_id"] == request.evidence_id), None)
            if record is None or record["room_matches"] is False:
                raise ValueError("UNKNOWN_EVIDENCE: select matching room evidence")
            place = self.place(record["place_id"])
            self.store.review_room(self.home.identity, request.evidence_id)
            place.update(identity_status="operator_confirmed", evidence_id=request.evidence_id)
            self.home.annotations_dirty = True
            self.room_records = self.store.room_observations(self.home.identity)
        elif request.action == "remember_object":
            from backend.spatial import measure_visible_region
            self.require_localized()
            paired = selected_evidence or worker.spatial_frames.get(request.spatial_sequence)
            maximum_age = 15. if selected_evidence else 1.
            if (not request.object_label.strip() or paired is None or paired[0].run_id != sim.run_id
                    or paired[0].episode_epoch != sim.epoch or paired[0].sequence != request.spatial_sequence
                    or not 0 <= time.monotonic() - paired[0].captured_at <= maximum_age):
                raise ValueError("STALE_OBSERVATION: object memory needs a labelled current paired camera observation")
            measurement = measure_visible_region(paired[0], request.object_bounds)
            measured = measurement["surface_median_xyz_m"]
            position = transform_pose([*measured[:2], 0.], self.transform)[:2] + [measured[2]]
            identity = str(uuid4())
            record = {"observation_id": identity, "map_id": self.home.identity, "frame": "map", "units": "m",
                "label": request.object_label.strip(), "position_m": position, "confidence": request.confidence,
                "observed_unix_s": time.time() - (time.monotonic() - paired[0].captured_at),
                "run_id": sim.run_id, "episode_epoch": sim.epoch, "spatial_sequence": paired[0].sequence,
                "head_rad": paired[0].head_rad, "object_bounds": request.object_bounds,
                "source": "unverified_semantic_annotation_on_measured_surface", "identity_verified": False,
                "supporting_images": [f"/api/home/{self.home.identity}/objects/{identity}/image.png"]}
            self.store.remember_object(record, paired[1])
            self.object_records = self.store.object_observations(self.home.identity)
        elif request.action in {"navigate_to", "guided_to", "explore", "explore_frontier"}:
            if not self.home.revision and request.action not in {"guided_to", "explore"} and not self.mission_owner:
                raise ValueError("SAVE_REQUIRED: review and save the initial map before autonomous tasks")
            if not worker.spatial_enabled or worker.spatial_map is None:
                raise ValueError("SPATIAL_REQUIRED: enable spatial sensing before mapped navigation")
            self.sample(force=True)
            self.require_localized()
            if sim.held or any(any(gripper.contact) for gripper in sim.observe(render=False).grippers.values()):
                raise ValueError("OBJECT_HELD: release objects before mapped navigation")
            target = self.place(request.place_id)["pose_m_rad"][:2] if request.action == "navigate_to" else None
            if request.action == "guided_to":
                if not self.allow_expansion:
                    raise ValueError("MAP_READ_ONLY: guided survey is disabled for this map")
                if request.pose_m_rad is None or not np.isfinite(request.pose_m_rad).all():
                    raise ValueError("WAYPOINT_REQUIRED: select a finite observed map position")
                target = request.pose_m_rad[:2]
            if request.return_place_id:
                if request.action != "navigate_to" or self.place(request.place_id)["kind"] != "room" or request.return_place_id == request.place_id:
                    raise ValueError("ROOM_WORKFLOW_REQUIRED: select a room and a distinct saved return destination")
                self.place(request.return_place_id)
            chosen_frontier = None
            radius = sim.robot_footprint()["radius_m"]
            if request.action == "explore_frontier":
                if selected_frontier is not None:
                    selected_frontier.check(self.mission_owner, sim.run_id, sim.epoch, self.home.identity, sim.odometry, time.monotonic())
                    if request.frontier_id != selected_frontier.frontier_id:
                        raise ValueError("FRONTIER_SELECTION_CHANGED: supplied identity changed")
                    chosen_frontier = {"frontier_id": selected_frontier.frontier_id, "position_m": list(selected_frontier.position_m)}
                else:
                    choices = self.home.frontiers(self.pose, radius, self.obstacles(), request.region_id)
                    chosen_frontier = next((item for item in choices if item["frontier_id"] == request.frontier_id), None)
                if chosen_frontier is None:
                    raise ValueError("UNKNOWN_FRONTIER: select a currently reachable frontier")
                if not self.allow_expansion:
                    raise ValueError("MAP_READ_ONLY: benchmark map cannot be expanded")
                target = chosen_frontier["position_m"]
                self.home.saved = False
            if target is not None:
                self.home.route(self.pose[:2], target, radius, self.obstacles())
                if request.action == "guided_to":
                    self.home.saved = False
            else:
                if not self.allow_expansion:
                    raise ValueError("MAP_READ_ONLY: scenario evaluations cannot expand the frozen map")
                self.home.route(self.pose[:2], self.pose[:2], radius, self.obstacles())
                self.home.saved = False
            self.room_verification = None
            if request.return_place_id:
                self.workflow = {"workflow_id": str(uuid4()), "room_place_id": request.place_id,
                    "return_place_id": request.return_place_id, "status": "navigating_to_room",
                    "deadline": time.monotonic() + request.time_budget, "reason": "Navigate, fresh room report, then return",
                    "room_evidence_id": None, "identity_verified": False}
            elif self.workflow and not (self.workflow["status"] == "returning" and request.place_id == self.workflow["return_place_id"]):
                self.fail("Room workflow replaced by another task", "cancelled")
                self.workflow = None
            if worker.continuous and not worker.continuous.active and getattr(worker.continuous, "home_owned", False):
                worker.continuous = None
            self.task = {"task_id": str(uuid4()), "kind": "navigate" if request.action in {"navigate_to", "guided_to"} else "explore",
                "guided_mapping": request.action == "guided_to",
                "status": "running", "reason": "Planning on observed map", "place_id": request.place_id,
                "region_id": request.region_id, "target_m": target, "started_at": time.monotonic(),
                "deadline": time.monotonic() + request.time_budget, "retries": 0, "segments": 0,
                "frontier_id": chosen_frontier["frontier_id"] if chosen_frontier else None,
                "single_frontier": chosen_frontier is not None, "visited_frontiers": 0, "completion_verified": False}
            if self.mission_owner:
                self.task.update(mission_id=self.mission_owner, local_exploration=True)
            self.stage = "mapping" if request.action == "guided_to" else "navigation" if request.action == "navigate_to" else "expansion"
            self.recheck_at = 0.
        return self.state()

    def enable_sensing(self):
        from backend.spatial import ObservedMap
        self.worker.spatial_enabled = True
        if self.worker.spatial_map is None:
            self.worker.spatial_map = ObservedMap(self.worker.sim.run_id, self.worker.sim.epoch)
        self.worker._sample_spatial(force=True)

    def place(self, identity):
        place = next((place for place in self.home.places if place["place_id"] == identity), None)
        if place is None:
            raise ValueError("UNKNOWN_PLACE: select an existing named destination")
        return place

    def room_view_matches(self, pose, head):
        import pybullet as bullet
        sim = self.worker.sim
        current_pose = transform_pose(sim.odometry.tolist(), self.transform)
        current_head = [bullet.getJointState(sim.robot, sim.joints[name], physicsClientId=sim.client)[0]
            for name in ("head_yaw", "head_pitch")]
        heading = current_pose[2] - pose[2]
        return (math.dist(current_pose[:2], pose[:2]) <= .05
            and abs(math.atan2(math.sin(heading), math.cos(heading))) <= .05
            and max(abs(actual - observed) for actual, observed in zip(current_head, head)) <= .05)

    @timed("map.clearance_mask")
    def allowed(self, radius):
        if self.allowed_cache is None or getattr(self, "allowed_radius", None) != radius:
            self.allowed_cache = self.home.allowed(radius, self.obstacles())
            self.allowed_radius = radius
        return self.allowed_cache

    @timed("map.path_valid")
    def path_valid(self, pose, linear, angular, *, guided=False):
        if not self.active and not (guided and self.stage == "mapping"):
            return False
        self.require_localized()
        radius = self.worker.sim.robot_footprint()["radius_m"]
        allowed = self.allowed(radius)
        velocity = self.worker.navigation.velocity if self.worker.navigation else (0., 0.)
        candidates = np.asarray([(linear, angular), velocity])
        horizons = np.linspace(0., 1., 16)
        headings = pose[2] + candidates[:, 1, None] * horizons / 2
        points = (np.asarray(pose[:2]) + candidates[:, 0, None, None] * horizons[None, :, None]
            * np.stack((np.cos(headings), np.sin(headings)), axis=-1)).reshape(-1, 2)
        cosine, sine = math.cos(self.transform[2]), math.sin(self.transform[2])
        points = points @ np.array([[cosine, sine], [-sine, cosine]]) + self.transform[:2]
        indices = self.home.indices(points)
        return bool(self.home.inside(indices).all() and allowed[indices[:, 1], indices[:, 0]].all())

    def guard_guided_drive(self, arguments):
        from backend.navigation import NavigationRuntime
        try:
            self.require_localized()
            if self.captured_at is None or not 0 <= time.monotonic() - self.captured_at <= self.max_frame_age_s:
                raise ValueError("SPATIAL_STALE: fresh depth required for guided mapping")
            if not self.path_valid(self.worker.sim.odometry, arguments.linear_mps, arguments.angular_radps, guided=True):
                raise ValueError("GUIDED_MAP_BLOCKED: predicted footprint leaves observed clear space")
            NavigationRuntime().check_clearance(self.worker.sim, arguments.linear_mps, arguments.angular_radps)
        except ValueError as error:
            raise MotionError("GUIDED_MAP_BLOCKED", str(error)) from error

    @timed("map.tick")
    def tick(self):
        if self.workflow and self.workflow["status"] in {"navigating_to_room", "awaiting_room_report", "returning"}:
            if self.worker.sim.cancel.is_set() or self.stop_revision != self.worker.stop_revision or self.task_revision != self.worker.task_revision:
                self.fail("CANCELLED: room workflow authority changed", "cancelled")
                return
            if time.monotonic() >= self.workflow["deadline"]:
                self.fail("TIMEOUT: room workflow budget expired", "limited")
                return
        if not self.active:
            return
        worker, sim, task = self.worker, self.worker.sim, self.task
        try:
            if sim.cancel.is_set() or self.stop_revision != worker.stop_revision or self.task_revision != worker.task_revision:
                self.fail("CANCELLED: task authority changed", "cancelled")
                return
            if time.monotonic() >= task["deadline"]:
                self.fail("TIMEOUT: task budget expired", "limited")
                return
            self.sample()
            self.require_localized()
            if worker.continuous and worker.continuous.active:
                self.continue_local_exploration()
                return
            if time.monotonic() < self.recheck_at:
                return
            if worker.continuous and getattr(worker.continuous, "home_owned", False):
                if worker.continuous.status != "arrived":
                    self.record_route_failure("controller", worker.continuous.reason)
                    if worker.continuous.reason.startswith(("SPATIAL_STALE:", "CANCELLED")):
                        raise ValueError(worker.continuous.reason)
                    task["retries"] += 1
                    if task["retries"] > 2:
                        raise ValueError("BLOCKED: bounded route retries exhausted; last route: " + worker.continuous.reason)
                    task["reason"] = "Replanning after " + worker.continuous.reason
                    timing_failure = worker.continuous.reason.startswith(("BUFFER_EXPIRED:", "BUFFER_EMPTY:", "MOTION_LEASE_EXPIRED:"))
                    if not timing_failure and task["kind"] == "explore" and not task.get("single_frontier") and task["frontier_id"]:
                        task.setdefault("rejected_frontiers", []).append(task["frontier_id"])
                        self.home.mark_frontier(task["frontier_id"])
                        task["target_m"] = None
                worker.continuous = None
            if task["target_m"] is not None and math.dist(self.pose[:2], task["target_m"]) <= .15:
                sim.hold_current()
                if task["kind"] == "navigate":
                    import pybullet as bullet
                    before = sim.odometry.copy()
                    previous = before.copy()
                    stable_since = None
                    settle_started = sim.ticks
                    while stable_since is None or sim.ticks - stable_since < 132:
                        if time.monotonic() >= task["deadline"]:
                            self.fail("TIMEOUT: arrival inspection exceeded the task budget", "limited")
                            return
                        sim._ticks(12)
                        linear, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
                        grounded = any(contact[2] != sim.robot and contact[7][2] > .8 and contact[9] > .01
                            for contact in bullet.getContactPoints(bodyA=sim.robot, physicsClientId=sim.client))
                        settled = (math.dist(sim.odometry[:2], previous[:2]) <= .025 * .05
                            and abs(sim.odometry[2] - previous[2]) <= .15 * .05
                            and math.hypot(*linear[:2]) <= .025 and abs(angular[2]) <= .15
                            and grounded and not sim.proximity_sensors().collisions)
                        stable_since = (sim.ticks if stable_since is None else stable_since) if settled else None
                        previous = sim.odometry.copy()
                        if sim.ticks - settle_started >= 480:
                            raise ValueError("ARRIVAL_UNSTABLE: robot did not settle at the destination")
                    self.sample(force=True)
                    self.require_localized()
                    if time.monotonic() >= task["deadline"]:
                        self.fail("TIMEOUT: arrival inspection exceeded the task budget", "limited")
                        return
                    if sim.cancel.is_set() or self.stop_revision != worker.stop_revision or self.task_revision != worker.task_revision:
                        self.fail("CANCELLED: arrival inspection interrupted", "cancelled")
                        return
                    if math.dist(sim.odometry[:2], before[:2]) > .02 or abs(sim.odometry[2] - before[2]) > .05 or math.dist(self.pose[:2], task["target_m"]) > .15:
                        raise ValueError("ARRIVAL_UNSTABLE: robot did not remain at the named destination")
                    task.update(status="completed", reason="Observed survey waypoint reached; mapping remains a draft" if task.get("guided_mapping") else "Named destination reached within 0.15 m; scenario completion is separate",
                        completion_verified=True, arrived_simulated_s=sim.ticks / 240, arrived_at=time.monotonic())
                    if self.workflow and self.workflow["status"] == "navigating_to_room":
                        self.workflow.update(status="awaiting_room_report", reason="Physical room arrival; fresh visual match or mismatch required")
                    elif self.workflow and self.workflow["status"] == "returning":
                        self.workflow.update(status="completed", reason="Room arrival, reported visual match and Home return completed; semantic identity unverified")
                    return
                self.home.mark_frontier(task["frontier_id"])
                task["visited_frontiers"] += 1
                task["target_m"] = None
                self.sample(force=True)
                if task.get("single_frontier"):
                    task.update(status="completed", reason="Reached selected frontier; fresh room observations required")
                    self.stage = "review"
                    return
            radius = sim.robot_footprint()["radius_m"]
            if task["target_m"] is None:
                frontiers = self.home.frontiers(self.pose, radius, self.obstacles(), task["region_id"],
                    excluded=task.get("rejected_frontiers", ()))
                if not frontiers:
                    task.update(status="completed", reason="No untried reachable frontiers; closed and unknown areas remain unexplored", completion_verified=True)
                    self.stage = "review"
                    return
                task["target_m"], task["frontier_id"] = frontiers[0]["position_m"], frontiers[0]["frontier_id"]
                if math.dist(self.pose[:2], task["target_m"]) <= .15:
                    self.home.mark_frontier(task["frontier_id"])
                    task["visited_frontiers"] += 1
                    task["target_m"] = None
                    self.sample(force=True)
                    self.recheck_at = time.monotonic() + .2
                    return
            try:
                self.route = self.home.route(self.pose[:2], task["target_m"], radius, self.obstacles())
            except ValueError as error:
                self.record_route_failure("map_replan", str(error))
                self.home.route(self.pose[:2], self.pose[:2], radius, self.obstacles())
                if task["kind"] != "explore" or task.get("single_frontier") or task["retries"] >= 2:
                    raise
                task.setdefault("rejected_frontiers", []).append(task["frontier_id"])
                self.home.mark_frontier(task["frontier_id"])
                task["target_m"] = None
                task["retries"] += 1
                task["reason"] = "Selecting another frontier after map rejection: " + str(error)
                self.recheck_at = time.monotonic() + .2
                return
            allowed = self.allowed(radius)
            endpoint = self.route[1]
            self.task["route_index"] = 1
            for index, candidate in enumerate(self.route[2:], 2):
                if math.dist(self.pose[:2], candidate) > 1.:
                    break
                if not self.home.segment_allowed(self.pose[:2], candidate, allowed):
                    break
                endpoint = candidate
                self.task["route_index"] = index
            target = transform_pose([*endpoint, 0.], inverse_pose(self.transform))[:2]
            worker._sample_spatial(force=True)
            if not worker.spatial_enabled or worker.spatial_map is None or worker.spatial_error or worker.spatial_map.captured_at is None or time.monotonic() - worker.spatial_map.captured_at > 1.:
                raise ValueError("SPATIAL_STALE: fresh paired depth required before moving")
            if sim.cancel.is_set() or self.stop_revision != worker.stop_revision or self.task_revision != worker.task_revision or time.monotonic() >= task["deadline"]:
                raise ValueError("CANCELLED: authority expired during route preparation")
            worker._start_navigation_path(np.array([sim.odometry[:2], target]), recover_clearance=True, home_owned=True)
            task["segments"] += 1
            self.recheck_at = time.monotonic() + .2
        except (ValueError, MotionError) as error:
            self.fail(str(error))
        except Exception as error:
            self.error = f"MAPPING_EXECUTION_FAILED: {type(error).__name__}: {error}"
            self.fail(self.error)
            sim.stop()

    def record_route_failure(self, phase, reason):
        failures = self.task.setdefault("route_failures", [])
        failures.append({"phase": phase, "reason": reason, "segment": self.task.get("segments", 0),
            "frontier_id": self.task.get("frontier_id"),
            "elapsed_s": round(time.monotonic() - self.task.get("started_at", time.monotonic()), 3)})
        runtime = getattr(self.worker, "navigation", None)
        if runtime is not None and hasattr(runtime, "diagnostic_state"):
            failures[-1]["motion_diagnostics"] = runtime.diagnostic_state()
        failures[-1]["clock"] = "monotonic"
        del failures[:-4]

    def continue_local_exploration(self):
        worker, task = self.worker, self.task
        control, runtime, now = worker.continuous, worker.navigation, time.monotonic()
        if (not task or not task.get("local_exploration") or not self.active or not control or not control.active
                or not getattr(control, "home_owned", False) or not runtime or not runtime.buffer
                or runtime.status != "running" or now >= runtime.expires_at or now >= runtime.skill_deadline
                or now >= task["deadline"] or worker.sim.cancel.is_set()
                or self.stop_revision != worker.stop_revision or self.task_revision != worker.task_revision
                or now - task.get("continued_at", 0.) < .25
                or self.captured_at is None or not 0 <= now - self.captured_at <= self.max_frame_age_s):
            return
        if np.linalg.norm(control.path[-1] - worker.sim.odometry[:2]) > .65:
            return
        radius = worker.sim.robot_footprint()["radius_m"]
        if radius > control.radius + .01 or worker.sim.held:
            return
        self.require_localized()
        task["continued_at"] = now
        allowed = self.allowed(radius)
        chosen = task["route_index"]
        for index in range(chosen + 1, len(self.route)):
            target = self.route[index]
            if math.dist(self.pose[:2], target) > 1.:
                break
            if not self.home.segment_allowed(self.pose[:2], target, allowed):
                break
            delta = np.asarray(target) - self.pose[:2]
            bearing = math.atan2(delta[1], delta[0]) - self.pose[2]
            if abs(math.atan2(math.sin(bearing), math.cos(bearing))) <= .35:
                chosen = index
        if chosen <= task["route_index"]:
            return
        target = transform_pose([*self.route[chosen], 0.], inverse_pose(self.transform))[:2]
        path = np.array([worker.sim.odometry[:2], target])
        if (time.monotonic() >= min(runtime.expires_at, runtime.skill_deadline, task["deadline"])
                or worker.sim.cancel.is_set() or self.stop_revision != worker.stop_revision or self.task_revision != worker.task_revision):
            return
        control.continue_path(path)
        control.start_position = worker.sim.odometry[:2].copy()
        runtime.authorized_travel_m = runtime.travel + float(np.linalg.norm(path[1] - path[0])) + .15
        task["route_index"] = chosen
        task["continuations"] = task.get("continuations", 0) + 1

    def state(self, compact=False, lightweight=False):
        home = self.home
        result = {"environment_id": self.environment_id, "stage": self.stage, "map_id": home.identity if home else None,
            "controller": "builtin_mapped_navigation", "design": "guided-waypoints-v1", "version": "0.4.0",
            "name": home.name if home else None, "revision": home.revision if home else 0, "expansion_allowed": self.allow_expansion,
            "localization": {**self.localization, "age_s": max(0., time.monotonic() - self.sampled_at),
                "age_basis": "time_since_last_laser_sample", "sample_clock": "monotonic",
                "last_sample_at_s": self.sampled_at, "last_consistency_check_at_s": self.validated_at,
                "tracking_method": "wheel_odometry_with_fixed_pose_scan_consistency",
                "continuous_pose_correction": False,
                "pose_m_rad": self.pose, "map_from_odometry_m_rad": self.transform},
            "frame": "map", "units": "m_rad", "timestamp_unix_s": time.time(), "error": self.error,
            "task": {key: value for key, value in self.task.items() if key not in {"deadline", "started_at", "arrived_at"}} if self.task else None,
            "places": [], "coverage": None}
        result["room_workflow"] = {key: value for key, value in self.workflow.items() if key != "deadline"} if self.workflow else None
        if lightweight:
            from copy import deepcopy
            result.pop("environment_id", None)
            result["task"] = deepcopy(result["task"])
            result["room_observations"] = deepcopy(self.room_records[:8])
            result["frontiers"] = []
            result["frontier_selection_available"] = False
            if home:
                result["coverage"] = {"free_m2": float(np.count_nonzero(home.cells == 0) * home.resolution_m ** 2),
                    "known_cells": int(np.count_nonzero(home.evidence)), "visited_cells": int(np.count_nonzero(home.visits)),
                    "scan_count": home.scan_count, "whole_home_percentage": None}
            return result
        if home:
            radius = self.worker.sim.robot_footprint()["radius_m"]
            fresh = (self.pose is not None and self.localization["status"] == "localized"
                and time.monotonic() - self.sampled_at <= .75 and time.monotonic() - self.validated_at <= 2.)
            allowed = home.allowed(radius, self.obstacles()) if fresh else None
            components = None
            if allowed is not None:
                from scipy.ndimage import label
                components, _ = label(allowed)
                position = home.indices(self.pose[:2])
                component = components[position[1], position[0]] if home.inside(position) else 0
            for place in home.places:
                column, row = home.indices(place["pose_m_rad"][:2])
                reachable = bool(components is not None and component and home.inside([column, row]) and components[row, column] == component)
                result["places"].append({**place, "reachable": reachable,
                    "identity_status": place.get("identity_status", "operator_named_unreviewed")})
            result["room_graph"] = {"map_id": home.identity, "frame": "map", "timestamp_unix_s": time.time(),
                **home.graph_summary(allowed)}
            result["frontier_selection_available"] = fresh and not self.active
            result["frontiers"] = home.frontiers(self.pose, radius, self.obstacles())[:8] if result["frontier_selection_available"] else []
            record = self.room_verification
            current = bool(record and self.task and record["task_id"] == self.task["task_id"]
                and record["run_id"] == self.worker.sim.run_id and record["episode_epoch"] == self.worker.sim.epoch
                and not self.worker.sim.cancel.is_set() and record["stop_revision"] == self.worker.stop_revision
                and record["task_revision"] == self.worker.task_revision and fresh and time.monotonic() <= self.room_report_deadline
                and self.room_view_matches(record["pose_m_rad"], record["head_rad"]))
            if record and not current:
                self.room_verification = None
            result["room_verification"] = {"status": "visual_match_reported" if current and record["room_matches"] else "visual_mismatch_reported" if current else "not_currently_verified",
                "evidence_id": record["observation_id"] if record else None, "place_id": record["place_id"] if record else None,
                "physical_arrival_is_room_identity": False, "identity_verified": False,
                "source": "camera_annotation_not_independent_semantic_evaluation"}
            result["room_observations"] = self.room_records[:8 if compact else 100]
            result["coverage"] = {"free_m2": float(np.count_nonzero(home.cells == 0) * home.resolution_m ** 2),
                "known_cells": int(np.count_nonzero(home.evidence)), "visited_cells": int(np.count_nonzero(home.visits)),
                "scan_count": home.scan_count, "whole_home_percentage": None}
            result["objects"] = [{**record, "currently_observed": bool(record["run_id"] == self.worker.sim.run_id
                and record["episode_epoch"] == self.worker.sim.epoch and self.worker.spatial_map
                and record["spatial_sequence"] == self.worker.spatial_map.sequence
                and 0 <= time.time() - record["observed_unix_s"] <= 2.)} for record in self.object_records[:8 if compact else 64]]
            if not compact:
                result.update(map={"cells": home.cells.ravel().tolist(), "width": home.size, "height": home.size,
                    "origin_m": home.origin.tolist(), "resolution_m": home.resolution_m},
                    edges=home.edges, live_obstacles_m=self.obstacles(), route_m=self.route,
                    frontier_attempts=home.frontier_attempts, dirty=not home.saved or home.annotations_dirty)
        if not compact:
            result["maps"] = self.store.catalog()
        else:
            result.pop("environment_id", None)
        return result