import math
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import Field

from backend.contracts import StrictModel
from backend.home_mapping import HomeMap, MapStore, inverse_pose, laser_points, transform_pose
from backend.simulation import MotionError


DEFAULT_MAP_PATH = Path(__file__).resolve().parents[1] / ".runtime" / "maps" / "homes.sqlite3"


class HomeRequest(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    action: Literal["start_mapping", "review", "save_map", "load_map", "localize", "add_place", "navigate_to", "explore", "cancel_task", "get_spatial_state", "remember_object"]
    object_label: str = Field(default="", max_length=80)
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    confidence: float = Field(default=.5, ge=0., le=1., allow_inf_nan=False)
    spatial_sequence: int | None = Field(default=None, ge=1)
    map_id: str | None = Field(default=None, max_length=80)
    name: str = Field(default="", max_length=80)
    place_id: str | None = Field(default=None, max_length=80)
    kind: Literal["room", "doorway", "destination"] = "destination"
    pose_m_rad: list[float] | None = Field(default=None, min_length=3, max_length=3)
    connects: list[str] = Field(default_factory=list, max_length=20)
    region_id: str | None = Field(default=None, max_length=80)
    time_budget: float = Field(default=60., ge=1., le=300., allow_inf_nan=False)


class HomeMission:
    def __init__(self, worker, store=None):
        self.worker = worker
        self.store = store or MapStore(DEFAULT_MAP_PATH)
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
        self.route = []
        self.allowed_cache = None
        self.stop_revision = worker.stop_revision
        self.task_revision = worker.task_revision
        self.recheck_at = 0.
        self.last_laser = None
        self.error = None
        self.object_records = []
        self.allow_expansion = True

    @property
    def active(self):
        return bool(self.task and self.task["status"] == "running")

    def fail(self, reason, status="failed"):
        worker = self.worker
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
        self.transform = None
        self.pose = None
        self.localization["status"] = "unlocalized"
        self.live.clear()
        if self.stage == "mapping":
            self.stage = "review"

    def sample(self, force=False):
        try:
            self._sample(force)
        except Exception as error:
            self.error = f"MAPPING_SENSOR_FAILED: {type(error).__name__}: {error}"
            self.invalidate(self.error)
            self.worker.sim.stop()

    def _sample(self, force=False):
        from backend.ros_navigation import capture_laser
        from backend.spatial import point_cloud
        if self.home is None:
            return
        now = time.monotonic()
        if not force and now - self.sampled_at < .2:
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
        mapped = self.stage == "mapping" or (self.active and self.task["kind"] == "explore")
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
        if spatial and spatial.observation and spatial.sequence != self.last_depth_sequence and now - spatial.captured_at <= 1.:
            _, points = point_cloud(spatial.observation, stride=4)
            points = points[(points[:, 2] > .04) & (points[:, 2] < 1.3)]
            for point in points:
                column, row = self.home.indices(transform_pose([*point[:2], 0.], self.transform)[:2])
                self.live[(int(column), int(row))] = now
                if mapped and self.home.inside([column, row]):
                    self.home.evidence[row, column] = max(4, self.home.evidence[row, column])
            self.last_depth_sequence = spatial.sequence
        self.localization["status"] = "localized"
        self.allowed_cache = None

    def obstacles(self):
        now = time.monotonic()
        return [(self.home.origin + (np.array(cell) + .5) * self.home.resolution_m).tolist()
            for cell, timestamp in self.live.items() if now - timestamp <= 2.]

    def require_localized(self):
        if self.pose is None or self.localization["status"] != "localized" or time.monotonic() - self.sampled_at > .75 or time.monotonic() - self.validated_at > 2.:
            raise ValueError("LOCALIZATION_REQUIRED: localize against the saved map with fresh sensors")

    def command(self, request, stop_revision, task_revision, selected_evidence=None):
        worker, sim = self.worker, self.worker.sim
        if request.run_id != sim.run_id or request.episode_epoch != sim.epoch:
            raise MotionError("STALE_STATE", "Map request belongs to another episode")
        if request.action == "get_spatial_state":
            return self.state()
        if request.action == "cancel_task":
            self.fail("Cancelled by operator", "cancelled")
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
            self.object_records = self.store.object_observations(self.home.identity)
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
            self.pose, self.last_odometry = pose, sim.odometry.tolist()
            self.localization = {"status": "localized", "method": "scan_to_saved_map", "quality": quality}
            self.validated_at = time.monotonic()
            self.error = None
            self.sample(force=True)
        elif request.action == "review":
            self.stage = "review"
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
        elif request.action in {"navigate_to", "explore"}:
            if not self.home.revision:
                raise ValueError("SAVE_REQUIRED: review and save the initial map before autonomous tasks")
            if not worker.spatial_enabled or worker.spatial_map is None:
                raise ValueError("SPATIAL_REQUIRED: enable spatial sensing before mapped navigation")
            self.sample(force=True)
            self.require_localized()
            if sim.held or any(any(gripper.contact) for gripper in sim.observe(render=False).grippers.values()):
                raise ValueError("OBJECT_HELD: release objects before mapped navigation")
            target = self.place(request.place_id)["pose_m_rad"][:2] if request.action == "navigate_to" else None
            radius = sim.robot_footprint()["radius_m"]
            if target is not None:
                self.home.route(self.pose[:2], target, radius, self.obstacles())
            else:
                if not self.allow_expansion:
                    raise ValueError("MAP_READ_ONLY: scenario evaluations cannot expand the frozen map")
                if not self.home.frontiers(self.pose, radius, self.obstacles(), request.region_id):
                    raise ValueError("NO_FRONTIERS: no untried reachable frontier in this region")
                self.home.saved = False
            self.task = {"task_id": str(uuid4()), "kind": "navigate" if target is not None else "explore",
                "status": "running", "reason": "Planning on observed map", "place_id": request.place_id,
                "region_id": request.region_id, "target_m": target, "started_at": time.monotonic(),
                "deadline": time.monotonic() + request.time_budget, "retries": 0, "segments": 0,
                "frontier_id": None, "visited_frontiers": 0, "completion_verified": False}
            self.stage = "navigation" if target is not None else "expansion"
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

    def path_valid(self, pose, linear, angular):
        if not self.active:
            return False
        self.require_localized()
        radius = self.worker.sim.robot_footprint()["radius_m"]
        if self.allowed_cache is None:
            self.allowed_cache = self.home.allowed(radius, self.obstacles())
        points = []
        for speed, turn in [(linear, angular), self.worker.navigation.velocity]:
            for horizon in np.linspace(0., 1., 16):
                heading = pose[2] + turn * horizon / 2
                point = np.array(pose[:2]) + speed * horizon * np.array([math.cos(heading), math.sin(heading)])
                points.append(transform_pose([*point, 0.], self.transform)[:2])
        indices = self.home.indices(points)
        return bool(self.home.inside(indices).all() and self.allowed_cache[indices[:, 1], indices[:, 0]].all())

    def tick(self):
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
                return
            if time.monotonic() < self.recheck_at:
                return
            if worker.continuous and getattr(worker.continuous, "home_owned", False):
                if worker.continuous.status != "arrived":
                    if worker.continuous.reason.startswith(("SPATIAL_STALE:", "CANCELLED")):
                        raise ValueError(worker.continuous.reason)
                    task["retries"] += 1
                    if task["retries"] > 2:
                        raise ValueError("BLOCKED: bounded route retries exhausted")
                worker.continuous = None
            if task["target_m"] is not None and math.dist(self.pose[:2], task["target_m"]) <= .15:
                sim.hold_current()
                if task["kind"] == "navigate":
                    before = sim.odometry.copy()
                    sim._ticks(120)
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
                    task.update(status="completed", reason="Named destination reached within 0.15 m; scenario completion is separate", completion_verified=True)
                    return
                self.home.mark_frontier(task["frontier_id"])
                task["visited_frontiers"] += 1
                task["target_m"] = None
                self.sample(force=True)
            radius = sim.robot_footprint()["radius_m"]
            if task["target_m"] is None:
                frontiers = self.home.frontiers(self.pose, radius, self.obstacles(), task["region_id"])
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
                self.home.route(self.pose[:2], self.pose[:2], radius, self.obstacles())
                if task["kind"] != "explore" or task["retries"] >= 2:
                    raise
                self.home.mark_frontier(task["frontier_id"])
                task["target_m"] = None
                task["retries"] += 1
                task["reason"] = "Frontier blocked by fresh observations; selecting another reachable boundary"
                self.recheck_at = time.monotonic() + .2
                return
            allowed = self.home.allowed(radius, self.obstacles())
            endpoint = self.route[1]
            for candidate in self.route[2:]:
                if math.dist(self.pose[:2], candidate) > 1.:
                    break
                indices = self.home.indices(np.linspace(self.pose[:2], candidate, max(2, math.ceil(math.dist(self.pose[:2], candidate) / .025))))
                if not allowed[indices[:, 1], indices[:, 0]].all():
                    break
                endpoint = candidate
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

    def state(self, compact=False):
        home = self.home
        result = {"environment_id": self.environment_id, "stage": self.stage, "map_id": home.identity if home else None,
            "controller": "builtin_mapped_navigation", "design": "persistent-home-map-v1", "version": "0.1.0",
            "name": home.name if home else None, "revision": home.revision if home else 0,
            "localization": {**self.localization, "age_s": max(0., time.monotonic() - self.sampled_at),
                "pose_m_rad": self.pose, "map_from_odometry_m_rad": self.transform},
            "frame": "map", "units": "m_rad", "timestamp_unix_s": time.time(), "error": self.error,
            "task": {key: value for key, value in self.task.items() if key not in {"deadline", "started_at"}} if self.task else None,
            "places": [], "coverage": None}
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
                result["places"].append({**place, "reachable": reachable})
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
        return result