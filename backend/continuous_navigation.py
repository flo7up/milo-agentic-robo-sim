import math
import time
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from backend.contracts import StrictModel
from backend.navigation import CONTINUOUS_SPEED_MPS, LINEAR_ACCELERATION_MPS2, DriveSegment, LocalSubgoal, ReplaceMotionBuffer
from backend.simulation import MotionError


CONTINUOUS_ARRIVAL_M = .04
CONTINUOUS_APPROACH_GAIN = .7
CONTINUOUS_ANGULAR_SPEED_RADPS = .5


class ContinuousScan(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    compact_arms: bool = False


class ContinuousTarget(ContinuousScan):
    spatial_sequence: int = Field(ge=1)
    pixel: list[float] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_pixel(self):
        if not all(0 <= value <= 1 for value in self.pixel):
            raise ValueError("Select a pixel inside the paired camera image")
        return self


def target_from_depth(observation, pixel):
    from backend.spatial import camera_in_odometry
    intrinsics = observation.calibration
    column = min(intrinsics.width - 1, int(pixel[0] * intrinsics.width))
    row = min(intrinsics.height - 1, int(pixel[1] * intrinsics.height))
    depth = observation.depth_m[row * intrinsics.width + column]
    if depth is None:
        raise ValueError("Selected pixel has no valid floor depth")
    optical = np.array([(column + .5 - intrinsics.cx) / intrinsics.fx * depth,
                        (row + .5 - intrinsics.cy) / intrinsics.fy * depth, depth])
    eye, rotation = camera_in_odometry(observation.head_rad, observation.odometry_m_rad)
    point = eye + rotation @ optical
    if abs(point[2]) > .04:
        raise ValueError("Select observed floor, not an object, wall or robot part")
    distance = np.linalg.norm(point[:2] - observation.odometry_m_rad[:2])
    if not .15 <= distance <= 1.8:
        raise ValueError("Select a floor destination between 0.15 and 1.8 m away")
    return point[:2].tolist()


class ContinuousDriveSegment(DriveSegment):
    linear_mps: float = Field(ge=-CONTINUOUS_SPEED_MPS, le=CONTINUOUS_SPEED_MPS)


class ContinuousMotionBuffer(ReplaceMotionBuffer):
    segments: list[ContinuousDriveSegment] = Field(min_length=1, max_length=4)


MOTION_SKILL_MIN_LENGTH_M = {"cruise": .2, "curve": .2, "approach": .15}


class CruiseSkill(StrictModel):
    kind: Literal["cruise"]
    length_m: float = Field(ge=MOTION_SKILL_MIN_LENGTH_M["cruise"], le=1.8, allow_inf_nan=False)
    view: Literal["forward", "left", "right"] = "forward"


class CurveSkill(CruiseSkill):
    kind: Literal["curve"]
    angle_rad: float = Field(ge=-1.2, le=1.2, allow_inf_nan=False)


class ApproachSkill(StrictModel):
    kind: Literal["approach"]
    candidate_id: int = Field(ge=0, le=40, strict=True)
    view: Literal["forward", "left", "right"] = "forward"


class InspectSkill(StrictModel):
    kind: Literal["inspect"]
    yaw_rad: float = Field(default=0., ge=-1.2, le=1.2, allow_inf_nan=False)
    pitch_rad: float = Field(default=.2, ge=0., le=.85, allow_inf_nan=False)
    duration_s: float = Field(default=1., ge=.5, le=2., allow_inf_nan=False)


class MotionSkillPlan(StrictModel):
    components: list[Annotated[CruiseSkill | CurveSkill | ApproachSkill | InspectSkill,
                               Field(discriminator="kind")]] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def ordered_components(self):
        moving = [component for component in self.components if component.kind != "inspect"]
        if not moving:
            raise ValueError("A motion plan needs at least one moving component")
        if any(component.kind == "inspect" for component in self.components[:-1]):
            raise ValueError("Inspection must be terminal; review fresh feedback before further motion")
        if any(component.kind == "approach" for component in moving[:-1]):
            raise ValueError("Approach must be the last moving component")
        return self


def compile_motion_skills(pose, plan, candidates):
    from scipy.integrate import cumulative_trapezoid
    from scipy.interpolate import BPoly
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("Skill plan requires a finite observed pose")
    point, heading = pose[:2].copy(), float(pose[2])
    path, boundaries = [point.copy()], []
    travelled = 0.
    for component in plan.components:
        if component.kind == "inspect":
            break
        if component.kind == "approach":
            candidate = next((item for item in candidates if item["id"] == component.candidate_id), None)
            if candidate is None:
                raise ValueError("Approach requires a current observed candidate")
            target = np.asarray(candidate["target_m"], dtype=float)
            if target.shape != (2,) or not np.isfinite(target).all():
                raise ValueError("Approach candidate has invalid observed geometry")
            delta = target - point
            length = float(np.linalg.norm(delta))
            if not MOTION_SKILL_MIN_LENGTH_M["approach"] <= length <= 1.8:
                raise ValueError("Approach must be 0.15-1.8 m from the preceding component")
            bearing = math.atan2(delta[1], delta[0])
            if abs(math.atan2(math.sin(bearing - heading), math.cos(bearing - heading))) > .65:
                raise ValueError("A sharp approach needs a stopped reorientation")
            curve = BPoly.from_derivatives([0., 1.], [
                [point, length * np.array([math.cos(heading), math.sin(heading)]), np.zeros(2)],
                [target, delta, np.zeros(2)]])
            points = curve(np.linspace(0., 1., max(20, math.ceil(length / .02) + 1)))
            heading = bearing
        else:
            length = component.length_m
            fraction = np.linspace(0., 1., max(20, math.ceil(length / .02) + 1))
            angle = component.angle_rad if component.kind == "curve" else 0.
            headings = heading + angle * (10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5)
            directions = np.column_stack((np.cos(headings), np.sin(headings)))
            points = point + cumulative_trapezoid(directions, fraction * length, axis=0, initial=0.)
            heading += angle
        path.extend(points[1:])
        travelled += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if travelled > 1.8 + 1e-6:
            raise ValueError("Composed motion exceeds the 1.8 m planning horizon")
        boundaries.append({"kind": component.kind, "end_index": len(path) - 1,
            "distance_m": travelled, "view": component.view})
        point = points[-1]
    points = np.asarray(path)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    headings = np.unwrap(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
    curvature = np.abs(np.diff(headings)) / np.maximum((lengths[:-1] + lengths[1:]) / 2, 1e-6)
    if not np.isfinite(points).all() or np.any(lengths < 1e-6) or np.max(curvature, initial=0.) > 2.5:
        raise ValueError("Composed route has an infeasible curvature or degenerate join")
    return points, boundaries


class MotionSkillBudgetError(MotionError):
    def __init__(self, message, timing):
        super().__init__("PLAN_BUDGET", message)
        self.timing = timing


def motion_skill_timing(path, captured_at, now, *, wall_seconds_per_sim_second=1.):
    if not math.isfinite(wall_seconds_per_sim_second) or wall_seconds_per_sim_second < 1.:
        raise ValueError("Motion timing requires a finite wall/simulation ratio of at least one")
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    headings = np.unwrap(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])))
    curvature = np.abs(np.diff(headings)) / np.maximum((lengths[:-1] + lengths[1:]) / 2, 1e-6)
    planning_speed = min(CONTINUOUS_SPEED_MPS, CONTINUOUS_ANGULAR_SPEED_RADPS / max(float(np.max(curvature, initial=0.)), .01))
    slowdown_distance = planning_speed / CONTINUOUS_APPROACH_GAIN
    distance = float(lengths.sum())
    tail_distance = min(distance, slowdown_distance)
    tail_sim_s = math.log(max(tail_distance, CONTINUOUS_ARRIVAL_M) / CONTINUOUS_ARRIVAL_M) / CONTINUOUS_APPROACH_GAIN
    cruise_sim_s = max(0., distance - slowdown_distance) / planning_speed
    acceleration_sim_s = planning_speed / (2 * LINEAR_ACCELERATION_MPS2)
    steering_sim_s = float(np.abs(np.diff(headings)).sum()) / CONTINUOUS_ANGULAR_SPEED_RADPS
    reserve_s = 1.
    remaining_s = max(0., captured_at + 20. - now)
    estimated_sim_s = cruise_sim_s + tail_sim_s + acceleration_sim_s + steering_sim_s
    estimated_s = estimated_sim_s * wall_seconds_per_sim_second + reserve_s
    available_sim_s = max(0., (remaining_s - reserve_s) / wall_seconds_per_sim_second - acceleration_sim_s - steering_sim_s)
    full_tail_sim_s = math.log(slowdown_distance / CONTINUOUS_ARRIVAL_M) / CONTINUOUS_APPROACH_GAIN
    maximum_length = min(1.8, CONTINUOUS_ARRIVAL_M * math.exp(CONTINUOUS_APPROACH_GAIN * min(available_sim_s, full_tail_sim_s))
        + max(0., available_sim_s - full_tail_sim_s) * planning_speed) if remaining_s > reserve_s else 0.
    timing = {"estimated_motion_s": estimated_s, "remaining_lease_s": remaining_s,
        "planning_speed_mps": planning_speed, "maximum_length_m": maximum_length,
        "estimated_motion_sim_s": estimated_sim_s, "terminal_slowdown_sim_s": tail_sim_s,
        "acceleration_allowance_sim_s": acceleration_sim_s, "wall_reserve_s": reserve_s,
        "steering_allowance_sim_s": steering_sim_s,
        "wall_seconds_per_sim_second": wall_seconds_per_sim_second,
        "estimate_only": True, "inspection_in_motion_lease": False}
    if estimated_s > remaining_s:
        ceiling = math.floor(maximum_length * 100) / 100
        options = {kind: {"minimum_length_m": minimum, "maximum_length_m": ceiling}
            for kind, minimum in MOTION_SKILL_MIN_LENGTH_M.items() if ceiling >= minimum}
        timing.update(admission="shorter_plan_required" if options else "no_feasible_plan",
            scope="rejected_path_timing_estimate", motion_authorized=False,
            shorter_component_options=options, next_actions=["look", "scan", "wait"])
        guidance = ("Only the shorter_component_options ranges are legal sizing suggestions. Select a fresh observed candidate or route within those ranges, or reobserve. "
            "Changing geometry requires a new timing and clearance check; no alternative path is authorized. " if options else
            "No legal shorter moving component fits this rejected-path timing estimate. Use look, scan or wait for fresh stopped feedback. "
            "Do not request sub-minimum components. Reassess the next fresh observation; this does not prove every possible path is infeasible. ")
        raise MotionSkillBudgetError(f"Composed motion needs an estimated {estimated_s:.2f} s; only {remaining_s:.2f} s remain. "
            + guidance + "The estimate is not a timing guarantee; no motion or inspection was started.", timing)
    return timing


class ObservedOrbit:
    def __init__(self, sensor, bounds, label, direction, footprint_radius):
        from backend.spatial import camera_in_odometry
        if len(bounds) != 4 or not all(0 <= value <= 1 for value in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Select a finite object bounding box in the current camera")
        calibration = sensor.calibration
        rows, columns = np.mgrid[0:calibration.height, 0:calibration.width]
        depth = np.asarray(sensor.depth_m, dtype=float).reshape(rows.shape)
        selected = ((columns + .5 >= bounds[0] * calibration.width) & (columns + .5 <= bounds[2] * calibration.width)
                    & (rows + .5 >= bounds[1] * calibration.height) & (rows + .5 <= bounds[3] * calibration.height) & np.isfinite(depth))
        eye, rotation = camera_in_odometry(sensor.head_rad, sensor.odometry_m_rad)
        optical = np.stack(((columns + .5 - calibration.cx) * depth / calibration.fx,
                            (rows + .5 - calibration.cy) * depth / calibration.fy, depth), axis=-1)
        points = optical[selected] @ rotation.T + eye
        points = points[(points[:, 2] > .08) & (points[:, 2] < 1.6)]
        if len(points) < 8:
            raise ValueError("Object selection has insufficient measured surface depth")
        distances = np.linalg.norm(points - eye, axis=1)
        points = points[distances < np.percentile(distances, 10) + 1.2]
        lower, upper = np.percentile(points[:, :2], [5, 95], axis=0)
        self.center = (lower + upper) / 2
        self.radius = float(np.clip(np.linalg.norm(upper - lower) / 2 + footprint_radius + .65, 1.2, 2.1))
        self.run_id, self.episode_epoch = sensor.run_id, sensor.episode_epoch
        self.label, self.direction = label, direction
        self.sign = -1 if direction == "clockwise" else 1
        self.angle = None
        self.swept = 0.
        self.travel_m = 0.
        self.started = time.monotonic()

    def path(self, pose):
        delta = np.asarray(pose[:2]) - self.center
        distance = float(np.linalg.norm(delta))
        angle = math.atan2(delta[1], delta[0])
        if abs(distance - self.radius) > .3:
            target = self.center + delta / max(distance, .001) * self.radius
            displacement = target - pose[:2]
            if np.linalg.norm(displacement) > 1.4:
                target = np.asarray(pose[:2]) + displacement / np.linalg.norm(displacement) * 1.4
            return [list(pose[:2]), target.tolist()], False
        if self.angle is not None:
            difference = math.atan2(math.sin(angle - self.angle), math.cos(angle - self.angle))
            self.swept = max(0., self.swept + self.sign * difference)
        self.angle = angle
        if self.swept >= 2 * math.pi:
            return [], True
        step = min(.65, 2 * math.pi + .08 - self.swept)
        angles = angle + self.sign * np.linspace(step / 12, step, 12)
        points = self.center + self.radius * np.column_stack((np.cos(angles), np.sin(angles)))
        return [list(pose[:2]), *points.tolist()], False


def smooth_observed_route(pose, waypoints, moving=False):
    from scipy.interpolate import CubicSpline
    points = np.asarray([list(pose[:2]), *waypoints], dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 25 or not np.isfinite(points).all():
        raise ValueError("Route requires 1-24 finite observed XY waypoints")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(distances < .05) or not .3 <= distances.sum() <= 6.:
        raise ValueError("Route segments must be distinct and total length between 0.3 and 6 m")
    initial = points[1] - points[0]
    bearing = math.atan2(initial[1], initial[0])
    difference = math.atan2(math.sin(bearing - pose[2]), math.cos(bearing - pose[2]))
    if moving and abs(difference) > .65:
        raise ValueError("A sharp route change requires braking before reorientation")
    heading = pose[2] if moving or abs(difference) <= .65 else bearing
    knots = np.r_[0., np.cumsum(distances)]
    curve = CubicSpline(knots, points, axis=0,
        bc_type=((1, np.array([math.cos(heading), math.sin(heading)])), (1, (points[-1] - points[-2]) / distances[-1])))
    path = curve(np.linspace(0., knots[-1], max(2, math.ceil(knots[-1] / .025) + 1)))
    path[0], path[-1] = points[0], points[-1]
    if not np.isfinite(path).all() or np.linalg.norm(np.diff(path, axis=0), axis=1).sum() > 6.:
        raise ValueError("Smoothed route exceeds the bounded planning horizon")
    return path


class ObservedExploration:
    def __init__(self, heading, travel=0., clock=time.monotonic):
        if not math.isfinite(heading):
            raise ValueError("Exploration heading must be finite")
        self.heading = heading
        self.started = clock()
        self.expires_at = self.started + 20.
        self.travel_limit = travel + 6.
        self.revision = 0
        self.last_attempt = -math.inf
        self.ending = False

    def renew(self, heading, captured_at, captured_travel, now, travel):
        if (self.ending or now >= self.expires_at or now >= self.started + 180 or travel >= self.travel_limit
                or not 0 <= now - captured_at <= 15 or not math.isfinite(heading)
                or not 0 <= captured_travel <= travel or travel >= captured_travel + 6.):
            raise ValueError("Exploration review lost its time or distance authority")
        self.heading = heading
        self.expires_at = min(captured_at + 20., self.started + 180.)
        self.travel_limit = captured_travel + 6.
        self.revision += 1

    def paths(self, pose):
        for offset in (0., -.15, .15, -.3, .3):
            difference = math.atan2(math.sin(self.heading + offset - pose[2]), math.cos(self.heading + offset - pose[2]))
            if abs(difference) > .65:
                continue
            headings = pose[2] + np.linspace(0., difference, 13)[1:]
            steps = 1.4 / 12 * np.column_stack((np.cos(headings), np.sin(headings)))
            yield np.vstack((pose[:2], np.asarray(pose[:2]) + np.cumsum(steps, axis=0)))


class ContinuousNavigation:
    def __init__(self, path, clock=time.monotonic):
        self.path = np.asarray(path, dtype=float)
        if self.path.ndim != 2 or self.path.shape[1] != 2 or len(self.path) < 2 or not np.isfinite(self.path).all():
            raise ValueError("A finite path with a start and destination is required")
        self.clock = clock
        self.identity = str(uuid4())
        self.revision = 0
        self.handoffs = 0
        self.status = "running"
        self.reason = "Following observed path"
        self.index = 1
        self.updates = 0
        self.last_update = -math.inf
        self.maximum_update_gap_s = 0.
        self.started = clock()
        self.distance_m = float(np.linalg.norm(self.path[-1] - self.path[0]))
        self.arrival_m = CONTINUOUS_ARRIVAL_M
        self.minimum_cruise_speed = CONTINUOUS_SPEED_MPS
        self.buffer_stops = 0
        self.replans = 0
        self.refill_after_replan = False
        self.circuit_following = False
        self.skill_components = []
        self.skill_timing = None
        self.exploration = None
        self.ai_route = False
        self.progress_at = clock()
        self.progress_pose = None

    @property
    def active(self):
        return self.status == "running"

    def continue_path(self, path):
        points = np.asarray(path, dtype=float)
        if (not self.active or points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.isfinite(points).all() or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > 1.8):
            raise ValueError("Continuation requires an active route and a finite bounded path")
        self.path = points
        self.index = 1
        self.revision += 1
        self.handoffs += 1
        self.distance_m = float(np.linalg.norm(points[-1] - points[0]))
        self.reason = "Following a freshly validated rolling continuation"

    def state(self):
        return {"status": self.status, "reason": self.reason, "path_m": self.path.tolist(),
            "target_m": self.path[-1].tolist(), "remaining_m": round(self.distance_m, 3),
            "updates": self.updates, "buffer_stops": self.buffer_stops,
            "replans": self.replans,
            "handoffs": self.handoffs, "revision": self.revision,
            "ai_generated_route": self.ai_route,
            "skill_components": self.skill_components,
            "skill_timing": self.skill_timing,
            "rolling_exploration": self.exploration is not None and not self.exploration.ending,
            "exploration_reviews": self.exploration.revision if self.exploration else 0,
            "exploration_review_remaining_s": max(0., self.exploration.expires_at - self.clock()) if self.exploration else None,
            "elapsed_s": round(self.clock() - self.started, 3), "controller": "continuous_local",
            "speed_limit_mps": CONTINUOUS_SPEED_MPS,
            "minimum_cruise_speed_mps": self.minimum_cruise_speed,
            "maximum_update_gap_s": self.maximum_update_gap_s}

    def finish(self, sim, runtime, status, reason):
        self.status, self.reason = status, reason
        runtime.cancel(sim, reason)

    def start(self, sim, runtime):
        runtime.continuous_motion = True
        observation = sim.observe(render=False)
        runtime.observe(sim, observation)
        runtime.apply(sim, "begin_local_subgoal", LocalSubgoal(expected_revision=runtime.revision,
            goal="Operator-selected observed floor destination"), observation.seq)

    def update(self, sim, runtime, observed_map, path_valid, replan=None):
        if not self.active:
            return
        runtime.command_sensor = {"kind": "observed_depth", "clock": "monotonic",
            "captured_at_s": observed_map.captured_at,
            "age_s": self.clock()-observed_map.captured_at if observed_map.captured_at is not None else None}
        if sim.cancel.is_set():
            self.finish(sim, runtime, "cancelled", "Stopped by operator")
            return
        if runtime.status in {"failed", "cancelled"}:
            self.status, self.reason = "blocked", runtime.reason
            return
        if runtime.status == "awaiting_feedback" and runtime.reason.startswith("CLEARANCE_STOP:"):
            self.finish(sim, runtime, "blocked", runtime.reason)
            return
        if self.updates and runtime.buffer and self.clock() >= runtime.expires_at:
            self.finish(sim, runtime, "blocked", "BUFFER_EXPIRED: Motion authorization expired")
            return
        if self.clock() >= runtime.skill_deadline:
            if runtime.recover_deadline:
                runtime.expire_skill(sim)
                self.status, self.reason = "blocked", runtime.reason
            else:
                self.finish(sim, runtime, "blocked", "SKILL_TIMEOUT: Local goal deadline expired")
            return
        if observed_map.captured_at is None or not 0 <= self.clock() - observed_map.captured_at <= observed_map.max_frame_age_s:
            self.finish(sim, runtime, "blocked", "SPATIAL_STALE: Fresh depth is required")
            return
        if self.clock() - self.last_update < .05:
            return
        if self.updates:
            self.maximum_update_gap_s = max(self.maximum_update_gap_s, self.clock() - self.last_update)
        self.last_update = self.clock()
        pose = sim.odometry.copy()
        if self.progress_pose is None or np.linalg.norm(pose[:2] - self.progress_pose[:2]) > .025 or abs(pose[2] - self.progress_pose[2]) > .08:
            self.progress_at, self.progress_pose = self.clock(), pose.copy()
        elif self.clock() - self.progress_at > 3.:
            self.finish(sim, runtime, "blocked", "NO_PROGRESS: No clearance-valid motion toward this path; inspect another route")
            return
        self.distance_m = float(np.linalg.norm(self.path[-1] - pose[:2]))
        if self.distance_m < self.arrival_m and (not self.ai_route or self.index >= len(self.path) - 2):
            self.finish(sim, runtime, "arrived", "Reached selected floor point")
            return
        while self.index < len(self.path) - 1:
            incoming = self.path[self.index] - self.path[self.index - 1]
            outgoing = self.path[self.index + 1] - self.path[self.index]
            cosine = float(np.dot(incoming, outgoing) / max(1e-9, np.linalg.norm(incoming) * np.linalg.norm(outgoing)))
            threshold = .35 if self.ai_route or self.circuit_following or (self.exploration is not None and self.handoffs) else .2 if cosine > .96 else .05
            if np.linalg.norm(self.path[self.index] - pose[:2]) >= threshold:
                break
            self.index += 1
        delta = self.path[self.index] - pose[:2]
        if self.ai_route:
            self.distance_m = float(np.linalg.norm(delta) + np.linalg.norm(np.diff(self.path[self.index:], axis=0), axis=1).sum())
        bearing = math.atan2(delta[1], delta[0]) - pose[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        angular = float(np.clip(2 * bearing, -CONTINUOUS_ANGULAR_SPEED_RADPS, CONTINUOUS_ANGULAR_SPEED_RADPS))
        approach = self.distance_m if self.ai_route or self.circuit_following or (self.exploration is not None and self.handoffs) or (self.index < len(self.path) - 1 and cosine > .96) else float(np.linalg.norm(delta))
        linear = min(CONTINUOUS_SPEED_MPS, CONTINUOUS_APPROACH_GAIN * approach) * max(0., math.cos(bearing)) if abs(bearing) < .7 else 0.
        if self.skill_components and self.index < len(self.path) - 2:
            ahead = self.path[max(0, self.index - 1):min(len(self.path), self.index + 20)]
            lengths = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
            headings = np.unwrap(np.arctan2(np.diff(ahead[:, 1]), np.diff(ahead[:, 0])))
            curvature = np.abs(np.diff(headings)) / np.maximum((lengths[:-1] + lengths[1:]) / 2, 1e-6)
            linear = min(linear, CONTINUOUS_ANGULAR_SPEED_RADPS / max(.01, float(np.max(curvature, initial=0.))))
        clearances = [reading.distance_m for reading in sim.proximity_sensors().distances
                      if reading.direction in {"front", "front_left", "front_right"} and reading.status == "hit"]
        if clearances:
            stopping_room = max(0., min(clearances) - .1)
            reaction_speed = LINEAR_ACCELERATION_MPS2 * .05
            clearance_speed = math.sqrt(reaction_speed ** 2 + 2 * LINEAR_ACCELERATION_MPS2 * stopping_room) - reaction_speed
            linear = min(linear, clearance_speed, .2 if min(clearances) < 1. else CONTINUOUS_SPEED_MPS)
        if not path_valid(pose, linear, angular):
            alternatives = [(linear * .5, angular)]
            if getattr(self, "home_owned", False) and linear > 0. and abs(bearing) < .2:
                alternatives.append((linear * .5, 0.))
            alternatives.append((0., angular))
            for reduced_linear, reduced_angular in alternatives:
                if path_valid(pose, reduced_linear, reduced_angular):
                    linear, angular = reduced_linear, reduced_angular
                    break
            else:
                if replan is not None and self.replans < 2:
                    runtime.brake(sim, "awaiting_feedback", "Replanning the same observed destination")
                    self.replans += 1
                    self.revision += 1
                    try:
                        path = replan(pose[:2], self.path[-1])
                    except ValueError:
                        path = None
                    if path is not None:
                        remaining_length = sum(float(np.linalg.norm(np.array(end) - begin)) for begin, end in zip(path, path[1:]))
                        if runtime.travel + remaining_length <= 1.8:
                            self.path = np.asarray(path, dtype=float)
                            self.index = 1
                            self.refill_after_replan = True
                            self.reason = "Following locally replanned route"
                            return
                self.finish(sim, runtime, "blocked", "OBSERVED_PATH_BLOCKED: Inspect the route before continuing")
                return
        if self.updates and not runtime.buffer and not self.refill_after_replan:
            self.buffer_stops += 1
            self.finish(sim, runtime, "blocked", "BUFFER_EMPTY: Stopped; select a fresh destination")
            return
        if self.updates > 10 and self.distance_m > .25 and abs(bearing) < .2:
            self.minimum_cruise_speed = min(self.minimum_cruise_speed, float(runtime.velocity[0]))
        observation = sim.observe(render=False)
        runtime.observe(sim, observation)
        if runtime.command_sensor and runtime.command_sensor["captured_at_s"] is not None:
            runtime.command_sensor["age_s"] = self.clock()-runtime.command_sensor["captured_at_s"]
        runtime.apply(sim, "replace_motion_buffer", ContinuousMotionBuffer(expected_revision=runtime.revision,
            segments=[{"kind": "drive", "linear_mps": linear, "angular_radps": angular, "duration_s": 1.}] * 2), observation.seq)
        self.refill_after_replan = False
        self.updates += 1