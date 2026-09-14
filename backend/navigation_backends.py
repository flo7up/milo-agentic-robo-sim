import math
from collections import OrderedDict, deque
from io import BytesIO
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import Field, model_validator

from backend.contracts import StrictModel
from backend.simulation import MotionError


class NavigationCapabilities(StrictModel):
    name: str
    goals: list[Literal["point", "image", "explore"]]
    sensors: list[Literal["rgb", "depth", "odometry"]]
    fixed_head: bool = False
    history_frames: int = Field(default=1, ge=1)
    head_pitch_rad: float = .3


class NavigationGoal(StrictModel):
    goal_id: str = Field(default_factory=lambda: str(uuid4()))
    kind: Literal["point", "image", "explore"]
    evidence: Literal["depth_pixel", "observed_map", "remembered_view", "exploration"]
    point_m: list[float] | None = None
    view_id: str | None = None
    tolerance_m: float = Field(default=.04, ge=.04, le=.04)
    timeout_s: float = Field(default=60., gt=0, le=60)

    @model_validator(mode="after")
    def grounded_goal(self):
        if self.kind == "point":
            if self.evidence not in {"depth_pixel", "observed_map"} or self.point_m is None or len(self.point_m) != 2 or not np.isfinite(self.point_m).all() or self.view_id is not None:
                raise ValueError("Point goals require two measured odometry coordinates and sensor evidence")
        elif self.kind == "image":
            if self.evidence != "remembered_view" or not self.view_id or self.point_m is not None:
                raise ValueError("Image goals require a retained goal view, not invented coordinates")
        elif self.evidence != "exploration" or self.point_m is not None or self.view_id is not None:
            raise ValueError("Exploration does not accept a fabricated destination")
        return self


class NavigationFrame(StrictModel):
    run_id: str
    episode_epoch: int = Field(ge=0)
    sequence: int = Field(ge=1)
    captured_at: float
    stop_revision: int = Field(ge=0)
    goal_id: str
    odometry_m_rad: list[float] = Field(min_length=3, max_length=3)
    head_rad: list[float] = Field(min_length=2, max_length=2)
    frame: Literal["wheel_odometry"] = "wheel_odometry"
    camera_frame: Literal["head_optical"] = "head_optical"


class NavigationProposal(StrictModel):
    ticket: NavigationFrame
    backend: str
    path_m: list[list[float]] = Field(min_length=2, max_length=128)
    timing: Literal["geometric_retimed"] = "geometric_retimed"
    poses_m_rad: list[list[float]] | None = None

    @model_validator(mode="after")
    def finite_path(self):
        points = np.asarray(self.path_m, dtype=float)
        if points.shape != (len(self.path_m), 2) or not np.isfinite(points).all():
            raise ValueError("A path contains finite XY coordinates in meters")
        if self.poses_m_rad is not None:
            poses = np.asarray(self.poses_m_rad, dtype=float)
            if poses.shape != (len(points), 3) or not np.isfinite(poses).all() or not np.allclose(points, poses[:, :2]):
                raise ValueError("Trajectory poses must match path XY and use radians")
            for start, end in zip(poses, poses[1:]):
                delta = end[:2] - start[:2]
                if np.linalg.norm(delta) < .005:
                    continue
                turn = math.atan2(math.sin(end[2] - start[2]), math.cos(end[2] - start[2]))
                bearing = math.atan2(delta[1], delta[0]) - start[2] - turn / 2
                error = abs(math.atan2(math.sin(bearing), math.cos(bearing)))
                if min(error, abs(math.pi - error)) > .35:
                    raise ValueError("NONHOLONOMIC_PATH: lateral pose translation is not differential drive")
        return self

    def checked_path(self, current, goal, capabilities, now, path_valid):
        ticket = self.ticket
        if (ticket.run_id, ticket.episode_epoch, ticket.stop_revision, ticket.goal_id) != (
                current.run_id, current.episode_epoch, current.stop_revision, current.goal_id) or goal.goal_id != current.goal_id:
            raise MotionError("STALE_PLAN", "Navigation goal or cancellation revision changed")
        if self.backend != capabilities.name or goal.kind not in capabilities.goals:
            raise MotionError("UNSUPPORTED_GOAL", "Backend does not support this goal representation")
        if not 0 <= now - ticket.captured_at <= 1. or ticket.sequence > current.sequence:
            raise MotionError("STALE_OBSERVATION", "Navigation prediction requires a fresh paired frame")
        displacement = np.array(current.odometry_m_rad) - ticket.odometry_m_rad
        if np.linalg.norm(displacement[:2]) > .05 or abs(math.atan2(math.sin(displacement[2]), math.cos(displacement[2]))) > .05:
            raise MotionError("STALE_OBSERVATION", "Robot moved since trajectory acquisition")
        if capabilities.fixed_head and (np.max(np.abs(np.array(current.head_rad) - ticket.head_rad)) > .02 or abs(ticket.head_rad[0]) > .02 or abs(ticket.head_rad[1] - capabilities.head_pitch_rad) > .02):
            raise MotionError("CAMERA_CHANGED", "Pause navigation and reset policy context after head movement")
        path = np.asarray(self.path_m)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if np.linalg.norm(path[0] - current.odometry_m_rad[:2]) > .05 or not .05 <= length <= 1.8:
            raise MotionError("INVALID_TRAJECTORY", "Trajectory start or local travel budget is invalid")
        if not path_valid(path):
            raise MotionError("OBSERVED_PATH_BLOCKED", "Trajectory crosses unknown or obstructed space")
        return self.path_m


class ConventionalBackend:
    capabilities = NavigationCapabilities(name="conventional", goals=["point"], sensors=["depth", "odometry"])

    def propose(self, frame, goal, observed_map, radius):
        if goal.kind not in self.capabilities.goals:
            raise MotionError("UNSUPPORTED_GOAL", "Conventional local planning requires an observed point")
        return NavigationProposal(ticket=frame, backend=self.capabilities.name,
            path_m=observed_map.plan(frame.odometry_m_rad[:2], goal.point_m, radius))


NOMAD_CAPABILITIES = NavigationCapabilities(name="nomad", goals=["image", "explore"], sensors=["rgb"], fixed_head=True, history_frames=4)
NAVDP_CAPABILITIES = NavigationCapabilities(name="navdp", goals=["point", "image", "explore"], sensors=["rgb", "depth"], fixed_head=True, history_frames=8)


class NavigationImages:
    def __init__(self, history_frames, max_gap_s=.75):
        self.frames = deque(maxlen=history_frames)
        self.views = OrderedDict()
        self.connections = deque(maxlen=64)
        self.max_gap_s = max_gap_s

    def append(self, frame, image):
        if self.frames:
            previous = self.frames[-1][0]
            if (frame.run_id, frame.episode_epoch, frame.goal_id) != (previous.run_id, previous.episode_epoch, previous.goal_id) or np.max(np.abs(np.array(frame.head_rad) - previous.head_rad)) > .02 or frame.captured_at - previous.captured_at > self.max_gap_s:
                self.frames.clear()
            elif frame.sequence <= previous.sequence or frame.captured_at <= previous.captured_at:
                raise ValueError("History requires ordered distinct observations")
        if abs(frame.head_rad[0]) > .02:
            self.frames.clear()
            raise ValueError("Navigation camera must face forward")
        self.frames.append((frame, image))

    def remember(self, frame, image):
        identity = str(uuid4())
        self.views[identity] = (frame, image)
        while len(self.views) > 32:
            self.views.popitem(last=False)
        return identity

    def goal_image(self, identity, frame, now):
        remembered, image = self.views[identity]
        if (remembered.run_id, remembered.episode_epoch) != (frame.run_id, frame.episode_epoch) or not 0 <= now - remembered.captured_at <= 1800:
            raise MotionError("GOAL_VIEW_EXPIRED", "Retained view is from another episode or has expired")
        if np.max(np.abs(np.array(frame.head_rad) - remembered.head_rad)) > .02:
            raise MotionError("CAMERA_CHANGED", "Goal view uses a different head-camera pose")
        return image

    def connect(self, source, destination, *, status, length_m):
        if status != "arrived" or not math.isfinite(length_m) or length_m <= 0:
            raise ValueError("Only successfully traversed routes form remembered connections")
        first, second = self.views[source][0], self.views[destination][0]
        if (first.run_id, first.episode_epoch) != (second.run_id, second.episode_epoch):
            raise ValueError("Remembered connections cannot cross episodes")
        self.connections.append({"source": source, "destination": destination, "length_m": length_m,
            "uncertainty": "wheel odometry; no loop closure", "requires_fresh_clearance": True})


def navdp_depth_png(sensor, image):
    from PIL import Image
    with Image.open(BytesIO(image)) as rgb:
        if rgb.size != (sensor.calibration.width, sensor.calibration.height):
            raise ValueError("RGB and metric depth must be aligned at the same resolution")
    depth = np.array(sensor.depth_m, dtype=float).reshape(sensor.calibration.height, sensor.calibration.width)
    valid = np.isfinite(depth) & (depth >= .1) & (depth <= 5.)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = np.rint(depth[valid] * 10000).astype(np.uint16)
    output = BytesIO()
    Image.fromarray(encoded).save(output, format="PNG")
    return output.getvalue()


def relative_proposal(frame, backend, waypoints):
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite metric XY waypoints, forward X and left Y")
    heading = frame.odometry_m_rad[2]
    rotation = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    origin = np.array(frame.odometry_m_rad[:2])
    path = np.concatenate([origin[None], origin + points @ rotation.T])
    return NavigationProposal(ticket=frame, backend=backend, path_m=path.tolist())


def navdp_camera_proposal(frame, poses):
    from backend.spatial import camera_in_odometry
    values = np.asarray(poses, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("NavDP requires finite camera-local XY/yaw waypoints")
    if np.linalg.norm(values[:, :2], axis=1).max() < .05:
        raise MotionError("POLICY_HOLD", "NavDP predicted no translational route")
    eye, optical = camera_in_odometry(frame.head_rad, frame.odometry_m_rad)
    camera_flu = np.column_stack((optical[:, 2], -optical[:, 0], -optical[:, 1]))
    camera_points = np.column_stack((values[:, :2], np.zeros(len(values))))
    waypoints = (eye + camera_points @ camera_flu.T)[:, :2]
    return NavigationProposal(ticket=frame, backend="navdp",
        path_m=np.concatenate([np.array(frame.odometry_m_rad[:2])[None], waypoints]).tolist())


def navdp_camera_goal(frame, goal):
    from backend.spatial import camera_in_odometry
    eye, optical = camera_in_odometry(frame.head_rad, frame.odometry_m_rad)
    camera_flu = np.column_stack((optical[:, 2], -optical[:, 0], -optical[:, 1]))
    return ((np.array([*goal.point_m, 0.]) - eye) @ camera_flu)[:2].tolist()