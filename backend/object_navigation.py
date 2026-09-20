import math
import time
from uuid import uuid4

import numpy as np

from backend.spatial import measure_visible_region, point_cloud


def circle_candidates(sensor, now=None):
    from scipy.ndimage import binary_dilation, label
    now = time.monotonic() if now is None else now
    if not 0 <= now-sensor.captured_at <= 1.:
        raise ValueError("Circle candidates require fresh paired depth")
    intrinsics = sensor.calibration
    depth = np.asarray(sensor.depth_m, dtype=float).reshape(intrinsics.height, intrinsics.width)
    rows, columns = np.where(np.isfinite(depth))
    eye, points = point_cloud(sensor, stride=1)
    selected = (points[:, 2] > .08) & (points[:, 2] < 1.6) & (np.linalg.norm(points-eye, axis=1) <= 4.)
    points, rows, columns = points[selected], rows[selected], columns[selected]
    if len(points) < 12:
        return []
    cells = np.floor(points[:, :2]/.08).astype(int)
    origin = cells.min(axis=0)
    cells -= origin
    size = cells.max(axis=0)+1
    occupied = np.zeros((size[1], size[0]), dtype=bool)
    occupied[cells[:, 1], cells[:, 0]] = True
    components, count = label(binary_dilation(occupied, iterations=1))
    membership = components[cells[:, 1], cells[:, 0]]
    candidates = []
    for component in range(1, count+1):
        mask = membership == component
        if np.count_nonzero(mask) < 12:
            continue
        surface = points[mask]
        extent = np.ptp(surface[:, :2], axis=0)
        if max(extent) > 2.5:
            continue
        lower = [int(columns[mask].min()), int(rows[mask].min())]
        upper = [int(columns[mask].max())+1, int(rows[mask].max())+1]
        if min(upper[0]-lower[0], upper[1]-lower[1]) < 3:
            continue
        candidates.append({"bounds": [lower[0]/intrinsics.width, lower[1]/intrinsics.height,
            upper[0]/intrinsics.width, upper[1]/intrinsics.height], "measured_points": int(mask.sum()),
            "distance_m": float(np.median(np.linalg.norm(surface-eye, axis=1))),
            "clipped": lower[0] == 0 or lower[1] == 0 or upper[0] == intrinsics.width or upper[1] == intrinsics.height,
            "source": "paired_head_depth", "frame": "wheel_odometry", "sequence": sensor.sequence,
            "run_id": sensor.run_id, "episode_epoch": sensor.episode_epoch, "captured_at": sensor.captured_at,
            "semantic_identity": "unverified", "motion_authorized": False,
            "limitation": "Connected visible surfaces may merge objects or omit hidden parts; identify from the paired image"})
    candidates.sort(key=lambda item: (-item["measured_points"], item["distance_m"]))
    return [{"id": f"{sensor.run_id}:{sensor.episode_epoch}:{sensor.sequence}:object-{index}", **item}
        for index, item in enumerate(candidates[:6])]


def resolve_circle_candidate(sensor, candidates, identity, now=None):
    now = time.monotonic() if now is None else now
    selected = next((item for item in candidates if item["id"] == identity), None)
    if selected is None:
        raise ValueError("UNKNOWN_OBJECT_CANDIDATE: select an ID from the current paired image")
    if ((selected["run_id"], selected["episode_epoch"], selected["sequence"], selected["captured_at"])
            != (sensor.run_id, sensor.episode_epoch, sensor.sequence, sensor.captured_at)
            or not 0 <= now-sensor.captured_at <= 15.):
        raise ValueError("STALE_OBJECT_CANDIDATE: inspect the object again")
    return list(selected["bounds"])


class ObservedObjectGoal:
    def __init__(self, sensor, bounds, label, standoff_m, stop_revision, task_revision):
        if not label.strip() or len(label) > 60 or not math.isfinite(standoff_m) or not .4 <= standoff_m <= 1.:
            raise ValueError("Object goals require a label and a 0.4-1.0 m base-to-visible-surface distance")
        measured = measure_visible_region(sensor, bounds)
        self.identity = str(uuid4())
        self.run_id, self.epoch = sensor.run_id, sensor.episode_epoch
        self.stop_revision, self.task_revision = stop_revision, task_revision
        self.surface = np.asarray(measured["surface_median_xyz_m"][:2])
        delta = np.asarray(sensor.odometry_m_rad[:2]) - self.surface
        if np.linalg.norm(delta) < standoff_m:
            raise ValueError("Object is already closer than the requested stopping distance; inspect before retreating")
        self.bearing = math.atan2(delta[1], delta[0])
        self.label, self.standoff_m = label, standoff_m
        self.captured_at, self.sequence = sensor.captured_at, sensor.sequence
        self.selected = None
        self.verified = False
        self.verified_pose = None
        self.verified_head = None
        self.verified_at = None
        self.arrived_at = None
        self.arrived_sequence = None

    def check(self, run_id, epoch, stop_revision, task_revision, now):
        if (run_id, epoch, stop_revision, task_revision) != (self.run_id, self.epoch, self.stop_revision, self.task_revision):
            raise ValueError("Object goal lost its episode or motion authority")
        if not 0 <= now - self.captured_at <= 120.:
            raise ValueError("Object sighting expired; identify it in a fresh image")

    def pose(self, side):
        offset = {"front": 0., "left": -.5, "right": .5}[side]
        angle = self.bearing + offset
        point = self.surface + self.standoff_m * np.array([math.cos(angle), math.sin(angle)])
        heading = math.atan2(self.surface[1] - point[1], self.surface[0] - point[0])
        return [*point.tolist(), heading]

    def options(self, pose, plan, valid):
        options = []
        for side in ("front", "left", "right"):
            target = self.pose(side)
            option = {"id": side, "target_pose_m_rad": target, "status": "blocked", "path_m": None}
            try:
                if np.linalg.norm(np.asarray(target[:2]) - pose[:2]) <= .04:
                    path = [list(pose[:2]), target[:2]]
                else:
                    path = plan(pose[:2], target[:2])
                points = np.concatenate([np.linspace(begin, end, max(2, math.ceil(np.linalg.norm(np.asarray(end) - begin) / .025) + 1))
                    for begin, end in zip(path, path[1:])])
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if length > 1.8 or not valid(points):
                    raise ValueError("Approach has no bounded observed footprint corridor")
                option.update(status="reachable", path_m=np.asarray(path).tolist(), path_length_m=length)
            except ValueError as error:
                option["reason"] = str(error)
            options.append(option)
        return options

    def arrival(self, sensor, bounds, pose, now):
        if self.selected is None:
            raise ValueError("Select an object approach before verification")
        if (sensor.run_id, sensor.episode_epoch) != (self.run_id, self.epoch) or not 0 <= now - sensor.captured_at <= 1.:
            raise ValueError("Object verification requires a fresh paired image")
        if sensor.sequence <= self.sequence or np.linalg.norm(np.asarray(sensor.odometry_m_rad) - pose) > .05:
            raise ValueError("Object verification must follow arrival at the current pose")
        measured = measure_visible_region(sensor, bounds)
        surface = np.asarray(measured["surface_median_xyz_m"][:2])
        if np.linalg.norm(surface - self.surface) > .25:
            raise ValueError("Object surface changed or association is uncertain; select a new goal")
        target = self.pose(self.selected)
        position_error = float(np.linalg.norm(np.asarray(target[:2]) - pose[:2]))
        heading_error = abs(math.atan2(math.sin(target[2] - pose[2]), math.cos(target[2] - pose[2])))
        distance = float(np.linalg.norm(surface - pose[:2]))
        return {"position_error_m": position_error, "heading_error_rad": heading_error,
            "visible_surface_distance_m": distance, "pose_ok": position_error <= .04 and heading_error <= .08
                and abs(distance - self.standoff_m) <= .15,
            "semantic_identity": "model_reported_not_independently_verified"}

    def state(self):
        return {"goal_id": self.identity, "label": self.label, "standoff_m": self.standoff_m,
            "surface_m": self.surface.tolist(), "selected_approach": self.selected, "verified": self.verified,
            "target_pose_m_rad": self.pose(self.selected) if self.selected else None,
            "source": "paired_head_depth", "limitation": "Static visible surface estimate, not full object extent or automatic identity tracking"}