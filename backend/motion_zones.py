import math
import time

import numpy as np
import pybullet as bullet

from backend.navigation import distance_stop_threshold


def clearance_sectors(cells, allowed, indices_for, inside, pose, resolution_m):
    rings = [.25, .5, 1.]
    coordinates = np.arange(-1., 1. + resolution_m / 4, resolution_m / 2)
    horizontal, lateral = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((horizontal.ravel(), lateral.ravel()))
    distances = np.linalg.norm(points, axis=1)
    bearings = np.arctan2(points[:, 1], points[:, 0])
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    mapped = points @ np.array([[cosine, sine], [-sine, cosine]]) + pose[:2]
    indices = indices_for(mapped)
    valid = inside(indices)
    values = np.full(len(points), -1, dtype=np.int8)
    admitted = np.zeros(len(points), dtype=bool)
    columns, rows = indices[valid].T
    values[valid] = cells[rows, columns]
    admitted[valid] = allowed[rows, columns]
    start = indices_for(np.asarray(pose[:2]))
    start_allowed = bool(inside(start) and allowed[start[1], start[0]])
    sectors = []
    for sector in range(16):
        center = sector * math.pi / 8
        difference = np.arctan2(np.sin(bearings - center), np.cos(bearings - center))
        for ring, outer in enumerate(rings):
            selected = (np.abs(difference) <= math.pi / 16) & (distances <= outer + 1e-9)
            if np.any(values[selected] == 100):
                status, reason = "restricted", "Observed obstacle in sampled center corridor"
            elif not selected.any() or np.any(values[selected] == -1):
                status, reason = "unknown", "Unobserved or out-of-map center corridor"
            elif not start_allowed or not admitted[selected].all():
                status, reason = "restricted", "Whole-footprint clearance mask rejects corridor (obstacle or unknown margin)"
            else:
                status, reason = "clear", "Sampled corridor passes observed footprint mask; not motion authorization"
            sectors.append({"sector": sector, "inner_m": 0. if ring == 0 else rings[ring - 1], "outer_m": outer,
                "start_rad": center - math.pi / 16, "end_rad": center + math.pi / 16,
                "status": status, "reason": reason})
    return sectors


def observed_motion_zones(worker, footprint, now=None):
    now = time.monotonic() if now is None else now
    sim = worker.sim
    navigation = worker.navigation
    continuous = worker.continuous
    home = worker.home_mission
    mapped = worker.spatial_map
    pose = sim.odometry.copy()
    moving = continuous is not None and continuous.active
    home_source = home is not None and home.home is not None and (not moving or getattr(continuous, "home_owned", False))
    source, age, margin = "unavailable", None, None
    maximum_age = home.max_frame_age_s if home_source else mapped.max_frame_age_s if mapped is not None else 1.
    planning_radius = footprint["radius_m"]
    sectors = []
    reason = "No observed map available"
    if home_source:
        source = "home_map"
        age = None if home.captured_at is None else now - home.captured_at
        try:
            home.require_localized()
            localized = home.transform is not None
        except ValueError:
            localized = False
        if not localized:
            reason = "LOCALIZATION_REQUIRED: map pose is unavailable"
        else:
            from backend.home_mapping import transform_pose
            pose = np.asarray(transform_pose(pose.tolist(), home.transform))
            cells = home.home.cells
            obstacles = home.obstacles()
            if len(obstacles):
                for column, row in home.home.indices(obstacles):
                    if home.home.inside(np.array([column, row])):
                        cells[row, column] = 100
            allowed = home.allowed(footprint["radius_m"])
            margin = home.home.resolution_m
            sectors = clearance_sectors(cells, allowed, home.home.indices, home.home.inside, pose, margin)
            reason = "Sampled home-map footprint clearance"
    elif mapped is not None and mapped.observation is not None:
        source = "rolling_depth_map"
        age = None if mapped.captured_at is None else now - mapped.captured_at
        start = continuous.start_position if moving else sim.odometry[:2]
        radius = continuous.radius if moving else footprint["radius_m"]
        planning_radius = radius
        allowed = mapped.traversable(start, radius)
        indices_for = lambda points: np.floor(np.asarray(points) / mapped.resolution_m).astype(int) - mapped.origin
        inside = lambda indices: np.all((indices >= 0) & (indices < mapped.size), axis=-1)
        margin = mapped.resolution_m / 2
        sectors = clearance_sectors(mapped.cells, allowed, indices_for, inside, pose, mapped.resolution_m)
        reason = "Sampled rolling-map footprint clearance"
    stale = age is None or not 0 <= age <= maximum_age
    valid_for = max(0., maximum_age - age) if not stale else 0.
    if home_source:
        valid_for = max(0., min(valid_for, .75 - (now - home.sampled_at), 2. - (now - home.validated_at)))
    if not worker.powered or not worker.spatial_enabled or stale:
        for sector in sectors:
            sector["map_status"] = sector["status"]
            sector["status"] = "unavailable"
        reason = "SENSING_OFF" if not worker.powered or not worker.spatial_enabled else "SPATIAL_STALE: fresh depth required"
    velocity = float(navigation.velocity[0]) if navigation is not None else 0.
    command = velocity
    if navigation is not None and navigation.buffer:
        segment = navigation.buffer[0][0]
        command = float(getattr(segment, "linear_mps", 0.))
    executing = navigation is not None and bool(navigation.buffer)
    speed = max(abs(command), abs(velocity)) if executing else .15
    checked_directions = set()
    for candidate in (command, velocity) if executing else (.15, -.15):
        if abs(candidate) > .001:
            checked_directions.update(("front", "front_left", "front_right") if candidate > 0 else ("rear", "rear_left", "rear_right"))
    stop = navigation.diagnostic_state().get("stop") if navigation is not None else None
    position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
    rotation = bullet.getMatrixFromQuaternion(orientation)
    return {"run_id": sim.run_id, "episode_epoch": sim.epoch, "frame": "robot_base", "clock": "monotonic",
        "captured_at_s": now, "sensor_age_s": age, "stale": stale, "source": source, "reason": reason,
        "display_pose": {"frame": "world", "position_m": list(position[:2]), "yaw_rad": math.atan2(rotation[3], rotation[0])},
        "odometry_m_rad": sim.odometry.tolist(), "maximum_sensor_age_s": maximum_age, "valid_for_s": valid_for,
        "motion_authorized": False, "footprint": footprint, "planning_radius_m": planning_radius, "map_margin_m": margin, "sectors": sectors,
        "preview_speed_mps": speed, "speed_basis": "current_command_and_velocity" if executing else "idle_reference",
        "beam_checked_directions": sorted(checked_directions),
        "beam_stop_distance_m": distance_stop_threshold(speed), "controller_stop": stop,
        "sector_meaning": "Sampled robot-center corridors, not swept-volume validation or permission to move"}