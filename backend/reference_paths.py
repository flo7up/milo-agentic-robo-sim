import hashlib
import json
import math


def reference_path(challenge):
    if challenge is None:
        return None
    if challenge.orbit is not None:
        orbit = challenge.orbit
        radius = (orbit.minimum_radius_m + orbit.maximum_radius_m) / 2
        angle = math.atan2(challenge.initial_xy[1] - orbit.center_m[1], challenge.initial_xy[0] - orbit.center_m[0])
        sign = -1 if orbit.direction == "clockwise" else 1
        points = [[orbit.center_m[0] + radius * math.cos(angle + sign * 2 * math.pi * index / 120),
                   orbit.center_m[1] + radius * math.sin(angle + sign * 2 * math.pi * index / 120)] for index in range(121)]
        points.insert(0, list(challenge.initial_xy))
    elif challenge.id in {"park", "local_park"}:
        points = [list(challenge.initial_xy), list(challenge.objectives[0].center[:2])]
    else:
        return None
    return {"version": "scenario-reference-v1", "label": "Scenario reference path", "points": points,
        "coordinate_frame": "world_xy_m", "evaluation_only": True, "optimal": False,
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest(),
        "note": "Geometric scenario reference, not a proven shortest path or motion authorization. Operator only."}