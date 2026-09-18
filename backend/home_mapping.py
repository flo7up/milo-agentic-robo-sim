import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Protocol
from uuid import uuid4

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from backend.spatial_memory import ClosingConnection, SpatialMemoryStore


class ObjectMemory(Protocol):
    def remember_object(self, observation: dict, image: bytes) -> None: ...
    def object_observations(self, map_id: str) -> list[dict]: ...


def transform_pose(pose, transform):
    cosine, sine = math.cos(transform[2]), math.sin(transform[2])
    return [transform[0] + cosine * pose[0] - sine * pose[1],
        transform[1] + sine * pose[0] + cosine * pose[1],
        math.atan2(math.sin(pose[2] + transform[2]), math.cos(pose[2] + transform[2]))]


def inverse_pose(pose):
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    return [-cosine * pose[0] - sine * pose[1], sine * pose[0] - cosine * pose[1], -pose[2]]


def laser_points(laser, pose):
    ranges = np.array([np.nan if value is None else value for value in laser["ranges_m"]])
    angles = laser["angle_min"] + np.arange(len(ranges)) * laser["angle_increment"] + pose[2]
    origin = transform_pose([*laser["origin_m"][:2], 0.], pose)[:2]
    valid = np.isfinite(ranges) & (ranges >= laser["range_min"])
    hits = valid & (ranges <= laser["range_max"])
    distances = np.minimum(ranges[valid], laser["range_max"])
    endpoints = np.array(origin) + np.column_stack((np.cos(angles[valid]), np.sin(angles[valid]))) * distances[:, None]
    return np.array(origin), endpoints, hits[valid]


class HomeMap:
    resolution_m = .1
    size = 400

    def __init__(self, environment_id, map_id=None):
        self.identity = map_id or str(uuid4())
        self.environment_id = environment_id
        self.memory_environment_id = environment_id
        self.environment_revision = "legacy"
        self.frame_revision = "1"
        self.profile_id = None
        self.name = "Unsaved home"
        self.origin = np.array([-20., -20.])
        self.evidence = np.zeros((self.size, self.size), dtype=np.int16)
        self.visits = np.zeros_like(self.evidence, dtype=np.uint16)
        self.places = []
        self.edges = []
        self.objects = []
        self.frontier_attempts = {}
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.revision = 0
        self.scan_count = 0
        self.saved = False
        self.annotations_dirty = False

    @property
    def cells(self):
        return np.where(self.evidence > 0, 100, np.where(self.evidence < 0, 0, -1)).astype(np.int8)

    def indices(self, points):
        scaled = (np.asarray(points) - self.origin) / self.resolution_m
        return np.floor(np.nextafter(scaled, np.inf)).astype(int)

    def inside(self, indices):
        indices = np.asarray(indices)
        return np.all((indices >= 0) & (indices < self.size), axis=-1)

    def observe(self, laser, pose, timestamp):
        if self.saved:
            raise ValueError("SAVED_MAP_READ_ONLY: scenario observations belong to the live layer")
        if not all(math.isfinite(value) for value in [*pose, timestamp]):
            raise ValueError("Map pose and timestamp must be finite")
        origin, endpoints, hits = laser_points(laser, pose)
        free_indices = []
        for endpoint in endpoints:
            samples = np.linspace(origin, endpoint, max(2, math.ceil(np.linalg.norm(endpoint - origin) / .04) + 1))[:-1]
            free_indices.append(self.indices(samples))
        if free_indices:
            free = np.concatenate(free_indices)
            free = free[self.inside(free)]
            codes = np.unique(free[:, 1] * self.size + free[:, 0])
            rows, columns = np.divmod(codes, self.size)
            self.evidence[rows, columns] = np.maximum(-20, self.evidence[rows, columns] - 1)
        occupied = np.unique(self.indices(endpoints[hits]), axis=0)
        occupied = occupied[self.inside(occupied)]
        self.evidence[occupied[:, 1], occupied[:, 0]] = np.minimum(20, self.evidence[occupied[:, 1], occupied[:, 0]] + 4)
        position = self.indices(pose[:2])
        if not self.inside(position):
            raise ValueError("MAP_BOUNDS: mapping area exceeded")
        self.visits[position[1], position[0]] = min(65535, int(self.visits[position[1], position[0]]) + 1)
        self.scan_count += 1
        self.updated_at = timestamp

    def allowed(self, radius_m, obstacles=()):
        known = self.cells == 0
        if len(obstacles):
            indices = self.indices(obstacles)
            indices = indices[self.inside(indices)]
            known[indices[:, 1], indices[:, 0]] = False
        clearance = distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1] * self.resolution_m
        return known & (clearance > radius_m + self.resolution_m)

    def segment_allowed(self, start, goal, allowed):
        endpoints = (np.asarray([start, goal], dtype=float) - self.origin) / self.resolution_m
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            return False
        if not self.inside(np.floor(endpoints).astype(int)).all():
            return False
        delta = endpoints[1] - endpoints[0]
        fractions = [np.array([0., 1.])]
        for axis in (0, 1):
            if abs(delta[axis]) > 1e-12:
                lower, upper = sorted(endpoints[:, axis])
                boundaries = np.arange(math.floor(lower) + 1, math.ceil(upper))
                fractions.append((boundaries - endpoints[0, axis]) / delta[axis])
        crossings = np.unique(np.concatenate(fractions))
        samples = np.r_[crossings, (crossings[:-1] + crossings[1:]) / 2]
        positions = endpoints[0] + samples[:, None] * delta
        on_boundary = np.isclose(positions, np.rint(positions), atol=1e-9, rtol=0.)
        upper = np.where(on_boundary, np.rint(positions), np.floor(positions)).astype(int)
        lower = upper - on_boundary.astype(int)
        indices = np.vstack((lower, upper, np.column_stack((lower[:, 0], upper[:, 1])),
            np.column_stack((upper[:, 0], lower[:, 1]))))
        return bool(self.inside(indices).all() and allowed[indices[:, 1], indices[:, 0]].all())

    def route(self, start, goal, radius_m, obstacles=()):
        allowed = self.allowed(radius_m, obstacles)
        indices = self.indices([start, goal])
        if not self.inside(indices).all() or not allowed[indices[:, 1], indices[:, 0]].all():
            raise ValueError("UNREACHABLE: destination or current footprint is not in mapped free space")
        nodes = np.arange(self.size ** 2).reshape(allowed.shape)
        sources, targets = [], []
        for row_delta, column_delta in ((1, 0), (0, 1)):
            source = nodes[:self.size - row_delta, :self.size - column_delta]
            target = nodes[row_delta:, column_delta:]
            valid = allowed[:self.size - row_delta, :self.size - column_delta] & allowed[row_delta:, column_delta:]
            sources.extend(source[valid])
            targets.extend(target[valid])
        graph = coo_matrix((np.ones(len(sources)), (sources, targets)), shape=(self.size ** 2, self.size ** 2)).tocsr()
        start_node, goal_node = indices[:, 1] * self.size + indices[:, 0]
        distances, predecessors = dijkstra(graph, directed=False, indices=start_node, return_predecessors=True)
        if not np.isfinite(distances[goal_node]):
            raise ValueError("UNREACHABLE: no connected mapped route")
        route = [list(goal)]
        current = int(goal_node)
        while current != start_node:
            current = int(predecessors[current])
            row, column = divmod(current, self.size)
            route.append((self.origin + (np.array([column, row]) + .5) * self.resolution_m).tolist())
        route[-1] = list(start)
        return route[::-1]

    def document(self):
        return {"schema": "milo-home-map-v1", "map_id": self.identity, "environment_id": self.environment_id,
            "name": self.name, "frame": "map", "units": "m_rad", "resolution_m": self.resolution_m,
            "origin_m": self.origin.tolist(), "width": self.size, "height": self.size,
            "created_unix_s": self.created_at, "updated_unix_s": self.updated_at, "revision": self.revision,
            "source": "simulated_laser_and_wheel_odometry", "scan_count": self.scan_count,
            "evidence": self.evidence.ravel().tolist(), "visits": self.visits.ravel().tolist(),
            "places": self.places, "edges": self.edges, "objects": self.objects,
            "frontier_attempts": self.frontier_attempts}

    def add_place(self, name, kind, pose, radius_m, connects=()):
        if not name.strip() or len(name) > 80 or kind not in {"room", "doorway", "destination"}:
            raise ValueError("A place needs a name and a valid kind")
        if len(self.places) >= 100 or any(place["name"].casefold() == name.strip().casefold() for place in self.places):
            raise ValueError("Place name already exists or the map has 100 places")
        self.route(pose[:2], pose[:2], radius_m)
        connections = []
        for identity in connects:
            neighbour = next((place for place in self.places if place["place_id"] == identity), None)
            if neighbour is None:
                raise ValueError("UNKNOWN_PLACE: graph connection does not exist")
            path = self.route(pose[:2], neighbour["pose_m_rad"][:2], radius_m)
            connections.append({"to": identity, "path_m": path})
        place = {"place_id": str(uuid4()), "name": name.strip(), "kind": kind,
            "pose_m_rad": list(pose), "frame": "map", "map_id": self.identity,
            "created_unix_s": time.time(), "source": "operator_annotation"}
        self.places.append(place)
        self.annotations_dirty = True
        self.edges.extend({"from": place["place_id"], **connection, "source": "operator_connected_observed_route"}
            for connection in connections)
        return place

    def graph_summary(self, allowed, limit=64):
        places = {place["place_id"]: place for place in self.places}
        connections = []
        for edge in self.edges[:limit]:
            if edge["from"] not in places or edge["to"] not in places:
                continue
            path = edge.get("path_m")
            status = "unknown"
            if allowed is not None and path:
                points = np.concatenate([np.linspace(start, end, max(2, math.ceil(math.dist(start, end) / .025) + 1))
                    for start, end in zip(path, path[1:])]) if len(path) > 1 else np.asarray(path)
                indices = self.indices(points)
                clear = self.inside(indices).all() and allowed[indices[:, 1], indices[:, 0]].all()
                status = "reachable" if clear else "blocked"
            connections.append({"from": edge["from"], "to": edge["to"], "status": status,
                "basis": "recorded_observed_connection_path" if path else "legacy_link_without_recorded_path"})
        return {"connections": connections, "truncated": len(self.edges) > limit,
            "claim": "Path clearance only; room identity and doorway opening are separate observations"}

    def frontiers(self, pose, radius_m, obstacles=(), region_id=None, *, excluded=()):
        allowed = self.allowed(radius_m, obstacles)
        position = self.indices(pose[:2])
        components, _ = label(allowed)
        if not self.inside(position) or not components[position[1], position[0]]:
            return []
        connected = components == components[position[1], position[0]]
        nearby_unknown = binary_dilation(self.cells == -1, iterations=math.ceil((radius_m + .4) / self.resolution_m))
        groups, count = label(connected & nearby_unknown)
        visited_count = int(np.count_nonzero(self.visits))
        distance_from_visited = (distance_transform_edt(self.visits == 0) * self.resolution_m
            if visited_count >= 5 else np.zeros_like(self.visits, dtype=float))
        region = None
        if region_id:
            region = next((place for place in self.places if place["place_id"] == region_id and place["kind"] == "room"), None)
            if region is None:
                raise ValueError("UNKNOWN_REGION: select a named room or the whole mapped area")
        result = []
        known = int(np.count_nonzero(self.evidence))
        for group_id in range(1, count + 1):
            rows, columns = np.where(groups == group_id)
            if len(rows) < 3:
                continue
            points = self.origin + (np.column_stack((columns, rows)) + .5) * self.resolution_m
            distances = np.linalg.norm(points - pose[:2], axis=1)
            headings = np.arctan2(points[:, 1] - pose[1], points[:, 0] - pose[0]) - pose[2]
            turns = np.abs(np.arctan2(np.sin(headings), np.cos(headings)))
            novelty = np.minimum(distance_from_visited[rows, columns], 1.5)
            scores = np.abs(distances - 1.) + .4 * turns + self.visits[rows, columns] * .2 - .35 * novelty
            for selected in np.argsort(scores):
                if distances[selected] <= .15 + self.resolution_m:
                    continue
                point = points[selected].tolist()
                if region and math.dist(point, region["pose_m_rad"][:2]) > 3.:
                    continue
                key = ":".join(str(int(math.floor(value / .5))) for value in point)
                if key in excluded:
                    continue
                previous = self.frontier_attempts.get(key, {})
                if previous.get("attempts", 0) >= 2 and known - previous.get("known_cells", known) < 25:
                    continue
                result.append({"frontier_id": key, "position_m": point, "distance_m": float(distances[selected]),
                    "attempts": previous.get("attempts", 0), "distance_from_visited_m": float(novelty[selected]),
                    "priority": float(scores[selected] + previous.get("attempts", 0)),
                    "region_scope": "within_3m_of_room_annotation" if region else "all_connected"})
                break
        return sorted(result, key=lambda item: item["priority"])[:20]

    def mark_frontier(self, identity):
        previous = self.frontier_attempts.get(identity, {})
        self.frontier_attempts[identity] = {"attempts": previous.get("attempts", 0) + 1,
            "known_cells": int(np.count_nonzero(self.evidence)), "last_attempt_unix_s": time.time()}

    def match_scan(self, laser, seed=None, radius_m=.25):
        _, local, hits = laser_points(laser, [0., 0., 0.])
        local = local[hits][::4]
        if len(local) < 20 or np.count_nonzero(self.cells == 100) < 20:
            raise ValueError("LOCALIZATION_UNRELIABLE: insufficient measured wall returns")
        field = distance_transform_edt(self.cells != 100) * self.resolution_m
        allowed = self.allowed(radius_m)

        def score(poses):
            results = []
            for pose in poses:
                position = self.indices(pose[:2])
                if not self.inside(position) or not allowed[position[1], position[0]]:
                    continue
                cosine, sine = math.cos(pose[2]), math.sin(pose[2])
                points = local @ np.array([[cosine, sine], [-sine, cosine]]) + pose[:2]
                indices = self.indices(points)
                inside = self.inside(indices)
                residuals = np.ones(len(points))
                residuals[inside] = field[indices[inside, 1], indices[inside, 0]]
                results.append((float(np.minimum(residuals, .6).mean()), float(np.mean(residuals <= .2)), list(pose)))
            return sorted(results, key=lambda item: item[0])

        if seed is None:
            rows, columns = np.where(allowed)
            positions = np.unique(np.floor((self.origin + np.column_stack((columns, rows)) * self.resolution_m) / .4) * .4, axis=0)
            if len(positions) > 6000:
                raise ValueError("LOCALIZATION_SEED_REQUIRED: select an approximate saved place")
            candidates = ([*point, heading] for point in positions for heading in np.arange(-math.pi, math.pi, math.pi / 12))
        else:
            if len(seed) != 3 or not np.isfinite(seed).all():
                raise ValueError("Localization seed must be a finite map pose")
            candidates = ([seed[0] + horizontal, seed[1] + lateral, seed[2] + heading]
                for horizontal in np.arange(-.3, .301, .1) for lateral in np.arange(-.3, .301, .1)
                for heading in np.arange(-.2, .201, .05))
        coarse = score(candidates)
        if not coarse:
            raise ValueError("LOCALIZATION_UNRELIABLE: no known free seed positions")
        seeds = []
        for item in coarse:
            if all(math.dist(item[2][:2], other[:2]) > .3 or abs(math.atan2(math.sin(item[2][2] - other[2]), math.cos(item[2][2] - other[2]))) > .3 for other in seeds):
                seeds.append(item[2])
            if len(seeds) == 4:
                break
        refined = score([point[0] + horizontal, point[1] + lateral, point[2] + heading]
            for point in seeds for horizontal in np.arange(-.1, .101, .05)
            for lateral in np.arange(-.1, .101, .05) for heading in np.arange(-.06, .061, .02))
        best = refined[0]
        ambiguous = any(item[0] < best[0] + .025 and (math.dist(item[2][:2], best[2][:2]) > .5
            or abs(math.atan2(math.sin(item[2][2] - best[2][2]), math.cos(item[2][2] - best[2][2]))) > .4) for item in refined[1:])
        if best[0] > .16 or best[1] < .65 or ambiguous:
            raise ValueError("LOCALIZATION_UNRELIABLE: scan mismatch or ambiguous location; select a known place and rescan")
        return best[2], {"mean_residual_m": best[0], "matched_fraction": best[1], "seeded": seed is not None,
            "method": "scan_pose_search", "pose_correction_estimated": True, "calibrated_confidence": False}

    def scan_quality(self, laser, pose):
        _, endpoints, hits = laser_points(laser, pose)
        indices = self.indices(endpoints[hits])
        indices = indices[self.inside(indices)]
        if len(indices) < 40:
            raise ValueError("LOCALIZATION_UNRELIABLE: insufficient measured returns")
        field = distance_transform_edt(self.cells != 100) * self.resolution_m
        residuals = field[indices[:, 1], indices[:, 0]]
        fraction = float(np.mean(residuals <= .25))
        residual = float(np.minimum(residuals, .6).mean())
        if fraction < .5 or residual > .25:
            raise ValueError("LOCALIZATION_LOST: live scan disagrees with mapped pose")
        return {"mean_residual_m": residual, "matched_fraction": fraction, "method": "fixed_pose_scan_consistency",
            "pose_correction_estimated": False, "calibrated_confidence": False}

    @classmethod
    def restore(cls, data):
        if (data["schema"] != "milo-home-map-v1" or data["width"] != cls.size or data["height"] != cls.size
                or data["resolution_m"] != cls.resolution_m or data["frame"] != "map"):
            raise ValueError("Unsupported saved map schema")
        result = cls(data["environment_id"], data["map_id"])
        result.name, result.origin = data["name"], np.array(data["origin_m"])
        result.evidence = np.array(data["evidence"], dtype=np.int16).reshape(cls.size, cls.size)
        result.visits = np.array(data["visits"], dtype=np.uint16).reshape(cls.size, cls.size)
        result.created_at, result.updated_at = data["created_unix_s"], data["updated_unix_s"]
        result.revision, result.scan_count = data["revision"], data["scan_count"]
        result.places, result.edges, result.objects = data["places"], data["edges"], data["objects"]
        result.frontier_attempts = data["frontier_attempts"]
        result.saved = True
        return result


class MapStore(SpatialMemoryStore):
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS maps (map_id TEXT PRIMARY KEY, environment_id TEXT UNIQUE, revision INTEGER, document TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS object_observations (observation_id TEXT PRIMARY KEY, map_id TEXT, timestamp REAL, document TEXT, image BLOB)")
            connection.execute("CREATE TABLE IF NOT EXISTS room_observations (observation_id TEXT PRIMARY KEY, map_id TEXT, place_id TEXT, timestamp REAL, document TEXT, image BLOB)")
        self.migrate_memory()

    def connect(self):
        return sqlite3.connect(self.path, timeout=5., factory=ClosingConnection)

    def catalog(self):
        with self.connect() as connection:
            return [{**{key: data[key] for key in ("map_id", "environment_id", "name", "revision", "updated_unix_s")},
                    "profile_id": profile_id, "environment_revision": revision}
                for document, profile_id, revision in connection.execute("SELECT document, profile_id, environment_revision FROM maps ORDER BY environment_id")
                for data in [json.loads(document)]]

    def save(self, home, name, *, scope=None):
        if not name.strip() or len(name) > 80 or not home.scan_count:
            raise ValueError("A name and observed map are required")
        data = home.document()
        data.update(name=name.strip(), revision=home.revision + 1, updated_unix_s=time.time())
        profile_id = home.profile_id or self.selected_profile(home.memory_environment_id, home.environment_revision)["profile_id"]
        home.profile_id = profile_id
        if scope and (home.identity, profile_id, home.memory_environment_id, home.environment_revision) != (
                scope.map_id, scope.profile_id, scope.environment_id, scope.environment_revision):
            raise ValueError("MAP_SCOPE_MISMATCH: cannot save another profile's map")
        self.bind_map(home, frame_revision=scope.frame_revision if scope else home.frame_revision)
        with self.connect() as connection:
            if scope:
                self._check_scope(connection, scope)
            self._profile(connection, home.memory_environment_id, home.environment_revision, profile_id)
            if home.revision:
                changed = connection.execute("UPDATE maps SET revision=?, document=? WHERE map_id=? AND revision=? AND environment_id=? AND environment_revision=? AND profile_id=?",
                    (data["revision"], json.dumps(data, allow_nan=False), home.identity, home.revision, home.environment_id, home.environment_revision, profile_id)).rowcount
                if changed != 1:
                    raise ValueError("MAP_CONFLICT: reload the current revision before saving")
            else:
                try:
                    connection.execute("INSERT INTO maps VALUES (?, ?, ?, ?, ?, ?)",
                        (home.identity, home.environment_id, data["revision"], json.dumps(data, allow_nan=False), profile_id, home.environment_revision))
                except sqlite3.IntegrityError as error:
                    raise ValueError("MAP_EXISTS: load the saved home instead of mapping this environment again") from error
        home.name, home.revision, home.updated_at, home.saved = data["name"], data["revision"], data["updated_unix_s"], True
        home.annotations_dirty = False
        home.profile_id = profile_id

    def load(self, map_id, environment_id, *, profile_id=None, environment_revision="legacy"):
        profile_id = profile_id or self.selected_profile(environment_id, environment_revision)["profile_id"]
        with self.connect() as connection:
            row = connection.execute("SELECT document FROM maps WHERE map_id=? AND environment_id=? AND environment_revision=? AND profile_id=?", (map_id, environment_id, environment_revision, profile_id)).fetchone()
        if row is None:
            raise ValueError("MAP_NOT_FOUND: map is unknown or belongs to another environment")
        home = HomeMap.restore(json.loads(row[0]))
        home.profile_id, home.environment_revision = profile_id, environment_revision
        with self.connect() as connection:
            row = connection.execute("SELECT environment_id, frame_revision FROM memory_map_scopes WHERE map_id=?", (map_id,)).fetchone()
        home.memory_environment_id = row[0] if row else home.environment_id
        home.frame_revision = row[1] if row else "1"
        return home

    def remember_object(self, observation, image):
        self.remember_legacy("object", observation, image)

    def object_observations(self, map_id):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT document FROM object_observations WHERE map_id=? ORDER BY timestamp DESC LIMIT 64", (map_id,))]

    def object_image(self, map_id, observation_id):
        with self.connect() as connection:
            row = connection.execute("SELECT image FROM object_observations WHERE map_id=? AND observation_id=?", (map_id, observation_id)).fetchone()
        if row is None:
            raise ValueError("Object image not found")
        return row[0]

    def remember_room(self, observation, image, *, allow_draft=False):
        self.remember_legacy("room", observation, image, allow_draft=allow_draft)

    def room_observations(self, map_id):
        with self.connect() as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT document FROM room_observations WHERE map_id=? ORDER BY timestamp DESC LIMIT 100", (map_id,))]

    def review_room(self, map_id, observation_id):
        with self.connect() as connection:
            row = connection.execute("SELECT document FROM room_observations WHERE map_id=? AND observation_id=?", (map_id, observation_id)).fetchone()
            if row is None:
                raise ValueError("UNKNOWN_EVIDENCE: select recorded room evidence")
            record = json.loads(row[0])
            record.update(review_status="operator_confirmed", reviewed_unix_s=time.time())
            patch = {"review_status": "operator_confirmed", "reviewed_unix_s": record["reviewed_unix_s"]}
            connection.execute("INSERT INTO memory_reviews VALUES (?, ?) ON CONFLICT(event_id) DO UPDATE SET document=excluded.document",
                (observation_id, json.dumps(patch)))
        return record

    def room_image(self, map_id, observation_id):
        with self.connect() as connection:
            row = connection.execute("SELECT image FROM room_observations WHERE map_id=? AND observation_id=?", (map_id, observation_id)).fetchone()
        if row is None:
            raise ValueError("Room image not found")
        return row[0]