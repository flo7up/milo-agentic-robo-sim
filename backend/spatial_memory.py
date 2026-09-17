import hashlib
import json
import math
import sqlite3
import time
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, model_validator

from backend.contracts import StrictModel


class MemoryScope(StrictModel):
    environment_id: str
    environment_revision: str
    profile_id: str
    map_id: str
    frame_revision: str = "1"
    run_id: str
    episode_epoch: int
    context_id: str


class MemoryObservation(StrictModel):
    kind: Literal["room", "object", "search", "entrance", "traversal"]
    source_key: str = Field(min_length=1, max_length=160)
    source: Literal["luna_camera", "worker_depth", "worker_odometry", "operator_camera", "fixture"]
    sequence: int = Field(ge=0)
    frame_ref: str = Field(min_length=1, max_length=120)
    observed_unix_s: float = Field(gt=0)
    pose_m_rad: list[float] = Field(min_length=3, max_length=3)
    pose_frame: Literal["map", "wheel_odometry"] = "map"
    head_rad: list[float] | None = Field(default=None, min_length=2, max_length=2)
    label: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=500)
    confidence: float = Field(default=.5, ge=0, le=1)
    entity_id: str | None = Field(default=None, max_length=80)
    place_id: str | None = Field(default=None, max_length=80)
    position_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    metric_valid: bool = True
    requires_revalidation: bool = False
    target: str = Field(default="", max_length=160)
    inspection_scope: Literal["viewpoint", "surface", "entrance"] = "viewpoint"
    visibility: Literal["partial", "adequate", "unknown"] = "unknown"
    visibility_limits: str = Field(default="", max_length=500)
    result: Literal["seen", "not_seen", "inconclusive"] = "inconclusive"
    from_entity_id: str | None = Field(default=None, max_length=80)
    to_entity_id: str | None = Field(default=None, max_length=80)
    path_m: list[list[float]] = Field(default_factory=list, max_length=1024)
    duration_s: float | None = Field(default=None, ge=0)
    room_matches: bool | None = None

    @model_validator(mode="after")
    def validate_evidence(self):
        numbers = self.pose_m_rad + (self.position_m or []) + (self.head_rad or [])
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Memory positions must be finite")
        if (not self.metric_valid or self.pose_frame != "map") and self.position_m is not None:
            raise ValueError("Unlocalized sightings cannot invent a metric position")
        if self.metric_valid and self.pose_frame != "map":
            raise ValueError("Metric memory requires a localized map frame")
        if self.kind in {"room", "object"} and not self.label.strip():
            raise ValueError("Semantic evidence requires a label")
        if self.kind == "search" and not self.target.strip():
            raise ValueError("Search evidence requires its target")
        if self.kind == "traversal":
            if (self.source not in {"worker_odometry", "fixture"} or not self.from_entity_id or not self.to_entity_id
                    or self.from_entity_id == self.to_entity_id or len(self.path_m) < 2 or not self.metric_valid):
                raise ValueError("A traversal requires distinct observed endpoints and measured travel")
            if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in self.path_m):
                raise ValueError("Traversal points must be finite map coordinates")
        elif self.path_m:
            raise ValueError("Only measured traversals can contain a path")
        return self


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *arguments):
        try:
            return super().__exit__(*arguments)
        finally:
            self.close()


def default_profile_id(environment_id, revision="legacy"):
    return str(uuid5(NAMESPACE_URL, json.dumps(["milo-memory", environment_id, revision])))


def environment_identity(challenge, instance_id=None, scene=None):
    definition = (challenge.environment if challenge and challenge.environment != "standalone"
        else "standalone:" + (challenge.id if challenge else "bench"))
    layout = challenge.scene() if challenge else scene or []
    geometry = [{key: item[key] for key in ("shape", "size", "position", "orientation", "mass", "marker") if key in item}
        for item in layout]
    fingerprint = hashlib.sha256(json.dumps(geometry, sort_keys=True, allow_nan=False).encode()).hexdigest()
    identity = instance_id or str(uuid5(NAMESPACE_URL, "milo-preset-instance:" + definition))
    return {"environment_id": identity, "revision": fingerprint, "scenario_id": definition,
        "fingerprint": fingerprint, "generation": {"kind": "fixed_preset", "seed": None, "layout_fingerprint": fingerprint}}


class SpatialMemoryStore:
    def migrate_memory(self):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise ValueError("MEMORY_SCHEMA_NEWER: update the application before opening this store")
            if version >= 1:
                self._migrate_observations(connection, version)
                self._migrate_reviews(connection, version)
                return
            connection.execute("CREATE TABLE memory_environments (environment_id TEXT, revision TEXT, scenario_id TEXT, fingerprint TEXT, generation TEXT, PRIMARY KEY(environment_id, revision))")
            connection.execute("CREATE TABLE knowledge_profiles (profile_id TEXT PRIMARY KEY, environment_id TEXT NOT NULL, environment_revision TEXT NOT NULL, name TEXT NOT NULL, created_unix_s REAL NOT NULL, parent_snapshot_id TEXT)")
            connection.execute("CREATE TABLE selected_knowledge (environment_id TEXT, environment_revision TEXT, profile_id TEXT NOT NULL, PRIMARY KEY(environment_id, environment_revision))")
            connection.execute("CREATE TABLE knowledge_snapshots (snapshot_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, environment_id TEXT NOT NULL, environment_revision TEXT NOT NULL, name TEXT NOT NULL, created_unix_s REAL NOT NULL, document TEXT NOT NULL)")
            connection.execute("ALTER TABLE maps RENAME TO maps_legacy")
            connection.execute("CREATE TABLE maps (map_id TEXT PRIMARY KEY, environment_id TEXT NOT NULL, revision INTEGER, document TEXT, profile_id TEXT NOT NULL, environment_revision TEXT NOT NULL, UNIQUE(environment_id, environment_revision, profile_id))")
            for map_id, environment_id, revision, document in connection.execute("SELECT map_id, environment_id, revision, document FROM maps_legacy").fetchall():
                profile_id = default_profile_id(environment_id)
                connection.execute("INSERT OR IGNORE INTO memory_environments VALUES (?, 'legacy', ?, 'legacy', '{}')", (environment_id, environment_id))
                connection.execute("INSERT OR IGNORE INTO knowledge_profiles VALUES (?, ?, 'legacy', 'Legacy knowledge', ?, NULL)", (profile_id, environment_id, time.time()))
                connection.execute("INSERT OR IGNORE INTO selected_knowledge VALUES (?, 'legacy', ?)", (environment_id, profile_id))
                connection.execute("INSERT INTO maps VALUES (?, ?, ?, ?, ?, 'legacy')", (map_id, environment_id, revision, document, profile_id))
            connection.execute("DROP TABLE maps_legacy")
            connection.execute("PRAGMA user_version=1")
            self._migrate_observations(connection, 1)
            self._migrate_reviews(connection, 1)

    @staticmethod
    def _migrate_reviews(connection, version):
        if version >= 3:
            return
        connection.execute("CREATE TABLE memory_reviews (event_id TEXT PRIMARY KEY, document TEXT NOT NULL)")
        connection.execute("DROP VIEW room_observations")
        connection.execute("CREATE VIEW room_observations AS SELECT events.event_id AS observation_id, events.map_id, json_extract(events.document, '$.place_id') AS place_id, events.observed_unix_s AS timestamp, CASE WHEN reviews.document IS NULL THEN events.document ELSE json_patch(events.document, reviews.document) END AS document, evidence.image FROM memory_events events LEFT JOIN memory_evidence evidence ON events.evidence_id=evidence.evidence_id LEFT JOIN memory_reviews reviews ON events.event_id=reviews.event_id WHERE events.kind='room'")
        connection.execute("PRAGMA user_version=3")

    def _migrate_observations(self, connection, version):
        if version >= 2:
            return
        connection.execute("CREATE TABLE memory_map_scopes (map_id TEXT PRIMARY KEY, environment_id TEXT NOT NULL, environment_revision TEXT NOT NULL, profile_id TEXT NOT NULL, frame_revision TEXT NOT NULL)")
        connection.execute("INSERT INTO memory_map_scopes SELECT map_id, environment_id, environment_revision, profile_id, '1' FROM maps")
        connection.execute("CREATE TABLE memory_evidence (evidence_id TEXT PRIMARY KEY, image BLOB NOT NULL)")
        connection.execute("CREATE TABLE memory_events (event_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, map_id TEXT NOT NULL, frame_revision TEXT NOT NULL, ingestion_key TEXT NOT NULL, kind TEXT NOT NULL, entity_id TEXT, observed_unix_s REAL NOT NULL, document TEXT NOT NULL, evidence_id TEXT, UNIQUE(profile_id, ingestion_key))")
        connection.execute("CREATE INDEX memory_events_lookup ON memory_events(profile_id, kind, observed_unix_s)")
        connection.execute("CREATE TABLE memory_entities (profile_id TEXT, entity_id TEXT, kind TEXT NOT NULL, document TEXT NOT NULL, PRIMARY KEY(profile_id, entity_id))")
        connection.execute("CREATE TABLE memory_contexts (context_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, document TEXT NOT NULL, active INTEGER NOT NULL)")
        for kind in ("room", "object"):
            table = kind + "_observations"
            records = connection.execute(f"SELECT observation_id, map_id, timestamp, document, image FROM {table}").fetchall()
            for identity, map_id, timestamp, document, image in records:
                scope = connection.execute("SELECT profile_id FROM memory_map_scopes WHERE map_id=?", (map_id,)).fetchone()
                profile_id = scope[0] if scope else "unbound:" + map_id
                evidence_id = hashlib.sha256(image).hexdigest() if image else None
                if image:
                    connection.execute("INSERT OR IGNORE INTO memory_evidence VALUES (?, ?)", (evidence_id, image))
                record = json.loads(document)
                entity_id = record.get("place_id") if kind == "room" else None
                connection.execute("INSERT INTO memory_events VALUES (?, ?, ?, '1', ?, ?, ?, ?, ?, ?)",
                    (identity, profile_id, map_id, "legacy:" + identity, kind, entity_id, timestamp, document, evidence_id))
            connection.execute(f"DROP TABLE {table}")
            place_column = ", json_extract(events.document, '$.place_id') AS place_id" if kind == "room" else ""
            connection.execute(f"CREATE VIEW {table} AS SELECT events.event_id AS observation_id, events.map_id{place_column}, events.observed_unix_s AS timestamp, events.document, evidence.image FROM memory_events events LEFT JOIN memory_evidence evidence ON events.evidence_id=evidence.evidence_id WHERE events.kind='{kind}'")
        connection.execute("PRAGMA user_version=2")

    def bind_map(self, home, *, environment_id=None, frame_revision="1"):
        environment_id = environment_id or getattr(home, "memory_environment_id", home.environment_id)
        profile_id = home.profile_id or self.selected_profile(environment_id, home.environment_revision)["profile_id"]
        with self.connect() as connection:
            self._profile(connection, environment_id, home.environment_revision, profile_id)
            existing = connection.execute("SELECT environment_id, environment_revision, profile_id, frame_revision FROM memory_map_scopes WHERE map_id=?", (home.identity,)).fetchone()
            expected = (environment_id, home.environment_revision, profile_id, frame_revision)
            if existing and existing != expected:
                raise ValueError("MAP_SCOPE_MISMATCH: map identity cannot move between knowledge profiles")
            connection.execute("INSERT OR IGNORE INTO memory_map_scopes VALUES (?, ?, ?, ?, ?)", (home.identity, *expected))
        home.profile_id = profile_id

    def open_context(self, home, run_id, episode_epoch, *, environment_id=None, frame_revision=None):
        environment_id = environment_id or getattr(home, "memory_environment_id", home.environment_id)
        frame_revision = frame_revision or getattr(home, "frame_revision", "1")
        self.bind_map(home, environment_id=environment_id, frame_revision=frame_revision)
        scope = MemoryScope(environment_id=environment_id, environment_revision=home.environment_revision,
            profile_id=home.profile_id, map_id=home.identity, frame_revision=frame_revision,
            run_id=run_id, episode_epoch=episode_epoch, context_id=str(uuid4()))
        with self.connect() as connection:
            connection.execute("UPDATE memory_contexts SET active=0 WHERE profile_id=?", (scope.profile_id,))
            connection.execute("INSERT INTO memory_contexts VALUES (?, ?, ?, 1)", (scope.context_id, scope.profile_id, scope.model_dump_json()))
        return scope

    def close_context(self, scope):
        with self.connect() as connection:
            connection.execute("UPDATE memory_contexts SET active=0 WHERE context_id=?", (scope.context_id,))

    def _check_scope(self, connection, scope):
        self._profile(connection, scope.environment_id, scope.environment_revision, scope.profile_id)
        row = connection.execute("SELECT document, active FROM memory_contexts WHERE context_id=?", (scope.context_id,)).fetchone()
        if row is None or not row[1] or MemoryScope.model_validate_json(row[0]) != scope:
            raise ValueError("STALE_MEMORY_CONTEXT: episode, profile or map changed")
        row = connection.execute("SELECT environment_id, environment_revision, profile_id, frame_revision FROM memory_map_scopes WHERE map_id=?", (scope.map_id,)).fetchone()
        if row != (scope.environment_id, scope.environment_revision, scope.profile_id, scope.frame_revision):
            raise ValueError("INCOMPATIBLE_MAP_FRAME: spatial memory requires revalidation")

    @staticmethod
    def _image(connection, image):
        if image is None:
            return None
        if len(image) > 1024 * 1024 or not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Memory evidence requires a bounded PNG camera image")
        identity = hashlib.sha256(image).hexdigest()
        connection.execute("INSERT OR IGNORE INTO memory_evidence VALUES (?, ?)", (identity, image))
        return identity

    def ingest(self, scope, observation, image=None, *, allow_new_entity=False, event_id=None, validate=None):
        observation = MemoryObservation.model_validate(observation)
        key = hashlib.sha256(json.dumps([scope.run_id, scope.episode_epoch, observation.sequence,
            observation.frame_ref, observation.kind, observation.source_key]).encode()).hexdigest()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_scope(connection, scope)
            if validate and not validate():
                raise ValueError("STALE_MEMORY_CONTEXT: observation authority was revoked")
            previous = connection.execute("SELECT document FROM memory_events WHERE profile_id=? AND ingestion_key=?", (scope.profile_id, key)).fetchone()
            if previous:
                return json.loads(previous[0])
            identity = event_id or str(uuid4())
            entity_id = observation.entity_id
            if observation.kind in {"room", "object"}:
                entity_id = entity_id or str(uuid4())
                if observation.entity_id and not allow_new_entity and not connection.execute("SELECT 1 FROM memory_entities WHERE profile_id=? AND entity_id=? AND kind=?", (scope.profile_id, entity_id, observation.kind)).fetchone():
                    raise ValueError("UNKNOWN_MEMORY_ENTITY: association must reference a scoped entity")
            if observation.kind in {"traversal", "entrance"}:
                for endpoint in (observation.from_entity_id, observation.to_entity_id):
                    if endpoint and not connection.execute("SELECT 1 FROM memory_entities WHERE profile_id=? AND entity_id=? AND kind='room'", (scope.profile_id, endpoint)).fetchone():
                        raise ValueError("UNKNOWN_REGION: connection endpoint is not an observed room")
            evidence_id = self._image(connection, image)
            record = {**observation.model_dump(), "observation_id": identity, "entity_id": entity_id,
                "map_id": scope.map_id, "frame": observation.pose_frame, "frame_revision": scope.frame_revision,
                "environment_id": scope.environment_id, "environment_revision": scope.environment_revision,
                "profile_id": scope.profile_id, "run_id": scope.run_id, "episode_epoch": scope.episode_epoch,
                "evidence_id": evidence_id, "image_sha256": evidence_id, "identity_verified": False, "confidence_calibrated": False,
                "arrival_verified": False,
                "requires_revalidation": not observation.metric_valid or observation.requires_revalidation, "absence_supported": False}
            if observation.kind == "room":
                record.update(place_id=observation.place_id, kind="room", evidence=observation.description, review_status="tentative", room_matches=observation.room_matches)
            if observation.kind in {"room", "object"}:
                collection = "rooms" if observation.kind == "room" else "objects"
                url = f"/api/home/{scope.map_id}/{collection}/{identity}/image.png"
                record.update(image_url=url, supporting_images=[url], spatial_sequence=observation.sequence)
            connection.execute("INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (identity, scope.profile_id, scope.map_id, scope.frame_revision, key, observation.kind, entity_id,
                    observation.observed_unix_s, json.dumps(record, allow_nan=False), evidence_id))
            if observation.kind in {"room", "object"}:
                self._update_belief(connection, scope, observation.kind, entity_id, record)
            if observation.kind == "traversal":
                connection_id = str(uuid5(NAMESPACE_URL, json.dumps([scope.map_id, observation.from_entity_id, observation.to_entity_id])))
                row = connection.execute("SELECT document FROM memory_entities WHERE profile_id=? AND entity_id=?", (scope.profile_id, connection_id)).fetchone()
                belief = json.loads(row[0]) if row else {"entity_id": connection_id, "success_count": 0, "total_distance_m": 0.}
                distance = sum(math.dist(start, end) for start, end in zip(observation.path_m, observation.path_m[1:]))
                belief.update(kind="connection", from_entity_id=observation.from_entity_id, to_entity_id=observation.to_entity_id,
                    map_id=scope.map_id, frame_revision=scope.frame_revision, requires_revalidation=False,
                    success_count=belief["success_count"] + 1, total_distance_m=belief["total_distance_m"] + distance,
                    last_distance_m=distance, last_duration_s=observation.duration_s, path_m=observation.path_m,
                    last_observed_unix_s=observation.observed_unix_s, evidence_id=identity, source=observation.source)
                connection.execute("INSERT INTO memory_entities VALUES (?, ?, 'connection', ?) ON CONFLICT(profile_id, entity_id) DO UPDATE SET document=excluded.document", (scope.profile_id, connection_id, json.dumps(belief)))
            if validate and not validate():
                raise ValueError("STALE_MEMORY_CONTEXT: observation authority changed before commit")
            return record

    @staticmethod
    def _update_belief(connection, scope, kind, identity, record):
        row = connection.execute("SELECT document FROM memory_entities WHERE profile_id=? AND entity_id=?", (scope.profile_id, identity)).fetchone()
        previous = json.loads(row[0]) if row else None
        if previous and record["observed_unix_s"] < previous["last_observed_unix_s"]:
            return
        labels = list(dict.fromkeys([*(previous or {}).get("labels", []), record["label"]]))[-8:]
        belief = {"entity_id": identity, "kind": kind, "label": record["label"], "labels": labels,
            "status": "conflicting_labels" if len(labels) > 1 else "tentative", "identity_verified": False,
            "confidence": record["confidence"], "confidence_calibrated": False,
            "position_m": record.get("position_m"), "pose_m_rad": record["pose_m_rad"], "place_id": record.get("place_id"),
            "pose_frame": record["pose_frame"],
            "map_id": scope.map_id, "frame_revision": scope.frame_revision,
            "requires_revalidation": record["requires_revalidation"], "last_observed_unix_s": record["observed_unix_s"],
            "evidence_id": record["observation_id"], "image_evidence_id": record["evidence_id"],
            "observation_count": (previous or {}).get("observation_count", 0) + 1}
        belief.update(source_key=record["source_key"], last_source_sequence=record["sequence"],
            last_run_id=scope.run_id, head_rad=record.get("head_rad"))
        connection.execute("INSERT INTO memory_entities VALUES (?, ?, ?, ?) ON CONFLICT(profile_id, entity_id) DO UPDATE SET document=excluded.document", (scope.profile_id, identity, kind, json.dumps(belief, allow_nan=False)))

    def memory_summary(self, scope, *, target="", limit=12):
        limit = min(64, max(1, limit))
        with self.connect() as connection:
            self._check_scope(connection, scope)
            counts = dict(connection.execute("SELECT kind, count(*) FROM memory_entities WHERE profile_id=? GROUP BY kind", (scope.profile_id,)))
            groups = {}
            for kind in ("room", "object", "connection"):
                groups[kind] = [json.loads(row[0]) for row in connection.execute(
                    "SELECT document FROM memory_entities WHERE profile_id=? AND kind=? AND (?='' OR kind!='object' OR instr(lower(json_extract(document, '$.label')),lower(?))>0) ORDER BY json_extract(document, '$.last_observed_unix_s') DESC LIMIT ?",
                    (scope.profile_id, kind, target, target, limit))]
            events = [json.loads(row[0]) for row in connection.execute("SELECT document FROM memory_events WHERE profile_id=? AND kind IN ('search', 'entrance') AND (?='' OR lower(json_extract(document,'$.target'))=lower(?)) ORDER BY observed_unix_s DESC LIMIT ?", (scope.profile_id, target, target, limit * 2))]
            for item in [item for group in groups.values() for item in group]:
                age = time.time() - item["last_observed_unix_s"]
                item["age_s"] = age if age >= 0 else None
            return {"scope": scope.model_dump(), "rooms": groups["room"], "objects": groups["object"],
                "connections": groups["connection"], "search_history": [item for item in events if item.get("kind") == "search"][:limit],
                "entrances": [item for item in events if item.get("kind") == "entrance"][:limit],
                "counts": {"rooms": counts.get("room", 0), "objects": counts.get("object", 0), "connections": counts.get("connection", 0)},
                "truncated": any(count > limit for count in counts.values()) or len(events) >= limit,
                "source": "observation_memory", "motion_authorized": False}

    def lookup_room(self, scope, query):
        with self.connect() as connection:
            self._check_scope(connection, scope)
            return [json.loads(row[0]) for row in connection.execute("SELECT document FROM memory_entities WHERE profile_id=? AND kind='room' AND (entity_id=? OR EXISTS(SELECT 1 FROM json_each(json_extract(document,'$.labels')) WHERE instr(lower(value),lower(?))>0)) ORDER BY json_extract(document,'$.last_observed_unix_s') DESC LIMIT 8", (scope.profile_id, query.strip(), query.strip()))]

    def find_object_sightings(self, scope, query, pose=None):
        summary = self.memory_summary(scope, target=query, limit=64)
        now = time.time()
        with self.connect() as connection:
            self._check_scope(connection, scope)
            inspections = [json.loads(row[0]) for row in connection.execute("SELECT document FROM memory_events WHERE profile_id=? AND kind='search' AND observed_unix_s>=? ORDER BY observed_unix_s DESC LIMIT 128", (scope.profile_id, now - 300.))]
        candidates = []
        for item in summary["objects"]:
            compatible = not item["requires_revalidation"] and item.get("pose_frame", "map") == "map" and item["map_id"] == scope.map_id and item["frame_revision"] == scope.frame_revision
            distance = math.dist(pose[:2], item["pose_m_rad"][:2]) if pose and compatible else None
            recent = any(record.get("target", "").casefold() == query.strip().casefold()
                and record.get("result") == "not_seen" and compatible and record.get("frame", "map") == "map"
                and record.get("frame_revision") == scope.frame_revision and record.get("head_rad") is not None and item.get("head_rad") is not None
                and max(abs(first-second) for first, second in zip(record["head_rad"], item["head_rad"])) <= .1
                and abs(math.atan2(math.sin(record["pose_m_rad"][2]-item["pose_m_rad"][2]), math.cos(record["pose_m_rad"][2]-item["pose_m_rad"][2]))) <= .1
                and record["observed_unix_s"] >= item["last_observed_unix_s"]
                and math.dist(record["pose_m_rad"][:2], item["pose_m_rad"][:2]) <= .25 for record in inspections)
            candidates.append({**item, "viewpoint_distance_m": distance, "recently_inspected": recent,
                "ranking_score": item["confidence"] - min(1., max(0., now-item["last_observed_unix_s"])/86400.) - (distance or 0.)*.05 - int(recent)*2 - int(item["requires_revalidation"])*4})
        return sorted(candidates, key=lambda item: item["ranking_score"], reverse=True)[:8]

    def search_history(self, scope, target):
        with self.connect() as connection:
            self._check_scope(connection, scope)
            return [json.loads(row[0]) for row in connection.execute("SELECT document FROM memory_events WHERE profile_id=? AND kind='search' AND lower(json_extract(document, '$.target'))=lower(?) ORDER BY observed_unix_s DESC LIMIT 16", (scope.profile_id, target.strip()))]

    def invalidate_frame(self, scope, new_frame_revision):
        if not new_frame_revision or new_frame_revision == scope.frame_revision:
            raise ValueError("Map correction requires a new frame revision")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_scope(connection, scope)
            connection.execute("UPDATE memory_map_scopes SET frame_revision=? WHERE map_id=?", (new_frame_revision, scope.map_id))
            connection.execute("UPDATE memory_contexts SET active=0 WHERE profile_id=?", (scope.profile_id,))
            for identity, document in connection.execute("SELECT entity_id, document FROM memory_entities WHERE profile_id=?", (scope.profile_id,)).fetchall():
                belief = json.loads(document)
                belief.update(requires_revalidation=True, status="requires_revalidation")
                connection.execute("UPDATE memory_entities SET document=? WHERE profile_id=? AND entity_id=?", (json.dumps(belief), scope.profile_id, identity))
            row = connection.execute("SELECT document FROM maps WHERE map_id=?", (scope.map_id,)).fetchone()
            if row:
                document = json.loads(row[0])
                for place in document["places"]:
                    place["requires_revalidation"] = True
                document["frontier_attempts"] = {}
                connection.execute("UPDATE maps SET document=? WHERE map_id=?", (json.dumps(document), scope.map_id))

    def checkpoint(self, scope, home, name):
        if not name.strip() or len(name) > 80 or home.identity != scope.map_id or home.profile_id != scope.profile_id:
            raise ValueError("CHECKPOINT_SCOPE_MISMATCH: select a named checkpoint of the active map")
        document = home.document()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_scope(connection, scope)
            linked = {place["place_id"] for place in document["places"]}
            if connection.execute("SELECT 1 FROM memory_entities WHERE profile_id=? AND json_extract(document,'$.place_id') IS NOT NULL AND json_extract(document,'$.place_id') NOT IN (SELECT value FROM json_each(?)) LIMIT 1", (scope.profile_id, json.dumps(sorted(linked)))).fetchone():
                raise ValueError("CHECKPOINT_PENDING: an observation viewpoint has not been published into the map")
            snapshot = {"map": document, "scope": scope.model_dump(), "entities": [json.loads(row[0]) for row in connection.execute("SELECT document FROM memory_entities WHERE profile_id=?", (scope.profile_id,))],
                "event_ids": [row[0] for row in connection.execute("SELECT event_id FROM memory_events WHERE profile_id=?", (scope.profile_id,))],
                "reviews": dict(connection.execute("SELECT reviews.event_id, reviews.document FROM memory_reviews reviews JOIN memory_events events ON reviews.event_id=events.event_id WHERE events.profile_id=?", (scope.profile_id,)).fetchall())}
            identity = str(uuid4())
            timestamp = time.time()
            connection.execute("INSERT INTO knowledge_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", (identity, scope.profile_id,
                scope.environment_id, scope.environment_revision, name.strip(), timestamp, json.dumps(snapshot, allow_nan=False)))
        return {"snapshot_id": identity, "name": name.strip(), "created_unix_s": timestamp, "immutable": True}

    def snapshots(self, environment_id, revision, profile_id):
        with self.connect() as connection:
            self._profile(connection, environment_id, revision, profile_id)
            return [{"snapshot_id": row[0], "name": row[1], "created_unix_s": row[2], "immutable": True}
                for row in connection.execute("SELECT snapshot_id, name, created_unix_s FROM knowledge_snapshots WHERE environment_id=? AND environment_revision=? AND profile_id=? ORDER BY created_unix_s DESC", (environment_id, revision, profile_id))]

    def fork_snapshot(self, environment_id, revision, snapshot_id, name):
        if not name.strip() or len(name) > 80:
            raise ValueError("A fork needs a name of at most 80 characters")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT document FROM knowledge_snapshots WHERE snapshot_id=? AND environment_id=? AND environment_revision=?", (snapshot_id, environment_id, revision)).fetchone()
            if row is None:
                raise ValueError("SNAPSHOT_SCOPE_MISMATCH: checkpoint belongs to another environment")
            snapshot = json.loads(row[0])
            profile_id, map_id = str(uuid4()), str(uuid4())
            connection.execute("INSERT INTO knowledge_profiles VALUES (?, ?, ?, ?, ?, ?)", (profile_id, environment_id, revision, name.strip(), time.time(), snapshot_id))
            document = snapshot["map"]
            document.update(map_id=map_id, revision=1, name=name.strip())
            for place in document["places"]:
                place["map_id"] = map_id
            connection.execute("INSERT INTO maps VALUES (?, ?, 1, ?, ?, ?)", (map_id, document["environment_id"], json.dumps(document), profile_id, revision))
            connection.execute("INSERT INTO memory_map_scopes VALUES (?, ?, ?, ?, ?)", (map_id, environment_id, revision, profile_id, snapshot["scope"]["frame_revision"]))
            event_ids = {identity: str(uuid4()) for identity in snapshot["event_ids"]}
            for place in document["places"]:
                if place.get("evidence_id") in event_ids:
                    place["evidence_id"] = event_ids[place["evidence_id"]]
            connection.execute("UPDATE maps SET document=? WHERE map_id=?", (json.dumps(document), map_id))
            for old_id, new_id in event_ids.items():
                event = connection.execute("SELECT frame_revision, ingestion_key, kind, entity_id, observed_unix_s, document, evidence_id FROM memory_events WHERE event_id=?", (old_id,)).fetchone()
                record = json.loads(event[5])
                record.update(observation_id=new_id, profile_id=profile_id, map_id=map_id)
                if event[2] in {"room", "object"}:
                    collection = "rooms" if event[2] == "room" else "objects"
                    url = f"/api/home/{map_id}/{collection}/{new_id}/image.png"
                    record.update(image_url=url, supporting_images=[url])
                connection.execute("INSERT INTO memory_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (new_id, profile_id, map_id, *event[:5], json.dumps(record), event[6]))
            for belief in snapshot["entities"]:
                belief.update(map_id=map_id, evidence_id=event_ids.get(belief.get("evidence_id")))
                connection.execute("INSERT INTO memory_entities VALUES (?, ?, ?, ?)", (profile_id, belief["entity_id"], belief["kind"], json.dumps(belief)))
            for event_id, review in snapshot.get("reviews", {}).items():
                connection.execute("INSERT INTO memory_reviews VALUES (?, ?)", (event_ids[event_id], review))
            return {**self._profile(connection, environment_id, revision, profile_id), "map_id": map_id}

    def remember_legacy(self, kind, observation, image, *, allow_draft=False):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT profile_id FROM maps WHERE map_id=?", (observation["map_id"],)).fetchone()
            if row is None and not allow_draft:
                raise ValueError("SAVE_REQUIRED: observations require a saved map")
            profile_id = row[0] if row else "unbound:" + observation["map_id"]
            evidence_id = self._image(connection, image)
            identity = observation["observation_id"]
            connection.execute("INSERT OR IGNORE INTO memory_events VALUES (?, ?, ?, '1', ?, ?, ?, ?, ?, ?)", (identity, profile_id,
                observation["map_id"], "legacy:" + identity, kind, observation.get("place_id"), observation["observed_unix_s"], json.dumps(observation, allow_nan=False), evidence_id))

    def ensure_environment(self, environment_id, revision, scenario_id, fingerprint, generation=None):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT fingerprint FROM memory_environments WHERE environment_id=? AND revision=?", (environment_id, revision)).fetchone()
            if existing and existing[0] != fingerprint:
                raise ValueError("ENVIRONMENT_REVISION_CONFLICT: changed layouts need a new revision")
            connection.execute("INSERT OR IGNORE INTO memory_environments VALUES (?, ?, ?, ?, ?)",
                (environment_id, revision, scenario_id, fingerprint, json.dumps(generation or {}, allow_nan=False, sort_keys=True)))

    @staticmethod
    def _profile(connection, environment_id, revision, profile_id):
        row = connection.execute("SELECT profile_id, name, created_unix_s, parent_snapshot_id FROM knowledge_profiles WHERE profile_id=? AND environment_id=? AND environment_revision=?",
            (profile_id, environment_id, revision)).fetchone()
        if row is None:
            raise ValueError("MEMORY_SCOPE_MISMATCH: knowledge belongs to another environment or revision")
        return {"profile_id": row[0], "name": row[1], "created_unix_s": row[2], "parent_snapshot_id": row[3],
            "environment_id": environment_id, "environment_revision": revision}

    def profiles(self, environment_id, revision):
        with self.connect() as connection:
            identities = connection.execute("SELECT profile_id FROM knowledge_profiles WHERE environment_id=? AND environment_revision=? ORDER BY created_unix_s, profile_id", (environment_id, revision)).fetchall()
            return [self._profile(connection, environment_id, revision, row[0]) for row in identities]

    def create_profile(self, environment_id, revision, name, *, profile_id=None, select=False):
        if not name.strip() or len(name) > 80:
            raise ValueError("A knowledge profile needs a name of at most 80 characters")
        identity = profile_id or str(uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute("SELECT 1 FROM memory_environments WHERE environment_id=? AND revision=?", (environment_id, revision)).fetchone():
                raise ValueError("UNKNOWN_ENVIRONMENT: register the environment revision first")
            connection.execute("INSERT INTO knowledge_profiles VALUES (?, ?, ?, ?, ?, NULL)", (identity, environment_id, revision, name.strip(), time.time()))
            if select:
                connection.execute("INSERT INTO selected_knowledge VALUES (?, ?, ?) ON CONFLICT(environment_id, environment_revision) DO UPDATE SET profile_id=excluded.profile_id", (environment_id, revision, identity))
            return self._profile(connection, environment_id, revision, identity)

    def selected_profile(self, environment_id, revision="legacy"):
        with self.connect() as connection:
            row = connection.execute("SELECT profile_id FROM selected_knowledge WHERE environment_id=? AND environment_revision=?", (environment_id, revision)).fetchone()
            if row:
                return self._profile(connection, environment_id, revision, row[0])
        if revision == "legacy":
            self.ensure_environment(environment_id, revision, environment_id, "legacy")
        identity = default_profile_id(environment_id, revision)
        try:
            return self.create_profile(environment_id, revision, "Default knowledge", profile_id=identity, select=True)
        except sqlite3.IntegrityError:
            return self.select_profile(environment_id, revision, identity)

    def select_profile(self, environment_id, revision, profile_id):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._profile(connection, environment_id, revision, profile_id)
            connection.execute("INSERT INTO selected_knowledge VALUES (?, ?, ?) ON CONFLICT(environment_id, environment_revision) DO UPDATE SET profile_id=excluded.profile_id", (environment_id, revision, profile_id))
            return result