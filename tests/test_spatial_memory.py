import json
import sqlite3

import pytest

from backend.home_mapping import HomeMap, MapStore


def test_profile_scoped_maps_and_revision_isolation(tmp_path):
    store = MapStore(tmp_path / "memory.sqlite3")
    store.ensure_environment("flat-one", "layout-a", "apartment", "fingerprint-a", {"seed": 713})
    first = store.create_profile("flat-one", "layout-a", "Learned")
    second = store.create_profile("flat-one", "layout-a", "Fresh")
    for profile in (first, second):
        home = HomeMap("flat-one")
        home.profile_id, home.environment_revision = profile["profile_id"], "layout-a"
        home.scan_count = 1
        store.save(home, profile["name"])
    maps = store.catalog()
    assert len(maps) == 2 and maps[0]["map_id"] != maps[1]["map_id"]
    first_map = next(item for item in maps if item["profile_id"] == first["profile_id"])
    loaded = store.load(first_map["map_id"], "flat-one", profile_id=first["profile_id"], environment_revision="layout-a")
    assert loaded.profile_id == first["profile_id"]
    with pytest.raises(ValueError, match="MAP_NOT_FOUND"):
        store.load(loaded.identity, "flat-one", profile_id=second["profile_id"], environment_revision="layout-a")
    with pytest.raises(ValueError, match="SCOPE_MISMATCH"):
        store.select_profile("flat-two", "layout-a", first["profile_id"])
    with pytest.raises(ValueError, match="REVISION_CONFLICT"):
        store.ensure_environment("flat-one", "layout-a", "apartment", "changed")
    store.select_profile("flat-one", "layout-a", second["profile_id"])
    assert MapStore(store.path).selected_profile("flat-one", "layout-a")["profile_id"] == second["profile_id"]
    store.path.unlink()


def test_memory_migration_preserves_legacy_map_document_and_image(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    home = HomeMap("legacy-flat")
    home.scan_count, home.revision = 1, 3
    document = json.dumps(home.document())
    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute("CREATE TABLE maps (map_id TEXT PRIMARY KEY, environment_id TEXT UNIQUE, revision INTEGER, document TEXT)")
            connection.execute("INSERT INTO maps VALUES (?, ?, ?, ?)", (home.identity, home.environment_id, 3, document))
            connection.execute("CREATE TABLE object_observations (observation_id TEXT PRIMARY KEY, map_id TEXT, timestamp REAL, document TEXT, image BLOB)")
            connection.execute("INSERT INTO object_observations VALUES ('sighting', ?, 1, '{}', ?)", (home.identity, b"preserve-image"))
    finally:
        connection.close()
    store = MapStore(path)
    with store.connect() as connection:
        assert connection.execute("SELECT document FROM maps").fetchone()[0] == document
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert store.load(home.identity, "legacy-flat").revision == 3
    assert store.object_image(home.identity, "sighting") == b"preserve-image"
    assert MapStore(path).selected_profile("legacy-flat")["name"] == "Legacy knowledge"
    path.unlink()


def memory_fixture(tmp_path):
    store = MapStore(tmp_path / "knowledge.sqlite3")
    home = HomeMap("flat")
    home.scan_count = 1
    store.save(home, "Observed flat")
    return store, home, store.open_context(home, "run-one", 1)


def observation(kind="room", source_key="first", **values):
    from backend.spatial_memory import MemoryObservation
    return MemoryObservation(kind=kind, source_key=source_key, source="fixture", sequence=1,
        frame_ref="camera-one", observed_unix_s=1700000000., pose_m_rad=[0., 0., 0.],
        label="Kitchen" if kind == "room" else "Cup", **values)


def test_memory_idempotency_beliefs_search_and_stale_context(tmp_path):
    store, home, scope = memory_fixture(tmp_path)
    first = store.ingest(scope, observation())
    assert store.ingest(scope, observation()) == first
    cup = store.ingest(scope, observation("object", "cup-one", position_m=[1., 0., .8]))
    other = store.ingest(scope, observation("object", "cup-two", position_m=[2., 0., .8]))
    assert cup["entity_id"] != other["entity_id"]
    revised = observation("object", "cup-reobserved", entity_id=cup["entity_id"], position_m=[1.5, 0., .8])
    store.ingest(scope, revised.model_copy(update={"label": "Mug", "observed_unix_s": 1700000001.}))
    store.ingest(scope, observation("search", "counter", target="Cup", inspection_scope="surface", visibility="partial", result="not_seen"))
    summary = store.memory_summary(scope)
    belief = next(item for item in summary["objects"] if item["entity_id"] == cup["entity_id"])
    assert belief["status"] == "conflicting_labels" and belief["position_m"] == [1.5, 0., .8]
    assert len(summary["rooms"]) == 1 and len(summary["objects"]) == 2
    assert summary["search_history"][0]["absence_supported"] is False
    assert summary["search_history"][0]["inspection_scope"] == "surface"
    store.close_context(scope)
    with pytest.raises(ValueError, match="STALE_MEMORY_CONTEXT"):
        store.ingest(scope, observation(source_key="late"))


def test_checkpoint_fork_is_independent_and_retains_evidence(tmp_path):
    from io import BytesIO
    from PIL import Image
    store, home, scope = memory_fixture(tmp_path)
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, format="PNG")
    room = store.ingest(scope, observation(), image.getvalue())
    second = store.ingest(scope, observation(source_key="second").model_copy(update={"label": "Hall", "pose_m_rad": [1., 0., 0.]}))
    store.ingest(scope, observation("traversal", "trip", from_entity_id=room["entity_id"], to_entity_id=second["entity_id"], path_m=[[0., 0.], [.5, 0.], [1., 0.]], duration_s=5.))
    checkpoint = store.checkpoint(scope, home, "Two rooms")
    store.review_room(home.identity, room["observation_id"])
    store.ingest(scope, observation(source_key="after-checkpoint").model_copy(update={"label": "Bedroom"}))
    fork = MapStore(store.path).fork_snapshot("flat", "legacy", checkpoint["snapshot_id"], "Independent")
    fork_map = store.load(fork["map_id"], "flat", profile_id=fork["profile_id"])
    fork_scope = store.open_context(fork_map, "run-two", 2)
    summary = store.memory_summary(fork_scope)
    assert len(summary["rooms"]) == 2 and len(store.memory_summary(scope)["rooms"]) == 3
    assert summary["connections"][0]["success_count"] == 1
    evidence = next(item for item in summary["rooms"] if item["entity_id"] == room["entity_id"])
    assert store.room_image(fork_map.identity, evidence["evidence_id"]) == image.getvalue()
    assert next(item for item in store.room_observations(fork_map.identity) if item["observation_id"] == evidence["evidence_id"])["review_status"] == "tentative"
    store.ingest(fork_scope, observation("search", "fork-search", target="Cup"))
    assert not store.memory_summary(scope)["search_history"]
    with pytest.raises(ValueError, match="SNAPSHOT_SCOPE_MISMATCH"):
        store.fork_snapshot("another-flat", "legacy", checkpoint["snapshot_id"], "Wrong environment")
    store.invalidate_frame(fork_scope, "corrected")
    with pytest.raises(ValueError, match="STALE_MEMORY_CONTEXT"):
        store.ingest(fork_scope, observation(source_key="old-frame"))
    corrected = store.open_context(fork_map, "run-three", 3, frame_revision="corrected")
    assert all(item["requires_revalidation"] for item in store.memory_summary(corrected)["rooms"])


def test_memory_observation_rejects_hidden_truth_and_unlocated_coordinates():
    from pydantic import ValidationError
    from backend.spatial_memory import MemoryObservation
    data = observation("object").model_dump()
    with pytest.raises(ValidationError):
        MemoryObservation(**data, simulator_objects=[{"name": "secret"}])
    with pytest.raises(ValidationError):
        MemoryObservation(**{**data, "metric_valid": False, "position_m": [1., 2., 3.]})


def test_environment_identity_uses_instance_and_layout_without_semantic_payload():
    from backend.challenges import get_challenge, shared_apartment
    from backend.spatial_memory import environment_identity
    first = environment_identity(shared_apartment("flat_kitchen"))
    assert first == environment_identity(shared_apartment("apartment"))
    other = environment_identity(shared_apartment("flat_kitchen"), "another-flat")
    assert other["environment_id"] != first["environment_id"] and other["revision"] == first["revision"]
    challenge = get_challenge("flat_kitchen")
    before = environment_identity(challenge)
    challenge.objects[0]["position"][0] += .1
    after = environment_identity(challenge)
    assert before["environment_id"] == after["environment_id"] and before["revision"] != after["revision"]
    assert "position" not in json.dumps(first) and first["generation"]["seed"] is None


async def test_background_memory_flush_and_obsolete_session_rejection(tmp_path):
    import asyncio
    from backend.memory_session import MemorySession
    store, home, scope = memory_fixture(tmp_path)
    session = MemorySession(store, scope)
    try:
        await asyncio.wrap_future(session.observe(observation()))
        await session.flush()
        assert len(session.summary["rooms"]) == 1
    finally:
        await session.close()
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        session.observe(observation(source_key="late"))
    with pytest.raises(ValueError, match="STALE_MEMORY_CONTEXT"):
        store.ingest(scope, observation(source_key="late"))


async def test_worker_profiles_persist_sensor_rooms_and_lookup_existing_navigation(tmp_path):
    import asyncio
    from backend.home_mission import HomeMission, HomeRequest
    from backend.memory_session import configure_memory, lookup_memory, persist_memory, record_observation
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "worker-memory.sqlite3"))
        await configure_memory(worker, instance_id="fixture-flat")
        scope = worker.memory.scope
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Fixture observation map"))
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        room = await record_observation(worker, context_id=scope.context_id, kind="room", label="Fixture room", description="Scripted semantic label on actual camera evidence", selected_evidence=paired, source="fixture")
        repeated = await record_observation(worker, context_id=scope.context_id, kind="room", label="Fixture room", description="Scripted semantic label on actual camera evidence", selected_evidence=paired, source="fixture")
        assert repeated["observation_id"] == room["observation_id"] and len(worker.home_mission.home.places) == 1
        assert not room["identity_verified"] and room["place_id"]
        lookup = await lookup_memory(worker, "lookup_room", "Fixture room")
        assert lookup["records"][0]["place_id"] == room["place_id"]
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=room["place_id"], time_budget=10))
        async with asyncio.timeout(15):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed"
        await persist_memory(worker)
        profile_id = worker.memory.scope.profile_id
        await configure_memory(worker, fresh=True, name="Empty fixture")
        assert worker.home_mission.home is None and not worker.memory.summary["rooms"]
        assert not worker.spatial_frames and not (worker.spatial_map.cells != -1).any()
        with pytest.raises(ValueError, match="STALE_MEMORY_CONTEXT"):
            await record_observation(worker, context_id=scope.context_id, kind="room", label="Delayed")
        await configure_memory(worker, profile_id=profile_id)
        assert worker.memory.summary["rooms"][0]["label"] == "Fixture room"
        assert worker.home_mission.localization["status"] == "unlocalized"
        await configure_memory(worker, instance_id="other-fixture-flat")
        assert not worker.memory.summary["rooms"] and worker.home_mission.home is None
    finally:
        await worker.close()