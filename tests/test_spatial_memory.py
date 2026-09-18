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


def test_repeated_nearby_object_matches_only_in_the_same_valid_map_frame(tmp_path):
    from backend.memory_session import matching_object
    store, _, scope = memory_fixture(tmp_path)
    existing = {"entity_id": "tv-one", "label": "TV", "labels": ["Television"],
        "position_m": [1., 2., .8], "pose_frame": "map", "map_id": scope.map_id,
        "frame_revision": scope.frame_revision, "requires_revalidation": False}
    summary = {"objects": [existing]}
    assert matching_object(summary, scope, "tv", [1.1, 2., .8]) == existing
    assert matching_object(summary, scope, "Television", [1.1, 2., .8]) == existing
    assert matching_object(summary, scope, "TV", [1.4, 2., .8]) is None
    assert matching_object(summary, scope, "Monitor", [1.1, 2., .8]) is None
    assert matching_object({"objects": [{**existing, "requires_revalidation": True}]}, scope, "TV", [1.1, 2., .8]) is None
    assert matching_object({"objects": [{**existing, "frame_revision": "corrected"}]}, scope, "TV", [1.1, 2., .8]) is None


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


@pytest.mark.parametrize("fault", ["empty_label", "foreign_entity", "write_failure"])
async def test_rejected_memory_observation_does_not_publish_a_place(tmp_path, monkeypatch, fault):
    import asyncio
    from backend.home_mission import HomeMission, HomeRequest
    from backend.memory_session import configure_memory, record_observation
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "rejected-memory.sqlite3"))
        await configure_memory(worker, instance_id="rejected-evidence-flat")
        await worker.home_command(HomeRequest(run_id=worker.latest["run_id"], episode_epoch=worker.epoch, action="start_mapping"))
        before = await worker.call(lambda sim: list(worker.home_mission.home.places))
        if fault == "write_failure":
            def fail(*arguments, **keywords):
                raise ValueError("Injected evidence storage failure")
            monkeypatch.setattr(worker.memory.store, "ingest", fail)
        with pytest.raises(ValueError):
            await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object",
                label="" if fault == "empty_label" else "Fixture object",
                entity_id="foreign" if fault == "foreign_entity" else None, source="fixture")
        assert await worker.call(lambda sim: worker.home_mission.home.places) == before
    finally:
        await worker.close()


async def test_revocation_during_evidence_commit_rolls_back(tmp_path, monkeypatch):
    import asyncio
    from backend.memory_session import MemorySession
    store, home, scope = memory_fixture(tmp_path)
    session = MemorySession(store, scope)
    valid = True
    original = store._update_belief
    def revoke(*arguments):
        nonlocal valid
        original(*arguments)
        valid = False
    monkeypatch.setattr(store, "_update_belief", revoke)
    try:
        with pytest.raises(ValueError, match="before commit"):
            await asyncio.wrap_future(session.observe(observation(), validate=lambda: valid))
        assert store.memory_summary(scope)["rooms"] == []
        with store.connect() as connection:
            assert connection.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0
    finally:
        await session.close()


async def test_executed_path_between_observed_regions_is_retained(tmp_path):
    import asyncio
    import math
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mission import HomeMission, HomeRequest
    from backend.memory_session import configure_memory, persist_memory, record_observation
    from backend.worker import SimulationWorker
    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]}]})
    worker = SimulationWorker(pace=False, challenge=challenge, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "path-memory.sqlite3"))
        await configure_memory(worker, instance_id="path-fixture")
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        first = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="room", label="Fixture region A", source="fixture")
        await worker.home_command(HomeRequest(**identity, action="guided_to", pose_m_rad=[.6, 0., 0.], time_budget=45.))
        async with asyncio.timeout(50):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        second = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="room", label="Fixture region B", source="fixture")
        assert math.dist(first["pose_m_rad"][:2], second["pose_m_rad"][:2]) >= .5
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Observed transitions"))
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=first["place_id"], time_budget=45.))
        async with asyncio.timeout(50):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed", worker.home_mission.task
        await persist_memory(worker)
        connections = worker.memory.summary["connections"]
        assert connections, worker.memory.summary
        returning = next(item for item in connections if item["from_entity_id"] == second["entity_id"] and item["to_entity_id"] == first["entity_id"])
        assert returning["success_count"] == 1 and returning["last_distance_m"] >= .35
        assert len(returning["path_m"]) > 3 and returning["source"] == "worker_odometry"
        assert not worker.sim.proximity_sensors().collisions
    finally:
        await worker.close()


def test_frame_revision_reload_and_unpublished_checkpoint_fail_closed(tmp_path):
    store, home, scope = memory_fixture(tmp_path)
    home.evidence[185:215, 185:215] = -2
    place = home.add_place("Observed room", "room", [0., 0., 0.], .3)
    store.save(home, home.name, scope=scope)
    store.ingest(scope, observation(place_id=place["place_id"]))
    store.invalidate_frame(scope, "replacement-frame")
    loaded = store.load(home.identity, home.environment_id)
    assert loaded.frame_revision == "replacement-frame"
    assert loaded.places[0]["requires_revalidation"]
    new_scope = store.open_context(loaded, "new-run", 2)
    assert store.memory_summary(new_scope)["rooms"][0]["requires_revalidation"]
    with pytest.raises(ValueError, match="STALE_MEMORY_CONTEXT"):
        store.ingest(scope, observation(source_key="late"))
    store.ingest(new_scope, observation(source_key="unpublished", place_id="missing"))
    with pytest.raises(ValueError, match="CHECKPOINT_PENDING"):
        store.checkpoint(new_scope, loaded, "Incomplete checkpoint")


async def test_close_drains_thread_and_discards_queued_observation(tmp_path):
    import asyncio
    import threading
    from backend.memory_session import MemorySession
    store, home, scope = memory_fixture(tmp_path)
    session = MemorySession(store, scope)
    entered, release = threading.Event(), threading.Event()
    def delay():
        entered.set()
        assert release.wait(5)
    held = session.submit(delay)
    assert await asyncio.to_thread(entered.wait, 5)
    queued = session.observe(observation())
    closing = asyncio.create_task(session.close())
    await asyncio.wait_for(asyncio.to_thread(lambda: release.set() if not session.active else None), 5)
    release.set()
    await closing
    assert held.done() and queued.done()
    assert isinstance(queued.exception(), ValueError)
    assert not any(thread.is_alive() for thread in session.executor._threads)
    with store.connect() as connection:
        assert connection.execute("SELECT count(*) FROM memory_events").fetchone()[0] == 0


def test_search_history_deprioritizes_same_view_without_claiming_room_absence(tmp_path):
    import time
    store, home, scope = memory_fixture(tmp_path)
    timestamp = time.time()
    sighting = observation("object", "cup", position_m=[1., 0., .7], head_rad=[0., .2]).model_copy(update={"observed_unix_s": timestamp-10})
    store.ingest(scope, sighting)
    search = observation("search", "partial-view", target="Cup", result="not_seen", visibility="partial", head_rad=[0., .2]).model_copy(update={"observed_unix_s": timestamp})
    report = store.ingest(scope, search)
    assert not report["absence_supported"]
    assert store.find_object_sightings(scope, "Cup", [0., 0., 0.])[0]["recently_inspected"]
    newer = sighting.model_copy(update={"source_key": "new-sighting", "entity_id": store.memory_summary(scope)["objects"][0]["entity_id"], "observed_unix_s": timestamp+1})
    store.ingest(scope, newer)
    assert not store.find_object_sightings(scope, "Cup", [0., 0., 0.])[0]["recently_inspected"]


async def test_memory_ingestion_does_not_consult_hidden_scene_metadata(tmp_path, monkeypatch):
    import asyncio
    from backend.challenges import get_challenge
    from backend.home_mission import HomeMission, HomeRequest
    from backend.memory_session import configure_memory, record_observation
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, challenge=get_challenge("park"), rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "truth-isolation.sqlite3"))
        await configure_memory(worker)
        await worker.home_command(HomeRequest(run_id=worker.latest["run_id"], episode_epoch=worker.epoch, action="start_mapping"))
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        monkeypatch.setattr(worker.challenge, "objects", [{"name": "HIDDEN_TRUTH_SENTINEL", "position": [999, 999, 999]}])
        monkeypatch.setattr(worker.challenge, "title", "HIDDEN_TRUTH_SENTINEL")
        record = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object", label="Fixture surface",
            bounds=[.4,.4,.6,.6], selected_evidence=paired, source="fixture")
        assert "HIDDEN_TRUTH_SENTINEL" not in json.dumps(record)
        assert record["position_m"] is None or all(abs(value)<10 for value in record["position_m"])
        assert worker.memory.store.object_image(record["map_id"], record["observation_id"]) == paired[1]
    finally:
        await worker.close()