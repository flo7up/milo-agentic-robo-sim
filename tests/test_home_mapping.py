import math
import asyncio
from uuid import uuid4

import numpy as np
import pytest

from backend.home_mapping import HomeMap, MapStore, inverse_pose, transform_pose


def test_sensor_rays_preserve_occlusion_and_unknown():
    home = HomeMap("test-home")
    laser = {"origin_m": [0., 0., .3], "angle_min": 0., "angle_increment": math.pi / 2,
        "range_min": .03, "range_max": 8., "ranges_m": [2., None, 8.01]}
    home.observe(laser, [0., 0., 0.], 100.)
    def cell(point):
        column, row = home.indices(point)
        return home.cells[row, column]
    assert cell([1., 0.]) == 0
    assert cell([2., 0.]) == 100
    assert cell([3., 0.]) == -1
    assert cell([0., 2.]) == -1
    assert cell([-3., 0.]) == 0


def test_save_once_reload_for_scenarios_and_revision_conflict(tmp_path):
    store = MapStore(tmp_path / "maps.sqlite3")
    home = HomeMap("shared_apartment_v1")
    home.scan_count = 1
    home.evidence[190:220, 190:240] = -2
    home.evidence[199:210, 220] = 4
    home.places = [{"place_id": "home", "name": "Home", "kind": "destination", "pose_m_rad": [0., 0., 0.]}]
    store.save(home, "My home")
    restored = MapStore(store.path).load(home.identity, "shared_apartment_v1")
    np.testing.assert_array_equal(restored.cells, home.cells)
    assert restored.places == home.places
    with pytest.raises(ValueError, match="READ_ONLY"):
        restored.observe({}, [0., 0., 0.], 101.)
    duplicate = HomeMap("shared_apartment_v1")
    duplicate.scan_count = 1
    with pytest.raises(ValueError, match="MAP_EXISTS"):
        store.save(duplicate, "Again")
    store.save(home, "Renamed")
    with pytest.raises(ValueError, match="MAP_CONFLICT"):
        store.save(restored, "Stale")
    with pytest.raises(ValueError, match="MAP_NOT_FOUND"):
        store.load(home.identity, "different-house")


def test_route_uses_footprint_unknown_and_live_obstacles():
    home = HomeMap("test-home")
    home.evidence[190:215, 190:230] = -2
    route = home.route([0., 0.], [2., 0.], .3)
    assert route[0] == [0., 0.] and route[-1] == [2., 0.]
    before = home.evidence.copy()
    barrier = [[1., lateral] for lateral in np.arange(-1., 1.5, .05)]
    with pytest.raises(ValueError, match="UNREACHABLE"):
        home.route([0., 0.], [2., 0.], .3, barrier)
    np.testing.assert_array_equal(home.evidence, before)
    with pytest.raises(ValueError, match="UNREACHABLE"):
        home.route([0., 0.], [8., 0.], .3)


def test_map_odometry_transform_round_trip():
    pose, transform = [1., 2., .4], [3., -2., 1.2]
    np.testing.assert_allclose(transform_pose(transform_pose(pose, transform), inverse_pose(transform)), pose)


def test_object_memory_is_separate_persistent_and_image_supported(tmp_path):
    from io import BytesIO
    from PIL import Image
    store = MapStore(tmp_path / "objects.sqlite3")
    home = HomeMap("home")
    home.scan_count = 1
    store.save(home, "Home")
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, format="PNG")
    record = {"observation_id": str(uuid4()), "map_id": home.identity, "observed_unix_s": 10.,
        "label": "toy", "confidence": .6, "position_m": [1., 2., .1], "frame": "map", "supporting_images": ["image.png"]}
    store.remember_object(record, image.getvalue())
    assert MapStore(store.path).object_observations(home.identity) == [record]
    assert store.object_image(home.identity, record["observation_id"]) == image.getvalue()
    restored = store.load(home.identity, "home")
    np.testing.assert_array_equal(home.evidence, restored.evidence)
    assert restored.revision == 1 and not restored.objects
    with pytest.raises(ValueError, match="not found"):
        store.object_image("wrong-map", record["observation_id"])


async def test_evaluator_map_options_are_explicit_and_do_not_create_maps():
    from types import SimpleNamespace
    from scripts.evaluate_supervised import argument_parser, prepare_home_map
    options = argument_parser().parse_args(["--output", "unused", "--mode", "luna_continuous",
        "--home-map-id", "saved-id", "--localization-pose", "1", "2", ".3"])
    assert options.home_map_id == "saved-id" and options.localization_pose == [1., 2., .3]
    calls = []
    async def command(request):
        calls.append(request)
        return {"localization": {"status": "localized"}}
    home = HomeMap("shared_apartment_v1", "saved-id")
    home.revision = 2
    worker = SimpleNamespace(latest={"run_id": "run"}, epoch=3, home_command=command,
        home_mission=SimpleNamespace(home=home))
    result = await prepare_home_map(worker, options)
    assert [call.action for call in calls] == ["load_map", "localize"]
    assert result["revision"] == 2 and result["seed_source"] == "operator_supplied" and not result["geometry_updates"]
    assert len(result["sha256"]) == 64


def test_named_graph_and_frontier_attempts_survive_reload(tmp_path):
    home = HomeMap("home")
    home.evidence[170:230, 170:230] = -2
    home.scan_count = 1
    room = home.add_place("Living room", "room", [0., 0., 0.], .3)
    door = home.add_place("Kitchen entrance", "doorway", [1., 0., 0.], .3, [room["place_id"]])
    assert home.edges[0]["from"] == door["place_id"]
    frontier = home.frontiers([0., 0., 0.], .3)[0]
    home.mark_frontier(frontier["frontier_id"])
    home.mark_frontier(frontier["frontier_id"])
    assert home.frontiers([0., 0., 0.], .3) == []
    store = MapStore(tmp_path / "map.sqlite3")
    store.save(home, "Home")
    restored = store.load(home.identity, "home")
    assert restored.frontier_attempts == home.frontier_attempts
    assert restored.edges == home.edges
    assert restored.frontiers([0., 0., 0.], .3) == []


def test_scan_matching_uses_sensor_returns_and_rejects_missing_data():
    home = HomeMap("home")
    ranges = [2. + .5 * math.sin(angle * 3) + .2 * math.cos(angle * 5)
        for angle in np.linspace(-math.pi, math.pi, 720, endpoint=False)]
    laser = {"origin_m": [.19, 0., .305], "angle_min": -math.pi, "angle_increment": 2 * math.pi / 720,
        "range_min": .03, "range_max": 8., "ranges_m": ranges}
    home.observe(laser, [1., -.5, .4], 100.)
    pose, quality = home.match_scan(laser, [1.1, -.6, .45])
    assert math.dist(pose[:2], [1., -.5]) < .12
    assert abs(pose[2] - .4) < .06
    assert quality["matched_fraction"] > .9
    with pytest.raises(ValueError, match="insufficient"):
        home.match_scan({**laser, "ranges_m": [None] * 720}, [1., -.5, .4])


async def test_guided_map_save_reload_localize_and_navigate_real_physics(tmp_path):
    from backend.challenges import get_challenge
    from backend.contracts import Command
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker

    store = MapStore(tmp_path / "home.sqlite3")
    worker = SimulationWorker(pace=False, challenge=get_challenge("park"), rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, store)
        def request(action, **kwargs):
            return HomeRequest(run_id=worker.latest["run_id"], episode_epoch=0, action=action, **kwargs)
        await worker.scan_continuous(ContinuousScan(run_id=worker.latest["run_id"], episode_epoch=0, compact_arms=True))
        mapped = await worker.home_command(request("start_mapping"))
        assert mapped["coverage"]["scan_count"] == 1
        await worker.home_command(request("add_place", name="Home"))
        home_place = worker.home_mission.home.places[0]["place_id"]
        for _ in range(2):
            observation, _ = await worker.feedback()
            result = await worker.execute(Command(run_id=observation.run_id, episode_epoch=0,
                observation_seq=observation.seq, action_id=str(uuid4()), tool="drive_base",
                arguments={"linear_mps": .15, "angular_radps": 0., "duration_s": 1.25}))
            assert result.status == "ok", result
            await worker.call(lambda sim: worker.home_mission.sample(force=True))
        state = await worker.home_state()
        assert state["localization"]["status"] == "localized", state["error"]
        assert state["localization"]["pose_m_rad"][0] > .2
        await worker.home_command(request("add_place", name="Kitchen entrance", kind="doorway", connects=[home_place]))
        await worker.home_command(request("review"))
        saved = await worker.home_command(request("save_map", name="Physics home"))
        identity = saved["map_id"]
        geometry = worker.home_mission.home.evidence.copy()
        target = worker.home_mission.home.places[1]["place_id"]
    finally:
        await worker.close()
    worker = SimulationWorker(pace=False, challenge=get_challenge("park"), rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, store)
        await worker.scan_continuous(ContinuousScan(run_id=worker.latest["run_id"], episode_epoch=0, compact_arms=True))
        loaded = await worker.home_command(request("load_map", map_id=identity))
        assert loaded["localization"]["status"] == "unlocalized"
        with pytest.raises(ValueError, match="LOCALIZATION_REQUIRED"):
            await worker.home_command(request("navigate_to", place_id=target))
        localized = await worker.home_command(request("localize", place_id=home_place))
        assert localized["localization"]["status"] == "localized"
        with pytest.raises(ValueError, match="UNKNOWN_PLACE"):
            await worker.home_command(request("navigate_to", place_id="unknown"))
        await worker.home_command(request("navigate_to", place_id=target, time_budget=45.))
        async with asyncio.timeout(50):
            while worker.home_mission.active:
                await asyncio.sleep(.05)
        state = await worker.home_state()
        assert state["task"]["status"] == "completed", state["task"]
        assert state["task"]["completion_verified"]
        assert state["localization"]["pose_m_rad"][0] > .2
        np.testing.assert_array_equal(worker.home_mission.home.evidence, geometry)
        assert not worker.sim.proximity_sensors().collisions
        await worker.home_command(request("navigate_to", place_id=home_place, time_budget=45.))
        async with asyncio.timeout(15):
            while not (worker.continuous and worker.continuous.active and worker.navigation.buffer):
                assert worker.home_mission.active, worker.home_mission.task
                await asyncio.sleep(.02)
        worker.stop()
        await worker.hold_stopped()
        cancelled = await worker.home_state()
        assert cancelled["task"]["status"] == "cancelled"
        assert not worker.navigation.buffer
        ticks = worker.sim.ticks
        await asyncio.sleep(.15)
        assert worker.sim.ticks == ticks
        await worker.resume_manual()
        await asyncio.sleep(.15)
        assert not worker.home_mission.active and not worker.navigation.buffer and worker.sim.ticks == ticks
        with pytest.raises(Exception, match="invalidated"):
            await worker.home_command(request("navigate_to", place_id=home_place), 0, 0)
    finally:
        await worker.close()


def test_home_api_reuses_one_saved_map_across_shared_scenarios(tmp_path, monkeypatch):
    from importlib import import_module
    from fastapi.testclient import TestClient
    from backend.app import app, lab
    from backend import home_mission
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(import_module("backend.app"), "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(home_mission, "DEFAULT_MAP_PATH", tmp_path / "api-home.sqlite3")
    with TestClient(app) as client:
        state = client.post("/api/challenges/load", json={"challenge_id": "apartment", "environment": "shared_apartment_v1"}).json()
        body = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"]}
        assert client.post("/api/home", json={**body, "action": "start_mapping"}).status_code == 409
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            mapped = client.post("/api/home", json={**body, "action": "start_mapping"})
            assert mapped.status_code == 200, mapped.text
            assert mapped.json()["coverage"]["scan_count"] > 0
            saved = client.post("/api/home", json={**body, "action": "save_map", "name": "One shared home"})
            assert saved.status_code == 200, saved.text
            identity = saved.json()["map_id"]
            original = lab.worker.home_mission.home.evidence.copy()
            assert client.post("/api/home", json={**body, "action": "start_mapping"}).status_code == 409
            second = client.post("/api/challenges/load", json={"challenge_id": "flat_kitchen", "environment": "shared_apartment_v1"}).json()
            loaded = client.get("/api/home")
            assert loaded.status_code == 200 and loaded.headers["cache-control"] == "no-store"
            assert loaded.json()["map_id"] == identity and loaded.json()["revision"] == 1
            assert loaded.json()["localization"]["status"] == "unlocalized"
            np.testing.assert_array_equal(lab.worker.home_mission.home.evidence, original)
            assert client.post("/api/home", json={**body, "action": "localize"}).status_code == 409
            body = {"run_id": second["run_id"], "episode_epoch": second["episode_epoch"]}
            assert client.post("/api/home", json={**body, "action": "navigate_to", "place_id": "invented"}).status_code == 409
            assert client.post("/api/home", json={**body, "action": "cancel_task"}).status_code == 200
            assert client.get("/api/state").json()["stopped"]
            assert client.post("/api/home", json={**body, "action": "localize", "pose_m_rad": [0., 0., 0.]}).status_code == 409
            assert client.post("/api/home", json={**body, "action": "save_map", "name": "bad"}, headers={"origin": "https://untrusted.invalid"}).status_code == 403


async def test_luna_home_dispatch_is_grounded_bounded_and_compact(tmp_path):
    import time
    from types import SimpleNamespace
    from backend.continuous_supervisor import GuideContinuous, execute_home_capability
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "luna-home.sqlite3"))
        settings = SimpleNamespace(run_id=worker.latest["run_id"], episode_epoch=0)
        controller = SimpleNamespace(_check_live=lambda *args: None)
        sensor = SimpleNamespace(captured_at=time.monotonic())
        state = await execute_home_capability(controller, worker, settings, GuideContinuous(action="spatial_state"), sensor, 0, 0)
        assert "map" not in state and "maps" not in state and "geometry" not in state
        with pytest.raises(ValueError, match="MAP_REQUIRED"):
            await execute_home_capability(controller, worker, settings, GuideContinuous(action="navigate_place", place_id="invented"), sensor, 0, 0)
        sensor.captured_at -= 16
        with pytest.raises(ValueError, match="STALE_PLAN"):
            await execute_home_capability(controller, worker, settings, GuideContinuous(action="navigate_place", place_id="invented"), sensor, 0, 0)
        with pytest.raises(ValueError):
            GuideContinuous(action="explore_map", time_budget=301)
        worker.stop()
        await worker.resume_manual()
        with pytest.raises(Exception, match="invalidated"):
            await worker.home_command(HomeRequest(run_id=settings.run_id, episode_epoch=0, action="start_mapping"), 0, 0)
    finally:
        await worker.close()


@pytest.mark.parametrize("fault", ["timeout", "sensor", "task_revision", "stop"])
def test_frontier_executive_is_bounded_and_fails_closed(tmp_path, fault):
    import threading
    import time
    from types import SimpleNamespace
    from backend.home_mission import HomeMission, HomeRequest
    holds, paths = [], []
    sim = SimpleNamespace(run_id="run", epoch=0, odometry=np.zeros(3), cancel=threading.Event(), held={},
        robot_footprint=lambda: {"radius_m": .3}, hold_current=lambda: holds.append(True),
        observe=lambda **kwargs: SimpleNamespace(grippers={}))
    worker = SimpleNamespace(sim=sim, challenge=None, stop_revision=0, task_revision=0, latest={},
        continuous=None, ros_navigation=None, renderer=None, skill=None, spatial_enabled=True,
        spatial_map=SimpleNamespace(captured_at=time.monotonic(), sequence=1), spatial_error=None,
        _sample_spatial=lambda **kwargs: None, _start_navigation_path=lambda path, **kwargs: paths.append(path))
    mission = HomeMission(worker, MapStore(tmp_path / "frontier.sqlite3"))
    mission.home = HomeMap("standalone:bench")
    mission.home.evidence[170:230, 170:230] = -2
    mission.home.scan_count, mission.home.revision, mission.home.saved = 3, 1, True
    mission.pose, mission.transform = [0., 0., 0.], [0., 0., 0.]
    mission.sampled_at = mission.validated_at = time.monotonic()
    mission.localization["status"] = "localized"
    mission.sample = lambda **kwargs: None
    mission.command(HomeRequest(run_id="run", episode_epoch=0, action="explore", time_budget=10), 0, 0)
    mission.tick()
    assert len(paths) == 1 and mission.task["segments"] == 1
    assert np.linalg.norm(paths[0][-1] - paths[0][0]) <= 1.01
    if fault == "timeout":
        mission.task["deadline"] = 0.
    elif fault == "sensor":
        mission.sampled_at -= 2.
    elif fault == "task_revision":
        worker.task_revision += 1
    else:
        sim.cancel.set()
    mission.tick()
    assert not mission.active and holds
    assert mission.task["status"] == ("limited" if fault == "timeout" else "failed" if fault == "sensor" else "cancelled")
    mission.tick()
    assert len(paths) == 1


async def test_object_observation_uses_paired_depth_and_becomes_last_seen(tmp_path):
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "observations.sqlite3"))
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": 0}
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Object evidence map"))
        def remember(sim):
            worker._sample_spatial(force=True)
            paired = worker.spatial_frames[worker.spatial_map.sequence]
            request = HomeRequest(**identity, action="remember_object", object_label="Visible surface",
                object_bounds=[0., 0., 1., 1.], confidence=.4, spatial_sequence=paired[0].sequence)
            state = worker.home_mission.command(request, worker.stop_revision, worker.task_revision, paired[:2])
            record = state["objects"][0]
            assert record["currently_observed"] and not record["identity_verified"]
            assert record["head_rad"] == paired[0].head_rad
            assert worker.home_mission.store.object_image(record["map_id"], record["observation_id"]) == paired[1]
            worker._sample_spatial(force=True)
            assert not worker.home_mission.state()["objects"][0]["currently_observed"]
            return record
        record = await worker.call(remember)
        restored = MapStore(tmp_path / "observations.sqlite3").object_observations(record["map_id"])
        assert restored[0]["position_m"] == record["position_m"] and restored[0]["confidence"] == .4
    finally:
        await worker.close()


async def test_mapping_sensor_exception_stops_without_killing_worker(tmp_path, monkeypatch):
    from backend.home_mission import HomeMission, HomeRequest
    from backend import ros_navigation
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "sensor-fault.sqlite3"))
        await worker.home_command(HomeRequest(run_id=worker.latest["run_id"], episode_epoch=0, action="start_mapping"))
        def broken_sensor(sim):
            raise RuntimeError("Injected sensor failure")
        monkeypatch.setattr(ros_navigation, "capture_laser", broken_sensor)
        await worker.call(lambda sim: worker.home_mission.sample(force=True))
        state = await worker.home_state()
        assert "Injected sensor failure" in state["error"]
        assert state["localization"]["status"] == "unlocalized"
        assert worker.sim.cancel.is_set() and worker.thread.is_alive()
        assert await worker.call(lambda sim: sim.run_id) == worker.latest["run_id"]
    finally:
        await worker.close()


async def test_shared_home_frontier_expansion_real_sensors_is_bounded(tmp_path, record_property):
    from backend.challenges import shared_apartment
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    store = MapStore(tmp_path / "shared-expansion.sqlite3")
    worker = SimulationWorker(pace=True, challenge=shared_apartment("flat_kitchen"), rendering="enhanced")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, store)
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": 0}
        await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        saved = await worker.home_command(HomeRequest(**identity, action="save_map", name="Shared sensor map"))
        original = store.load(saved["map_id"], "shared_apartment_v1").evidence.copy()
        await worker.home_command(HomeRequest(**identity, action="explore", time_budget=12.))
        async with asyncio.timeout(20):
            while worker.home_mission.active:
                await asyncio.sleep(.1)
        state = await worker.home_state()
        record_property("evidence", "scripted_executive_real_physics_enhanced")
        record_property("task_status", state["task"]["status"])
        record_property("route_segments", state["task"]["segments"])
        record_property("retries", state["task"]["retries"])
        record_property("net_displacement_m", math.hypot(*worker.sim.odometry[:2]))
        record_property("additional_known_cells", state["coverage"]["known_cells"] - int(np.count_nonzero(original)))
        assert state["task"]["segments"] > 0, state["task"]
        assert state["task"]["status"] in {"limited", "completed"}, {"task": state["task"], "localization": state["localization"], "error": state["error"]}
        assert math.hypot(*worker.sim.odometry[:2]) > .15, state["task"]
        assert not worker.sim.proximity_sensors().collisions
        assert not worker.navigation.buffer
        np.testing.assert_array_equal(store.load(saved["map_id"], "shared_apartment_v1").evidence, original)
    finally:
        await worker.close()