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


def test_flat_laser_cell_dedup_preserves_exact_ray_evidence():
    from backend.home_mapping import laser_points
    home = HomeMap("laser-equivalence")
    home.evidence[195:205, 195:205] = -20
    home.evidence[190, 207] = 20
    generator = np.random.default_rng(714)
    laser = {"origin_m": [.15, 0., .3], "angle_min": -math.pi, "angle_increment": 2 * math.pi / 720,
        "range_min": .03, "range_max": 8., "ranges_m": generator.uniform(.2, 9., 720).tolist()}
    laser["ranges_m"][7] = None
    pose = [18., -.1, .2]
    expected = home.evidence.copy()
    origin, endpoints, hits = laser_points(laser, pose)
    samples = [home.indices(np.linspace(origin, endpoint, max(2, math.ceil(np.linalg.norm(endpoint - origin) / .04) + 1))[:-1])
        for endpoint in endpoints]
    free = np.unique(np.concatenate(samples), axis=0)
    free = free[home.inside(free)]
    expected[free[:, 1], free[:, 0]] = np.maximum(-20, expected[free[:, 1], free[:, 0]] - 1)
    occupied = np.unique(home.indices(endpoints[hits]), axis=0)
    occupied = occupied[home.inside(occupied)]
    expected[occupied[:, 1], occupied[:, 0]] = np.minimum(20, expected[occupied[:, 1], occupied[:, 0]] + 4)
    home.observe(laser, pose, 100.)
    np.testing.assert_array_equal(home.evidence, expected)


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


def test_room_v2_preserves_source_database_and_keeps_semantics_separate(tmp_path):
    from scripts.benchmark_rooms import copy_store, definition, planned_cases, score_room
    store = MapStore(tmp_path / "original.sqlite3")
    home = HomeMap("shared_apartment_v1")
    home.scan_count = 1
    store.save(home, "Original")
    original = store.path.read_bytes()
    copy_store(store.path, tmp_path / "working.sqlite3")
    assert store.path.read_bytes() == original
    working = MapStore(tmp_path / "working.sqlite3")
    restored = working.load(home.identity, home.environment_id)
    working.save(restored, "Working")
    assert store.path.read_bytes() == original
    with pytest.raises(ValueError, match="new separate"):
        copy_store(store.path, working.path)
    assert definition()["suite_id"] == "household-room-v2"
    assert [case["task_id"] for case in planned_cases(definition()["tasks"])] == ["room_return", "blocked_doorway", "unknown_room"]
    metrics = {"recording_complete": True, "source_unchanged": True, "original_map_unchanged": True,
        "contacts": 0, "manual_placements": 0, "buffers_empty": True, "elapsed_s": 20., "budget_s": 180.,
        "arrivals": [{"status": "completed", "error_m": .03, "dwell_s": .5}] * 2}
    assert not score_room("room_return", metrics)["passed"]
    assert score_room("room_return", {**metrics, "post_arrival_report": True})["passed"]
    assert not score_room("room_return", {**metrics, "post_arrival_report": True, "contacts": 1})["passed"]
    assert score_room("room_return", {"blocked_prerequisite": "No camera-backed Kitchen"})["status"] == "blocked"


def test_room_graph_reports_current_connection_clearance_without_identity_claims():
    home = HomeMap("room-graph")
    home.evidence[170:230, 170:230] = -2
    living = home.add_place("Living room", "room", [0., 0., 0.], .3)
    kitchen = home.add_place("Kitchen", "room", [1., 0., 0.], .3, [living["place_id"]])
    before = home.document()
    graph = home.graph_summary(home.allowed(.3))
    assert graph["connections"][0]["from"] == kitchen["place_id"]
    assert graph["connections"][0]["status"] == "reachable"
    obstacles = [[.5, lateral] for lateral in np.arange(-.5, .6, .1)]
    assert home.graph_summary(home.allowed(.3, obstacles))["connections"][0]["status"] == "blocked"
    assert home.graph_summary(None)["connections"][0]["status"] == "unknown"
    assert "path_m" not in graph["connections"][0]
    assert home.document() == before


async def test_room_observations_require_post_arrival_camera_and_operator_review(tmp_path):
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "rooms.sqlite3"))
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": 0}
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Room test"))
        def current(sim):
            worker._sample_spatial(force=True)
            return worker.spatial_frames[worker.spatial_map.sequence][:2]
        paired = await worker.call(current)
        state = await worker.home_command(HomeRequest(**identity, action="observe_room", name="Test room",
            evidence_text="Visible walls and floor in the current camera", confidence=.6, spatial_sequence=paired[0].sequence), selected_evidence=paired)
        place = state["places"][0]
        record = state["room_observations"][0]
        assert place["identity_status"] == "tentative" and not record["identity_verified"]
        review = HomeRequest(**identity, action="review_room", evidence_id=record["observation_id"])
        with pytest.raises(ValueError, match="OPERATOR_REVIEW_REQUIRED"):
            await worker.home_command(review)
        state = await worker.home_command(review, operator_review=True)
        assert state["places"][0]["identity_status"] == "operator_confirmed"
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Room test"))
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=place["place_id"], time_budget=10))
        async with asyncio.timeout(15):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert worker.home_mission.task["status"] == "completed"
        with pytest.raises(ValueError, match="ARRIVAL_REQUIRED"):
            await worker.home_command(HomeRequest(**identity, action="observe_room", place_id=place["place_id"], room_matches=True,
                evidence_text="Old image cannot verify arrival", spatial_sequence=paired[0].sequence), selected_evidence=paired)
        paired = await worker.call(current)
        state = await worker.home_command(HomeRequest(**identity, action="observe_room", place_id=place["place_id"], room_matches=False,
            evidence_text="The fresh image does not support the requested room", spatial_sequence=paired[0].sequence), selected_evidence=paired)
        assert state["room_verification"]["status"] == "visual_mismatch_reported"
        assert not state["room_verification"]["identity_verified"]
        assert worker.home_mission.store.room_image(state["map_id"], state["room_observations"][0]["observation_id"]) == paired[1]
        compact = await worker.home_state(compact=True)
        assert "room_graph" in compact and "frontiers" in compact and "map" not in compact
        def changed_view(sim):
            import pybullet as bullet
            bullet.resetJointState(sim.robot, sim.joints["head_yaw"], .3, physicsClientId=sim.client)
            state = worker.home_mission.state(compact=True)
            assert state["room_verification"]["status"] == "not_currently_verified"
            bullet.resetJointState(sim.robot, sim.joints["head_yaw"], paired[0].head_rad[0], physicsClientId=sim.client)
            assert worker.home_mission.state(compact=True)["room_verification"]["status"] == "not_currently_verified"
        await worker.call(changed_view)
        restored = worker.home_mission.store.load(state["map_id"], worker.home_mission.environment_id)
        assert restored.places[0]["identity_status"] == "operator_confirmed"
        await worker.home_command(HomeRequest(**identity, action="localize", place_id=place["place_id"]))
        with pytest.raises(ValueError, match="STALE_OBSERVATION"):
            await worker.home_command(HomeRequest(**identity, action="observe_room", place_id=place["place_id"],
                evidence_text="Image predates relocalization", spatial_sequence=paired[0].sequence), selected_evidence=paired)
        worker.stop()
        await worker.hold_stopped()
        with pytest.raises(Exception, match="invalidated"):
            await worker.home_command(HomeRequest(**identity, action="observe_room", place_id=place["place_id"], evidence_text="late"), 0, 0)
    finally:
        await worker.close()


@pytest.mark.parametrize("report", [True, False, "timeout", "stop"])
async def test_room_round_trip_requires_visual_report_and_preserves_deadline(tmp_path, report):
    from types import SimpleNamespace
    from backend.continuous_supervisor import GuideContinuous, execute_home_capability
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "roundtrip.sqlite3"))
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": 0}
        async def command(action, **values):
            return await worker.home_command(HomeRequest(**identity, action=action, **values))
        await command("start_mapping")
        await command("add_place", name="Home")
        state = await command("add_place", name="Scripted room", kind="room")
        home, room = state["places"]
        await command("save_map")
        import time
        await execute_home_capability(SimpleNamespace(_check_live=lambda *args: None), worker, SimpleNamespace(**identity),
            GuideContinuous(action="navigate_place", place_id=room["place_id"], return_place_id=home["place_id"], time_budget=15.),
            SimpleNamespace(captured_at=time.monotonic()), worker.stop_revision, worker.task_revision)
        async with asyncio.timeout(20):
            while worker.home_mission.active:
                await asyncio.sleep(.02)
        assert (await worker.home_state(compact=True))["room_workflow"]["status"] == "awaiting_room_report"
        deadline = worker.home_mission.workflow["deadline"]
        if report == "timeout":
            await worker.call(lambda sim: worker.home_mission.workflow.update(deadline=0.))
            await worker.call(lambda sim: worker.home_mission.tick())
        elif report == "stop":
            worker.stop()
            await worker.hold_stopped()
            await worker.call(lambda sim: worker.home_mission.tick())
        else:
            def observe(sim):
                worker._sample_spatial(force=True)
                sensor, image = worker.spatial_frames[worker.spatial_map.sequence][:2]
                return worker.home_mission.command(HomeRequest(**identity, action="observe_room", place_id=room["place_id"],
                    room_matches=report, spatial_sequence=sensor.sequence, evidence_text="Scripted current-view contract fixture"),
                    worker.stop_revision, worker.task_revision, (sensor, image))
            await worker.call(observe)
            async with asyncio.timeout(20):
                while worker.home_mission.active:
                    await asyncio.sleep(.02)
            assert worker.home_mission.workflow["deadline"] == deadline
        state = await worker.home_state(compact=True)
        assert state["room_workflow"]["status"] == ("completed" if report is True else "failed" if report is False else "limited" if report == "timeout" else "cancelled")
        assert state["room_workflow"]["identity_verified"] is False
        if report is True:
            assert state["task"]["place_id"] == home["place_id"]
            assert state["room_workflow"]["room_evidence_id"]
        assert worker.navigation is None or not worker.navigation.buffer
    finally:
        await worker.close()


def test_grid_boundary_rounding_preserves_real_left_side_and_unknown_space():
    home = HomeMap("boundary-test")
    np.testing.assert_array_equal(home.indices([[-.1, 0.], [-.1 - 1e-10, 0.], [-.1 + 1e-10, 0.]]),
        [[199, 200], [198, 200], [199, 200]])
    home.evidence[180:220, 195:230] = -2
    before = home.evidence.copy()
    assert home.route([-.1, 0.], [.7, 0.], .397)[0] == [-.1, 0.]
    with pytest.raises(ValueError, match="UNREACHABLE"):
        home.route([-.1 - 1e-10, 0.], [.7, 0.], .397)
    np.testing.assert_array_equal(home.evidence, before)


@pytest.mark.parametrize("mapping", [True, False])
def test_batched_home_depth_preserves_obstacles_and_grid_evidence(tmp_path, monkeypatch, mapping):
    import time
    from types import SimpleNamespace
    from backend.home_mission import HomeMission
    from backend import spatial
    worker = SimpleNamespace(challenge=None, stop_revision=0, task_revision=0)
    mission = HomeMission(worker, MapStore(tmp_path / "batch.sqlite3"))
    mission.home = HomeMap("test")
    mission.home.evidence[190:220, 190:220] = -5
    mission.home.evidence[201, 202] = 17
    mission.transform = [.7, -.2, .45]
    mission.stage = "mapping" if mapping else "loaded"
    generator = np.random.default_rng(713)
    points = generator.uniform([-25, -25, -.1], [25, 25, 1.5], (4000, 3))
    points = np.concatenate((points, points[:1000], [[.1, .2, .04], [.1, .2, 1.3]]))
    monkeypatch.setattr(spatial, "point_cloud", lambda sensor, stride: (None, points))
    expected = mission.home.evidence.copy()
    keys = set()
    for point in points:
        if not .04 < point[2] < 1.3:
            continue
        column, row = mission.home.indices(transform_pose([*point[:2], 0.], mission.transform)[:2])
        keys.add((int(column), int(row)))
        if mapping and mission.home.inside([column, row]):
            expected[row, column] = max(4, expected[row, column])
    mission.observe_depth(SimpleNamespace(sequence=1, captured_at=time.monotonic()))
    assert set(mission.live) == keys
    np.testing.assert_array_equal(mission.home.evidence, expected)
    assert mission.last_depth_sequence == 1


async def test_mapped_rgbd_capture_does_not_block_worker_and_stop_discards_frame(tmp_path):
    from io import BytesIO
    import threading
    import time
    from types import SimpleNamespace
    from PIL import Image
    from backend.contracts import SpatialSettings
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    release, entered = threading.Event(), threading.Event()
    finished = None
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.latest["run_id"], episode_epoch=0, enabled=True))
        image = BytesIO()
        Image.new("RGB", (160, 120)).save(image, format="PNG")
        def capture(packet, depth):
            entered.set()
            assert release.wait(5.)
            return image.getvalue(), np.ones((120, 160))
        def install(sim):
            worker.home_mission = HomeMission(worker, MapStore(tmp_path / "async.sqlite3"))
            worker.home_mission.home = HomeMap(worker.home_mission.environment_id)
            sim.rendering = "enhanced"
            sim.camera_renderer = SimpleNamespace(capture=capture)
            sim.owns_camera_renderer = False
            worker.spatial_sampled_at = 0.
            before = worker.spatial_map.sequence
            began = time.monotonic()
            worker._sample_spatial()
            return before, time.monotonic() - began, worker.spatial_capture_pending
        before, duration, finished = await worker.call(install)
        assert duration < .3
        assert await asyncio.to_thread(entered.wait, 2.)
        assert await asyncio.wait_for(worker.call(lambda sim: sim.run_id), .5) == worker.latest["run_id"]
        worker.stop()
        release.set()
        await asyncio.wrap_future(finished)
        await worker.call(lambda sim: worker._receive_mapped_capture())
        assert worker.spatial_map.sequence == before
        assert worker.mapped_depth_sensor is None
        assert worker.sim.cancel.is_set()
    finally:
        release.set()
        await worker.call(lambda sim: (setattr(sim, "rendering", "tiny"), setattr(sim, "camera_renderer", None)))
        await worker.close()


@pytest.mark.parametrize("fault", [None, "stale", "future", "episode", "generation", "stop"])
async def test_mapped_capture_validation_is_independent_of_pending_map_processing(monkeypatch, fault):
    from concurrent.futures import Future
    import time
    from backend.contracts import SpatialSettings
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    pending_map = Future()
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.latest["run_id"], episode_epoch=0, enabled=True))
        def check(sim):
            observation, image = sim.capture_spatial(worker.spatial_sequence + 1)
            if fault == "stale":
                observation = observation.model_copy(update={"captured_at": time.monotonic() - 2.})
            elif fault == "future":
                observation = observation.model_copy(update={"captured_at": time.monotonic() + 2.})
            elif fault == "episode":
                observation = observation.model_copy(update={"episode_epoch": 99})
            worker.spatial_capture_pending = Future()
            worker.spatial_capture_pending.set_result((observation, image))
            worker.spatial_capture_stop_revision = worker.stop_revision - (fault == "stop")
            worker.spatial_capture_generation = worker.spatial_generation - (fault == "generation")
            previous = worker.spatial_map
            monkeypatch.setattr(worker.spatial_processor, "submit", lambda *args: pending_map)
            worker._receive_mapped_capture()
            assert worker.spatial_map is previous
            if fault is None:
                assert worker.mapped_depth_sensor is observation
                assert worker.spatial_pending is pending_map and not pending_map.done()
            else:
                assert worker.mapped_depth_sensor is None
            worker.spatial_enabled = False
        await worker.call(check)
    finally:
        pending_map.cancel()
        await worker.close()


@pytest.mark.parametrize("fault", ["unknown", "stale", "shield"])
async def test_guided_mapping_drive_checks_observed_path_and_whole_robot_before_motion(tmp_path, monkeypatch, fault):
    from backend.contracts import Command
    from backend.home_mission import HomeMission, HomeRequest
    from backend.navigation import NavigationRuntime
    from backend.simulation import MotionError
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "guided.sqlite3"))
        identity = {"run_id": worker.latest["run_id"], "episode_epoch": 0}
        await worker.home_command(HomeRequest(**identity, action="start_mapping"))
        def inject(sim):
            worker.home_mission.home.evidence[:] = -2
            worker.home_mission.allowed_cache = None
            if fault == "unknown":
                worker.home_mission.home.evidence[195:205, 201:203] = 0
            elif fault == "stale":
                worker.spatial_map.captured_at -= 5.
            return sim.ticks, sim.odometry.copy(), sim.seq
        if fault == "shield":
            def blocked(*args):
                raise MotionError("CLEARANCE_STOP", "Injected whole-robot obstruction")
            monkeypatch.setattr(NavigationRuntime, "check_clearance", blocked)
        if fault == "stale":
            monkeypatch.setattr(worker, "_sample_spatial", lambda **kwargs: None)
        if fault == "unknown":
            monkeypatch.setattr(worker.home_mission, "sample", lambda **kwargs: None)
        ticks, before, sequence = await worker.call(inject)
        result = await worker.execute(Command(**identity, observation_seq=sequence, action_id=str(uuid4()), tool="drive_base",
            arguments={"linear_mps": .15, "angular_radps": 0., "duration_s": 1.}))
        assert result.status == "error" and result.error in {"GUIDED_MAP_BLOCKED", "CLEARANCE_STOP"}
        assert worker.sim.ticks == ticks
        np.testing.assert_array_equal(worker.sim.odometry, before)
    finally:
        await worker.close()


async def test_mapped_capture_continues_while_map_job_is_delayed(tmp_path):
    from concurrent.futures import Future
    import time
    from types import SimpleNamespace
    from backend.contracts import SpatialSettings
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    pending_map, capture = Future(), Future()
    submissions = []
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.latest["run_id"], episode_epoch=0, enabled=True))
        def check(sim):
            sensor, image = sim.capture_spatial(worker.spatial_sequence + 1)
            worker.home_mission = HomeMission(worker, MapStore(tmp_path / "delayed-map.sqlite3"))
            worker.home_mission.home = HomeMap(worker.home_mission.environment_id)
            worker.spatial_pending = pending_map
            worker.spatial_pending_map = worker.spatial_map
            worker.spatial_capture_executor = SimpleNamespace(submit=lambda *args: submissions.append(args) or capture,
                shutdown=lambda **kwargs: None)
            worker.inference_owner = "delayed-capture-test"
            sim.rendering = "enhanced"
            worker.spatial_sampled_at = time.monotonic() - 1.
            worker._sample_spatial()
            assert len(submissions) == 1 and worker.spatial_capture_pending is capture
            worker._sample_spatial()
            assert len(submissions) == 1
            capture.set_result((sensor, image))
            worker._receive_mapped_capture()
            assert worker.mapped_depth_sensor is sensor
            assert worker.spatial_pending is pending_map and not pending_map.done()
            assert worker.spatial_capture_pending is None
            worker.stop()
            revoked = Future()
            revoked.set_result((sensor.model_copy(update={"sequence": sensor.sequence + 1}), image))
            worker.spatial_capture_pending = revoked
            worker._receive_mapped_capture()
            assert worker.mapped_depth_sensor is sensor
            worker.spatial_enabled = False
            sim.rendering = "tiny"
            worker.spatial_pending = None
        await worker.call(check)
    finally:
        pending_map.cancel()
        await worker.call(lambda sim: (setattr(sim, "rendering", "tiny"), setattr(worker, "spatial_pending", None)))
        await worker.close()


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
    remaining = home.frontiers([0., 0., 0.], .3)
    assert remaining and all(item["frontier_id"] != frontier["frontier_id"] for item in remaining)
    store = MapStore(tmp_path / "map.sqlite3")
    store.save(home, "Home")
    restored = store.load(home.identity, "home")
    assert restored.frontier_attempts == home.frontier_attempts
    assert restored.edges == home.edges
    assert restored.frontiers([0., 0., 0.], .3) == remaining


def test_frontier_selection_requires_new_reachable_motion_and_preserves_unknown():
    home = HomeMap("home")
    home.evidence[190:211, 190:211] = -2
    before = home.cells.copy()
    pose = [0., 0., 0.]
    frontier = home.frontiers(pose, .3)[0]
    assert frontier["distance_m"] > .15 + home.resolution_m
    assert len(home.route(pose[:2], frontier["position_m"], .3)) >= 2
    home.mark_frontier(frontier["frontier_id"])
    home.mark_frontier(frontier["frontier_id"])
    replacement = home.frontiers(pose, .3)[0]
    assert replacement["frontier_id"] != frontier["frontier_id"]
    assert len(home.route(pose[:2], replacement["position_m"], .3)) >= 2
    assert home.frontiers(pose, .3, obstacles=[[0., 0.]]) == []
    np.testing.assert_array_equal(home.cells, before)


def test_frontier_ranking_prefers_useful_forward_progress_without_crossing_unknown():
    home = HomeMap("home")
    home.evidence[182:219, 182:219] = -2
    pose = [0., 0., 0.]
    frontier = home.frontiers(pose, .3)[0]
    assert .7 <= frontier["distance_m"] <= 1.4
    assert frontier["position_m"][0] > .7 and abs(frontier["position_m"][1]) < .3
    path = home.route(pose[:2], frontier["position_m"], .3)
    allowed = home.allowed(.3)
    indices = home.indices(path)
    assert allowed[indices[:, 1], indices[:, 0]].all()
    alternatives = home.frontiers(pose, .3, excluded=[frontier["frontier_id"]])
    assert alternatives and all(candidate["frontier_id"] != frontier["frontier_id"] for candidate in alternatives)
    assert all(len(home.route(pose[:2], candidate["position_m"], .3)) >= 2 for candidate in alternatives)


async def test_named_arrival_holds_measured_velocity_for_full_dwell(tmp_path):
    from types import SimpleNamespace
    import pybullet as bullet
    import time
    from backend.challenges import get_challenge
    from backend.home_mission import HomeMission
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    try:
        worker = SimpleNamespace(sim=sim, challenge=get_challenge("park"), stop_revision=0, task_revision=0,
            continuous=None, latest={})
        home = HomeMission(worker, MapStore(tmp_path / "dwell.sqlite3"))
        home.pose = sim.odometry.tolist()
        home.sample = lambda **kwargs: None
        home.require_localized = lambda: None
        home.task = {"status": "running", "kind": "navigate", "target_m": home.pose[:2],
            "deadline": time.monotonic() + 10., "guided_mapping": False}
        original_ticks = sim._ticks
        sampled = []

        def ticks(count):
            original_ticks(count)
            if not sampled:
                bullet.resetBaseVelocity(sim.robot, [0.03, 0., 0.], [0., 0., 0.], physicsClientId=sim.client)
            velocity, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
            sampled.append((sim.ticks, math.hypot(*velocity[:2]), abs(angular[2])))

        sim._ticks = ticks
        home.tick()
        assert home.task["status"] == "completed", home.task
        last_unstable = max(tick for tick, speed, angular in sampled if speed > .025 or angular > .15)
        assert sampled[-1][0] - last_unstable >= 132
    finally:
        sim.close()


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
        with pytest.raises(ValueError, match="UNREACHABLE"):
            await worker.home_command(request("guided_to", pose_m_rad=[19., 19., 0.]))
        await worker.home_command(request("guided_to", pose_m_rad=[.35, 0., 0.], time_budget=45.))
        async with asyncio.timeout(50):
            while worker.home_mission.active:
                await asyncio.sleep(.05)
        state = await worker.home_state()
        assert state["task"]["status"] == "completed", state["task"]
        assert state["task"]["guided_mapping"] and state["stage"] == "mapping"
        assert state["design"] == "guided-waypoints-v1" and state["version"] == "0.4.0"
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
        destination = worker.home_mission.place(target)["pose_m_rad"]
        assert math.dist(state["localization"]["pose_m_rad"][:2], destination[:2]) <= .15
        assert math.hypot(*state["localization"]["pose_m_rad"][:2]) > .15
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
    from backend import session_recording
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
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
            skipped = client.post("/api/challenges/load", json={"challenge_id": "flat_kitchen", "environment": "shared_apartment_v1", "reuse_saved_map": False})
            assert skipped.status_code == 200
            assert skipped.json()["map_setup"]["reuse_saved_map"] is False
            assert skipped.json()["map_setup"]["map_id"] is None
            assert client.get("/api/home").json()["maps"][0]["map_id"] == identity
            reset = client.post("/api/reset").json()
            assert reset["map_setup"]["map_id"] is None and reset["map_setup"]["reuse_saved_map"] is False
            second = client.post("/api/challenges/load", json={"challenge_id": "flat_kitchen", "environment": "shared_apartment_v1", "reuse_saved_map": True}).json()
            assert second["map_setup"]["map_id"] == identity and second["map_setup"]["revision"] == 1
            assert second["map_setup"]["localization"] == "unlocalized"
            np.testing.assert_array_equal(lab.worker.home_mission.home.evidence, original)
            assert client.post("/api/home", json={**body, "action": "localize"}).status_code == 409
            body = {"run_id": second["run_id"], "episode_epoch": second["episode_epoch"]}
            assert client.post("/api/home", json={**body, "action": "navigate_to", "place_id": "invented"}).status_code == 409
            assert client.post("/api/home", json={**body, "action": "cancel_task"}).status_code == 200
            assert client.get("/api/state").json()["stopped"]
            assert client.post("/api/home", json={**body, "action": "localize", "pose_m_rad": [0., 0., 0.]}).status_code == 409
            assert client.post("/api/home", json={**body, "action": "save_map", "name": "bad"}, headers={"origin": "https://untrusted.invalid"}).status_code == 403


@pytest.mark.parametrize("blocked", [None, "stop", "stale", "unknown"])
def test_local_exploration_continuation_preserves_velocity_deadline_and_safety(tmp_path, blocked):
    import threading
    import time
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation
    from backend.home_mission import HomeMission
    now = time.monotonic()
    sim = SimpleNamespace(odometry=np.array([.2, 0., 0.]), held={}, cancel=threading.Event(), robot_footprint=lambda: {"radius_m": .3})
    control = ContinuousNavigation([[0., 0.], [.6, 0.]])
    control.home_owned, control.radius = True, .3
    runtime = SimpleNamespace(status="running", buffer=[True], expires_at=now + 10., skill_deadline=now + 20., travel=.2, velocity=np.array([.2,0.]))
    worker = SimpleNamespace(sim=sim, challenge=None, continuous=control, navigation=runtime, stop_revision=0, task_revision=0,
        spatial_map=SimpleNamespace(captured_at=now))
    mission = HomeMission(worker, MapStore(tmp_path / "continue.sqlite3"))
    mission.home = HomeMap("test")
    mission.home.evidence[170:230, 170:230] = -2
    mission.pose, mission.transform = [.2,0.,0.], [0.,0.,0.]
    mission.localization["status"] = "localized"
    mission.sampled_at = mission.validated_at = now
    mission.route = [[0.,0.],[.6,0.],[1.,0.],[1.2,0.]]
    mission.task = {"status":"running", "local_exploration":True, "route_index":1, "deadline":now + 30.}
    if blocked == "stop": sim.cancel.set()
    if blocked == "stale": worker.spatial_map.captured_at = now - 2.
    if blocked == "unknown": mission.home.evidence[:,207:210] = 0
    mission.continue_local_exploration()
    assert control.handoffs == (0 if blocked else 1)
    assert runtime.expires_at == now + 10. and runtime.skill_deadline == now + 20.
    np.testing.assert_array_equal(runtime.velocity, [.2,0.])
    if not blocked:
        assert mission.task["continuations"] == 1
        assert np.linalg.norm(control.path[-1] - control.path[0]) <= 1.


async def test_explicit_local_exploration_builds_draft_without_model_or_saved_map_mutation(tmp_path):
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    store = MapStore(tmp_path / "local-explore.sqlite3")
    original = HomeMap("standalone:bench")
    original.scan_count = 1
    store.save(original, "Preserved map")
    before = original.document()
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, store)
        request = HomeRequest(run_id=worker.latest["run_id"], episode_epoch=0, action="start_exploration", time_budget=1.)
        state = await worker.home_command(request)
        assert state["task"]["local_exploration"] and state["task"]["model_calls"] == 0
        assert state["map_id"] != original.identity and state["revision"] == 0
        worker.stop()
        await worker.hold_stopped()
        state = await worker.home_state()
        if state["task"]["status"] == "completed":
            assert state["task"]["reason"] == "No untried reachable frontiers; closed and unknown areas remain unexplored"
            assert state["task"]["segments"] == 0
        else:
            assert state["task"]["status"] == "cancelled"
        assert worker.latest["stopped"] and not worker.home_mission.active
        assert store.load(original.identity, original.environment_id).document() == before
        await worker.resume_manual()
        assert not worker.home_mission.active
        worker.home_mission.allow_expansion = False
        with pytest.raises(ValueError, match="MAP_READ_ONLY"):
            await worker.home_command(request)
    finally:
        await worker.close()


def test_recorded_starting_map_distinguishes_none_draft_and_saved(tmp_path):
    from types import SimpleNamespace
    from backend.session_recording import map_provenance
    from scripts.benchmark_household import digest
    worker = SimpleNamespace(home_mission=None)
    assert map_provenance(worker)["mode"] == "none"
    home = HomeMap("shared_apartment_v1")
    home.scan_count = 1
    worker.home_mission = SimpleNamespace(home=home, localization={"status":"unlocalized"})
    assert map_provenance(worker)["mode"] == "draft"
    MapStore(tmp_path / "provenance.sqlite3").save(home, "Reused home")
    result = map_provenance(worker)
    assert result["mode"] == "saved" and result["revision"] == 1 and result["sha256"] == digest(home.document())
    assert result["localization"] == "unlocalized"


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


@pytest.mark.parametrize("fault", ["timeout", "sensor", "task_revision", "stop", "selected_blocked", "selected_arrived"])
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
    if fault.startswith("selected"):
        with pytest.raises(ValueError, match="UNKNOWN_FRONTIER"):
            mission.command(HomeRequest(run_id="run", episode_epoch=0, action="explore_frontier", frontier_id="invented"), 0, 0)
        frontier = mission.state(compact=True)["frontiers"][0]
        mission.command(HomeRequest(run_id="run", episode_epoch=0, action="explore_frontier", frontier_id=frontier["frontier_id"], time_budget=10), 0, 0)
        assert mission.stage == "expansion" and mission.task["single_frontier"]
    else:
        mission.command(HomeRequest(run_id="run", episode_epoch=0, action="explore", time_budget=10), 0, 0)
    assert mission.state(compact=True)["frontiers"] == []
    assert mission.state(compact=True)["frontier_selection_available"] is False
    mission.tick()
    assert len(paths) == 1 and mission.task["segments"] == 1
    assert np.linalg.norm(paths[0][-1] - paths[0][0]) <= 1.01
    if fault == "timeout":
        mission.task["deadline"] = 0.
    elif fault == "sensor":
        mission.sampled_at -= 2.
    elif fault == "task_revision":
        worker.task_revision += 1
    elif fault == "selected_blocked":
        column, row = mission.home.indices(mission.task["target_m"])
        mission.home.evidence[row, column] = 4
        mission.recheck_at = 0.
    elif fault == "selected_arrived":
        mission.pose = [*mission.task["target_m"], 0.]
        mission.recheck_at = 0.
    else:
        sim.cancel.set()
    mission.tick()
    assert not mission.active and holds
    assert mission.task["status"] == ("limited" if fault == "timeout" else "failed" if fault in {"sensor", "selected_blocked"} else "completed" if fault == "selected_arrived" else "cancelled")
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


@pytest.mark.parametrize("position,linear,angular", [([-6.8, 1.8], .35, 0.), ([-6.8, 1.8], 0., .5),
    ([-.9, 2.], .15, 0.), ([-7.4, 1.8], -.15, 0.)])
def test_clearance_broadphase_matches_all_object_checks(monkeypatch, position, linear, angular):
    import pybullet as bullet
    from backend.challenges import shared_apartment
    from backend.navigation import NavigationRuntime
    from backend.simulation import BulletSimulation, MotionError
    challenge = shared_apartment("flat_kitchen")
    challenge.initial_xy = position
    sim = BulletSimulation(challenge=challenge, rendering="tiny", width=160, height=120)
    runtime = NavigationRuntime()
    original = bullet.getClosestPoints
    counts = [0]
    def count(*args, **kwargs):
        counts[0] += 1
        return original(*args, **kwargs)
    def outcome():
        try:
            runtime.check_clearance(sim, linear, angular)
            return "clear"
        except MotionError as error:
            return error.code
    try:
        monkeypatch.setattr(bullet, "getClosestPoints", count)
        filtered = outcome()
        filtered_count = counts[0]
        counts[0] = 0
        monkeypatch.setattr(runtime, "bounds_overlap", lambda *args: True)
        reference = outcome()
        assert filtered == reference
        if filtered == "clear":
            assert filtered_count < counts[0] / 2
    finally:
        sim.close()