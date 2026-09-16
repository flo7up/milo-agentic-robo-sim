import asyncio
from io import BytesIO
import threading
import time
from uuid import uuid4

import pytest
import pybullet as bullet
from PIL import Image

from backend.contracts import Command, ManualPlacement
from backend.simulation import MotionError
from backend.worker import CameraActivity, SimulationWorker


def test_worker_timing_separates_nested_wall_time_and_preserves_exception():
    from backend.worker_timing import WorkerTiming
    now = [0.]
    timing = WorkerTiming(clock=lambda: now[0])
    with timing.measure("control"):
        now[0] = .1
        with pytest.raises(ValueError, match="fixture"):
            with timing.measure("map"):
                now[0] = .4
                raise ValueError("fixture")
        now[0] = .5
    child, parent = timing.events
    assert child["phase"] == "map" and child["duration_s"] == pytest.approx(.3)
    assert child["error_type"] == "ValueError" and child["parent_id"] == parent["span_id"]
    assert parent["duration_s"] == .5 and parent["self_s"] == pytest.approx(.2)
    assert not timing.stack


def test_worker_timing_stop_snapshot_is_bounded_and_marks_unfinished_work():
    from backend.worker_timing import WorkerTiming
    now = [0.]
    timing = WorkerTiming(clock=lambda: now[0], capacity=3, incident_capacity=1)
    diagnostic = {"authorization_id": "buffer", "stop": {"reason": "BUFFER_EXPIRED"}}
    with timing.measure("queued_operation"):
        now[0] = .1
        with timing.measure("collision"):
            now[0] = .6
            for index in range(5):
                timing.event("renewal", revision=index)
            timing.stop(diagnostic, "run", 2, .05)
            diagnostic["stop"]["reason"] = "changed"
            snapshot = timing.incidents[0]
            assert snapshot["navigation"]["stop"]["reason"] == "BUFFER_EXPIRED"
            assert len(snapshot["events"]) == 3 and snapshot["events_evicted"] == 2
            assert snapshot["active_spans"][0]["self_so_far_s"] == pytest.approx(.1)
            assert snapshot["active_spans"][1]["duration_so_far_s"] == .5
    timing.stop(diagnostic, "run", 2, .05)
    assert timing.incidents_dropped == 1 and len(timing.incidents) == 1
    saved = list(timing.take_incidents())
    assert saved[0]["clock"] == "test" and saved[0]["incident_id"] == 2
    assert not list(timing.take_incidents())
    assert len(timing.events) == 3 and not timing.stack


@pytest.mark.parametrize("home_source", [False, True])
async def test_motion_zones_are_opt_in_observed_only_and_never_authorize_motion(home_source):
    from types import SimpleNamespace
    import numpy as np
    from backend.home_mapping import HomeMap
    from backend.home_mission import HomeMission
    from backend.motion_zones import observed_motion_zones
    from backend.spatial import ObservedMap
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        def check(sim):
            disabled = worker.spatial_state(include_motion_zones=True)
            assert disabled["footprint"] is None
            assert disabled["motion_zones"]["footprint"]["radius_m"] > .2
            assert disabled["motion_zones"]["sectors"] == []
            assert not disabled["motion_zones"]["motion_authorized"]
            worker.spatial_enabled = True
            worker.spatial_map = ObservedMap(sim.run_id, sim.epoch)
            worker.spatial_map.cells[:] = 0
            worker.spatial_map.captured_at = time.monotonic()
            worker.spatial_map.observation = SimpleNamespace(odometry_m_rad=sim.odometry.tolist())
            if home_source:
                worker.home_mission = HomeMission(worker)
                worker.home_mission.home = HomeMap("scripted-zone-fixture")
                worker.home_mission.home.evidence[:] = -1
                worker.home_mission.transform = [0., 0., 0.]
                worker.home_mission.pose = sim.odometry.tolist()
                worker.home_mission.sampled_at = time.monotonic()
                worker.home_mission.validated_at = time.monotonic()
                worker.home_mission.localization["status"] = "localized"
            before = sim.ticks, sim.odometry.copy(), worker.spatial_map.cells.copy()
            assert "motion_zones" not in worker.spatial_state()
            result = worker.spatial_state(include_motion_zones=True)["motion_zones"]
            assert result["source"] == ("home_map" if home_source else "rolling_depth_map")
            assert not result["motion_authorized"] and len(result["sectors"]) == 48
            base = next(pose for pose in sim.snapshot()["poses"] if pose["key"] == f"{sim.robot}:-1")
            assert result["display_pose"]["frame"] == "world"
            assert result["display_pose"]["position_m"] == base["position"][:2]
            assert result["display_pose"]["yaw_rad"] == pytest.approx(0., abs=.01)
            assert all(sector["status"] == "clear" for sector in result["sectors"])
            assert result["beam_stop_distance_m"] == .18
            assert result["speed_basis"] == "idle_reference"
            assert 0. < result["valid_for_s"] <= (.75 if home_source else 1.)
            expired = observed_motion_zones(worker, sim.robot_footprint(), worker.spatial_map.captured_at + 2.)
            assert expired["stale"] and all(sector["status"] == "unavailable" for sector in expired["sectors"])
            assert sim.ticks == before[0]
            np.testing.assert_array_equal(sim.odometry, before[1])
            np.testing.assert_array_equal(worker.spatial_map.cells, before[2])
            assert worker.navigation is None
            if home_source:
                worker.home_mission.sampled_at -= 1.
                unavailable = observed_motion_zones(worker, sim.robot_footprint())
                assert unavailable["reason"].startswith("LOCALIZATION_REQUIRED")
                assert unavailable["sectors"] == []
        await worker.call(check)
    finally:
        worker.stop()
        await worker.close()


async def test_worker_timing_expiry_is_saved_only_by_background_flush(tmp_path):
    import json
    from backend.challenges import get_challenge
    from backend.navigation import NavigationRuntime
    from backend.recording import RunRecorder
    from scripts.navigation_policy import apply
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=False)
    recorder = RunRecorder(tmp_path / "recording")
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.call(lambda sim: recorder.capture(worker))
        def expire(sim):
            worker.navigation = NavigationRuntime()
            apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Capture expiry timing"})
            apply(worker.navigation, sim, "replace_motion_buffer", {"segments": [
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0., "duration_s": 1.}]})
            worker.navigation.expires_at = time.monotonic() - .01
            worker.navigation.diagnostics.expires_at_s = worker.navigation.expires_at
            ticks = sim.ticks
            worker.navigation.tick(sim)
            assert not worker.navigation.buffer and not worker.navigation.velocity.any()
            assert sim.ticks == ticks
            recorder.capture(worker)
            return worker.navigation.diagnostics.authorization_id
        identity = await worker.call(expire)
        worker.stop()
        assert not list((recorder.directory / "media").glob("worker-timing-*.json"))
        await asyncio.to_thread(recorder.flush)
        files = list((recorder.directory / "media").glob("worker-timing-*.json"))
        assert len(files) == 1
        incident = json.loads(files[0].read_text())
        assert incident["navigation"]["authorization_id"] == identity
        assert "expired" in incident["navigation"]["stop"]["reason"]
        assert incident["clock"] == "monotonic" and incident["run_id"] == worker.sim.run_id
        assert any(event["kind"] == "queue_wait" for event in incident["events"])
        assert any(event.get("phase") == "safety.check_clearance" for event in incident["events"])
        assert any(span["phase"] == "navigation.tick" for span in incident["active_spans"])
        assert all(event.get("self_s", 0.) <= event.get("duration_s", 0.) + 1e-9 for event in incident["events"])
        samples = [json.loads(line) for line in (recorder.directory / "trajectory.jsonl").read_text().splitlines()]
        assert any(f"media/{files[0].name}" in sample.get("worker_timing_files", []) for sample in samples)
        score = await asyncio.to_thread(recorder.finish, {"real_model": False})
        assert score["worker_timing"] == {"scope": "Bounded stop-centered worker wall-time snapshots",
            "incidents": 1, "omitted": 0, "dropped": 0, "limit": 128}
    finally:
        recorder.stream.close()
        worker.stop()
        await worker.close()


async def test_worker_stop_during_buffer_update_is_not_an_execution_failure(monkeypatch):
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousNavigation
    from backend.navigation import NavigationRuntime
    from scripts.navigation_policy import apply
    from backend.worker_timing import timed
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        def check(sim):
            worker.navigation = NavigationRuntime()
            apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Stop during renewal"})
            apply(worker.navigation, sim, "replace_motion_buffer", {"segments": [
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0., "duration_s": 1.}]})
            worker.continuous = ContinuousNavigation([[0., 0.], [1., 0.]])
            @timed("fixture.renewal")
            def interrupted(current):
                sim.stop()
                raise MotionError("CANCELLED", "Navigation was stopped")
            monkeypatch.setattr(worker, "_update_continuous", lambda: interrupted(worker))
            before = sim.ticks
            worker._tick_navigation()
            assert sim.ticks == before and sim.cancel.is_set()
            assert worker.navigation.status == worker.continuous.status == "cancelled"
            assert worker.navigation.reason.startswith("CANCELLED:")
            assert not worker.navigation.buffer and not worker.navigation.velocity.any()
            incident = worker.worker_timing.incidents[-1]
            assert incident["navigation"]["stop"]["initiator"] == "external_stop"
            assert any(event.get("error_code") == "CANCELLED" for event in incident["events"])
        await worker.call(check)
    finally:
        worker.stop()
        await worker.close()


@pytest.mark.parametrize("interrupted", [False, True])
async def test_navigation_progress_and_stop_survive_nonempty_read_queue(interrupted):
    from concurrent.futures import Future
    from backend.challenges import get_challenge
    from backend.navigation import NavigationRuntime
    from scripts.navigation_policy import apply
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    requests = []
    try:
        await asyncio.wrap_future(worker.ready)

        def begin(sim):
            sim.width, sim.height = 160, 120
            worker.navigation = NavigationRuntime()
            apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Bounded drive under read pressure"})
            apply(worker.navigation, sim, "replace_motion_buffer", {"segments": [
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0., "duration_s": 1.}] * 2})

            def read(sim):
                threading.Event().wait(.01)
                return sim.ticks, sim.odometry.copy()

            for _ in range(80):
                future = Future()
                requests.append(future)
                worker.queue.put((future, read))

        await worker.call(begin)
        if interrupted:
            await asyncio.wrap_future(requests[10])
            worker.stop()
        snapshots = await asyncio.gather(*(asyncio.wrap_future(request) for request in requests))
        assert snapshots[30][0] > snapshots[0][0], "Read requests starved the physics executor"
        if interrupted:
            assert snapshots[-1][0] == snapshots[30][0]
            assert not worker.navigation.buffer and not worker.navigation.velocity.any()
        else:
            assert snapshots[-1][0] > snapshots[30][0], worker.navigation.reason
            assert snapshots[-1][1][0] > .015
            assert worker.navigation.status in {"running", "awaiting_feedback"}, worker.navigation.reason
            if worker.navigation.status == "awaiting_feedback":
                assert not worker.navigation.buffer and not worker.navigation.velocity.any()
                assert "expired" in worker.navigation.reason or "exhausted" in worker.navigation.reason
    finally:
        worker.stop()
        await worker.close()


async def test_ros_session_rejects_stale_commands_and_latches_watchdog():
    from backend.ros_navigation import RosNavigationSession, RosStart, RosVelocity
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        def check(sim):
            now = [10.]
            request = RosStart(run_id=sim.run_id, episode_epoch=sim.epoch, target_m_rad=[1., 0., 0.])
            session = RosNavigationSession(sim, request, worker.stop_revision, clock=lambda: now[0])
            assert session.state()["speed_limit_mps"] == .35
            session.observe(1, now[0])
            command = RosVelocity(session_id=session.identity, sensor_sequence=1,
                command_sequence=1, linear_mps=.2, angular_radps=0.)
            for speed in (-.5, .5):
                with pytest.raises(ValueError):
                    RosVelocity.model_validate({**command.model_dump(), "linear_mps": speed})
            session.apply(sim, command, worker.stop_revision)
            assert session.runtime.buffer
            with pytest.raises(MotionError, match="already consumed"):
                session.apply(sim, command, worker.stop_revision)
            with pytest.raises(MotionError, match="old session"):
                session.apply(sim, command.model_copy(update={"session_id": "old", "command_sequence": 2}), worker.stop_revision)
            now[0] += .51
            session.tick(sim, worker.stop_revision)
            assert not session.active and not session.runtime.buffer
            session.observe(2, now[0])
            with pytest.raises(MotionError, match="invalidated"):
                session.apply(sim, command.model_copy(update={"sensor_sequence": 2, "command_sequence": 2}), worker.stop_revision)
            fresh = RosNavigationSession(sim, request, worker.stop_revision, clock=lambda: now[0])
            fresh.observe(2, now[0] - 1.)
            with pytest.raises(MotionError, match="recent sensor"):
                fresh.apply(sim, command.model_copy(update={"session_id": fresh.identity, "sensor_sequence": 2}), worker.stop_revision)
            fresh.observe(3, now[0])
            with pytest.raises(MotionError, match="invalidated"):
                fresh.apply(sim, command.model_copy(update={"session_id": fresh.identity, "sensor_sequence": 3}), worker.stop_revision + 1)
            with pytest.raises(MotionError, match="invalidated"):
                fresh.apply(sim, command.model_copy(update={"session_id": fresh.identity, "sensor_sequence": 3}),
                    worker.stop_revision, task_revision=1)
            assert fresh.state()["task_revision"] == 0
            sim.stop()
            with pytest.raises(MotionError, match="invalidated"):
                fresh.apply(sim, command.model_copy(update={"session_id": fresh.identity, "sensor_sequence": 3}), worker.stop_revision)
        await worker.call(check)
    finally:
        await worker.close()


@pytest.mark.parametrize("operation", ["velocity", "result"])
async def test_ros_request_watchdog_publishes_terminal_state(monkeypatch, operation):
    from backend.ros_navigation import RosGoalResult, RosStart, RosVelocity
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        session = await worker.start_ros(RosStart(run_id=worker.latest["run_id"],
            episode_epoch=0, target_m_rad=[1., 0., 0.]))
        method = "apply" if operation == "velocity" else "complete"
        original = getattr(worker.ros_navigation, method)
        def expire_on_request(*args, **kwargs):
            worker.ros_navigation.command_deadline = 0.
            return original(*args, **kwargs)
        monkeypatch.setattr(worker.ros_navigation, method, expire_on_request)
        with pytest.raises(MotionError, match="ROS_WATCHDOG"):
            if operation == "velocity":
                await worker.ros_velocity(RosVelocity(session_id=session["session_id"],
                    sensor_sequence=1, command_sequence=1, linear_mps=0., angular_radps=0.))
            else:
                await worker.ros_result(RosGoalResult(session_id=session["session_id"], status="succeeded"))
        assert worker.latest["ros_navigation"]["status"] == "cancelled"
        assert "ROS_WATCHDOG" in worker.latest["ros_navigation"]["reason"]
        assert not worker.latest["busy"] and not worker.navigation.buffer
    finally:
        await worker.close()


@pytest.mark.parametrize("fault", [None, "offline", "stale", "task", "stop", "unknown"])
async def test_nav2_agent_goal_requires_ready_current_observed_authority(monkeypatch, fault):
    from types import SimpleNamespace
    from backend.ros_navigation import RosBridgeStatus, RosGoalResult
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        status = RosBridgeStatus(bridge_id="test-bridge", run_id=worker.sim.run_id, episode_epoch=0,
            ready=True, message="Scripted bridge readiness")
        if fault != "offline":
            await worker.ros_heartbeat(status)
        monkeypatch.setattr(worker, "_sample_spatial", lambda force=False: None)
        worker.spatial_map = SimpleNamespace(public=lambda: {"stale": False}, traversable=lambda *args: True,
            contains_path=lambda *args: fault != "unknown")
        sensor = SimpleNamespace(run_id=worker.sim.run_id, episode_epoch=0,
            captured_at=time.monotonic() - (16 if fault == "stale" else 0), odometry_m_rad=worker.sim.odometry.copy())
        if fault == "stop":
            worker.stop()
        if fault:
            with pytest.raises(MotionError):
                await worker.start_nav2_target(sensor, [.5, 0.], 0, 1 if fault == "task" else 0)
            assert worker.ros_navigation is None
        else:
            result = await worker.start_nav2_target(sensor, [.5, 0.], 0, 0)
            assert result["controller"] == "nav2_external" and result["owner"] == "agent"
            assert worker.continuous is None and worker.navigation.skill_deadline <= sensor.captured_at + 20.
            with pytest.raises(MotionError, match="old goal"):
                await worker.ros_result(RosGoalResult(session_id="old", status="succeeded"))
            result = await worker.ros_result(RosGoalResult(session_id=result["session_id"], status="succeeded"))
            assert result["status"] == "blocked" and not worker.navigation.buffer
            with pytest.raises(MotionError):
                await worker.ros_heartbeat(status.model_copy(update={"bridge_id": "another-bridge"}))
            worker.ros_bridge_at = time.monotonic() - 3
            assert not worker.nav2_status()["ready"]
    finally:
        await worker.close()


def test_nav2_backend_selection_preserves_builtin_backup():
    from backend.agent import AgentStart
    from backend.experiment_variants import variant_snapshot
    from types import SimpleNamespace
    settings = AgentStart(run_id="test", episode_epoch=0, goal="Find an observed room", execution_mode="luna_continuous")
    assert settings.navigation_backend == "builtin"
    nav2 = AgentStart(**{**settings.model_dump(), "navigation_backend": "nav2"})
    profile = SimpleNamespace(provider="foundry", id="luna", deployment="test")
    assert variant_snapshot(nav2, profile)["architecture"]["id"] == "nav2-supervised"
    assert variant_snapshot(settings, profile)["architecture"]["id"] == "observed-continuous"
    for incompatible in ({"skill_composer": True}, {"continuous_handoff": True}, {"execution_mode": "single_step"}):
        with pytest.raises(ValueError, match="Nav2 requires"):
            AgentStart(**{**nav2.model_dump(), **incompatible})


async def test_ros_odometry_burst_does_not_force_one_physics_tick_per_request(monkeypatch):
    from backend.ros_navigation import RosStart, RosVelocity
    worker = SimulationWorker(pace=True, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        session = await worker.start_ros(RosStart(run_id=worker.latest["run_id"],
            episode_epoch=0, target_m_rad=[1., 0., 0.]))
        sensor = await worker.ros_sensors()
        await worker.ros_velocity(RosVelocity(session_id=session["session_id"], sensor_sequence=sensor["sequence"],
            command_sequence=1, linear_mps=.1, angular_radps=0.))
        ticks = []
        original = worker.ros_navigation.tick
        def count_tick(*args, **kwargs):
            ticks.append(time.monotonic())
            return original(*args, **kwargs)
        monkeypatch.setattr(worker.ros_navigation, "tick", count_tick)
        packets = await asyncio.gather(*(worker.ros_odometry() for index in range(40)))
        assert len(packets) == 40 and len(ticks) < 10
        assert worker.ros_navigation.active
    finally:
        await worker.close()


async def test_ros_odometry_does_not_wait_for_camera_or_authorize_motion(monkeypatch):
    worker = SimulationWorker(pace=False, rendering="tiny")
    try:
        await asyncio.wrap_future(worker.ready)
        def forbid_capture(*args):
            raise AssertionError("Odometry must not capture the camera")
        monkeypatch.setattr(worker.sim, "capture_spatial", forbid_capture)
        packet = await worker.ros_odometry()
        assert packet["run_id"] == worker.latest["run_id"] and packet["episode_epoch"] == 0
        assert packet["odometry_m_rad"] == worker.sim.odometry.tolist()
        assert 0 <= time.time() - packet["captured_unix_s"] < .5
        assert worker.ros_sensor_sequence == 0 and worker.ros_navigation is None
        assert worker.sim.ticks == 0
    finally:
        await worker.close()


@pytest.mark.parametrize("rendering", ["tiny", "enhanced"])
async def test_ros_stream_reuses_paired_camera_and_keeps_motion_authorized(monkeypatch, rendering):
    from backend.challenges import get_challenge
    from backend.ros_navigation import RosStart, RosVelocity
    worker = SimulationWorker(challenge=get_challenge("park"), pace=True, rendering=rendering)
    try:
        await asyncio.wrap_future(worker.ready)
        def forbid_extra_capture():
            raise AssertionError("ROS motion must reuse the paired sensor camera")
        monkeypatch.setattr(worker.sim, "capture", forbid_extra_capture)
        def forbid_active_preview():
            assert not (worker.ros_navigation and worker.ros_navigation.active), "ROS must reuse paired head rendering"
        monkeypatch.setattr(worker, "_preview_update", forbid_active_preview)
        session = await worker.start_ros(RosStart(run_id=worker.latest["run_id"],
            episode_epoch=0, target_m_rad=[1., 0., 0.]))
        for sequence in range(1, 16):
            sensor = await worker.ros_sensors()
            state = await worker.ros_velocity(RosVelocity(session_id=session["session_id"],
                sensor_sequence=sensor["sequence"], command_sequence=sequence,
                linear_mps=.1, angular_radps=0.))
            assert state["status"] == "running", state
        assert worker.navigation.travel > .01
        assert worker.latest["camera"]["simulated_time_s"] > 0
        worker.stop()
        await worker.hold_stopped()
        assert not worker.ros_navigation.active and not worker.navigation.buffer
    finally:
        await worker.close()


@pytest.mark.parametrize("rendering", ["tiny", "enhanced"])
async def test_spatial_sampling_preserves_model_sequences_and_stop_during_motion(rendering):
    from backend.challenges import get_challenge
    from backend.contracts import SpatialSettings
    from scripts.navigation_policy import apply
    worker = SimulationWorker(challenge=get_challenge("park"), pace=True, rendering=rendering)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True)
        initial = await worker.configure_spatial(settings)
        await worker.begin_navigation(worker.stop_revision)
        await worker.call(lambda sim: apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Spatial sensor motion check"}))
        observation, _ = await worker.feedback()
        result = await worker.execute_navigation(Command(run_id=worker.sim.run_id, episode_epoch=0,
            observation_seq=observation.seq, action_id="spatial-drive", tool="replace_motion_buffer",
            arguments={"expected_revision": observation.navigation.revision, "segments": [
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0., "duration_s": 1.}]}))
        assert result.status == "ok"
        model_seq = worker.sim.seq
        async with asyncio.timeout(4):
            while worker.latest["navigation"]["remaining_s"] > 0:
                await asyncio.sleep(.02)
        spatial = await worker.call(lambda sim: worker.spatial_state())
        assert worker.sim.seq == model_seq
        assert spatial["frame"]["sequence"] > initial["frame"]["sequence"]
        assert spatial["frame"]["simulated_time_s"] > 0 and spatial["error"] is None
        assert worker.latest["navigation"]["status"] != "failed", worker.latest["navigation"]["reason"]
        observation, _ = await worker.feedback()
        await worker.execute_navigation(Command(run_id=worker.sim.run_id, episode_epoch=0,
            observation_seq=observation.seq, action_id="spatial-stop", tool="replace_motion_buffer",
            arguments={"expected_revision": observation.navigation.revision, "segments": [
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0., "duration_s": 1.}]}))
        worker.stop()
        await worker.call(lambda sim: sim.hold_current())
        assert worker.latest["stopped"] and not worker.navigation.buffer
        ticks = worker.sim.ticks
        await worker.call(lambda sim: worker._sample_spatial(force=True))
        assert worker.sim.ticks == ticks and worker.latest["stopped"]
    finally:
        await worker.close()
    assert not worker.spatial_frames


async def test_expired_spatial_result_is_measured_but_never_published():
    from concurrent.futures import Future
    from backend.contracts import SpatialSettings
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True))
        def check(sim):
            source = worker.spatial_map
            frames = list(worker.spatial_frames)
            observation, image, depth = next(reversed(worker.spatial_frames.values()))
            expired = observation.model_copy(update={"captured_at": time.monotonic() - source.max_frame_age_s - 1.})
            accepted = worker.spatial_timing["accepted_results"]
            worker.spatial_pending = Future()
            worker.spatial_pending.set_result((source, expired, image, depth))
            worker.spatial_pending_map = source
            worker.spatial_submitted_at = expired.captured_at
            worker._receive_spatial()
            assert worker.spatial_map is source and list(worker.spatial_frames) == frames
            assert worker.spatial_error == "Spatial processing result expired"
            assert worker.spatial_processing_max_s > source.max_frame_age_s
            assert worker.spatial_timing["post_capture_max_s"] > source.max_frame_age_s
            assert worker.spatial_timing["expired_results"] == 1
            assert worker.spatial_timing["accepted_results"] == accepted
            worker._sample_spatial(force=True)
            assert worker.spatial_error is None
            assert worker.spatial_timing["accepted_results"] == accepted + 1
            assert worker.spatial_timing["capture_max_s"] > 0
        await worker.call(check)
    finally:
        await worker.close()


async def test_spatial_processing_does_not_block_worker_or_publish_replaced_map(monkeypatch):
    from backend import camera
    from backend.contracts import SpatialSettings
    entered, release = threading.Event(), threading.Event()
    original = camera.process_spatial
    def delayed(*args):
        entered.set()
        assert release.wait(3)
        return original(*args)
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True)
        await worker.configure_spatial(settings)
        previous_map = worker.spatial_map
        monkeypatch.setattr(camera, "process_spatial", delayed)
        await worker.call(lambda sim: (setattr(worker, "spatial_sampled_at", 0), worker._sample_spatial()))
        assert await asyncio.to_thread(entered.wait, 1)
        assert await asyncio.wait_for(worker.call(lambda sim: sim.ticks), .5) == 0
        await worker.configure_spatial(settings.model_copy(update={"enabled": False}))
        release.set()
        await worker.call(lambda sim: worker._receive_spatial(wait=True))
        assert worker.spatial_map is None and not worker.spatial_frames
        assert previous_map.sequence > 0
    finally:
        release.set()
        await worker.close()


def test_floor_coverage_matches_original_rasterization_at_map_edges():
    import numpy as np
    from PIL import Image, ImageDraw
    from backend.spatial import floor_coverage
    random = np.random.default_rng(713)
    corners = random.integers(-12, 45, size=(2000, 4, 2))
    corners = np.concatenate((corners, np.full((1, 4, 2), 7), np.array([
        [[-1, -1], [0, -1], [0, 0], [-1, 0]], [[31, 31], [32, 31], [32, 32], [31, 32]],
        [[-20, -20], [-18, -20], [-18, -18], [-20, -18]], [[35, 35], [36, 35], [36, 36], [35, 36]]])))
    for batch in (corners, *(polygon[None] for polygon in corners)):
        expected = Image.new("1", (32, 32))
        draw = ImageDraw.Draw(expected)
        crosses = np.any(batch != batch[:, :1], axis=(1, 2))
        for polygon in np.unique(batch[crosses], axis=0):
            draw.polygon([tuple(point) for point in polygon], fill=1)
        np.testing.assert_array_equal(floor_coverage(batch, 32), np.asarray(expected, dtype=bool))
    assert not floor_coverage(np.empty((0, 4, 2), dtype=int), 32).any()


async def test_spatial_processing_preserves_source_and_matches_deep_copy():
    from copy import deepcopy
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from backend.camera import process_spatial
    from backend.contracts import SpatialSettings
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True))
        def check(sim):
            source = worker.spatial_map
            original = deepcopy(source)
            expected = deepcopy(source)
            sensor, image = sim.capture_spatial(worker.spatial_sequence + 1)
            expected.update(sensor, rgb=np.asarray(Image.open(BytesIO(image)).convert("RGB")))
            mapped, returned, rgb, depth = process_spatial(source, sensor, image)
            assert returned is sensor and mapped.observation is sensor and rgb == image and depth
            assert source.sequence == original.sequence and source.captured_at == original.captured_at
            assert source.observation == original.observation
            for name in ("cells", "floor_colors", "obstacle_low", "obstacle_high", "clear_votes", "origin"):
                np.testing.assert_array_equal(getattr(mapped, name), getattr(expected, name))
                np.testing.assert_array_equal(getattr(source, name), getattr(original, name))
                assert not np.shares_memory(getattr(mapped, name), getattr(source, name))
        await worker.call(check)
    finally:
        await worker.close()


async def test_navigation_proposal_goal_change_and_stop_reject_old_predictions():
    from backend.continuous_navigation import ContinuousScan
    from backend.navigation_backends import NavigationGoal, NOMAD_CAPABILITIES, relative_proposal
    from backend.challenges import get_challenge
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=0, compact_arms=True))
        goal = NavigationGoal(kind="explore", evidence="exploration")
        frame, _, _ = await worker.capture_navigation_frame(goal)
        proposal = relative_proposal(frame, "nomad", [[.1, 0.], [.2, 0.]])
        await worker.capture_navigation_frame(NavigationGoal(kind="explore", evidence="exploration"))
        before = worker.sim.ticks
        with pytest.raises(MotionError, match="superseded"):
            await worker.start_navigation_proposal(proposal, goal, NOMAD_CAPABILITIES)
        assert worker.sim.ticks == before and not worker.navigation
        worker.stop()
        with pytest.raises(MotionError, match="stopped"):
            await worker.start_navigation_proposal(proposal, goal, NOMAD_CAPABILITIES)
        assert worker.sim.ticks == before
    finally:
        await worker.close()


async def test_navigation_policy_head_change_brakes_installed_route():
    from backend.continuous_navigation import ContinuousScan
    from backend.navigation_backends import NavigationGoal, NOMAD_CAPABILITIES, relative_proposal
    from backend.challenges import get_challenge
    worker = SimulationWorker(challenge=get_challenge("local_park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=0, compact_arms=True))
        result = await worker.execute(Command(run_id=worker.sim.run_id, episode_epoch=0,
            observation_seq=worker.latest["observation"]["seq"], action_id=str(uuid4()), tool="set_head",
            arguments={"yaw_rad": 0., "pitch_rad": .3, "duration_s": 1.}))
        assert result.status == "ok"
        goal = NavigationGoal(kind="explore", evidence="exploration")
        frame, _, _ = await worker.capture_navigation_frame(goal)
        proposal = relative_proposal(frame, "nomad", [[.2, 0.], [.4, 0.]])
        await worker.start_navigation_proposal(proposal, goal, NOMAD_CAPABILITIES)
        def move_head(sim):
            bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .5, physicsClientId=sim.client)
            worker.continuous.last_update = 0.
            worker._update_continuous()
            assert worker.continuous.status == "blocked" and worker.continuous.reason.startswith("CAMERA_CHANGED")
            assert not worker.navigation.buffer and not worker.navigation.tickets
        await worker.call(move_head)
    finally:
        await worker.close()


async def test_continuous_collects_ready_depth_before_freshness_check():
    from concurrent.futures import Future
    from types import SimpleNamespace
    from backend.camera import process_spatial
    from backend.contracts import SpatialSettings
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True))
        def check(sim):
            sensor, image = sim.capture_spatial(worker.spatial_sequence + 1)
            completed = process_spatial(worker.spatial_map, sensor, image)
            worker.spatial_map.captured_at = time.monotonic() - 2
            worker.spatial_pending_map = worker.spatial_map
            worker.spatial_pending = Future()
            worker.spatial_pending.set_result(completed)
            worker.spatial_sampled_at = time.monotonic()
            observed = []
            worker.continuous = SimpleNamespace(last_update=0, active=False, skill_components=[], exploration=None,
                start_position=sim.odometry[:2], radius=sim.robot_footprint()["radius_m"],
                update=lambda sim, runtime, mapped, valid, replan: observed.append(mapped.captured_at))
            worker._update_continuous()
            assert observed == [sensor.captured_at]
            assert time.monotonic() - observed[0] < 1
        await worker.call(check)
    finally:
        await worker.close()


async def test_spatial_sampling_cost_and_manual_relocation_reset():
    import json
    import numpy as np
    from backend.contracts import SpatialSettings
    from backend.simulation import MotionError
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.configure_spatial(SpatialSettings(run_id=worker.sim.run_id, episode_epoch=0, enabled=True))
        durations = []
        for _ in range(10):
            def measure(sim):
                started = time.perf_counter()
                worker._sample_spatial(force=True)
                return time.perf_counter() - started
            durations.append(await worker.call(measure))
        assert worker.spatial_error is None and len(worker.spatial_frames) <= 8
        sensor = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[0])
        review, sheet = await worker.camera_history_review(sensor, worker.stop_revision)
        assert sheet is not None and review["source"] == "paired_head_camera"
        frame_id = review["frames"][0]["frame_id"]
        metadata, original = await worker.inspect_camera_history(frame_id, worker.stop_revision)
        assert original and metadata["historical"] and not metadata["authorizes_motion"]
        assert "depth_m" not in metadata and "objects" not in metadata
        assert await worker.call(lambda sim: len(worker.camera_history.recent)) > len(worker.spatial_frames)
        old_sequences = set(worker.spatial_frames)
        await worker.reposition(ManualPlacement(run_id=worker.sim.run_id, episode_epoch=0,
            observation_seq=worker.latest["observation"]["seq"], xy_m=[-.5, 0.]))
        assert not old_sequences.intersection(worker.spatial_frames)
        assert worker.latest["observation"]["odometry_m_rad"] == [0., 0., 0.]
        assert worker.spatial_map.sequence == worker.spatial_sequence
        with pytest.raises(ValueError, match="expired"):
            await worker.inspect_camera_history(frame_id, worker.stop_revision)
        with pytest.raises(ValueError, match="generation"):
            await worker.acknowledge_camera_history(review, worker.stop_revision)
        revision = worker.stop_revision
        worker.stop()
        with pytest.raises(MotionError, match="stopped"):
            await worker.inspect_camera_history(frame_id, revision)
        print(json.dumps({"spatial_capture_mapping_ms": {"median": float(np.median(durations) * 1000),
            "p95": float(np.percentile(durations, 95) * 1000), "max": max(durations) * 1000}, "samples": 10}))
    finally:
        await worker.close()


@pytest.mark.parametrize("column,rendering", [(72, "tiny"), (80, "tiny"), (88, "tiny"), (80, "enhanced")])
async def test_continuous_navigation_follows_a_camera_selected_floor_point(column, rendering):
    import json
    import numpy as np
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan, ContinuousTarget, target_from_depth
    worker = SimulationWorker(challenge=get_challenge("park"), pace=True, rendering=rendering)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=0, compact_arms=True))
        sensor = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[0])
        candidates = []
        for row in range(35, 100, 4):
            pixel = [column / 160, row / 120]
            try:
                target = target_from_depth(sensor, pixel)
                candidates.append((np.linalg.norm(np.array(target) - [1.2, 0]), pixel))
            except ValueError:
                pass
        pixel = min(candidates)[1]
        measured = []
        def install(sim):
            publish = sim.on_tick
            def sample():
                publish()
                if worker.continuous and worker.continuous.active:
                    speed = sum(bullet.getJointState(sim.robot, index, physicsClientId=sim.client)[1] for index in sim.wheels) * .09 / 2
                    measured.append((worker.continuous.distance_m, speed))
            sim.on_tick = sample
        await worker.call(install)
        await worker.start_continuous(ContinuousTarget(run_id=worker.sim.run_id, episode_epoch=0,
            spatial_sequence=sensor.sequence, pixel=pixel))
        async with asyncio.timeout(30):
            while worker.continuous.active:
                await asyncio.sleep(.1)
        assert worker.continuous.status == "arrived", json.dumps({**worker.continuous.state(),
            "spatial_error": worker.spatial_error, "spatial_processing_max_s": worker.spatial_processing_max_s,
            "map_age_s": time.monotonic() - worker.spatial_map.captured_at})
        assert worker.continuous.buffer_stops == 0 and worker.continuous.updates > 20
        assert worker.continuous.minimum_cruise_speed > .08
        assert worker.preview_renderer
        if rendering == "tiny":
            assert worker.preview_renderer.process.is_alive()
        else:
            assert worker.sim.camera_renderer.process.poll() is None
        assert np.linalg.norm(worker.sim.odometry[:2] - worker.continuous.path[-1]) < .06
        cruise = [speed for remaining, speed in measured if .25 < remaining < .9]
        assert len(cruise) > 10 and min(cruise) > .08
        print(json.dumps({"continuous_column": column, **worker.continuous.state(), "measured_min_cruise_mps": min(cruise)}))
    finally:
        await worker.close()


async def test_enhanced_preview_discards_frame_from_before_stop(monkeypatch):
    worker = SimulationWorker(rendering="enhanced", pace=False)
    entered, release = threading.Event(), threading.Event()
    try:
        await asyncio.wrap_future(worker.ready)
        original = worker.sim.camera_renderer.capture
        def delayed(packet, depth=True):
            entered.set()
            assert release.wait(3)
            return original(packet, depth)
        assert worker.preview_renderer is not None
        monkeypatch.setattr(worker.sim.camera_renderer, "capture", delayed)
        await worker.call(lambda sim: worker._preview_update())
        assert await asyncio.to_thread(entered.wait, 1)
        pending = worker.preview_renderer.pending
        worker.stop()
        ticks, sequence = worker.sim.ticks, worker.camera_seq
        release.set()
        await asyncio.wrap_future(pending)
        await worker.call(lambda sim: (setattr(worker, "preview_requested_at", time.monotonic()), worker._preview_update()))
        assert worker.camera_seq == sequence and worker.sim.ticks == ticks
        assert worker.latest["stopped"]
        await worker.call(lambda sim: sim.camera_renderer.close())
        with pytest.raises(MotionError, match="Reset the episode"):
            await worker.resume_manual()
        assert worker.latest["stopped"] and worker.sim.ticks == ticks
        monkeypatch.setattr(worker.sim.camera_renderer, "capture", original)
        await worker.call(lambda sim: (setattr(worker, "preview_requested_at", 0), worker._preview_update()))
        pending = worker.preview_renderer.pending
        with pytest.raises(RuntimeError, match="closed"):
            await asyncio.wrap_future(pending)
        await worker.call(lambda sim: worker._preview_update())
        assert "RENDERER_FAILED" in worker.latest["rendering_error"]
        assert await asyncio.wait_for(worker.call(lambda sim: sim.ticks), 1) == ticks
    finally:
        release.set()
        await worker.close()


async def test_kitchen_downward_sweep_connects_observed_routes():
    import math
    import numpy as np
    from scipy.ndimage import label
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan
    from backend.continuous_supervisor import rotate, settle

    class Controller:
        def _check_live(self, worker, settings):
            assert not worker.sim.cancel.is_set(), worker.latest["navigation"]

    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
        settings = SimpleNamespace(run_id=worker.sim.run_id, episode_epoch=0)
        await worker.scan_continuous(ContinuousScan(**vars(settings), compact_arms=True))
        def connected(sim):
            mapped = worker.spatial_map
            components, _ = label(mapped.traversable(sim.odometry[:2], sim.robot_footprint()["radius_m"]))
            current = components[mapped.cell_index(sim.odometry[:2])]
            return int(np.sum(components == current)) if current else 0
        before = await worker.call(connected)
        await rotate(Controller(), worker, settings, worker.stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
        after = await worker.call(connected)
        sensor, image, candidates, observation = await worker.continuous_candidates()
        assert after > before * 3 and len(candidates) > 3
        assert any(candidate["pixel"] is None and candidate["distance_m"] > 1. for candidate in candidates)
        assert sensor.run_id == observation.run_id and image and not worker.latest["proximity"]["collisions"]
        assert not worker.navigation.buffer
        parking = await worker.parking_clearance()
        assert parking["required_clearance_m"] >= .47
        assert parking["parking_margin_ok"] == (parking["observed_clearance_m"] >= parking["required_clearance_m"])
        ticks = worker.sim.ticks
        await settle(worker, settings)
        assert worker.sim.ticks == ticks + 120
        velocity, angular = await worker.call(lambda sim: bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client))
        assert np.linalg.norm(velocity) < .025 and np.linalg.norm(angular) < .15
        worker.stop()
        with pytest.raises(ValueError):
            await settle(worker, settings)
        assert worker.sim.ticks == ticks + 120
    finally:
        await worker.close()


@pytest.mark.parametrize("invalidate", [None, "stop", "revision", "arrival"])
async def test_moving_continuation_preserves_velocity_and_authorization(invalidate, monkeypatch):
    import numpy as np
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan, ContinuousTarget
    worker = SimulationWorker(challenge=get_challenge("park"), pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=0, compact_arms=True))
        sensor = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[0])
        await worker.start_continuous(ContinuousTarget(run_id=worker.sim.run_id, episode_epoch=0, spatial_sequence=sensor.sequence, pixel=[.5, .6]))
        async with asyncio.timeout(3):
            while worker.navigation.velocity[0] < .1:
                await asyncio.sleep(.02)
        def duplicate_capture():
            raise AssertionError("Moving proposals must reuse their paired RGB/depth capture")
        with monkeypatch.context() as patch:
            patch.setattr(worker.sim, "capture", duplicate_capture)
            sensor, image, candidates, observation, proposal = await worker.moving_candidates()
        assert Image.open(BytesIO(image)).size == (sensor.calibration.width, sensor.calibration.height)
        assert worker.sim.frame(observation.frame_ref) == image
        assert observation.head_rad == sensor.head_rad
        assert observation.odometry_m_rad == sensor.odometry_m_rad
        assert observation.simulated_time_s == sensor.simulated_time_s
        assert observation.navigation is not None
        choice = min(candidates, key=lambda item: abs(item["target_m"][1]) + abs(item["distance_m"] - 1.4))
        deadline = worker.navigation.skill_deadline
        if invalidate == "stop":
            worker.stop()
        elif invalidate == "revision":
            await worker.call(lambda sim: setattr(worker.continuous, "revision", worker.continuous.revision + 1))
        elif invalidate == "arrival":
            await worker.call(lambda sim: worker.continuous.finish(sim, worker.navigation, "arrived", "Test reached prior goal"))
        if invalidate:
            with pytest.raises(ValueError):
                await worker.retarget_continuous(proposal, choice["id"])
        else:
            result = await worker.retarget_continuous(proposal, choice["id"])
            assert result["handoffs"] == 1 and worker.navigation.velocity[0] > .08
            assert worker.navigation.skill_deadline == deadline
            async with asyncio.timeout(25):
                while worker.continuous.active:
                    await asyncio.sleep(.05)
            assert worker.continuous.status == "arrived", worker.continuous.state()
            assert worker.continuous.buffer_stops == 0 and worker.navigation.travel < 2
            assert np.linalg.norm(worker.sim.odometry[:2] - choice["target_m"]) < .06
    finally:
        await worker.close()


def test_camera_activity_uses_elapsed_time_and_ignores_tiny_pixel_noise():
    def png(color):
        output = BytesIO()
        Image.new("RGB", (64, 48), color).save(output, format="PNG")
        return output.getvalue()

    activity = CameraActivity()
    assert activity.sample(png((100, 100, 100)), 10) == {"revision": 1, "unchanged_s": 0}
    assert activity.sample(png((102, 102, 102)), 14.9)["unchanged_s"] == pytest.approx(4.9)
    assert activity.sample(png((100, 100, 100)), 15) == {"revision": 1, "unchanged_s": 5}
    assert activity.sample(png((130, 100, 100)), 16) == {"revision": 2, "unchanged_s": 0}


@pytest.mark.asyncio
async def test_camera_monitor_detects_real_head_changes_without_observing_or_stepping():
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        before = await worker.sample_camera_activity()
        await worker.sample_camera_activity()
        assert worker.latest["camera"] == initial["camera"]
        await worker.call(lambda sim: bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .8, physicsClientId=sim.client))
        changed = await worker.sample_camera_activity()
        assert changed["revision"] == before["revision"] + 1
        assert worker.latest["camera"]["seq"] > initial["camera"]["seq"]
        assert worker.sim.seq == initial["observation"]["seq"] and worker.sim.ticks == 0
        assert len(worker.sim.frames) == 1
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_live_camera_and_poses_update_during_motion_without_extra_model_observations():
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        initial_seq = worker.latest["observation"]["seq"]
        initial_frame = worker.sim.frame(worker.latest["observation"]["frame_ref"])
        assert worker.camera_frames[worker.latest["camera"]["frame_ref"]] == initial_frame
        envelope = Command(run_id=worker.sim.run_id, episode_epoch=0, action_id="live-head",
                           observation_seq=initial_seq, tool="set_head",
                           arguments={"yaw_rad": .8, "pitch_rad": .7, "duration_s": 2})
        started = time.monotonic()
        task = asyncio.create_task(worker.execute(envelope, assisted=False))
        images, poses, observation_sequences = {}, {}, set()
        async with asyncio.timeout(5):
            while not task.done():
                state = worker.latest
                if state["busy"]:
                    camera = state["camera"]
                    if image := worker.camera_frames.get(camera["frame_ref"]):
                        images[camera["seq"]] = image
                    poses[state["snapshot"]["simulated_time_s"]] = state["snapshot"]["poses"]
                    observation_sequences.add(state["observation"]["seq"])
                await asyncio.sleep(.02)
        result = await task
        assert result.status == "ok"
        assert len(images) >= 5 and len(set(images.values())) >= 3
        assert len(poses) >= 8 and next(iter(poses.values())) != list(poses.values())[-1]
        assert observation_sequences == {initial_seq}
        assert result.observation.seq == initial_seq + 1
        assert len(worker.sim.frames) == 2
        assert len(worker.camera_frames) <= worker.camera_capacity
        assert worker.camera_frames[worker.latest["camera"]["frame_ref"]] == worker.sim.frame(result.observation.frame_ref)
        assert worker.latest["camera"]["simulated_time_s"] == 2
        assert time.monotonic() - started < 4
    finally:
        await worker.close()
    assert not worker.camera_frames


@pytest.mark.asyncio
async def test_manual_placement_updates_physics_and_camera_without_driving():
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        original_pose = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client))
        original_frame = worker.camera_frames[initial["camera"]["frame_ref"]]
        placement = ManualPlacement(run_id=initial["run_id"], episode_epoch=initial["episode_epoch"],
                                    observation_seq=initial["observation"]["seq"], xy_m=[-.6, .3])
        updated = await worker.reposition(placement)
        position, orientation = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client))
        assert position[:2] == pytest.approx([-.6, .3])
        assert position[2] == original_pose[0][2] and orientation == original_pose[1]
        assert updated["observation"]["seq"] == initial["observation"]["seq"] + 1
        assert updated["observation"]["odometry_m_rad"] == initial["observation"]["odometry_m_rad"]
        assert updated["snapshot"]["simulated_time_s"] == 0
        assert updated["manual_placements"] == 1 and not updated["busy"]
        assert worker.camera_frames[updated["camera"]["frame_ref"]] != original_frame
        assert "xy_m" not in updated["observation"] and "manual_placements" not in updated["observation"]
        with pytest.raises(MotionError, match="state changed"):
            await worker.reposition(placement)
        for destination, code in [([2.5, 0], "COLLISION_BLOCKED"), ([3, 0], "OUT_OF_BOUNDS")]:
            rejected = placement.model_copy(update={"xy_m": destination, "observation_seq": updated["observation"]["seq"]})
            with pytest.raises(MotionError) as failure:
                await worker.reposition(rejected)
            assert failure.value.code == code
            assert worker.latest == updated
            assert await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)) == (position, orientation)
        worker.stop()
        with pytest.raises(MotionError, match="Resume manual"):
            await worker.reposition(placement.model_copy(update={"observation_seq": updated["observation"]["seq"]}))
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_stop_cancels_a_queued_manual_placement():
    worker = SimulationWorker(pace=False)
    entered, release = threading.Event(), threading.Event()
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        placement = ManualPlacement(run_id=initial["run_id"], episode_epoch=initial["episode_epoch"],
                                    observation_seq=initial["observation"]["seq"], xy_m=[-.6, .3])

        def block_worker(sim):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Placement test worker was not released")

        blocker = asyncio.create_task(worker.call(block_worker))
        assert await asyncio.to_thread(entered.wait, 2)
        queued = asyncio.create_task(worker.reposition(placement))
        await asyncio.sleep(0)
        worker.stop()
        release.set()
        await blocker
        with pytest.raises(MotionError, match="Resume manual"):
            await queued
        assert worker.latest["snapshot"] == initial["snapshot"]
        assert worker.latest["manual_placements"] == 0
    finally:
        release.set()
        await worker.close()


@pytest.mark.asyncio
async def test_stop_interrupts_worker_and_reset_isolates_clients():
    worker = SimulationWorker(pace=True)
    await asyncio.wrap_future(worker.ready)
    envelope = Command(run_id=worker.sim.run_id, episode_epoch=0, action_id=str(uuid4()), observation_seq=worker.sim.seq,
                       tool="drive_base", arguments={"linear_mps": .2, "angular_radps": 0, "duration_s": 2})
    task = asyncio.create_task(worker.execute(envelope))
    await asyncio.sleep(.12)
    worker.stop()
    result = await asyncio.wait_for(task, 2)
    assert result.error == "CANCELLED"
    assert result.actual_duration_s < 2
    old_id = worker.sim.run_id
    await worker.close()
    replacement = SimulationWorker(epoch=1, pace=False)
    try:
        await asyncio.wrap_future(replacement.ready)
        assert replacement.sim.run_id != old_id
        result = await replacement.execute(envelope)
        assert result.error == "CANCELLED"
        assert replacement.sim.ticks == 0
    finally:
        await replacement.close()