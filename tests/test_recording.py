import asyncio
import json

import pytest

from backend.recording import RunRecorder, score_samples


@pytest.mark.parametrize("persistent", [False, True])
def test_recording_json_replacement_retries_without_losing_previous_report(tmp_path, monkeypatch, persistent):
    from pathlib import Path
    from backend import recording
    path = tmp_path / "report.json"
    path.write_text('{"previous": true}', encoding="utf-8")
    attempts, delays = [], []
    replace = Path.replace

    def guarded_replace(temporary, destination):
        attempts.append(temporary)
        assert recording.json.loads(path.read_text()) == {"previous": True}
        assert temporary != path.with_suffix(".tmp")
        if persistent or len(attempts) < 3:
            raise PermissionError("Injected Windows sharing violation")
        return replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", guarded_replace)
    monkeypatch.setattr(recording.time, "sleep", delays.append)
    if persistent:
        with pytest.raises(PermissionError):
            recording.write_recording_json(path, {"current": True})
        assert recording.read_recording_json(path) == {"previous": True}
        assert len(attempts) == 5
    else:
        recording.write_recording_json(path, {"current": True})
        assert recording.read_recording_json(path) == {"current": True}
        assert len(attempts) == 3
    assert len(delays) == len(attempts) - 1 and sum(delays) <= .31
    assert list(tmp_path.iterdir()) == [path]


def sample(wall, position=0., speed=0., activity="inference_wait", status="in_progress"):
    return {"wall_s": wall, "simulated_s": wall / 2, "position_m": [position, 0., .155],
        "linear_speed_mps": speed, "angular_speed_radps": 0., "activity": activity,
        "physics_status": status, "collisions": []}


def test_scripted_physics_recordings_are_visible_and_preserve_failures(tmp_path):
    from backend.challenges import get_challenge
    from backend.saved_results import saved_results, recorded_trajectory, terminal_image
    from backend.simulation import BulletSimulation
    from scripts.record_scenario_tests import PhysicsTestBatch
    from tests.test_simulation import command
    batch = PhysicsTestBatch(tmp_path / "performance/recorded-tests")
    for index, failure in enumerate((None, AssertionError)):
        sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
        sequence, ticks = sim.seq, sim.ticks
        recording = batch.record(sim, f"case-{index}", "Scripted non-completing motion; recorder regression")
        try:
            assert (sim.seq, sim.ticks) == (sequence, ticks)
            assert command(sim, "drive_base", linear_mps=.1, angular_radps=0, duration_s=.5).status == "ok"
            recording.finish(failure)
            assert sim.on_tick is None
        finally:
            sim.close()
    batch.finish()
    data = saved_results(tmp_path)["batches"]
    assert len(data) == 1 and data[0]["planned"] == 2 and data[0]["evidence"] == "scripted_test"
    assert data[0]["trials"][1]["status"] == "Test failed"
    assert all(not trial["verified_success"] and trial["recording_complete"] for trial in data[0]["trials"])
    for index in range(2):
        route = recorded_trajectory(data[0]["id"], index, tmp_path)
        assert route["scene"]["source"] == "recorded_initial"
        assert route["points"][-1]["x"] > route["points"][0]["x"] + .02
        assert route["sample_count"] >= 10
        assert terminal_image(data[0]["id"], index, tmp_path).read_bytes().startswith(b"\x89PNG")
    assert batch.manifest["source_changed_during_run"] is False
    report = batch.results[1]
    report["physics_success"] = True
    report["recording_scorecard"]["completion_time_s"] = 1.
    batch.write()
    assert not saved_results(tmp_path)["batches"][0]["trials"][1]["verified_success"]


@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("custom_folder", [False, True])
async def test_browser_control_sessions_save_replays_on_completion_and_stop(tmp_path, monkeypatch, stopped, custom_folder):
    from backend import session_recording
    from backend.challenges import get_challenge
    from backend.saved_results import saved_results, recorded_trajectory
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    model = ScriptedModel([model_response(arguments=json.dumps({"linear_mps": .1, "angular_radps": 0, "duration_s": .5})), "wait"])
    controller = controller_for(model)
    controller.record_sessions = True
    controller.recording_evidence = "scripted_test"
    if custom_folder:
        from backend.preferences import PreferenceStore, PreferencesPatch
        controller.recording_root = tmp_path / "Custom recordings"
        PreferenceStore().update(PreferencesPatch(recording_directory=str(controller.recording_root)))
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, max_turns=2 if stopped else 1, feedback_interval_s=.25))
        if stopped:
            async with asyncio.timeout(10):
                while worker.sim.ticks < 120:
                    await asyncio.sleep(.02)
            await controller.halt()
        else:
            await asyncio.wait_for(controller.task, 10)
        assert worker.recorder is None and not controller.active
        data = saved_results(controller.recording_root if custom_folder else tmp_path)["batches"]
        assert len(data) == 1 and data[0]["evidence"] == "scripted_test"
        assert data[0]["trials"][0]["recording_complete"]
        route = recorded_trajectory(data[0]["id"], 0, controller.recording_root if custom_folder else tmp_path)
        assert route["scene"]["source"] == "recorded_initial"
        assert route["points"][-1]["x"] > route["points"][0]["x"] + .02
        if stopped:
            assert data[0]["trials"][0]["termination"] == "interrupted"
        if custom_folder:
            from backend.saved_results import recorded_replay, recorded_replay_media, terminal_image
            assert controller.recording_directory.is_relative_to(controller.recording_root)
            assert not (tmp_path / "performance").exists()
            PreferenceStore().update(PreferencesPatch(recording_directory=str(tmp_path / "Next recordings")))
            assert any(batch["id"] == data[0]["id"] for batch in saved_results()["batches"])
            assert recorded_trajectory(data[0]["id"], 0)["points"] == route["points"]
            replay = recorded_replay(data[0]["id"], 0)
            assert replay["recording_complete"] and replay["sample_count"] > 0
            image_name = next(frame["camera_url"].rsplit("/", 1)[1] for frame in replay["frames"] if frame["camera_url"])
            assert recorded_replay_media(data[0]["id"], 0, image_name)[0].startswith(b"\x89PNG")
            assert terminal_image(data[0]["id"], 0).is_file()
            with pytest.raises(ValueError):
                recorded_replay_media(data[0]["id"], 0, "../private.json")
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("enabled", [False, True])
async def test_recording_disabled_or_unwritable_destination_never_silently_falls_back(tmp_path, monkeypatch, enabled):
    from backend import session_recording
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "default")
    target = tmp_path / "not-a-folder"
    target.write_text("preserve")
    model = ScriptedModel([model_response(name="wait", arguments=json.dumps({"duration_s": .1}))])
    controller = controller_for(model)
    controller.record_sessions = enabled
    controller.recording_root = target
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        ticks = worker.sim.ticks
        controller.start(worker, start_settings(worker, max_turns=1))
        await asyncio.wait_for(controller.task, 10)
        assert worker.recorder is None and not controller.recording_active
        assert target.read_text() == "preserve" and not (tmp_path / "default").exists()
        if enabled:
            assert controller.recording_error and worker.sim.cancel.is_set()
            assert worker.sim.ticks == ticks and controller.state["turns"] == 0
        else:
            assert controller.recording_error is None and controller.state["turns"] == 1
    finally:
        await controller.halt()
        await worker.close()


async def test_browser_recording_write_failure_brakes_and_detaches(tmp_path, monkeypatch):
    from backend import session_recording
    from backend.challenges import get_challenge
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, start_settings
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    def failed_write(self):
        raise OSError("Injected recording failure")
    monkeypatch.setattr(RunRecorder, "flush", failed_write)
    controller = controller_for(ScriptedModel(["wait"]))
    controller.record_sessions = True
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker))
        await asyncio.wait_for(controller.task, 10)
        assert worker.recorder is None and worker.latest["stopped"]
        assert not controller.active and not controller.recording_active
        assert controller.state["phase"] == "error" and "Recording failed" in controller.state["error"]
        assert controller.recording_error == controller.state["error"]
    finally:
        await controller.halt()
        await worker.close()


def test_scorecard_keeps_failures_and_deliberate_stops_separate():
    rows = [sample(0), sample(1, speed=.1, activity="driving"), sample(2, .1),
        sample(4, .1, speed=.1, activity="driving"), sample(5, .2, status="completed")]
    score = score_samples(rows)
    assert score["completion_time_s"] == 5
    assert score["actual_distance_m"] == pytest.approx(.2)
    assert score["inference_stationary_s"] == 3
    assert score["stationary_intervals"] == [{"start_s": 2, "duration_s": 2, "reason": "inference_wait"}]
    assert score_samples(rows + [sample(6, .2)])["completion_time_s"] is None
    assert score_samples(rows, dropped=1)["completion_time_s"] is None


async def test_home_recording_retains_immutable_snapshots_and_transition_evidence(tmp_path):
    import base64
    import zlib
    import numpy as np
    from types import SimpleNamespace
    from backend.home_mapping import HomeMap
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False, rendering="tiny")
    recorder = RunRecorder(tmp_path / "batch" / "case" / "recording", max_pending=2)
    try:
        await asyncio.wrap_future(worker.ready)
        def capture(sim):
            original = sim.seq, sim.ticks
            home = HomeMap("test")
            mission = SimpleNamespace(home=home, task=None, stage="mapping", localization={"status": "localized"},
                error=None, pose=[0., 0., 0.], transform=[0., 0., 0.], sampled_at=0., route=[], obstacles=lambda: [])
            worker.home_mission = mission
            home.evidence[200, 200] = -1
            home.visits[200, 200] = 1
            home.scan_count = 1
            recorder.capture(worker)
            first = recorder.home_media[0][1]["cells"]
            home.evidence[200, 201] = 4
            home.scan_count = 2
            mission.stage = "loaded"
            recorder.capture(worker)
            assert first != recorder.home_media[0][1]["cells"]
            mission.task = {"task_id": "trial", "status": "failed", "reason": "LOCALIZATION_LOST", "segments": 1}
            recorder.capture(worker)
            assert (sim.seq, sim.ticks) == original
            worker.home_mission = None
        await worker.call(capture)
        score = recorder.finish({"evidence": "scripted_test", "real_model": False})
        rows = [json.loads(line) for line in (recorder.directory / "trajectory.jsonl").read_text().splitlines()]
        assert recorder.dropped == 1 and not score["spatial_recording"]["complete"]
        assert rows[-1]["home_events"][0]["reason"] == "LOCALIZATION_LOST"
        for row in rows:
            document = json.loads((recorder.directory / row["home"]["map"]["path"]).read_text())
            raw = zlib.decompress(base64.b64decode(document["data"]))
            cells = np.frombuffer(raw[:160000], dtype=np.int8).reshape(400, 400)
            assert cells[200, 200] == 0 and cells[200, 201] == 100
        assert "home" not in worker.latest["observation"]
        from backend.saved_results import saved_results, recorded_replay, recorded_replay_media
        batch_directory = recorder.directory.parent.parent
        (batch_directory / "experiment.json").write_text(json.dumps({"mode": "home_mapping", "evidence": "scripted_test",
            "cases": [{"case_id": "case", "challenge": "bench"}]}))
        (recorder.directory.parent / "report.json").write_text(json.dumps({"evidence": "scripted_test", "recording_scorecard": score}))
        batch = saved_results(tmp_path)["batches"][0]
        replay = recorded_replay(batch["id"], 0, tmp_path)
        assert replay["evaluation_only"] and not replay["recording_complete"]
        assert replay["events"][-1]["reason"] == "LOCALIZATION_LOST"
        filename = replay["frames"][-1]["map_url"].rsplit("/", 1)[1]
        mapped, content_type = recorded_replay_media(batch["id"], 0, filename, tmp_path)
        assert content_type == "application/json" and mapped["cells"][200 * 400 + 201] == 100
        assert mapped["visited_indices"] == [200 * 400 + 200]
        for name in ("../scorecard.json", "camera-999.png", "home-999.json"):
            with pytest.raises(ValueError):
                recorded_replay_media(batch["id"], 0, name, tmp_path)
        document_path = recorder.directory / "media" / filename
        document = json.loads(document_path.read_text())
        document["sha256"] = "corrupt"
        document_path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="Corrupt"):
            recorded_replay_media(batch["id"], 0, filename, tmp_path)
        rows[-1]["manual_placements"] = 1
        rows[-1]["collisions"] = ["contact"]
        rows[-1]["wall_s"] = rows[0]["wall_s"] + 2.
        (recorder.directory / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        replay = recorded_replay(batch["id"], 0, tmp_path)
        assert replay["frames"][-1]["segment"] > replay["frames"][0]["segment"]
        assert {"Contact detected", "Pose continuity changed", "Recording gap: 2.00 s"}.issubset({event["reason"] for event in replay["events"]})
        from unittest.mock import patch
        with patch("backend.saved_results.saved_results", side_effect=AssertionError("Replay must not scan all report contents")):
            assert recorded_replay(batch["id"], 0, tmp_path)["sample_count"] == 2
        with patch("backend.saved_results.result_directories", side_effect=AssertionError("Known replay must reuse its validated batch location")):
            assert recorded_replay(batch["id"], 0, tmp_path)["sample_count"] == 2
    finally:
        worker.home_mission = None
        await worker.close()


async def test_worker_recording_is_observational_and_bounded(tmp_path):
    from backend.worker import SimulationWorker
    from backend.challenges import get_challenge
    from backend.contracts import Command
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    recorder = RunRecorder(tmp_path / "run", max_pending=128)
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.sim.seq
        await worker.call(lambda sim: (setattr(worker, "recorder", recorder), recorder.capture(worker)))
        assert worker.sim.ticks == 0 and worker.sim.seq == initial
        await worker.execute(Command(run_id=worker.sim.run_id, episode_epoch=0, observation_seq=initial,
            action_id="recorded-drive", tool="drive_base", arguments={"linear_mps": .1, "angular_radps": 0., "duration_s": 1.}), assisted=False)
        worker.stop()
        await worker.hold_stopped()
        await worker.call(lambda sim: (recorder.capture(worker), setattr(worker, "recorder", None)))
        score = recorder.finish({"real_model": False, "label": "scripted recording test"})
        assert score["actual_distance_m"] > .07 and score["complete_recording"]
        assert not score["autonomous_success"]
        rows = [json.loads(line) for line in (recorder.directory / "trajectory.jsonl").read_text().splitlines()]
        assert len(rows) >= 20 and rows[-1]["simulated_s"] == 1
        assert any(row["linear_speed_mps"] > .08 for row in rows)
        assert "position_m" not in worker.latest["observation"]
        assert (recorder.directory / rows[0]["camera"]["path"]).exists()
        scene = json.loads((recorder.directory / "scene.json").read_text())
        assert scene["evaluation_only"] and scene["geometry"]
        assert scene["snapshot"]["simulated_time_s"] == 0
        assert scene["robot_body_id"] == worker.sim.robot
        assert "geometry" not in worker.latest["observation"]
    finally:
        await worker.close()


def test_video_frames_use_latest_past_camera_sample_and_reject_external_paths(tmp_path):
    from PIL import Image
    from scripts.export_recording import RecordingVideo
    (tmp_path / "manifest.json").write_text(json.dumps({"challenge": "park", "real_model": True}))
    (tmp_path / "scorecard.json").write_text(json.dumps({"final_physics_success": False, "complete_recording": True}))
    Image.new("RGB", (32, 24), "red").save(tmp_path / "first.png")
    Image.new("RGB", (32, 24), "blue").save(tmp_path / "last.png")
    samples = [{**sample(seconds, position=seconds), "yaw_rad": 0., "camera": {"path": filename}}
        for seconds, filename in ((0., "first.png"), (2., "last.png"))]
    (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in samples))
    recording = RecordingVideo(tmp_path)
    assert recording.frame(1.9).getpixel((50, 150)) == (255, 0, 0)
    assert recording.frame(2.).getpixel((50, 150)) == (0, 0, 255)
    assert recording.frame(100.).size == (1280, 720)
    with pytest.raises(ValueError, match="escapes"):
        recording.camera("../outside.png")
    (tmp_path / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in reversed(samples)))
    with pytest.raises(ValueError, match="ordered"):
        RecordingVideo(tmp_path)


def test_exploration_benchmark_records_timing_recovery_and_comparison_flags():
    from scripts.evaluate_supervised import argument_parser, execution_metrics
    options = argument_parser().parse_args(["--output", "unused", "--exploration", "legacy", "--repeats", "3", "--continuous-handoff"])
    assert options.exploration == "legacy" and options.repeats == 3 and options.continuous_handoff
    events = [
        {"kind": "policy", "title": "Exploration preparation", "payload": {"wall_s": 2.}},
        {"kind": "response", "title": "Decision", "payload": {"latency_s": 1.}},
        {"kind": "policy", "title": "Local action timing", "payload": {"action": "scan", "wall_s": 3.}},
        {"kind": "policy", "title": "No-progress recovery", "payload": {"replacement": {"action": "turn"}}},
        {"kind": "policy", "title": "No-progress recovery", "payload": {"replacement": None}},
    ]
    metrics = execution_metrics(events)
    assert metrics["preparation_wall_s"] == 2.
    assert metrics["inference_wall_s"] == 1.
    assert metrics["local_action_wall_s"] == {"scan": 3.}
    assert metrics["recovery_interventions"] == 1 and metrics["recovery_exhausted"]


def test_benchmark_accepts_every_scored_scenario_and_excludes_unscored_bench():
    from backend.challenges import PRESETS
    from scripts.evaluate_supervised import argument_parser, benchmark_cases
    identifiers = sorted(PRESETS)
    options = argument_parser().parse_args(["--output", "unused", "--challenges", *identifiers])
    cases = benchmark_cases(options.challenges)
    assert [name for name, _ in cases] == identifiers
    assert all(challenge.id == name and challenge.objectives for name, challenge in cases)
    assert all(challenge is not PRESETS[name] for name, challenge in cases)
    with pytest.raises(SystemExit):
        argument_parser().parse_args(["--output", "unused", "--challenges", "bench"])


def test_fixed_benchmark_cases_and_failed_run_denominator():
    from scripts.evaluate_supervised import benchmark_cases, summarize_benchmark
    cases = benchmark_cases(["kitchen_bathroom"], "kitchen-v1")
    assert len(cases) == 10 and len({tuple(challenge.initial_xy) for _, challenge in cases}) == 10
    assert benchmark_cases(["kitchen_bathroom"], "kitchen-v1", 3)[0][1].initial_xy == cases[3][1].initial_xy
    results = [{"physics_success": True, "recording_scorecard": {"completion_time_s": 100, "complete_recording": True}},
        {"physics_success": False, "recording_scorecard": {"completion_time_s": None}}]
    assert summarize_benchmark(results)["success_rate"] == .5
    assert summarize_benchmark(results)["successful_completion_median_s"] == 100
    assert summarize_benchmark(results, 4)["success_rate"] == .25
    assert summarize_benchmark(results, 4)["missing_reports"] == 2
    results[0]["recording_scorecard"]["operator_assisted"] = True
    assert summarize_benchmark(results)["successes"] == 0
    with pytest.raises(ValueError):
        benchmark_cases(["park"], "kitchen-v1")


async def test_docking_benchmark_uses_camera_target_and_requires_stopped_containment(monkeypatch):
    from types import SimpleNamespace
    from scripts.audit_architecture import DOCKING_CASES, SensorDockingPolicy, docking_score
    import backend.navigation_supervisor as supervisor
    assert len(DOCKING_CASES) == 6 and len({case["heading_rad"] for case in DOCKING_CASES}) > 2
    monkeypatch.setattr(supervisor, "marker_target", lambda observation, image, color: [1., .2])
    policy = SensorDockingPolicy()
    result = await policy.predict(SimpleNamespace(odometry_m_rad=[0., 0., 0.]), b"camera")
    assert result["action"][0] > 0 and result["action"][1] > 0
    stopped = await policy.predict(SimpleNamespace(odometry_m_rad=[1., .2, 0.]), b"camera")
    assert stopped["action"] == [0., 0.]
    final = {"parked": True, "position": [1., .2, .15]}
    requests = [{"status": "ok", "execution_status": "awaiting_feedback"}]
    assert docking_score(final, [1., .2], True, [0., 0.], requests)["success"]
    assert not docking_score(final, [1., .2], False, [0., 0.], requests)["success"]
    assert not docking_score(final, [1., .2], True, [.1, 0.], requests)["success"]
    assert not docking_score(final, [1., .2], True, [0., 0.], [{"status": "error"}])["success"]


async def test_waypoint_docking_passes_sensor_derived_goal_and_preserves_model_actions(monkeypatch):
    from scripts.audit_architecture import WaypointDockingPolicy
    from tests.test_agent import observation
    import backend.navigation_supervisor as supervisor
    monkeypatch.setattr(supervisor, "marker_target", lambda observation, image, color: [1., .2])
    class Client:
        async def predict(self, observation, image):
            assert image == b"paired-head"
            return {"action": [.1, .2], "raw_action": [.1, .2], "saturated_axes": []}
    client = Client()
    policy = WaypointDockingPolicy(client)
    initial = observation().model_copy(update={"odometry_m_rad": [0., 0., 0.]})
    first = await policy.predict(initial, b"paired-head")
    assert first["action"] == [.1, .2] and first["observed_target_m"] == [1., .2]
    assert "forward +1.00 m, left +0.20 m" in client.instruction
    await policy.predict(initial.model_copy(update={"odometry_m_rad": [.5, .1, 0.]}), b"paired-head")
    assert "forward +0.50 m, left +0.10 m" in client.instruction
    with pytest.raises(ValueError, match="another episode"):
        await policy.predict(initial.model_copy(update={"episode_epoch": initial.episode_epoch + 1}), b"paired-head")


def test_audit_primitive_postconditions_reject_wrong_direction_and_no_effect():
    from scripts.audit_architecture import primitive_pass, distribution
    assert primitive_pass("forward", [0, 0, 0], [.2, 0, 0])
    assert not primitive_pass("forward", [0, 0, 0], [-.2, 0, 0])
    assert not primitive_pass("forward", [0, 0, 0], [0, 0, 0])
    assert primitive_pass("left", [0, 0, 0], [0, 0, .5])
    assert primitive_pass("hold", [0, 0, 0], [0, 0, 0])
    assert distribution([1, 2, 3])["median"] == 2


def test_audit_summary_retains_late_stop_feedback_and_failed_trial(tmp_path):
    from scripts.audit_architecture import summarize_artifacts
    batch = tmp_path / "batch"
    trial = batch / "trial"
    trial.mkdir(parents=True)
    (batch / "manifest.json").write_text(json.dumps({"checkpoint": "test-checkpoint"}))
    (trial / "report.json").write_text(json.dumps({"source": "smolvla", "success": False}))
    (trial / "actions.json").write_text(json.dumps([{"event": "accepted", "command": {
        "arguments": {"segments": [{"kind": "drive"}]}}, "result": {"error": None}}]))
    feedback = {"title": "SmolVLA motion feedback", "kind": "feedback", "payload": {
        "observation": {"navigation": {"revision": 7, "reason": "CLEARANCE_STOP"}}}}
    (trial / "trace.json").write_text(json.dumps([feedback, feedback]))
    summarize_artifacts(tmp_path)
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == 1 and rows[0]["success"] is False
    assert rows[0]["checkpoint"] == "test-checkpoint"
    assert rows[0]["drive_accepted"] == 1 and rows[0]["sampled_nonzero_drive_sim_s"] == 0
    assert rows[0]["observed_navigation_reasons_by_revision"] == {"CLEARANCE_STOP": 1}


async def test_dropped_recording_retains_referenced_media_and_is_not_scored_complete(tmp_path):
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=False)
    recorder = RunRecorder(tmp_path / "overflow", max_pending=1)
    try:
        await asyncio.wrap_future(worker.ready)
        for _ in range(3):
            await worker.call(lambda sim: recorder.capture(worker))
        score = recorder.finish({"real_model": True, "label": "<script>unsafe text</script>"})
        assert score["dropped_records"] == 2 and not score["complete_recording"]
        assert score["completion_time_s"] is None and not score["autonomous_success"]
        row = json.loads((recorder.directory / "trajectory.jsonl").read_text())
        assert (recorder.directory / row["camera"]["path"]).exists()
        replay = (recorder.directory / "replay.html").read_text()
        assert "<script>unsafe text</script>" not in replay and "__RECORDING_DATA__" not in replay
    finally:
        await worker.close()


@pytest.mark.parametrize("ending", ["save_map", "stop", "reset", "disconnect", "reject", "agent/takeover"])
def test_operator_home_sessions_record_all_endings(tmp_path, monkeypatch, ending):
    from importlib import import_module
    from fastapi.testclient import TestClient
    from backend.app import app, lab
    from backend import session_recording, saved_results, home_mission
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(import_module("backend.app"), "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    monkeypatch.setattr(saved_results, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(home_mission, "DEFAULT_MAP_PATH", tmp_path / "maps.sqlite3")
    with TestClient(app) as client:
        if ending == "stop":
            assert client.post("/api/preferences", json={"recording_directory": str(tmp_path / "Custom map recordings")}).status_code == 200
        state = client.get("/api/state").json()
        identity = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"]}
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            response = client.post("/api/home", json={**identity, "action": "navigate_to" if ending == "reject" else "start_mapping", "place_id": "unknown"})
            assert response.status_code == (409 if ending == "reject" else 200), response.text
            if ending == "save_map":
                assert client.post("/api/home", json={**identity, "action": "save_map", "name": "Recorded map"}).status_code == 200
            elif ending in {"stop", "reset", "agent/takeover"}:
                assert client.post("/api/" + ending).status_code == 200
        assert lab.worker.recorder is None
        if ending == "stop":
            status = client.get("/api/recording").json()
            assert not status["active"] and status["run_directory"].startswith(str(tmp_path / "Custom map recordings"))
            assert not (tmp_path / "performance").exists()
            assert client.post("/api/preferences", json={"recording_directory": ""}).status_code == 200
        batches = client.get("/api/test-results").json()["batches"]
        assert len(batches) == 1 and batches[0]["evidence"] == "operator_session"
        trial = batches[0]["trials"][0]
        assert not trial["verified_success"]
        assert trial["recording_complete"]
        replay = client.get(trial["replay_url"])
        assert replay.status_code == 200, replay.text
        assert replay.json()["sample_count"] > 0
        if ending != "reject":
            assert any(frame["map_url"] for frame in replay.json()["frames"])
            filename = next(frame["map_url"] for frame in replay.json()["frames"] if frame["map_url"])
            assert client.get(filename).status_code == 200
        assert client.get(trial["replay_url"] + "/media/home-9999.json").status_code == 404


async def test_operator_recording_writer_failure_stops_and_retains_error(tmp_path, monkeypatch):
    from backend import session_recording
    from backend.home_mission import HomeRequest
    from backend.worker import SimulationWorker
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    worker = SimulationWorker(pace=False, rendering="tiny")
    session = None
    try:
        await asyncio.wrap_future(worker.ready)
        session = session_recording.HomeSessionRecording(worker, HomeRequest(run_id=worker.sim.run_id, episode_epoch=0, action="start_mapping"))
        await session.start()
        session.executing = False
        def fail_flush():
            raise OSError("Injected persistent recording failure")
        monkeypatch.setattr(session.recorder, "flush", fail_flush)
        await asyncio.wait_for(session.writer, 5)
        assert session.finished and worker.sim.cancel.is_set() and worker.recorder is None
        report = json.loads((session.directory / "bench" / "report.json").read_text())
        assert "Injected persistent" in report["recording_error"]
        assert not report["recording_scorecard"]["complete_recording"]
        assert report["termination_reason"] == "recording_error"
    finally:
        if session and not session.finished:
            await session.finish("test_cleanup")
        await worker.close()


def test_stop_during_recording_start_cannot_authorize_mapping_after_resume(tmp_path, monkeypatch):
    from importlib import import_module
    from fastapi.testclient import TestClient
    from backend.app import app, lab
    from backend import session_recording, home_mission
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(import_module("backend.app"), "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(home_mission, "DEFAULT_MAP_PATH", tmp_path / "maps.sqlite3")
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    original = session_recording.HomeSessionRecording.start
    async def interrupted_start(self):
        await original(self)
        self.worker.stop()
        await self.worker.resume_manual()
    monkeypatch.setattr(session_recording.HomeSessionRecording, "start", interrupted_start)
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            response = client.post("/api/home", json={"action": "start_mapping", "run_id": state["run_id"], "episode_epoch": state["episode_epoch"]})
            assert response.status_code == 409 and "invalidated" in response.text
            assert lab.worker.home_mission.home is None
            assert lab.worker.recorder is None