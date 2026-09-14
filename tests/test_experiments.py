from types import SimpleNamespace

from backend.experiment_variants import variant_snapshot


def test_variant_versions_are_stable_and_separate_implementation_from_model(tmp_path):
    backend = tmp_path / "backend"
    backend.mkdir()
    implementation = backend / "controller.py"
    implementation.write_text("version = 1\n", encoding="utf-8")
    settings = SimpleNamespace(execution_mode="luna_continuous", reasoning="high", goal="First task")
    profile = SimpleNamespace(id="luna", provider="foundry", deployment="deployment-a")
    first = variant_snapshot(settings, profile, tmp_path)
    assert first == variant_snapshot(settings, profile, tmp_path)
    settings.goal = "Another task"
    assert first == variant_snapshot(settings, profile, tmp_path)
    profile.deployment = "deployment-b"
    second = variant_snapshot(settings, profile, tmp_path)
    assert first["architecture"] == second["architecture"]
    assert first["model_variant"]["revision"] != second["model_variant"]["revision"]
    implementation.write_text("version = 2\n", encoding="utf-8")
    third = variant_snapshot(settings, profile, tmp_path)
    assert second["architecture"]["revision"] != third["architecture"]["revision"]
    assert second["model_variant"] == third["model_variant"]
    assert not third["model_variant"]["underlying_identity_verified"]
    configured = variant_snapshot(settings, profile, tmp_path, {"camera_history": "disabled", "max_turns": None})
    assert configured["architecture"] == third["architecture"]
    assert configured["model_variant"]["revision"] != third["model_variant"]["revision"]
    settings.ai_generated_routes = True
    assert variant_snapshot(settings, profile, tmp_path)["architecture"]["id"] == "ai-waypoints"
    settings.navigation_backend = "nav2"
    ros = tmp_path / "ros"
    ros.mkdir()
    configuration = ros / "nav2.yaml"
    configuration.write_text("controller_frequency: 10\n", encoding="utf-8")
    nav2 = variant_snapshot(settings, profile, tmp_path)
    assert "ros/nav2.yaml" in nav2["code_sha256"]
    configuration.write_text("controller_frequency: 20\n", encoding="utf-8")
    changed = variant_snapshot(settings, profile, tmp_path)
    assert changed["architecture"]["revision"] != nav2["architecture"]["revision"]
    assert changed["model_variant"] == nav2["model_variant"]
    settings.navigation_backend = "builtin"
    assert "ros/nav2.yaml" not in variant_snapshot(settings, profile, tmp_path)["code_sha256"]


def test_live_run_is_archived_provisionally_with_version_and_finalizes(tmp_path):
    import json
    import time
    from backend.saved_results import saved_results
    from backend.session_recording import publish_live_report
    directory = tmp_path / "performance/model-run"
    (directory / "park").mkdir(parents=True)
    variant = variant_snapshot(SimpleNamespace(execution_mode="luna_continuous"), SimpleNamespace(deployment="luna"), tmp_path)
    manifest = {"schema_version": 2, "stage": "challenges", "evidence": "real_model", "session_id": "live-session",
        "cases": [{"case_id": "park", "challenge": "park", "environment": "standalone", "challenge_sha256": "a" * 64}],
        "architecture": variant["architecture"], "model_variant": variant["model_variant"], "variant_id": variant["variant_id"]}
    (directory / "experiment.json").write_text(json.dumps(manifest), encoding="utf-8")
    recorder = SimpleNamespace(clock=time.monotonic, started=time.monotonic() - 1, dropped=0, sample_sequence=5)
    publish_live_report(directory, recorder, manifest, {"phase": "thinking", "turns": 1, "input_tokens": 12}, {"title": "Park"})
    batch = saved_results(tmp_path)["batches"][0]
    assert batch["running"] and batch["session_id"] == "live-session"
    assert batch["architecture"]["version_key"] == variant["architecture"]["version_key"]
    assert batch["trials"][0]["status"] == "Running" and not batch["trials"][0]["verified_success"]
    path = directory / "park/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["updated_at"] = time.time() - 30
    path.write_text(json.dumps(report), encoding="utf-8")
    assert saved_results(tmp_path)["batches"][0]["trials"][0]["status"] == "Interrupted recording"
    report.update(run_status="finished", verification_eligible=True, physics_success=True,
        recording_scorecard={"complete_recording": True, "completion_time_s": 10.})
    path.write_text(json.dumps(report), encoding="utf-8")
    final = saved_results(tmp_path)["batches"][0]
    assert not final["running"] and final["successes"] == 1
    report.update(initial_goal="Original goal", final_goal="Custom revised goal", verification_eligible=False)
    path.write_text(json.dumps(report), encoding="utf-8")
    revised = saved_results(tmp_path)["batches"][0]
    assert revised["trials"][0]["task_sha256"] != final["trials"][0]["task_sha256"]
    assert revised["successes"] == 0


def test_live_reports_share_reader_writer_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from backend import recording
    from backend.saved_results import read_json
    original_read = Path.read_text
    original_replace = Path.replace
    operations = []

    def checked_read(path, *args, **kwargs):
        assert recording.recording_json_lock.locked()
        operations.append("read")
        return original_read(path, *args, **kwargs)

    def checked_replace(path, target):
        assert recording.recording_json_lock.locked()
        operations.append("replace")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "read_text", checked_read)
    monkeypatch.setattr(Path, "replace", checked_replace)
    path = tmp_path / "report.json"
    recording.write_recording_json(path, {"run_status": "running"})
    assert read_json(path, tmp_path) == {"run_status": "running"}
    recording.write_recording_json(path, {"run_status": "finished"})
    assert read_json(path, tmp_path) == {"run_status": "finished"}
    assert operations == ["replace", "read", "replace", "read"]
    assert not recording.recording_json_lock.locked()


def test_reference_paths_are_optional_operator_only_and_match_circle_direction():
    import math
    from backend.challenges import furniture_circuit, get_challenge
    from backend.reference_paths import reference_path
    assert reference_path(None) is None
    assert reference_path(get_challenge("apartment")) is None
    for direction, expected_sign in (("clockwise", -1), ("counterclockwise", 1)):
        challenge = furniture_circuit("table", direction)
        reference = reference_path(challenge)
        assert reference["evaluation_only"] and not reference["optimal"]
        assert reference["points"][0] == challenge.initial_xy
        assert len(reference["points"]) == 122
        first, second = reference["points"][1:3]
        first_angle = math.atan2(first[1] - challenge.orbit.center_m[1], first[0] - challenge.orbit.center_m[0])
        second_angle = math.atan2(second[1] - challenge.orbit.center_m[1], second[0] - challenge.orbit.center_m[0])
        turn = math.atan2(math.sin(second_angle-first_angle), math.cos(second_angle-first_angle))
        assert turn * expected_sign > 0
        assert "reference" not in challenge.public()


def test_reference_endpoint_is_read_only_and_not_in_agent_observations(monkeypatch):
    from fastapi.testclient import TestClient
    import backend.app as application
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(application, "load_dotenv", lambda *args, **kwargs: None)
    with TestClient(application.app) as client:
        state = client.post("/api/challenges/load", json={"challenge_id": "furniture_circuit"}).json()
        route = client.get("/api/reference-path", params={"run_id": state["run_id"]})
        assert route.status_code == 200 and route.json()["reference"]["evaluation_only"]
        assert client.get("/api/reference-path", params={"run_id": "old-episode"}).status_code == 409
        after = client.get("/api/state").json()
        assert after["snapshot"] == state["snapshot"] and after["agent"]["active"] is False
        assert "reference" not in str(after["observation"])
        preview = client.get("/api/test-variant")
        assert preview.status_code == 200
        assert preview.json()["architecture"]["id"] == "observed-continuous"
        assert preview.json()["supports_ai_generated_routes"] is False
        assert client.get("/api/test-variant", params={"execution_mode": "invalid"}).status_code == 422