from fastapi.testclient import TestClient

from backend.app import app, lab


import pytest


@pytest.fixture(autouse=True)
def isolate_foundry_environment(monkeypatch):
    from importlib import import_module
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(import_module("backend.app"), "load_dotenv", lambda *args, **kwargs: None)
    for name in ("FOUNDRY_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT", "project_endpoint", "FOUNDRY_MODELS_JSON",
                 "FOUNDRY_DEPLOYMENT", "deployment_name", "AZURE_AI_MODEL_DEPLOYMENT_NAME",
                 "FOUNDRY_REALTIME_ENDPOINT", "FOUNDRY_REALTIME_DEPLOYMENT", "FOUNDRY_REALTIME_VOICE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("challenge_id", ["bench", "park"])
def test_workspace_preferences_restore_scene_stopped_without_commands(challenge_id):
    from backend.preferences import PreferenceStore
    with TestClient(app) as client:
        before = client.get("/api/state").json()
        response = client.post("/api/preferences", json={"turns": 21, "inspector": "settings", "handoff": False,
            "challenge_selection": {"challenge_id": "tidy"}})
        assert response.status_code == 200
        assert client.get("/api/preferences").headers["cache-control"] == "no-store"
        after = client.get("/api/state").json()
        assert after["run_id"] == before["run_id"] and after["snapshot"] == before["snapshot"]
        assert not after["agent"]["active"]
        assert client.post("/api/preferences", json={"active": True}).status_code == 422
        assert client.post("/api/preferences", json={"turns": 20}, headers={"Origin": "https://untrusted.example"}).status_code == 403
        loaded = client.post("/api/challenges/load", json={"challenge_id": challenge_id, "reuse_saved_map": False})
        assert loaded.status_code == 200
        assert PreferenceStore().read()["scene"]["challenge_id"] == challenge_id
    with TestClient(app) as client:
        restored = client.get("/api/state").json()
        assert (restored["challenge"]["id"] if restored["challenge"] else "bench") == challenge_id
        assert restored["stopped"]
        assert restored["snapshot"]["simulated_time_s"] == 0
        assert not restored["agent"]["active"] and not restored["agent"]["auto_wake"] and not restored["busy"]
        assert restored["map_setup"]["reuse_saved_map"] is False
        preferences = client.get("/api/preferences").json()["preferences"]
        assert preferences["turns"] == 21 and preferences["inspector"] == "settings"
        assert preferences["challenge_selection"]["challenge_id"] == "tidy"


def test_knowledge_profiles_checkpoint_restart_and_obsolete_api_context():
    with TestClient(app) as client, client.websocket_connect("/api/live") as socket:
        initial = socket.receive_json()
        memory = client.get("/api/memory").json()
        def command(action, state=memory, **values):
            scope = state["scope"]
            return client.post("/api/memory", json={"run_id": scope["run_id"], "episode_epoch": scope["episode_epoch"],
                "context_id": scope["context_id"], "action": action, **values})
        assert memory["enabled"] and memory["scope"]["environment_id"] != "standalone:bench"
        profile_id = memory["scope"]["profile_id"]
        fresh = command("fresh_profile", name="Isolated fixture knowledge")
        assert fresh.status_code == 200
        fresh = fresh.json()
        assert fresh["scope"]["profile_id"] != profile_id and not fresh["rooms"]
        assert command("fresh_profile", name="Delayed").status_code == 409
        scope = fresh["scope"]
        identity = {"run_id": scope["run_id"], "episode_epoch": scope["episode_epoch"]}
        assert client.post("/api/home", json={**identity, "action": "start_mapping"}).status_code == 200
        assert client.post("/api/home", json={**identity, "action": "save_map", "name": "API fixture map"}).status_code == 200
        for label in ("Fixture kitchen", "Fixture hall"):
            report = command("record_observation", fresh, kind="room", label=label, description="Operator-supplied fixture label on current sensor evidence")
            assert report.status_code == 200, report.text
        recorded = client.get("/api/memory").json()
        assert len(recorded["rooms"]) == 2
        snapshot = command("checkpoint", fresh, name="Two fixture rooms")
        assert snapshot.status_code == 200, snapshot.text
        checkpoint_id = snapshot.json()["snapshots"][0]["snapshot_id"]
        fork = command("fork_checkpoint", fresh, snapshot_id=checkpoint_id, name="Fixture fork")
        assert fork.status_code == 200, fork.text
        fork = fork.json()
        assert fork["scope"]["profile_id"] != fresh["scope"]["profile_id"] and len(fork["rooms"]) == 2
        assert client.get("/api/state").json()["snapshot"] == initial["snapshot"]
        persisted_profile = fork["scope"]["profile_id"]
        persisted_map = fork["scope"]["map_id"]
    with TestClient(app) as client:
        restored = client.get("/api/memory").json()
        assert restored["scope"]["profile_id"] == persisted_profile
        assert restored["scope"]["map_id"] == persisted_map and len(restored["rooms"]) == 2
        assert restored["scope"]["context_id"] != fork["scope"]["context_id"]
        assert client.get("/api/home").json()["localization"]["status"] == "unlocalized"


def test_robot_power_preserves_scene_blocks_work_and_resumes_only_to_idle():
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        identity = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"]}
        off = client.post("/api/power", json={**identity, "on": False})
        assert off.status_code == 200
        assert off.json()["power"]["mode"] == "off" and off.json()["stopped"]
        assert client.post("/api/resume").status_code == 409
        assert client.post("/api/spatial", json={**identity, "enabled": True}).status_code == 409
        assert client.post("/api/agent/start", json={**identity, "goal": "No work while off"}).status_code == 409
        assert client.post("/api/power", json={**identity, "run_id": "old", "on": True}).status_code == 409
        assert client.post("/api/power", json={**identity, "on": True}, headers={"Origin": "https://untrusted.example"}).status_code == 403
        on = client.post("/api/power", json={**identity, "on": True}).json()
        assert on["power"]["mode"] == "idle" and not on["agent"]["active"]
        assert on["run_id"] == initial["run_id"] and on["snapshot"] == initial["snapshot"]
        assert not on["agent"]["auto_wake"]
        assert client.post("/api/power", json={**identity, "on": False}).status_code == 200
        reset = client.post("/api/reset").json()
        assert reset["power"]["mode"] == "off" and reset["stopped"]
        assert client.post("/api/power", json={**identity, "on": True}).status_code == 409
        assert client.post("/api/power", json={"run_id":reset["run_id"], "episode_epoch":reset["episode_epoch"], "on":True}).json()["power"]["mode"] == "idle"


def test_recording_folder_settings_are_checked_persisted_and_idle_only(tmp_path):
    from backend.app import lab
    from backend.preferences import PreferenceStore
    directory = tmp_path / "Recorded runs"
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        assert client.get("/api/recording").json()["enabled"]
        response = client.post("/api/preferences", json={"recording_enabled": False, "recording_directory": str(directory)})
        assert response.status_code == 200, response.text
        assert directory.is_dir() and not list(directory.iterdir())
        status = client.get("/api/recording").json()
        assert status["directory"] == str(directory) and status["status"] == "off" and not status["active"]
        assert not lab.agent.record_sessions and lab.agent.recording_root == directory
        assert client.get("/api/state").json()["snapshot"] == initial["snapshot"]
        blocked = tmp_path / "not-a-folder"
        blocked.write_text("preserve")
        assert client.post("/api/preferences", json={"recording_directory": str(blocked)}).status_code == 400
        assert client.get("/api/recording").json()["directory"] == str(directory)
        lab.agent.recording_active = True
        try:
            assert client.post("/api/preferences", json={"recording_enabled": True}).status_code == 409
        finally:
            lab.agent.recording_active = False
        assert client.post("/api/preferences", json={"recording_directory": "//server/share"}).status_code == 422
        assert client.post("/api/preferences", json={"recording_enabled": True}, headers={"Origin":"https://other.example"}).status_code == 403
        assert PreferenceStore().recording_directories() == [str(directory)]
    with TestClient(app) as client:
        status = client.get("/api/recording").json()
        assert not status["enabled"] and status["directory"] == str(directory)
        assert not lab.agent.record_sessions


def test_corrupt_preferences_do_not_replace_store_or_prevent_startup(tmp_path):
    path = tmp_path / "preferences.sqlite3"
    path.write_bytes(b"retained invalid preferences")
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        assert state["challenge"] is None and state["preference_error"]
        assert client.get("/api/preferences").status_code == 503
        assert client.post("/api/preferences", json={"turns": 12}).status_code == 503
        assert path.read_bytes() == b"retained invalid preferences"


@pytest.mark.parametrize("ending", ["stop", "agent/takeover", "reset", "disconnect"])
def test_browser_session_recording_survives_api_lifecycle(tmp_path, monkeypatch, ending):
    import time
    from backend import session_recording, saved_results
    from tests.test_agent import ScriptedModel, model_response
    monkeypatch.setattr(session_recording, "SESSION_RESULTS_ROOT", tmp_path / "performance")
    monkeypatch.setattr(saved_results, "RESULTS_ROOT", tmp_path)
    with TestClient(app) as client:
        state = client.post("/api/challenges/load", json={"challenge_id": "park"}).json()
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com", "models": [
            {"id": "luna", "label": "Scripted test", "deployment": "test-deployment"}]})
        lab.agent.model_factory = lambda config: ScriptedModel([model_response(arguments='{"linear_mps":0.1,"angular_radps":0,"duration_s":0.5}'), "wait"])
        lab.agent.recording_evidence = "scripted_test"
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            started = client.post("/api/agent/start", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                "goal": "Scripted recording lifecycle", "max_turns": 2, "feedback_interval_s": .25})
            assert started.status_code == 200
            deadline = time.monotonic() + 10
            while client.get("/api/state").json()["snapshot"]["simulated_time_s"] < .5:
                assert time.monotonic() < deadline
                time.sleep(.02)
            if ending != "disconnect":
                assert client.post(f"/api/{ending}").status_code == 200
        deadline = time.monotonic() + 10
        while True:
            data = client.get("/api/test-results").json()["batches"]
            if data and data[0]["trials"][0]["recording_complete"]:
                break
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert len(data) == 1 and data[0]["evidence"] == "scripted_test"
        trial = data[0]["trials"][0]
        assert trial["termination"] == "interrupted" and trial["distance_m"] > .02
        route = client.get(trial["trajectory_url"])
        assert route.status_code == 200 and route.json()["scene"]["source"] == "recorded_initial"
        assert client.get(trial["image_url"]).status_code == 200
        assert lab.worker.recorder is None and not lab.agent.recording_active


def test_saved_test_results_are_read_only_private_and_support_legacy_reports(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from backend import saved_results
    root = tmp_path / "runtime"
    directory = root / "past-run"
    (directory / "park").mkdir(parents=True)
    (directory / "experiment.json").write_text(json.dumps({"mode": "luna_continuous", "stage": "challenges",
        "cases": [{"case_id": "park", "challenge": "park"}, {"case_id": "missing", "challenge": "park"}], "secret": "not-public"}))
    (directory / "summary.json").write_text(json.dumps([{"case_id": "park", "challenge": "park", "real_luna": True,
        "physics_success": True, "input_tokens": 10, "secret": "not-public", "recording_scorecard": {
            "complete_recording": True, "completion_time_s": 12., "contact_episodes": 0}}]))
    Image.new("RGB", (160, 120), "red").save(directory / "park/terminal.png")
    (directory / "park/recording").mkdir()
    (directory / "park/recording/trajectory.jsonl").write_text("\n".join(json.dumps({
        "position_m": [-2 + index / 1000, .5, .1], "wall_s": index / 20, "collisions": ["contact"] if index == 7 else [],
        "manual_placements": int(index >= 10), "physics_status": "completed" if index == 3000 else "in_progress",
        "activity": "driving", "secret": "not-public"}) for index in range(3001)))
    (root / "older").mkdir()
    (root / "older/experiment.json").write_text(json.dumps({"mode": "luna_continuous", "challenges": ["park", "apartment"]}))
    (root / "older/summary.json").write_text(json.dumps([{"challenge": "park", "physics_success": True, "real_luna": True}]))
    (root / "broken").mkdir()
    (root / "broken/experiment.json").write_text("{")
    (root / "performance").mkdir()
    (root / "performance/index.jsonl").write_text(json.dumps({"output": str(tmp_path.parent)}) + "\ninvalid")
    monkeypatch.setattr(saved_results, "RESULTS_ROOT", root)
    with TestClient(app) as client:
        before = client.get("/api/state").json()
        connections = lab.connections
        response = client.get("/api/test-results")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        data = response.json()
        assert len(data["batches"]) == 2 and data["skipped"] == 1 and "not-public" not in response.text
        older = next(batch for batch in data["batches"] if batch["name"] == "older")
        assert older["planned"] == 2 and older["successes"] == 0 and older["trials"][0]["status"] == "Unverified pass"
        assert older["trials"][1]["status"] == "Missing report"
        batch = next(batch for batch in data["batches"] if batch["name"] == "past-run")
        assert batch["legacy"] and batch["planned"] == 2 and batch["successes"] == 1
        assert batch["trials"][1]["status"] == "Missing report"
        image = client.get(batch["trials"][0]["image_url"])
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        route = client.get(batch["trials"][0]["trajectory_url"])
        assert route.status_code == 200 and "not-public" not in route.text
        assert route.headers["cache-control"] == "no-store" and route.json()["evaluation_only"]
        trajectory = route.json()
        assert trajectory["sample_count"] == 3001 and trajectory["downsampled"] and len(trajectory["points"]) <= 1501
        assert trajectory["points"][0]["x"] == -2 and trajectory["points"][-1]["x"] == 1
        assert trajectory["points"][-1]["wall_s"] == 150 and trajectory["points"][-1]["status"] == "completed"
        assert trajectory["points"][0]["segment"] == 0 and trajectory["points"][-1]["segment"] == 1
        assert trajectory["contacts"][0]["wall_s"] == .35 and trajectory["bounds_m"] == [-2, .5, 1., .5]
        assert trajectory["scene"]["source"] == "reconstructed_current"
        assert trajectory["scene"]["evaluation_only"] and trajectory["scene"]["geometry"]
        assert batch["trials"][1]["trajectory_url"] is None
        assert client.get(f"/api/test-results/{batch['id']}/trajectories/-1").status_code == 404
        assert client.get("/api/test-results/unknown/trajectories/0").status_code == 404
        assert client.get(batch["trials"][0]["trajectory_url"], headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.get(f"/api/test-results/{batch['id']}/images/-1").status_code == 404
        assert client.get("/api/test-results/unknown/images/0").status_code == 404
        assert client.get("/api/test-results", headers={"Origin": "https://untrusted.example"}).status_code == 403
        after = client.get("/api/state").json()
        assert after["run_id"] == before["run_id"] and after["snapshot"] == before["snapshot"] and lab.connections == connections
    assert saved_results.saved_results(tmp_path / "empty")["batches"] == []


def test_saved_results_index_partial_reports_and_path_boundaries(tmp_path, monkeypatch):
    import json
    from backend import saved_results
    root = tmp_path / "runtime"
    nested = root / "performance/batches/deep/run"
    (nested / "park").mkdir(parents=True)
    (nested / "experiment.json").write_text(json.dumps({"schema_version": 2, "stage": "challenges", "mode": "reference",
        "design": {"label": "control-v2", "source_sha256": "a" * 64}, "evidence": "scripted_reference",
        "started_at": "2026-09-13T12:00:00Z", "cases": [{"case_id": "park", "challenge": "park"},
            {"case_id": "../outside", "challenge": "park"}]}))
    (nested / "park/report.json").write_text(json.dumps({"physics_success": True, "manual_placements": 1,
        "input_tokens": float("nan"), "recording_scorecard": {"complete_recording": True, "completion_time_s": 5.}}))
    (root / "performance/index.jsonl").write_text(json.dumps({"output": str(nested)}) + "\n" + json.dumps({"output": str(tmp_path)}))
    data = saved_results.saved_results(root)
    assert len(data["batches"]) == 1
    batch = data["batches"][0]
    assert batch["design"] == "control-v2" and not batch["legacy"] and batch["date_source"] == "recorded"
    assert batch["evidence"] == "scripted_reference" and batch["successes"] == 0
    assert batch["trials"][0]["status"] == "Assisted pass" and batch["trials"][0]["input_tokens"] is None
    assert batch["trials"][1]["image_url"] is None
    assert not saved_results.local_file(tmp_path / "outside.json", root)
    with pytest.raises(ValueError, match="unavailable"):
        saved_results.terminal_image(batch["id"], 1, root)
    monkeypatch.setattr(saved_results, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="Unavailable"):
        saved_results.read_json(nested / "experiment.json", root)


@pytest.mark.parametrize("content", ["", "{", "[]", '{"position_m":[NaN,0],"wall_s":0}',
    '{"position_m":[0,0],"wall_s":-1}', '{"position_m":[0,0],"wall_s":1}\n{"position_m":[0,0],"wall_s":0}'])
def test_saved_trajectory_rejects_invalid_records(tmp_path, monkeypatch, content):
    import hashlib
    import json
    from backend import saved_results
    directory = tmp_path / "run"
    (directory / "park/recording").mkdir(parents=True)
    (directory / "experiment.json").write_text(json.dumps({"cases": [{"case_id": "park", "challenge": "park"}]}))
    path = directory / "park/recording/trajectory.jsonl"
    path.write_text(content)
    identifier = hashlib.sha256(b"run").hexdigest()[:20]
    with pytest.raises(ValueError):
        saved_results.recorded_trajectory(identifier, 0, tmp_path)
    path.write_text('{"position_m":[0,0],"wall_s":0}')
    assert len(saved_results.recorded_trajectory(identifier, 0, tmp_path)["points"]) == 1
    monkeypatch.setattr(saved_results, "MAX_TRAJECTORY_BYTES", 1)
    with pytest.raises(ValueError):
        saved_results.recorded_trajectory(identifier, 0, tmp_path)


def test_saved_scene_uses_recorded_transforms_and_rejects_unsafe_data(tmp_path):
    import json
    from backend.saved_results import trajectory_scene
    scene = {"schema_version": 1, "coordinate_frame": "recorded_world_xy_m", "robot_body_id": 7,
        "snapshot": {"simulated_time_s": 0, "poses": [
            {"key": "1:-1", "position": [-2, 3, .5], "quaternion": [0, 0, 1, 0]},
            {"key": "7:-1", "position": [0, 0, 0], "quaternion": [0, 0, 0, 1]}]},
        "geometry": [{"key": "1:-1", "type": 3, "dimensions": [1, 2, 1], "color": [.5, .5, .5, 1],
            "position": [.1, 0, 0], "quaternion": [0, 0, 0, 1], "name": "table", "texture": "https://untrusted.example/secret.png", "secret": "private"},
            {"key": "7:-1", "name": "robot"}], "secret": "private"}
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    result = trajectory_scene(tmp_path, "furniture_circuit", tmp_path)
    assert result["source"] == "recorded_initial" and result["evaluation_only"]
    assert len(result["geometry"]) == 1 and result["geometry"][0]["texture"] is None
    assert result["poses"][0]["position"] == [-2, 3, .5] and result["poses"][0]["quaternion"] == [0, 0, 1, 0]
    assert "private" not in json.dumps(result) and "untrusted" not in json.dumps(result)
    scene["geometry"][0]["position"][0] = float("nan")
    path.write_text(json.dumps(scene))
    assert trajectory_scene(tmp_path, "furniture_circuit", tmp_path) is None
    path.write_text("{")
    assert trajectory_scene(tmp_path, "furniture_circuit", tmp_path) is None
    path.unlink()
    rebuilt = trajectory_scene(tmp_path, "furniture_circuit", tmp_path)
    assert rebuilt["source"] == "reconstructed_current"
    assert any("table" in asset["name"] for asset in rebuilt["geometry"])
    assert trajectory_scene(tmp_path, "unknown-scenario", tmp_path) is None
    shared = trajectory_scene(tmp_path, "apartment", tmp_path, "shared_apartment_v1")
    assert shared["source"] == "reconstructed_current"
    assert any(asset["name"] == "charger_pad" for asset in shared["geometry"])
    assert not any(asset["name"] == "charger_pad" for asset in trajectory_scene(tmp_path, "apartment", tmp_path)["geometry"])
    assert trajectory_scene(tmp_path, "apartment", tmp_path, "unknown-environment") is None
    from backend.saved_results import normalize_trial
    normalized = normalize_trial({}, {"challenge": "apartment", "environment": "shared_apartment_v1"}, 0, "batch", tmp_path, tmp_path, "scripted_test")
    assert normalized["environment"] == "shared_apartment_v1"


def test_enhanced_is_app_default_and_renderer_survives_episode_reset(monkeypatch):
    monkeypatch.delenv("MILO_RENDERER", raising=False)
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        assert initial["rendering"] == "enhanced"
        resources = lab.render_resources
        process = resources.renderer.process
        initial_image = client.get(initial["camera"]["url"])
        assert initial_image.status_code == 200 and initial_image.content.startswith(b"\x89PNG")
        reset = client.post("/api/reset")
        assert reset.status_code == 200
        current = reset.json()
        assert current["rendering"] == "enhanced" and current["run_id"] != initial["run_id"]
        assert lab.render_resources is resources and resources.renderer.process is process
        assert process.poll() is None and current["snapshot"]["simulated_time_s"] == 0
        assert client.get(initial["camera"]["url"]).status_code == 404
        assert client.get(current["camera"]["url"]).status_code == 200
    assert process.poll() is not None and lab.render_resources is None


@pytest.mark.parametrize("interruption", ["stop", "takeover", "reset", "disconnect", "stale_map", "blocked_map"])
def test_continuous_navigation_ownership_and_interruptions(interruption):
    import time
    with TestClient(app) as client:
        state = client.post("/api/challenges/load", json={"challenge_id": "park"}).json()
        envelope = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"]}
        assert client.post("/api/continuous/scan", json=envelope).status_code == 409
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            scanned = client.post("/api/continuous/scan", json=envelope | {"compact_arms": True})
            assert scanned.status_code == 200
            fresh = client.get("/api/spatial").json()
            selection = envelope | {"spatial_sequence": fresh["frame"]["sequence"], "pixel": [.5, .4]}
            started = client.post("/api/continuous/start", json=selection)
            assert started.status_code == 200, started.text
            assert client.post("/api/continuous/start", json=selection).status_code == 409
            assert client.post("/api/spatial", json=envelope | {"enabled": False}).status_code == 409
            worker = lab.worker
            if interruption in {"stale_map", "blocked_map"}:
                def invalidate(sim):
                    worker._sample_spatial = lambda force=False: None
                    if interruption == "stale_map":
                        worker.spatial_map.captured_at = time.monotonic() - 2
                    else:
                        worker.spatial_map.cells[:] = 100
                client.portal.call(worker.call, invalidate)
                deadline = time.monotonic() + 3
                while worker.continuous.active and time.monotonic() < deadline:
                    time.sleep(.02)
                assert worker.continuous.status == "blocked"
                assert "SPATIAL_STALE" in worker.continuous.reason if interruption == "stale_map" else "OBSERVED_PATH_BLOCKED" in worker.continuous.reason
                assert not worker.navigation.buffer
            elif interruption != "disconnect":
                path = {"stop": "/api/stop", "takeover": "/api/agent/takeover", "reset": "/api/reset"}[interruption]
                assert client.post(path).status_code == 200
        deadline = time.monotonic() + 3
        while worker.continuous.active and time.monotonic() < deadline:
            time.sleep(.02)
        assert not worker.continuous.active and not worker.navigation.buffer
        ticks = worker.sim.ticks
        time.sleep(.1)
        assert worker.sim.ticks == ticks
        assert client.post("/api/continuous/start", json=selection).status_code == 409


@pytest.mark.parametrize("interruption", ["stop", "takeover", "reset", "disconnect"])
def test_ros_navigation_ownership_and_interruptions(monkeypatch, interruption):
    import time
    monkeypatch.delenv("MILO_ROS_ENABLED", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/ros/sensors").status_code == 404
        assert client.get("/api/ros/status").json()["enabled"] is False
        monkeypatch.setenv("MILO_ROS_ENABLED", "1")
        state = client.post("/api/challenges/load", json={"challenge_id": "park"}).json()
        assert client.get("/api/ros/status").json()["enabled"] is True
        variant = client.get("/api/test-variant?navigation_backend=nav2").json()
        assert variant["supports_navigation_backend"] and variant["architecture"]["id"] == "nav2-supervised"
        assert client.get("/api/test-variant?navigation_backend=nav2&reasoning=none").status_code == 200
        odometry = client.get("/api/ros/odometry").json()
        assert odometry["run_id"] == state["run_id"] and "sequence" not in odometry and "head_rgb_png" not in odometry
        goal = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"], "target_m_rad": [1., 0., 0.]}
        assert client.post("/api/ros/start", json=goal).status_code == 409
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            result = client.post("/api/ros/start", json=goal)
            assert result.status_code == 200, result.text
            worker = lab.worker
            sensor = client.get("/api/ros/sensors").json()
            assert "snapshot" not in sensor and "objects" not in sensor and "challenge" not in sensor
            assert len(sensor["laser"]["ranges_m"]) == 720
            command = {"session_id": result.json()["session_id"], "sensor_sequence": sensor["sequence"],
                "command_sequence": 1, "linear_mps": .1, "angular_radps": 0.}
            accepted = client.post("/api/ros/velocity", json=command)
            assert accepted.status_code == 200, accepted.text
            assert client.post("/api/ros/velocity", json=command).status_code == 409
            assert client.post("/api/command", json={}).status_code == 409
            assert client.post("/api/agent/start", json={}).status_code == 409
            if interruption != "disconnect":
                path = {"stop": "/api/stop", "takeover": "/api/agent/takeover", "reset": "/api/reset"}[interruption]
                assert client.post(path).status_code == 200
        client.portal.call(worker.call, lambda sim: None) if not worker.closed else None
        assert not worker.ros_navigation.active and not worker.navigation.buffer
        ticks = worker.sim.ticks
        time.sleep(.1)
        assert worker.sim.ticks == ticks
        assert client.post("/api/ros/velocity", json=command | {"command_sequence": 2}).status_code == 409


@pytest.mark.parametrize("mode", ["luna_continuous", "luna_navigation"])
def test_run_chat_redirects_same_episode_and_preserves_memory(monkeypatch, mode):
    import asyncio
    import time
    from backend.agent import AgentController
    from backend.contracts import Command
    goals = []

    async def scripted(self, worker, settings, profile, stop_revision):
        goals.append(settings.goal)
        try:
            assert await worker.resume_manual(expected_stop_revision=stop_revision)
            before, _ = await worker.feedback()
            if len(goals) == 1:
                result = await worker.execute(Command(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
                    observation_seq=before.seq, action_id="chat-test-drive", tool="drive_base",
                    arguments={"linear_mps": .1, "angular_radps": 0., "duration_s": .5}), assisted=False)
                self.navigation_memory.remember(before, result.observation, {"action": "navigate"}, {"status": "ok"})
            self.state["phase"] = "thinking"
            await asyncio.Event().wait()
        finally:
            worker.stop()
            await worker.hold_stopped()

    monkeypatch.setattr(AgentController, "_run", scripted)
    with TestClient(app) as client:
        state = client.post("/api/challenges/load", json={"challenge_id": "park"}).json()
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com"})
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            assert client.post("/api/agent/start", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                "execution_mode": mode, "goal": "Original instruction"}).status_code == 200
            deadline = time.monotonic() + 4
            while lab.agent.state["phase"] != "thinking" and time.monotonic() < deadline:
                time.sleep(.02)
            previous = client.get("/api/state").json()
            instruction = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                "session_id": previous["agent"]["session_id"], "message": "Inspect the doorway on the right"}
            assert client.post("/api/agent/instruction", json=instruction | {"message": " "}).status_code == 422
            assert client.post("/api/agent/instruction", json=instruction, headers={"Origin":"https://untrusted.example"}).status_code == 403
            response = client.post("/api/agent/instruction", json=instruction)
            assert response.status_code == 200, response.text
            after = client.get("/api/state").json()
            assert after["run_id"] == previous["run_id"] and after["episode_epoch"] == previous["episode_epoch"]
            assert after["snapshot"]["simulated_time_s"] == previous["snapshot"]["simulated_time_s"] == .5
            assert after["agent"]["goal"] == instruction["message"]
            assert len(after["agent"]["run_memory"]["recent_actions"]) >= 2
            assert client.post("/api/agent/instruction", json=instruction).status_code == 409
            lab.agent.state["turns"] = lab.agent.state["max_turns"]
            paused = client.post("/api/agent/instruction", json=instruction | {"session_id": after["agent"]["session_id"], "message": "stop"})
            assert paused.status_code == 200 and not paused.json()["active"]
            assert client.get("/api/state").json()["stopped"]
            assert goals == ["Original instruction", instruction["message"]]
        reset = client.post("/api/reset").json()
        assert not reset["agent"]["run_memory"]["recent_actions"] and not reset["agent"]["run_messages"]


def test_spatial_sensor_opt_in_pairing_and_episode_reset():
    from io import BytesIO
    from PIL import Image
    with TestClient(app) as client:
        before = client.get("/api/state").json()
        assert not client.get("/api/spatial").json()["enabled"]
        payload = {"run_id": before["run_id"], "episode_epoch": before["episode_epoch"], "enabled": True}
        response = client.post("/api/spatial", json=payload)
        assert response.status_code == 200
        spatial = response.json()
        assert spatial["enabled"] and not spatial["map"]["stale"]
        frame = spatial["frame"]
        rgb = client.get(frame["rgb_url"])
        depth = client.get(frame["depth_url"])
        raw = client.get(frame["data_url"]).json()
        assert Image.open(BytesIO(rgb.content)).size == Image.open(BytesIO(depth.content)).size == (160, 120)
        assert raw["sequence"] == frame["sequence"] and raw["captured_at"] == frame["captured_at"]
        assert len(raw["depth_m"]) == 160 * 120
        assert client.get("/api/state").json()["observation"] == before["observation"]
        assert client.get("/api/state").json()["snapshot"]["simulated_time_s"] == 0
        assert client.post("/api/spatial", json=payload | {"episode_epoch": 99}).status_code == 409
        client.post("/api/reset")
        reset_spatial = client.get("/api/spatial").json()
        assert reset_spatial["enabled"] and reset_spatial["frame"]["run_id"] != raw["run_id"]
        assert client.get(frame["data_url"]).status_code == 404
        assert client.post("/api/challenges/load", json={"challenge_id": "kitchen_bathroom"}).status_code == 200
        assert client.get("/api/spatial").json()["enabled"]


def test_spatial_can_enable_but_not_disable_during_model_control():
    with TestClient(app) as client:
        before = client.get("/api/state").json()
        settings = {"run_id": before["run_id"], "episode_epoch": before["episode_epoch"], "enabled": False}
        assert client.post("/api/spatial", json=settings).status_code == 200
        lab.agent.active = True
        try:
            assert client.post("/api/spatial", json=settings | {"enabled": True}).status_code == 200
            assert client.post("/api/spatial", json=settings).status_code == 409
            assert client.get("/api/state").json()["snapshot"]["simulated_time_s"] == 0
        finally:
            lab.agent.active = False


def test_spatial_reads_during_episode_replacement_are_not_server_errors(monkeypatch):
    from types import SimpleNamespace

    async def closed_read(operation):
        raise RuntimeError("Episode closed")

    with TestClient(app) as client:
        with monkeypatch.context() as temporary:
            temporary.setattr(lab, "worker", SimpleNamespace(closed=True, call=closed_read))
            assert client.get("/api/spatial").status_code == 409
            assert client.get("/api/spatial/old-run/1/depth.json").status_code == 404


@pytest.mark.parametrize("policy_status", ["unavailable", "incompatible", "untrained", "stop_during_check"])
def test_policy_start_preflight_does_not_create_failed_sessions_or_ignore_stop(policy_status):
    from backend.policy import PolicyConnectionError
    from tests.test_policy import trained_metadata

    class Policy:
        async def describe(self):
            if policy_status in {"unavailable", "incompatible"}:
                raise PolicyConnectionError(policy_status, f"SmolVLA {policy_status}")
            metadata = trained_metadata()
            metadata.trained_for_milo = policy_status != "untrained"
            if policy_status == "stop_during_check":
                lab.worker.stop()
            return metadata

        async def close(self):
            pass

    with TestClient(app) as client:
        lab.agent.policy_factory = lambda config: Policy()
        with client.websocket_connect("/api/live") as socket:
            initial = socket.receive_json()
            if policy_status != "stop_during_check":
                check = client.post("/api/policy/check", json={"endpoint": "http://127.0.0.1:8085"})
                assert check.status_code == 200
                assert check.json()["status"] == policy_status and not check.json()["ready"]
            response = client.post("/api/agent/start", json={
                "run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "goal": "Pick up cube", "execution_mode": "supervised_policy"})
            assert response.status_code == 409
            state = client.get("/api/state").json()
            assert state["agent"]["session_id"] == initial["agent"]["session_id"]
            assert state["agent"]["error"] is None and not state["agent"]["active"]
            assert state["agent"]["turns"] == 0 and state["snapshot"]["simulated_time_s"] == 0
            assert client.post("/api/policy/check", json={"endpoint": "http://example.com"}).status_code == 422


@pytest.mark.parametrize("stage", ["loading", "inference"])
@pytest.mark.parametrize("interruption", ["stop", "takeover", "reset", "mode", "disconnect"])
def test_live_local_navigation_lifecycle_rejects_late_motion(stage, interruption):
    import asyncio
    import time
    closed = []
    waiting = []

    class LocalModel:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            if stage == "loading":
                waiting.append(True)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return

        async def predict(self, observation, image):
            if observation.navigation:
                waiting.append(True)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return {"action": [.15, 0.]}
            return {"action": [0., 0.]}

        async def close(self):
            closed.append(True)

    with TestClient(app) as client:
        lab.agent.local_navigation_factory = LocalModel
        initial = client.post("/api/challenges/load", json={"challenge_id": "local_park"}).json()
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            response = client.post("/api/agent/start", json={"run_id": initial["run_id"],
                "episode_epoch": initial["episode_epoch"], "goal": initial["challenge"]["goal"], "execution_mode": "local_navigation"})
            assert response.status_code == 200 and response.json()["local_model"]["phase"] == "loading"
            deadline = time.monotonic() + 12
            while not waiting and time.monotonic() < deadline:
                time.sleep(.02)
            assert waiting
            original = lab.worker
            before = original.latest["snapshot"]["simulated_time_s"]
            if interruption != "disconnect":
                path = {"stop": "/api/stop", "takeover": "/api/agent/takeover", "reset": "/api/reset", "mode": "/api/agent/mode"}[interruption]
                payload = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "mode": "voice"} if interruption == "mode" else {}
                assert client.post(path, json=payload).status_code == 200
        deadline = time.monotonic() + 3
        while not closed and time.monotonic() < deadline:
            time.sleep(.02)
        assert closed == [True] and not lab.agent.active
        assert original.latest["snapshot"]["simulated_time_s"] == before
        assert original.navigation is None or not original.navigation.buffer


@pytest.mark.parametrize("challenge_id", ["bench", "recharge", "warehouse"])
def test_live_local_navigation_uses_current_challenge_and_position_without_reset(challenge_id):
    import asyncio

    class LocalModel:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            await asyncio.Event().wait()

        async def close(self):
            pass

    with TestClient(app) as client:
        lab.agent.local_navigation_factory = LocalModel
        initial = client.post("/api/challenges/load", json={"challenge_id": challenge_id}).json()
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            assert client.post("/api/robot/placement", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "observation_seq": initial["observation"]["seq"], "xy_m": [-.2, 0]}).status_code == 200
            initial = client.get("/api/state").json()
            assert client.post("/api/command", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "observation_seq": initial["observation"]["seq"], "action_id": "wait-before-local", "tool": "wait", "arguments": {"duration_s": .1}}).status_code == 200
            before = client.get("/api/state").json()
            goal = initial["challenge"]["goal"] if initial["challenge"] else "Move forward carefully and stop"
            payload = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "goal": goal,
                "execution_mode": "local_navigation"}
            assert client.get("/api/tools?execution_mode=local_navigation").json() == []
            assert client.post("/api/agent/chat", json={**payload, "message": "Move"}).status_code == 400
            response = client.post("/api/agent/start", json=payload)
            assert response.status_code == 200
            assert response.json()["local_model"]["challenge_id"] == challenge_id
            assert response.json()["goal"] == goal
            after = client.get("/api/state").json()
            assert after["run_id"] == before["run_id"] and after["episode_epoch"] == before["episode_epoch"]
            assert after["challenge"] == before["challenge"] and after["manual_placements"] == 1
            assert after["snapshot"] == before["snapshot"]
            assert client.post("/api/agent/start", json=payload).status_code == 409
            assert client.post("/api/stop").status_code == 200


@pytest.mark.parametrize("unload", [False, True])
def test_resident_model_survives_sessions_reset_disconnect_and_shuts_down(monkeypatch, unload):
    from importlib import import_module
    from types import SimpleNamespace
    import time
    from backend.local_navigation import ResidentNavigationModel
    created, goals, resets = [], [], []

    class Client:
        alive = True
        process = SimpleNamespace(pid=12345)

        def __init__(self):
            created.append(self)

        @staticmethod
        def check_available():
            pass

        async def start(self):
            pass

        async def reset(self):
            resets.append(True)

        async def predict(self, observation, image):
            goals.append(self.instruction)
            return {"action": [0., 0.]}

        async def close(self):
            self.alive = False

    monkeypatch.setattr(import_module("backend.app"), "ResidentNavigationModel", lambda: ResidentNavigationModel(Client))
    with TestClient(app) as client:
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            for challenge_id in ("local_park", "recharge"):
                initial = client.post("/api/challenges/load", json={"challenge_id": challenge_id}).json()
                worker = lab.worker
                response = client.post("/api/agent/start", json={"run_id": initial["run_id"],
                    "episode_epoch": initial["episode_epoch"], "goal": initial["challenge"]["goal"],
                    "execution_mode": "local_navigation", "max_turns": 1})
                assert response.status_code == 200
                assert client.post("/api/local-navigation/unload").status_code == 409
                deadline = time.monotonic() + 15
                while lab.agent.active and time.monotonic() < deadline:
                    time.sleep(.02)
                assert not lab.agent.active and lab.agent.state["error"] is None
                current = client.get("/api/state").json()
                assert current["local_navigation_model"]["phase"] == "ready"
                assert current["local_navigation_model"]["load_count"] == 1
                assert current["local_navigation_model"]["process_id"] == 12345
                assert "local_navigation_model" not in current["observation"]
                assert lab.worker is worker and worker.latest["stopped"]
                assert goals[-1] == initial["challenge"]["goal"]
                assert client.post("/api/stop").status_code == 200
                reset = client.post("/api/reset").json()
                assert reset["run_id"] != current["run_id"]
                assert reset["local_navigation_model"] == current["local_navigation_model"]
                assert reset["agent"]["local_model"] is None
                assert reset["snapshot"]["simulated_time_s"] == 0
        assert len(created) == 1 and len(resets) == 2 and created[0].alive
        assert client.get("/api/state").json()["local_navigation_model"]["phase"] == "ready"
        assert client.post("/api/local-navigation/unload", headers={"Origin": "https://untrusted.example"}).status_code == 403
        if unload:
            response = client.post("/api/local-navigation/unload")
            assert response.status_code == 200 and response.json()["phase"] == "unloaded"
            assert not created[0].alive
    assert not created[0].alive


def test_local_navigation_status_is_read_only_and_never_enters_model_observations(tmp_path, monkeypatch):
    from backend import local_progress
    monkeypatch.setattr(local_progress, "PROGRESS_ROOT", tmp_path)
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        assert client.get("/api/local-navigation/status").json() == {"test": None}
        progress = local_progress.NavigationProgress(tmp_path / "candidate" / "checkpoint", 15, 30)
        progress.update(phase="loading")
        response = client.get("/api/local-navigation/status")
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        status = response.json()["test"]
        assert status["phase"] == "loading" and status["checkpoint"] == "candidate"
        assert str(tmp_path) not in response.text and "observation" not in status
        progress.update(phase="running", requests_completed=3)
        assert client.get("/api/local-navigation/status").json()["test"]["requests_completed"] == 3
        current = client.get("/api/state").json()
        assert current["run_id"] == initial["run_id"]
        assert current["observation"] == initial["observation"]
        assert current["snapshot"]["simulated_time_s"] == 0 and not current["agent"]["active"]
        assert client.post("/api/local-navigation/status").status_code == 405
        assert client.get("/api/local-navigation/status", headers={"Origin": "https://untrusted.example"}).status_code == 403
        progress.path.write_text("{", encoding="utf-8")
        assert client.get("/api/local-navigation/status").status_code == 503


@pytest.mark.parametrize("phase", ["loading", "warming", "running", "completed", "failed"])
def test_local_navigation_progress_expiry_and_latest_run(tmp_path, monkeypatch, phase):
    from backend import local_progress
    monkeypatch.setattr(local_progress, "PROGRESS_ROOT", tmp_path)
    monkeypatch.setattr(local_progress.time, "time_ns", lambda: 1)
    run_ids = iter(["ffffffff-ffff-ffff-ffff-ffffffffffff", "00000000-0000-0000-0000-000000000000"])
    monkeypatch.setattr(local_progress, "uuid4", lambda: next(run_ids))
    older = local_progress.NavigationProgress(tmp_path / "older" / "checkpoint", 12, 30)
    newer = local_progress.NavigationProgress(tmp_path / "newer" / "checkpoint", 15, 30)
    newer.update(phase=phase)
    older.update(phase="running")
    result = local_progress.read_navigation_progress(now=newer.state.updated_at + 16)
    assert result["checkpoint"] == "newer"
    assert result["phase"] == (phase if phase in local_progress.TERMINAL_PHASES else "interrupted")
    assert not list(tmp_path.glob("*.tmp"))


def test_local_navigation_progress_write_failure_is_nonfatal(tmp_path, monkeypatch):
    from backend import local_progress
    monkeypatch.setattr(local_progress, "PROGRESS_ROOT", tmp_path)
    progress = local_progress.NavigationProgress(tmp_path / "candidate" / "checkpoint", 15, 30)

    def unavailable(path, target):
        raise OSError("Progress file temporarily unavailable")

    monkeypatch.setattr(type(progress.path), "replace", unavailable)
    progress.update(phase="warming")
    assert progress.state.phase == "warming"
    assert local_progress.read_navigation_progress()["phase"] == "checking"


async def test_local_navigation_progress_records_startup_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from backend import local_progress
    from scripts.navigation_policy import evaluate
    monkeypatch.setattr(local_progress, "PROGRESS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        await evaluate(SimpleNamespace(checkpoint=tmp_path / "missing" / "checkpoint", case=15, requests=30))
    result = local_progress.read_navigation_progress()
    assert result["phase"] == "failed" and result["success"] is False


def test_shared_scene_texture_endpoints():
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        urls = {entry["texture"] for entry in state["geometry"] if entry["texture"]}
        assert urls == {"/api/textures/concrete.png", "/api/textures/plaster.png"}
        for url in urls:
            response = client.get(url)
            assert response.status_code == 200 and response.content.startswith(b"\x89PNG")
        assert client.get("/api/textures/missing.png").status_code == 404


def test_manual_api_camera_stop_and_reset():
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        initial_id = state["run_id"]
        image = client.get(f"/api/frames/{initial_id}/{state['observation']['frame_ref']}")
        assert image.status_code == 200
        assert image.content.startswith(b"\x89PNG")
        response = client.post("/api/command", json={
            "run_id": initial_id, "episode_epoch": state["episode_epoch"], "action_id": "api-drive",
            "observation_seq": state["observation"]["seq"], "tool": "drive_base",
            "arguments": {"linear_mps": .2, "angular_radps": 0, "duration_s": .5}})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert client.post("/api/stop").json()["stopped"]
        assert client.post("/api/resume").json()["stopped"] is False
        replacement = client.post("/api/reset").json()
        assert replacement["run_id"] != initial_id
        assert replacement["episode_epoch"] == state["episode_epoch"] + 1
        assert client.get(f"/api/frames/{initial_id}/{state['observation']['frame_ref']}").status_code == 404
        assert client.post("/api/stop", headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_live_camera_frames_are_separate_from_model_frames_and_reset_safely():
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        camera = client.get(initial["camera"]["url"])
        assert camera.status_code == 200 and camera.content.startswith(b"\x89PNG")
        assert camera.headers["cache-control"] == "no-store"
        model_url = f"/api/frames/{initial['run_id']}/{initial['observation']['frame_ref']}"
        assert camera.content == client.get(model_url).content
        result = client.post("/api/command", json={
            "run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "action_id": "camera-head",
            "observation_seq": initial["observation"]["seq"], "tool": "set_head",
            "arguments": {"yaw_rad": .7, "pitch_rad": .6, "duration_s": 1}}).json()
        current = client.get("/api/state").json()
        assert current["camera"]["seq"] > initial["camera"]["seq"] + 1
        assert current["camera"]["simulated_time_s"] == 1
        assert result["observation"]["seq"] == initial["observation"]["seq"] + 1
        assert client.get(current["camera"]["url"]).content != camera.content
        assert client.get(model_url).content == camera.content
        assert "camera" not in result["observation"]
        replacement = client.post("/api/reset").json()
        assert client.get(current["camera"]["url"]).status_code == 404
        assert client.get(replacement["camera"]["url"]).status_code == 200
        assert client.get(f"/api/camera/{replacement['run_id']}/missing.png").status_code == 404


def test_manual_placement_api_validates_destination_and_episode():
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        placement = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                     "observation_seq": initial["observation"]["seq"], "xy_m": [-.6, .3]}
        assert client.post("/api/robot/placement", json=placement, headers={"Origin": "https://untrusted.example"}).status_code == 403
        for target in ([], [0, 0, 0], ["NaN", 0], ["Infinity", 0]):
            assert client.post("/api/robot/placement", json={**placement, "xy_m": target}).status_code == 422
        assert client.post("/api/robot/placement", json={**placement, "height_m": 1}).status_code == 422
        moved = client.post("/api/robot/placement", json=placement)
        assert moved.status_code == 200
        state = moved.json()
        assert state["manual_placements"] == 1 and state["run_id"] == initial["run_id"]
        robot = next(pose for pose in state["snapshot"]["poses"] if pose["key"] == f"{state['robot_body_id']}:-1")
        assert robot["position"][:2] == pytest.approx(placement["xy_m"])
        assert client.get(state["camera"]["url"]).content != client.get(initial["camera"]["url"]).content
        assert client.post("/api/robot/placement", json=placement).status_code == 409
        placement["observation_seq"] = state["observation"]["seq"]
        assert client.post("/api/robot/placement", json={**placement, "xy_m": [2.5, 0]}).status_code == 422
        assert client.get("/api/state").json()["manual_placements"] == 1
        client.post("/api/stop")
        assert client.post("/api/robot/placement", json=placement).status_code == 409
        replacement = client.post("/api/reset").json()
        assert replacement["manual_placements"] == 0
        assert client.post("/api/robot/placement", json=placement).status_code == 409
        assert all(tool["name"] not in {"reposition", "placement"} for tool in client.get("/api/tools").json())


async def test_last_disconnect_finishes_cleanup_in_cancelled_socket_scope(monkeypatch):
    import anyio
    from types import SimpleNamespace
    from backend.app import live

    cleanup = []

    async def halt(reason):
        await anyio.lowlevel.checkpoint()
        cleanup.append(reason)

    class Socket:
        headers = {}

        async def accept(self):
            pass

        async def send_json(self, state):
            scope.cancel()
            await anyio.lowlevel.checkpoint()

    monkeypatch.setattr(lab, "connections", 0)
    monkeypatch.setattr(lab, "worker", SimpleNamespace(stop=lambda: cleanup.append("stop")))
    monkeypatch.setattr(lab, "agent", SimpleNamespace(halt=halt))
    monkeypatch.setattr(lab, "state", lambda: {})
    with anyio.CancelScope() as scope:
        await live(Socket())
    assert lab.connections == 0
    assert cleanup == ["stop", "Operator disconnected"]


@pytest.mark.parametrize("interruption", ["stop", "takeover", "reset", "mode", "disconnect"])
def test_navigation_plan_lifecycle_invalidates_motion(interruption):
    import asyncio
    import json
    import time
    from tests.test_agent import model_response

    class Planner:
        def __init__(self):
            self.calls = 0
            self.closed = False

        async def respond(self, profile, reasoning, goal, inputs):
            self.calls += 1
            revision = json.loads(inputs[-1]["content"][0]["text"])["navigation"]["revision"]
            if self.calls == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": revision,
                    "steps": [{"skill": "inspect_room", "goal": "Inspect the room"}]}), call_id="plan")
            if self.calls == 2:
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": revision,
                    "segments": [{"kind": "head", "yaw_rad": .8, "pitch_rad": .2, "duration_s": 1}] * 2}), call_id="buffer")
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    with TestClient(app) as client:
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com"})
        planner = Planner()
        lab.agent.model_factory = lambda config: planner
        assert "replace_motion_buffer" in {tool["name"] for tool in client.get("/api/tools?execution_mode=navigation_plan").json()}
        with client.websocket_connect("/api/live") as socket:
            state = socket.receive_json()
            started = client.post("/api/agent/start", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                "goal": "Inspect the room", "execution_mode": "navigation_plan", "feedback_interval_s": .25})
            assert started.status_code == 200
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                current = client.get("/api/state").json()
                if current["navigation"] and current["navigation"]["remaining_s"] > 0:
                    break
                time.sleep(.02)
            assert current["navigation"]["remaining_s"] > 0
            assert client.post("/api/robot/placement", json={"run_id": current["run_id"], "episode_epoch": current["episode_epoch"],
                "observation_seq": current["observation"]["seq"], "xy_m": [0, 0]}).status_code == 409
            original = lab.worker
            if interruption != "disconnect":
                path = {"stop": "/api/stop", "takeover": "/api/agent/takeover", "reset": "/api/reset", "mode": "/api/agent/mode"}[interruption]
                payload = {"run_id": current["run_id"], "episode_epoch": current["episode_epoch"], "mode": "voice"} if interruption == "mode" else {}
                assert client.post(path, json=payload).status_code == 200
        deadline = time.monotonic() + 3
        while lab.agent.active and time.monotonic() < deadline:
            time.sleep(.02)
        assert planner.closed and not lab.agent.active
        assert not original.latest["navigation"]["remaining_s"]
        assert original.latest["navigation"]["steps"][0]["status"] == "cancelled"
        if interruption == "reset":
            assert client.get("/api/state").json()["navigation"] is None


def test_luna_default_and_nano_selection_survive_episode_reset(monkeypatch):
    from tests.test_agent import ScriptedModel, text_response
    deployments = []

    class DeploymentProbe(ScriptedModel):
        async def respond(self, profile, reasoning, goal, inputs):
            deployments.append(profile.deployment)
            return await super().respond(profile, reasoning, goal, inputs)

    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://test.services.ai.azure.com/api/projects/test")
    monkeypatch.setenv("deployment_name", "GPT-5.2")
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        config = state["agent"]["configuration"]
        assert config["default_model_id"] == state["agent"]["model_id"] == "luna"
        assert [(profile["id"], profile["deployment"]) for profile in config["models"]] == [
            ("luna", "gpt-5.6-luna"), ("nano", "gpt-5.4-nano"), ("gemma", "gemma4:e2b-it-qat"), ("configured", "GPT-5.2")]
        assert all(profile["configured"] for profile in config["models"])
        lab.agent.model_factory = lambda config: DeploymentProbe([text_response()])
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            for model_id in ("luna", "nano"):
                request = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                           "model_id": model_id, "goal": "Observe only", "max_turns": 1}
                assert client.post("/api/agent/start", json=request).status_code == 200
                while socket.receive_json()["agent"]["active"]:
                    pass
            assert deployments == ["gpt-5.6-luna", "gpt-5.4-nano"]
            reset = client.post("/api/reset").json()
            assert reset["agent"]["model_id"] == "luna"
            assert reset["agent"]["configuration"] == config


def test_agent_configuration_and_manual_ownership(monkeypatch):
    from tests.test_agent import ScriptedModel

    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODELS_JSON", raising=False)
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        settings = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                    "goal": "Observe then stop", "feedback_interval_s": 2}
        assert not initial["agent"]["configuration"]["models"][0]["configured"]
        assert client.post("/api/agent/start", json=settings).status_code == 409
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            assert client.post("/api/agent/start", json=settings).status_code == 400
            assert client.post("/api/agent/config", json={"endpoint": "https://example.com"}).status_code == 422
            config = {"endpoint": "https://test.openai.azure.com", "models": [
                {"id": "luna", "label": "GPT-5.6 Luna", "deployment": "test-luna"},
                {"id": "other", "label": "Another model", "deployment": "other-model"}]}
            assert client.post("/api/agent/config", json=config).status_code == 200
            lab.agent.model_factory = lambda config: ScriptedModel(["wait"])
            assert client.post("/api/agent/start", json={**settings, "model_id": "unknown"}).status_code == 400
            assert client.post("/api/agent/start", json={**settings, "model_id": "other"}).status_code == 200
            assert client.get("/api/agent").json()["model_id"] == "other"
            assert client.post("/api/agent/start", json=settings).status_code == 409
            assert client.post("/api/agent/config", json=config).status_code == 409
            manual = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                      "action_id": "manual", "observation_seq": initial["observation"]["seq"], "tool": "observe"}
            assert client.post("/api/command", json=manual).status_code == 409
            assert client.post("/api/robot/placement", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "observation_seq": initial["observation"]["seq"], "xy_m": [-.6, .3]}).status_code == 409
            assert client.post("/api/agent/rate", json={"feedback_interval_s": .5}).json()["feedback_interval_s"] == .5
            assert client.post("/api/agent/rate", json={"feedback_interval_s": 0}).status_code == 422
            assert client.post("/api/agent/rate", json={}, headers={"Origin": "https://untrusted.example"}).status_code == 403
            takeover = client.post("/api/agent/takeover").json()
            assert not takeover["agent"]["active"] and not takeover["stopped"]
            assert client.post("/api/command", json=manual).json()["status"] == "ok"
            lab.agent.model_factory = lambda config: ScriptedModel(["wait"])
            assert client.post("/api/agent/start", json=settings).status_code == 200
            replacement = client.post("/api/reset").json()
            assert not replacement["agent"]["active"] and replacement["run_id"] != initial["run_id"]
            assert client.post("/api/agent/start", json=settings).status_code == 409
            settings.update(run_id=replacement["run_id"], episode_epoch=replacement["episode_epoch"])
            assert client.post("/api/agent/start", json=settings).status_code == 200
        assert not client.get("/api/agent").json()["active"]
        assert client.get("/api/state").json()["stopped"]


@pytest.mark.parametrize("action", ["stop", "agent/takeover", "agent/mode", "reset", "disconnect"])
def test_operator_lifecycle_preserves_zero_inference_idle(action):
    from tests.test_agent import ScriptedModel, text_response
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com", "models": [
            {"id": "luna", "label": "Test", "deployment": "test"}]})
        controller = lab.agent
        controller.idle_delay_s, controller.camera_poll_s = .05, .02
        controller.model_factory = lambda config: ScriptedModel([text_response()])
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            client.post("/api/agent/chat", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "goal": "Inspect the room", "message": "What do you see?"})
            for _ in range(20):
                if not socket.receive_json()["agent"]["active"]:
                    break
            else:
                pytest.fail("Agent did not enter idle")
            assert not controller.state["auto_wake"] and controller.idle_task is None
            if action != "disconnect":
                body = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "mode": "voice"} if action == "agent/mode" else {}
                assert client.post(f"/api/{action}", json=body).status_code == 200
                assert not client.get("/api/state").json()["agent"]["auto_wake"]
        assert not controller.state["auto_wake"] and controller.idle_task is None
        assert not controller.active


def test_chat_and_voice_modes_are_exclusive_and_switch_cancels_inference():
    from tests.test_agent import ScriptedModel, text_response
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        assert initial["interaction_mode"] == "chat"
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com", "models": [
            {"id": "luna", "label": "Test", "deployment": "test"}]})
        settings = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "goal": "Inspect the room", "message": "Hello"}
        assert client.post("/api/agent/chat", json=settings).status_code == 409
        assert client.post("/api/agent/chat", json={**settings, "message": " "}).status_code == 422
        mode = {"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"], "mode": "voice"}
        assert client.post("/api/agent/mode", json=mode, headers={"Origin": "https://untrusted.example"}).status_code == 403
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            model = ScriptedModel([text_response()])
            lab.agent.model_factory = lambda config: model
            assert client.post("/api/agent/chat", json=settings).status_code == 200
            while client.get("/api/agent").json()["active"]:
                socket.receive_json()
            replied = client.get("/api/agent").json()
            assert replied["mode"] == "chat" and replied["chat_messages"][-1]["text"] == "I can see the room."
            waiting = ScriptedModel(["wait"])
            lab.agent.model_factory = lambda config: waiting
            assert client.post("/api/agent/chat", json={**settings, "conversation_id": replied["session_id"]}).status_code == 200
            assert client.post("/api/agent/chat", json=settings).status_code == 409
            switched = client.post("/api/agent/mode", json=mode).json()
            assert switched["interaction_mode"] == "voice" and switched["stopped"] and not switched["agent"]["active"]
            assert client.post("/api/agent/chat", json=settings).status_code == 409
            assert client.post("/api/agent/mode", json={**mode, "mode": "chat"}).status_code == 200
            with client.websocket_connect("/api/voice") as voice:
                voice.send_json({"type": "start", "run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"]})
                assert voice.receive_json()["type"] == "ended"
            reset = client.post("/api/reset").json()
            assert reset["interaction_mode"] == "chat" and reset["agent"]["chat_messages"] == []
            assert client.post("/api/agent/mode", json=mode).status_code == 409


def test_reset_never_streams_an_uninitialized_worker():
    with TestClient(app) as client:
        with client.websocket_connect("/api/live") as socket:
            socket.receive_json()
            for _ in range(3):
                replacement = client.post("/api/reset").json()
                while True:
                    streamed = socket.receive_json()
                    assert {"snapshot", "observation", "geometry", "agent", "run_id"} <= streamed.keys()
                    if streamed["run_id"] == replacement["run_id"]:
                        assert not streamed["stopped"]
                        break


def test_backend_environment_file_is_loaded_first_without_overriding_process(monkeypatch):
    from importlib import import_module
    module = import_module("backend.app")
    paths = []
    monkeypatch.setattr(module, "load_dotenv", lambda path, **kwargs: paths.append((path, kwargs)))
    monkeypatch.delenv("FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODELS_JSON", raising=False)
    with TestClient(app):
        assert paths[0][0].parent.name == "backend"
        assert paths[1][0].parent.name == "robosim713"
        assert all(path.name == ".env" and options == {"override": False} for path, options in paths)


def test_exchange_feed_frames_survive_live_cache_rotation_and_expire_on_reset():
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        live_url = f"/api/frames/{state['run_id']}/{state['observation']['frame_ref']}"
        image = client.get(live_url).content
        lab.agent.state["session_id"] = "test-trace"
        identifier = lab.agent._trace("feedback", "Camera + sensors submitted", {
            "observation": state["observation"]}, image=image, images=[image, image])
        feed = client.get("/api/agent/trace", params={"session_id": "test-trace"}).json()
        trace_url = feed["events"][0]["image_url"]
        historical_url = feed["events"][0]["image_urls"][1]
        assert client.get(historical_url).content == image
        assert client.get(trace_url, params={"index": 2}).status_code == 404
        assert client.get(trace_url, params={"index": -1}).status_code == 422
        assert client.get(trace_url, params={"index": 8}).status_code == 422
        assert feed["events"][0]["payload"]["observation"] == state["observation"]
        assert client.get("/api/agent/trace", params={"after": identifier}).json()["events"] == []
        assert client.get("/api/agent/trace", params={"after": -1}).status_code == 422
        assert client.get("/api/agent/trace", params={"session_id": "obsolete"}).status_code == 409
        for index in range(17):
            client.post("/api/command", json={
                "run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                "action_id": f"cache-rotation-{index}", "observation_seq": 0, "tool": "observe"})
        assert client.get(live_url).status_code == 404
        retained = client.get(trace_url)
        assert retained.status_code == 200 and retained.content == image
        assert client.get(historical_url).content == image
        assert retained.headers["cache-control"] == "no-store"
        client.post("/api/reset")
        assert client.get("/api/agent/trace").json()["events"] == []
        assert client.get(trace_url).status_code == 404
        assert client.get(historical_url).status_code == 404
        assert client.get("/api/agent/trace", params={"session_id": "test-trace"}).status_code == 409


def test_voice_websocket_ownership_stop_and_private_config(monkeypatch):
    from tests.test_realtime import RealtimeModel
    monkeypatch.setattr(lab, "voice_factory", lambda config: RealtimeModel([]))
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        assert not state["realtime"]["configured"]
        config = {"endpoint": "https://test.openai.azure.com", "deployment": "test-voice"}
        assert client.post("/api/voice/config", json=config).status_code == 200
        assert client.post("/api/voice/config", json=config, headers={"Origin": "https://untrusted.example"}).status_code == 403
        with client.websocket_connect("/api/live") as live:
            live.receive_json()
            client.post("/api/agent/mode", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"], "mode": "voice"})
            with client.websocket_connect("/api/voice") as voice:
                voice.send_json({"type": "start", "run_id": state["run_id"], "episode_epoch": state["episode_epoch"]})
                assert voice.receive_json()["type"] == "ready"
                assert client.get("/api/agent").json()["mode"] == "voice"
                assert client.post("/api/voice/config", json=config).status_code == 409
                assert client.post("/api/robot/placement", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                    "observation_seq": state["observation"]["seq"], "xy_m": [-.6, .3]}).status_code == 409
                assert client.post("/api/command", json={"run_id": state["run_id"], "episode_epoch": state["episode_epoch"],
                    "action_id": "voice-manual", "observation_seq": 0, "tool": "observe"}).status_code == 409
                assert client.post("/api/stop").json()["stopped"]
                assert voice.receive_json()["type"] == "ended"
            assert not client.get("/api/agent").json()["active"]


def test_challenge_loading_resets_scene_goal_progress_and_preserves_model_config():
    with TestClient(app) as client:
        initial = client.get("/api/state").json()
        assert initial["challenge"] is None
        presets = client.get("/api/challenges").json()
        assert {preset["id"] for preset in presets} == {"park", "tidy", "sort", "recharge", "apartment", "kitchen_bathroom", "clinic_delivery", "warehouse", "inspection", "workshop", "local_park", "pedestrian_crossing", "flat_kitchen", "furniture_circuit"}
        assert all("objects" not in preset for preset in presets)
        config = {"endpoint": "https://test.openai.azure.com", "models": [
            {"id": "model", "label": "Model", "deployment": "test-model"}]}
        client.post("/api/agent/config", json=config)
        for preset in presets:
            loaded = client.post("/api/challenges/load", json={"challenge_id": preset["id"]}).json()
            assert loaded["run_id"] != initial["run_id"]
            assert loaded["challenge"]["goal"] == preset["goal"]
            assert loaded["challenge"]["status"] == "in_progress"
            assert not loaded["agent"]["active"] and not loaded["busy"] and not loaded["stopped"]
            assert loaded["agent"]["configuration"]["models"][0]["deployment"] == "test-model"
            assert "challenge" not in loaded["observation"] and "progress" not in loaded["observation"]
            reset = client.post("/api/reset").json()
            assert reset["challenge"]["id"] == preset["id"]
            assert reset["run_id"] != loaded["run_id"] and reset["challenge"]["completed_objectives"] == 0
        unchanged = client.get("/api/state").json()["run_id"]
        assert client.post("/api/challenges/load", json={"challenge_id": "unknown"}).status_code == 422
        assert client.get("/api/state").json()["run_id"] == unchanged
        assert client.post("/api/challenges/load", json={"challenge_id": "park"}, headers={"Origin": "https://untrusted.example"}).status_code == 403
        assert client.post("/api/challenges/load", json={"challenge_id": "bench"}).json()["challenge"] is None


def test_loading_challenge_invalidates_pending_model_and_previous_episode():
    from tests.test_agent import ScriptedModel
    with TestClient(app) as client:
        client.post("/api/agent/config", json={"endpoint": "https://test.openai.azure.com", "models": [
            {"id": "luna", "label": "Test model", "deployment": "test-model"}]})
        initial = client.get("/api/state").json()
        with client.websocket_connect("/api/live") as live:
            live.receive_json()
            lab.agent.model_factory = lambda config: ScriptedModel(["wait"])
            assert client.post("/api/agent/start", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                                                       "goal": "Move forward"}).status_code == 200
            old_controller = lab.agent
            loaded = client.post("/api/challenges/load", json={"challenge_id": "tidy"}).json()
            assert old_controller.cancelled and not old_controller.active
            assert old_controller.worker.closed
            assert loaded["challenge"]["id"] == "tidy" and not loaded["agent"]["active"]
            assert client.post("/api/command", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
                "action_id": "old-motion", "observation_seq": initial["observation"]["seq"], "tool": "drive_base",
                "arguments": {"linear_mps": .2, "angular_radps": 0, "duration_s": .5}}).json()["status"] == "cancelled"
            assert client.get("/api/state").json()["snapshot"]["simulated_time_s"] == 0


def test_recharge_sensor_and_mission_reset_without_location_leak():
    with TestClient(app) as client:
        loaded = client.post("/api/challenges/load", json={"challenge_id": "recharge"}).json()
        assert loaded["challenge"]["suggested_turn_limit"] == 100
        assert loaded["observation"]["battery"] == {"charge_pct": 100, "low": False, "charging": False}
        assert "center" not in str(loaded["observation"]) and "charger_beacon" not in str(loaded["observation"])
        result = client.post("/api/command", json={"run_id": loaded["run_id"], "episode_epoch": loaded["episode_epoch"],
            "action_id": "leave-dock", "observation_seq": loaded["observation"]["seq"], "tool": "drive_base",
            "arguments": {"linear_mps": -.2, "angular_radps": 0, "duration_s": 2}}).json()
        assert result["observation"]["battery"]["charge_pct"] < 100
        reset = client.post("/api/reset").json()
        assert reset["run_id"] != loaded["run_id"] and reset["challenge"]["id"] == "recharge"
        assert reset["observation"]["battery"]["charge_pct"] == 100
        assert reset["challenge"]["completed_objectives"] == 0
        assert client.post("/api/challenges/load", json={"challenge_id": "bench"}).json()["observation"]["battery"] is None