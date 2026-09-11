from fastapi.testclient import TestClient

from backend.app import app, lab


import pytest


@pytest.fixture(autouse=True)
def isolate_foundry_environment(monkeypatch):
    from importlib import import_module
    monkeypatch.setattr(import_module("backend.app"), "load_dotenv", lambda *args, **kwargs: None)
    for name in ("FOUNDRY_ENDPOINT", "FOUNDRY_PROJECT_ENDPOINT", "project_endpoint", "FOUNDRY_MODELS_JSON",
                 "FOUNDRY_DEPLOYMENT", "deployment_name", "AZURE_AI_MODEL_DEPLOYMENT_NAME",
                 "FOUNDRY_REALTIME_ENDPOINT", "FOUNDRY_REALTIME_DEPLOYMENT", "FOUNDRY_REALTIME_VOICE"):
        monkeypatch.delenv(name, raising=False)


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
def test_operator_lifecycle_invalidates_idle_camera_wake(action):
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
            for _ in range(100):
                if socket.receive_json()["agent"]["phase"] == "sleeping":
                    break
            else:
                pytest.fail("Agent did not enter idle")
            assert controller.state["auto_wake"]
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
            "observation": state["observation"]}, image=image)
        feed = client.get("/api/agent/trace", params={"session_id": "test-trace"}).json()
        trace_url = feed["events"][0]["image_url"]
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
        assert retained.headers["cache-control"] == "no-store"
        client.post("/api/reset")
        assert client.get("/api/agent/trace").json()["events"] == []
        assert client.get(trace_url).status_code == 404
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
        assert {preset["id"] for preset in presets} == {"park", "tidy", "sort", "recharge", "apartment", "kitchen_bathroom"}
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