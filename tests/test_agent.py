import asyncio
import base64
import json
import time
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
import pytest
from pydantic import ValidationError

from backend.agent import AgentController, AgentStart, ChatStart, FeedbackRate, FoundryConfig, FoundryModel, ModelProfile, feedback_message, robot_tools
from backend.contracts import AgentObservation
from backend.worker import SimulationWorker


def observation(seq=1):
    return AgentObservation(run_id="test-run", episode_epoch=1, seq=seq, wall_timestamp=0,
                            simulated_time_s=0, frame_ref=f"1-{seq}.png", joints=[], grippers={},
                            head_rad=[0, .7], odometry_m_rad=[0, 0, 0], bumpers=[])


def test_feedback_and_tools_expose_only_robot_inputs():
    message = feedback_message(observation(), b"camera-png")
    sensors = json.loads(message["content"][0]["text"])
    assert set(sensors) == set(AgentObservation.model_fields)
    assert not {"snapshot", "geometry", "bodies", "robot_position", "goals"} & sensors.keys()
    assert base64.b64decode(message["content"][1]["image_url"].split(",")[1]) == b"camera-png"
    assert {tool["name"] for tool in robot_tools()} == {
        "observe", "wait", "drive_base", "set_head", "set_arm_joints", "move_end_effector", "set_gripper", "stop"}


def test_model_feedback_rounds_numbers_without_mutating_physics_values():
    from backend.feedback import compact_numbers, feedback_json
    raw = observation().model_copy(update={"wall_timestamp": 1789115008.2482228,
        "head_rad": [.123456789, -.0000003], "odometry_m_rad": [89.06823068698428, 0.0, -.67891234]})
    before = raw.model_dump()
    text = feedback_message(raw, b"camera")["content"][0]["text"]
    payload = json.loads(text)
    assert payload["head_rad"] == [.123, 0]
    assert payload["odometry_m_rad"] == [89.068, 0, -.679]
    assert payload["wall_timestamp"] == 1789115008.248
    assert payload["run_id"] == raw.run_id and payload["seq"] == raw.seq
    assert raw.model_dump() == before
    assert compact_numbers({"enabled": True, "revision": 123456789, "empty": None}) == {
        "enabled": True, "revision": 123456789, "empty": None}
    assert feedback_json({"value": 1.0, "zero": -0.000001}) == '{"value":1,"zero":0}'
    assert len(text) < len(raw.model_dump_json())


def test_image_batch_and_retained_context_settings_are_bounded():
    fields = {"run_id": "test", "episode_epoch": 0, "goal": "Inspect"}
    defaults = AgentStart(**fields)
    assert defaults.images_per_request == 1 and defaults.context_tokens == 4096
    assert AgentStart(**fields, images_per_request=8, context_tokens=0).context_tokens == 0
    for invalid in ({"images_per_request": 0}, {"images_per_request": 9},
                    {"context_tokens": -1}, {"context_tokens": 32769}):
        with pytest.raises(ValidationError):
            AgentStart(**fields, **invalid)


def test_default_profile_requires_real_connection():
    config = FoundryConfig()
    assert config.public()["models"][0]["label"] == "GPT-5.6 Luna"
    assert [(profile.id, profile.deployment) for profile in config.models] == [
        ("luna", "gpt-5.6-luna"), ("nano", "gpt-5.4-nano"), ("gemma", "gemma4:e2b-it-qat")]
    assert config.models[0].reasoning_efforts[0] == "low"
    assert config.public()["models"][0]["reasoning_efforts"] == ["low", "none", "medium", "high"]
    assert config.models[1].reasoning_efforts == ["low", "medium", "high"]
    assert AgentStart(run_id="test-run", episode_epoch=1, goal="Look around").reasoning == "low"
    assert not config.public()["models"][0]["configured"]
    assert config.public()["models"][2]["configured"]
    assert config.models[2].provider == "ollama" and config.models[2].reasoning_efforts == ["none"]
    assert AgentController(config).state["reasoning"] == "none"
    configured = FoundryConfig(endpoint="https://test.openai.azure.com/", models=[
        ModelProfile(id="luna", label="GPT-5.6 Luna", deployment="actual-luna-deployment")])
    assert configured.endpoint == "https://test.openai.azure.com/openai/v1/"
    assert configured.public()["models"][0]["configured"]


@pytest.mark.parametrize("endpoint", ["https://example.com", "http://192.168.1.1:11434",
    "http://localhost:11434/api", "http://user:secret@127.0.0.1:11434", "http://localhost?token=x", "http://localhost:0"])
def test_ollama_endpoint_rejects_nonlocal_or_credential_bearing_urls(endpoint):
    with pytest.raises(ValidationError):
        FoundryConfig(ollama_endpoint=endpoint)


def test_ollama_config_does_not_require_cloud_credentials():
    config = FoundryConfig(ollama_endpoint="http://localhost:11434/")
    assert config.ollama_endpoint == "http://localhost:11434"
    assert config.public()["default_model_id"] == "gemma"
    assert config.public()["models"][2]["reasoning_efforts"] == ["none"]
    config.ollama_endpoint = ""
    assert not config.configured(config.models[2])


@pytest.mark.parametrize("endpoint", ["http://test.openai.azure.com", "https://example.com",
                                     "https://test.services.ai.azure.com/api/projects/test/extra",
                                     "https://user:secret@test.openai.azure.com",
                                     "https://test.openai.azure.com/?api-key=secret"])
def test_foundry_endpoint_rejects_wrong_or_credential_bearing_urls(endpoint):
    with pytest.raises(ValidationError):
        FoundryConfig(endpoint=endpoint)


async def test_project_endpoint_aliases_and_actual_deployment_label(monkeypatch):
    for name in ("FOUNDRY_ENDPOINT", "FOUNDRY_MODELS_JSON", "FOUNDRY_DEPLOYMENT", "AZURE_AI_MODEL_DEPLOYMENT_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://test.services.ai.azure.com/api/projects/test")
    monkeypatch.setenv("deployment_name", "GPT-5.2")
    config = FoundryConfig.from_environment()
    assert config.models[0].label == "GPT-5.6 Luna" and config.models[0].deployment == "gpt-5.6-luna"
    assert config.models[1].deployment == "gpt-5.4-nano"
    assert config.models[3].deployment == "GPT-5.2" and config.models[3].label == "GPT-5.2"
    assert config.public()["default_model_id"] == "luna"
    assert AgentController(config).state["model_id"] == "luna"
    model = FoundryModel(config)
    try:
        assert model.credential is not None
        assert str(model.client.base_url) == config.endpoint + "/openai/v1/"
    finally:
        await model.close()


@pytest.mark.parametrize("deployment", ["", "gpt-5.6-luna", "custom-luna", "gpt-5.4-nano"])
def test_model_defaults_preserve_legacy_aliases_without_duplicates(monkeypatch, deployment):
    monkeypatch.delenv("FOUNDRY_MODELS_JSON", raising=False)
    monkeypatch.setenv("FOUNDRY_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("FOUNDRY_DEPLOYMENT", deployment)
    monkeypatch.delenv("AZURE_AI_MODEL_DEPLOYMENT_NAME", raising=False)
    monkeypatch.delenv("deployment_name", raising=False)
    config = FoundryConfig.from_environment()
    assert len(config.models) == 3
    assert config.models[0].deployment == ("custom-luna" if deployment == "custom-luna" else "gpt-5.6-luna")
    assert "none" in config.models[0].reasoning_efforts
    assert config.models[1].deployment == "gpt-5.4-nano"
    assert config.public()["default_model_id"] == "luna"
    assert all(profile["configured"] for profile in config.public()["models"])


def test_custom_model_profiles_override_builtins(monkeypatch):
    profiles = [{"id": "custom", "label": "Custom model", "deployment": "my-deployment"}]
    monkeypatch.setenv("FOUNDRY_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("FOUNDRY_DEPLOYMENT", "gpt-5.6-luna")
    monkeypatch.setenv("FOUNDRY_MODELS_JSON", json.dumps(profiles))
    config = FoundryConfig.from_environment()
    assert len(config.models) == 1 and config.models[0].deployment == "my-deployment"
    assert config.public()["default_model_id"] == "custom"
    assert config.models[0].reasoning_efforts == ["low", "medium", "high"]


def test_default_prefers_first_configured_profile_in_explicit_lists():
    config = FoundryConfig(endpoint="https://test.openai.azure.com", models=[
        ModelProfile(id="luna", label="GPT-5.6 Luna", deployment="   "),
        ModelProfile(id="other", label="GPT-5.2", deployment="existing-deployment"),
    ])
    assert config.public()["default_model_id"] == "other"
    config.models[0].deployment = "luna-deployment"
    assert config.public()["default_model_id"] == "luna"
    config.endpoint = ""
    assert not any(profile["configured"] for profile in config.public()["models"])
    assert config.public()["default_model_id"] == "luna"


def model_response(name="drive_base", arguments=None, call_id="call-1", status="completed"):
    item = {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments if arguments is not None else json.dumps({
                "linear_mps": .2, "angular_radps": 0, "duration_s": .5})}
    return SimpleNamespace(status=status, usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                           output_text="Taking a short step", model_extra={},
                           output=[SimpleNamespace(model_dump=lambda **kwargs: item)])


def text_response(text="I can see the room."):
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}
    return SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                           output_text=text, model_extra={}, output=[SimpleNamespace(model_dump=lambda **kwargs: item)])


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.inputs = []
        self.times = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def respond(self, profile, reasoning, goal, inputs):
        self.inputs.append(inputs)
        self.times.append(time.monotonic())
        self.entered.set()
        reply = next(self.replies)
        if reply == "wait":
            await self.release.wait()
            return model_response()
        return reply

    async def close(self):
        self.closed = True


def controller_for(model, request_timeout_s=45):
    config = FoundryConfig(endpoint="https://test.openai.azure.com", models=[
        ModelProfile(id="luna", label="GPT-5.6 Luna", deployment="test-deployment")])
    return AgentController(config, lambda config: model, request_timeout_s=request_timeout_s)


def start_settings(worker, **kwargs):
    return AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.latest["episode_epoch"],
                      goal="Move forward cautiously", **kwargs)


async def wait_for_phase(controller, phase, timeout=8):
    async with asyncio.timeout(timeout):
        while controller.state["phase"] != phase:
            await asyncio.sleep(.02)


async def test_stalled_drive_history_is_explicit_and_repeated_pushes_stop():
    from backend.challenges import get_challenge
    worker = SimulationWorker(challenge=get_challenge("apartment"), pace=False)
    model = ScriptedModel([model_response(arguments=json.dumps({"linear_mps": .1, "angular_radps": 0, "duration_s": 1}),
                                         call_id=f"drive-{turn}") for turn in range(20)])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, feedback_interval_s=.25, max_turns=20))
        await controller.task
        assert controller.state["phase"] == "error" and "no progress" in controller.state["error"]
        assert len(model.inputs) < 20 and worker.latest["stopped"]
        assert not controller.state["auto_wake"]
        assert all(event["error"] == "NO_PROGRESS" for event in controller.state["events"][-3:])
        last_feedback = model.inputs[-1][-1]
        summary = json.loads(last_feedback["content"][-1]["text"])["recent_actions"]
        assert len(summary) == 3 and summary[-1]["error"] == "NO_PROGRESS"
        assert summary[-1]["arguments"]["linear_mps"] == .1
        assert abs(summary[-1]["odometry_delta_m_rad"][0]) < .005
        assert json.dumps(model.inputs[-1]).count("recent_actions") == 1
        recorded = [event for event in controller.trace()["events"] if event["kind"] == "feedback"][-1]
        assert recorded["payload"]["recent_actions"] == summary
    finally:
        await controller.halt()
        await worker.close()


async def test_finished_task_idles_after_five_seconds_and_camera_change_wakes_once():
    import pybullet as bullet
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([text_response(), "wait"])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, feedback_interval_s=.25))
        await controller.task
        assert controller.idle_delay_s == 5 and controller.state["auto_wake"]
        started = time.monotonic()
        await wait_for_phase(controller, "sleeping")
        assert time.monotonic() - started >= 4.9
        assert not controller.active and len(model.inputs) == 1 and worker.sim.ticks == 0
        sequence = worker.sim.seq
        await asyncio.sleep(.6)
        assert len(model.inputs) == 1 and worker.sim.seq == sequence
        await worker.call(lambda sim: bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .8, physicsClientId=sim.client))
        await wait_for_phase(controller, "thinking")
        assert len(model.inputs) == 2 and controller.active
        assert "scene changed while idle" in json.dumps(model.inputs[-1])
        assert controller.state["turns"] == 2 and controller.state["input_tokens"] == 10
        await controller.halt()
        assert not controller.state["auto_wake"] and controller.idle_task is None
    finally:
        await controller.halt()
        await worker.close()


async def test_unachievable_idle_is_cancelled_by_stop_and_chat_can_start_again():
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([model_response("stop", '{"task_status":"unachievable","reason":"No reachable target"}'), text_response()])
    controller = controller_for(model)
    controller.idle_delay_s, controller.camera_poll_s = .05, .02
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker))
        await controller.task
        await wait_for_phase(controller, "sleeping")
        assert controller.state["idle_reason"] == "Task unachievable"
        await controller.halt()
        assert controller.state["phase"] == "stopped" and not controller.state["auto_wake"]
        controller.start(worker, ChatStart(**start_settings(worker).model_dump(), message="What changed?"))
        await controller.task
        assert controller.state["chat_messages"][-1]["role"] == "assistant"
        assert len(model.inputs) == 2
    finally:
        await controller.halt()
        await worker.close()


async def test_static_camera_does_not_idle_pending_inference_and_limits_prevent_auto_restart():
    import pybullet as bullet
    worker = SimulationWorker(pace=False)
    model = ScriptedModel(["wait"])
    controller = controller_for(model)
    controller.idle_delay_s, controller.camera_poll_s = .05, .02
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, max_turns=1))
        await model.entered.wait()
        await asyncio.sleep(.15)
        assert controller.state["phase"] == "thinking" and not controller.state["auto_wake"]
        await controller.halt()
        model = ScriptedModel([text_response()])
        controller.model_factory = lambda config: model
        controller.start(worker, start_settings(worker, max_turns=1))
        await controller.task
        await wait_for_phase(controller, "sleeping")
        assert not controller.state["auto_wake"] and controller.idle_task is None
        await worker.call(lambda sim: bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .8, physicsClientId=sim.client))
        await asyncio.sleep(.15)
        assert len(model.inputs) == 1 and not controller.active
        generation = worker.stop_revision
        worker.stop()
        assert not await worker.resume_manual(expected_stop_revision=generation)
        assert worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


async def test_completed_challenge_stops_goal_loop_without_extra_model_request():
    from backend.challenges import get_challenge
    worker = SimulationWorker(pace=False, challenge=get_challenge("park"))
    arguments = '{"linear_mps":0.3,"angular_radps":0,"duration_s":2}'
    model = ScriptedModel([model_response(arguments=arguments), model_response(arguments=arguments, call_id="park-2")])
    controller = controller_for(model)
    controller.idle_delay_s, controller.camera_poll_s = .05, .02
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, feedback_interval_s=.25))
        await controller.task
        assert worker.latest["challenge"]["status"] == "completed"
        assert len(model.inputs) == 2
        await wait_for_phase(controller, "sleeping")
        assert controller.state["idle_reason"] == "Task completed"
        assert controller.state["outcome"]["source"] == "physics"
        assert "completed_objectives" not in json.dumps(model.inputs)
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("status", ["completed", "unachievable"])
async def test_reported_task_outcomes_are_structured_and_not_physics_verification(status):
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([model_response("stop", json.dumps({"task_status": status, "reason": "Target assessment"}))])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker))
        await controller.task
        outcome = controller.public()["outcome"]
        assert outcome["kind"] == status and outcome["source"] == "agent"
        assert outcome["message"] == "Target assessment" and outcome["timestamp"] > 0
        assert "outcome" not in json.dumps(model.inputs)
        await controller.halt("Manual takeover")
        assert controller.state["outcome"]["kind"] == "interrupted"
        assert controller.state["outcome"]["message"] == "Manual takeover"
    finally:
        await controller.halt()
        await worker.close()


async def test_chat_messages_use_real_feedback_and_bounded_followup_context():
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([text_response(), model_response(), text_response("Movement complete.")])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = ChatStart(**start_settings(worker, feedback_interval_s=.25).model_dump(), message="What do you see?")
        controller.start(worker, settings)
        await controller.task
        assert controller.state["mode"] == "chat" and not controller.active
        assert controller.state["outcome"]["kind"] == "ended"
        assert worker.sim.ticks == 0
        first_session = controller.state["session_id"]
        assert [entry["role"] for entry in controller.state["chat_messages"]] == ["user", "assistant"]
        followup = settings.model_copy(update={"message": "Move forward a little", "conversation_id": first_session})
        controller.start(worker, followup)
        with pytest.raises(RuntimeError):
            controller.start(worker, followup)
        await controller.task
        assert worker.latest["observation"]["odometry_m_rad"][0] > .02
        assert len(controller.state["chat_messages"]) == 5
        assert "What do you see?" in json.dumps(model.inputs[1])
        assert "I can see the room." in json.dumps(model.inputs[1])
        assert model.inputs[1][-1]["content"][1]["type"] == "input_image"
        assert "function_call_output" in json.dumps(model.inputs[2])
        assert all(secret not in json.dumps(model.inputs) for secret in ("snapshot", "geometry", "robot_position", "completed_objectives"))
        assert controller.chat_history.maxlen == 6
        with pytest.raises(RuntimeError, match="conversation changed"):
            controller.start(worker, followup)
        with pytest.raises(ValidationError):
            ChatStart(**start_settings(worker).model_dump(), message="   ")
    finally:
        await controller.halt()
        await worker.close()


async def test_model_drives_real_physics_with_paced_feedback_and_tool_continuation():
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([model_response(), model_response("stop", "{}", "call-2")])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        before = worker.latest["observation"]["odometry_m_rad"][0]
        controller.start(worker, start_settings(worker, feedback_interval_s=.5))
        await controller.task
        assert worker.latest["observation"]["odometry_m_rad"][0] > before + .02
        assert model.times[1] - model.times[0] >= .48
        assert controller.state["phase"] == "completed"
        assert worker.latest["stopped"] and not controller.active and model.closed
        assert controller.state["input_tokens"] == 20
        continuation = model.inputs[1]
        assert continuation[1]["type"] == "function_call"
        assert continuation[2]["type"] == "function_call_output"
        assert continuation[2]["call_id"] == "call-1"
        assert json.loads(continuation[2]["output"])["status"] == "ok"
        assert continuation[-1]["content"][1]["image_url"].startswith("data:image/png;base64,")
        assert all("geometry" not in json.dumps(inputs) and "snapshot" not in json.dumps(inputs) for inputs in model.inputs)
        feed = controller.trace()
        assert [entry["kind"] for entry in feed["events"]] == [
            "session", "feedback", "response", "tool", "result", "feedback", "response", "tool", "result", "session"]
        feedback = feed["events"][1]
        assert feedback["payload"]["observation"] == json.loads(model.inputs[0][0]["content"][0]["text"])
        assert controller.trace_images[feedback["id"]] == base64.b64decode(model.inputs[0][0]["content"][1]["image_url"].split(",")[1])
        assert feed["events"][2]["payload"]["text"] == "Taking a short step"
        assert feed["events"][5]["payload"]["tool_result_call_ids"] == ["call-1"]
        assert feed["events"][4]["payload"]["model_result"] == json.loads(continuation[2]["output"])
        assert "observation" not in feed["events"][4]["payload"]["model_result"]
        assert "observation" in feed["events"][4]["payload"]["result"]
        assert "trace_records" not in controller.public() and "trace_images" not in controller.public()
    finally:
        await controller.halt()
        await worker.close()


async def test_inference_cancel_cannot_move_robot_and_closes_client():
    worker = SimulationWorker(pace=False)
    model = ScriptedModel(["wait"])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        before = worker.latest["observation"]["odometry_m_rad"][:]
        controller.start(worker, start_settings(worker))
        await model.entered.wait()
        assert [entry["kind"] for entry in controller.trace()["events"]] == ["session", "feedback"]
        await controller.halt("Manual takeover")
        model.release.set()
        assert worker.latest["observation"]["odometry_m_rad"] == before
        assert not controller.active and model.closed
        assert controller.state["events"] == [] and worker.latest["stopped"]
        assert not any(entry["kind"] == "response" for entry in controller.trace()["events"])
        assert controller.trace()["events"][-2]["payload"]["reason"] == "Manual takeover"
    finally:
        await worker.close()


@pytest.mark.parametrize("reply, expected_phase", [
    (model_response(arguments='{"linear_mps": 100}'), "completed"),
    (model_response(arguments="not json"), "completed"),
    (model_response(arguments='{"linear_mps":NaN,"angular_radps":0,"duration_s":1}'), "completed"),
    (model_response(name="teleport", arguments="{}"), "completed"),
    (model_response(status="incomplete"), "error"),
    ("wait", "error"),
])
async def test_invalid_or_timed_out_inference_never_moves(reply, expected_phase):
    worker = SimulationWorker(pace=False)
    controller = controller_for(ScriptedModel([reply]), request_timeout_s=.05)
    try:
        await asyncio.wrap_future(worker.ready)
        before = worker.latest["observation"]["odometry_m_rad"][:]
        controller.start(worker, start_settings(worker, max_turns=1))
        await controller.task
        assert worker.latest["observation"]["odometry_m_rad"] == before
        assert controller.state["phase"] == expected_phase and worker.latest["stopped"]
        json.dumps(controller.public(), allow_nan=False)
        json.dumps(controller.trace(), allow_nan=False)
        if expected_phase == "error":
            assert controller.trace()["events"][-1]["payload"]["status"] == "error"
    finally:
        await worker.close()


def test_feedback_rate_is_bounded():
    assert FeedbackRate().feedback_interval_s == 2
    for value in (0, -.5, 31, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            FeedbackRate(feedback_interval_s=value)


async def test_rate_change_applies_to_current_wait():
    worker = SimulationWorker(pace=False)
    model = ScriptedModel([model_response("observe", "{}"), model_response("stop", "{}", "call-2")])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, feedback_interval_s=10))
        async with asyncio.timeout(3):
            while controller.state["phase"] != "waiting":
                await asyncio.sleep(.005)
            controller.set_rate(FeedbackRate(feedback_interval_s=.25))
            await controller.task
        assert .23 <= model.times[1] - model.times[0] < 2
    finally:
        await controller.halt()
        await worker.close()


async def test_repeated_stop_during_actual_motion():
    worker = SimulationWorker(pace=True)
    model = ScriptedModel([model_response(arguments=json.dumps({
        "linear_mps": .2, "angular_radps": 0, "duration_s": 2}))])
    controller = controller_for(model)
    moving = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wrap_future(worker.ready)
        def attach(sim):
            original = sim.on_tick
            def publish():
                original()
                loop.call_soon_threadsafe(moving.set)
            sim.on_tick = publish
        await worker.call(attach)
        controller.start(worker, start_settings(worker))
        await asyncio.wait_for(moving.wait(), 3)
        await asyncio.gather(controller.halt(), controller.halt())
        assert 0 < worker.latest["snapshot"]["simulated_time_s"] < 2
        assert worker.latest["stopped"] and not worker.latest["busy"]
        assert model.closed and not controller.active
        results = [entry for entry in controller.trace()["events"] if entry["kind"] == "result"]
        assert len(results) == 1 and results[0]["payload"]["result"]["status"] == "cancelled"
    finally:
        await worker.close()


@pytest.mark.parametrize("reasoning", ["low", "none"])
@pytest.mark.parametrize("execution_mode", ["single_step", "navigation_plan"])
async def test_responses_sdk_payload_uses_vision_tools_and_selected_reasoning(monkeypatch, reasoning, execution_mode):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only-not-a-secret")
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp-test", "object": "response", "created_at": 0, "status": "completed",
            "model": "test-luna", "output": [], "parallel_tool_calls": False,
            "tool_choice": "auto", "tools": []})
    config = FoundryConfig(endpoint="https://test.openai.azure.com")
    model = FoundryModel(config)
    model.execution_mode = execution_mode
    await model.client.close()
    model.client = AsyncOpenAI(base_url=config.endpoint, api_key="test-only-not-a-secret",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        await model.respond(ModelProfile(id="luna", label="Luna", deployment="test-luna"), reasoning, "Inspect",
                            [feedback_message(observation(), b"image")])
        payload = captured[0]
        assert payload["model"] == "test-luna"
        assert payload["reasoning"] == {"effort": reasoning}
        assert payload["store"] is False and payload["parallel_tool_calls"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["input"][0]["content"][1]["type"] == "input_image"
        assert payload["tools"] == robot_tools(execution_mode)
        assert ("Navigation-plan mode" in payload["instructions"]) == (execution_mode == "navigation_plan")
        assert "temperature" not in payload
    finally:
        await model.close()


async def test_duplicate_call_id_does_not_repeat_motion():
    worker = SimulationWorker(pace=False)
    controller = controller_for(ScriptedModel([model_response(), model_response()]))
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, feedback_interval_s=.25))
        await controller.task
        assert len(controller.state["events"]) == 1
        assert controller.state["phase"] == "error"
        assert worker.latest["snapshot"]["simulated_time_s"] == .5
    finally:
        await worker.close()


@pytest.mark.parametrize("kind", ["multiple", "refusal", "filtered"])
async def test_ambiguous_or_refused_response_never_moves(kind):
    response = model_response()
    if kind == "multiple":
        response.output.append(model_response("stop", "{}", "call-2").output[0])
    elif kind == "refusal":
        response.output.append(SimpleNamespace(model_dump=lambda **kwargs: {
            "type": "message", "content": [{"type": "refusal", "refusal": "Declined"}]}))
    else:
        response.model_extra = {"content_filters": [{"blocked": True}]}
    worker = SimulationWorker(pace=False)
    controller = controller_for(ScriptedModel([response]))
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker))
        await controller.task
        assert controller.state["phase"] == "error" and controller.state["events"] == []
        assert worker.latest["snapshot"]["simulated_time_s"] == 0
    finally:
        await worker.close()


def test_trace_retention_and_incremental_reads():
    controller = AgentController()
    controller.state["session_id"] = "trace-session"
    for index in range(controller.trace_capacity + 3):
        controller._trace("feedback", "Camera + sensors submitted", {"index": index}, image=b"png")
    feed = controller.trace()
    assert len(feed["events"]) == len(controller.trace_images) == controller.trace_capacity
    assert feed["first_id"] == 4 and feed["revision"] == controller.trace_capacity + 3
    assert 1 not in controller.trace_images
    assert len(controller.trace(after=feed["revision"] - 1)["events"]) == 1
    assert controller.trace(after=feed["revision"])["events"] == []


async def test_trace_never_exposes_encrypted_reasoning_or_credentials(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "credential-marker")
    response = model_response("stop", "{}")
    response.output.insert(0, SimpleNamespace(model_dump=lambda **kwargs: {
        "type": "reasoning", "encrypted_content": "private-reasoning-marker", "summary": []}))
    worker = SimulationWorker(pace=False)
    controller = controller_for(ScriptedModel([response]))
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker))
        await controller.task
        serialized = json.dumps(controller.trace())
        assert "private-reasoning-marker" not in serialized and "credential-marker" not in serialized
        assert "encrypted_content" not in serialized and "geometry" not in serialized
    finally:
        await worker.close()


if __name__ == "__main__":
    from contextlib import asynccontextmanager
    import uvicorn
    from backend.app import app, lab, lifespan
    from backend.realtime import RealtimeConfig
    from tests.test_realtime import RealtimeModel, tool_call

    class BrowserModel(ScriptedModel):
        async def respond(self, profile, reasoning, goal, inputs):
            if getattr(self, "execution_mode", "single_step") == "navigation_plan":
                self.inputs.append(inputs)
                state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
                revision = state["revision"]
                call_id = f"navigation-{len(self.inputs)}"
                if not state["steps"]:
                    return model_response("set_navigation_plan", json.dumps({"expected_revision": revision, "steps": [
                        {"skill": skill, "goal": f"Scripted {skill} fixture"} for skill in
                        ["inspect_room", "locate_doorway", "approach", "cross"]]}), call_id=call_id)
                if state["active_step"] == 0 and state["scan_span_rad"] >= .6:
                    return model_response("complete_navigation_skill", json.dumps({"expected_revision": revision,
                        "evidence": "Scripted camera review; not live semantic recognition"}), call_id=call_id)
                if state["active_step"] == 1 and state["remaining_s"] > 0:
                    await self.release.wait()
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": revision, "segments": [
                    {"kind": "head", "yaw_rad": -.8, "pitch_rad": .2, "duration_s": 1},
                    {"kind": "head", "yaw_rad": .8, "pitch_rad": .2, "duration_s": 1}]}), call_id=call_id)
            if not goal.startswith("Respond to the latest chat message."):
                return await super().respond(profile, reasoning, goal, inputs)
            messages = [(index, item["content"][0]["text"]) for index, item in enumerate(inputs)
                        if item.get("role") == "user" and len(item.get("content", [])) == 1 and item["content"][0]["type"] == "input_text"]
            index, message = messages[-1]
            if message == "Hold your reply":
                await self.release.wait()
            if message == "Move forward a little" and not any(item.get("type") == "function_call_output" for item in inputs[index:]):
                return model_response(arguments=json.dumps({"linear_mps": .2, "angular_radps": 0, "duration_s": .5}))
            return text_response("Movement complete." if message == "Move forward a little" else "I can see the room through my head camera.")

    @asynccontextmanager
    async def browser_fixture(application):
        async with lifespan(application):
            lab.agent = AgentController(FoundryConfig(), lambda config: BrowserModel([
                model_response(arguments=json.dumps({"linear_mps": .2, "angular_radps": .4, "duration_s": 2})), "wait"]))
            lab.realtime_config = RealtimeConfig()
            lab.voice_factory = lambda config: RealtimeModel([tool_call(), []])
            yield

    app.router.lifespan_context = browser_fixture
    uvicorn.run(app, host="127.0.0.1", port=8001)