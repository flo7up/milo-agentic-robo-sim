import asyncio
import base64
import json

import httpx
import pytest
from openai.types.responses import Response

from backend.agent import AgentController, ConfiguredModel, FoundryConfig, OllamaModel, feedback_message, ollama_messages, robot_tools
from backend.worker import SimulationWorker
from tests.test_agent import observation, start_settings


def local_response(name="drive_base", arguments=None, **extra):
    return {"model": "gemma4:e2b-it-qat", "done": True, "done_reason": "stop",
        "prompt_eval_count": 100, "eval_count": 20,
        "message": {"role": "assistant", "content": "Moving carefully.", "thinking": "private-marker",
                    "tool_calls": [{"function": {"name": name, "arguments": arguments or {
                        "linear_mps": .2, "angular_radps": 0, "duration_s": .5}}}]}, **extra}


async def adapter_for(handler):
    adapter = OllamaModel(FoundryConfig())
    await adapter.close()
    adapter.client = httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler))
    return adapter


def task_supervision_response(arguments):
    return Response.model_validate({"id": "task-review", "created_at": 1., "object": "response", "model": "luna",
        "output": [{"type": "function_call", "call_id": "task-review-1", "name": "review_task",
            "arguments": json.dumps(arguments)}], "tools": [], "tool_choice": "auto", "parallel_tool_calls": False,
        "usage": {"input_tokens": 80, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 100}, "status": "completed"})


@pytest.mark.parametrize("execution_mode", ["single_step", "navigation_plan"])
async def test_ollama_payload_history_privacy_and_tool_result_continuation(execution_mode):
    captured = []
    def handle(request):
        assert "authorization" not in request.headers and "api-key" not in request.headers
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=local_response())
    adapter = await adapter_for(handle)
    adapter.execution_mode = execution_mode
    profile = FoundryConfig().models[2]
    inputs = [feedback_message(observation(), b"camera-png")]
    try:
        response = await adapter.respond(profile, "none", "Move once", inputs)
        assert response.status == "completed" and response.output_text == "Moving carefully."
        assert response.usage.total_tokens == 120
        assert "private-marker" not in response.model_dump_json()
        call = next(item for item in response.output if item.type == "function_call")
        history = [*inputs, *[item.model_dump(exclude_none=True) for item in response.output],
            {"type": "function_call_output", "call_id": call.call_id, "output": '{"status":"ok"}'},
            {"type": "reasoning", "encrypted_content": "cloud-private-marker"}, feedback_message(observation(2), b"next-camera")]
        await adapter.respond(profile, "none", "Continue", history)
        payload = captured[-1]
        assert payload["think"] is False and payload["stream"] is False
        assert {tool["function"]["name"] for tool in payload["tools"]} == {tool["name"] for tool in robot_tools(execution_mode)}
        assert ("Navigation-plan mode" in payload["messages"][0]["content"]) == (execution_mode == "navigation_plan")
        assert base64.b64decode(payload["messages"][1]["images"][0]) == b"camera-png"
        assert json.loads(payload["messages"][1]["content"]) == observation().model_dump()
        assert any(item.get("tool_name") == "drive_base" and item["content"] == '{"status":"ok"}' for item in payload["messages"])
        assert "private-marker" not in json.dumps(payload)
    finally:
        await adapter.close()


@pytest.mark.parametrize("circuit", [False, True])
async def test_ollama_unified_mission_uses_shared_scoped_contract_and_paired_images(circuit):
    from types import SimpleNamespace
    from backend.agent import FoundryModel, mission_model_contract
    from backend.mission_supervisor import instructions_for, local_response_schema, tools as mission_tools
    captured = []
    decision = {"action": "navigate_frontier", "frontier_id": "observed-1", "reason": "Inspect the opening."}
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=local_response(message={"role": "assistant", "content": json.dumps(decision)}))
    adapter = await adapter_for(handle)
    adapter.unified_mission = True
    adapter.execution_mode = "luna_continuous"
    brief = {"mission": {"plan": {"kind": "room", "completion": "identify"}},
        "available_actions": ["navigate_frontier", "wait"],
        "observation": {"spatial": {"frontiers": [{"frontier_id": "observed-1"}]}}}
    candidate_ids = ["run:0:1:object-0"] if circuit else []
    goal = "Circle the table clockwise" if circuit else "Find Kitchen"
    if circuit:
        brief["mission"]["plan"] = {"kind": "circuit", "target": "table", "circle_direction": "clockwise"}
        brief["available_actions"] = ["circle", "look"]
        brief["observation"]["spatial"]["object_candidates"] = [{"id": candidate_ids[0], "bounds": [.3, .1, .7, .7]}]
        decision = {"action": "circle", "object_candidate_id": candidate_ids[0], "object_label": "table",
            "evidence_text": "Flat top and legs", "circle_direction": "clockwise"}
    inputs = [{"role": "user", "content": [{"type": "input_text", "text": json.dumps(brief)}] + [
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(image).decode()}
        for image in (b"paired-head", b"observed-map")]}]
    try:
        cloud_payloads = []
        async def cloud_respond(**payload):
            cloud_payloads.append(payload)
        cloud = FoundryModel.__new__(FoundryModel)
        cloud.unified_mission = True
        cloud.client = SimpleNamespace(responses=SimpleNamespace(create=cloud_respond))
        await cloud.respond(FoundryConfig().models[0], "low", goal, inputs)
        assert cloud_payloads[0] == {"model": "gpt-5.6-luna", "instructions": instructions_for(brief) + "\nUser goal: " + goal,
            "input": inputs, "tools": mission_tools(brief["available_actions"], ["observed-1"], candidate_ids),
            "parallel_tool_calls": False, "tool_choice": {"type": "function", "name": "guide_mission"},
            "reasoning": {"effort": "low"}, "max_output_tokens": 2048, "store": False}
        response = await adapter.respond(FoundryConfig().models[2], "none", goal, inputs)
        instructions, tools = mission_model_contract(goal, inputs, structured=True)
        payload = captured[0]
        assert payload["format"] == local_response_schema(brief["available_actions"], ["observed-1"], candidate_ids) and "tools" not in payload
        assert [branch["properties"]["action"]["const"] for branch in payload["format"]["anyOf"]] == brief["available_actions"]
        selection_field = "object_candidate_id" if circuit else "frontier_id"
        assert payload["format"]["anyOf"][0]["properties"][selection_field]["enum"] == (candidate_ids if circuit else ["observed-1"])
        if circuit:
            assert "object_bounds" not in payload["format"]["anyOf"][0]["properties"]
        assert payload["messages"][0]["content"].startswith(instructions)
        assert [base64.b64decode(image) for image in payload["messages"][-1]["images"]] == [b"paired-head", b"observed-map"]
        assert response.output[0].name == "guide_mission"
        assert json.loads(response.output[0].arguments) == decision
        assert response.usage.total_tokens == 120
    finally:
        await adapter.close()


@pytest.mark.parametrize("durations", [True, False])
async def test_ollama_retains_bounded_provider_duration_telemetry(durations):
    def handle(request):
        return httpx.Response(200, json=local_response(**({"total_duration": 2_000_000_000,
            "load_duration": 100_000_000, "prompt_eval_duration": 500_000_000, "eval_duration": 1_000_000_000}
            if durations else {"total_duration": "unknown", "load_duration": -1, "prompt_eval_duration": True, "eval_duration": 0})))
    adapter = await adapter_for(handle)
    try:
        response = await adapter.respond(FoundryConfig().models[2], "none", "Wait", [])
        timing = response.model_extra["local_timing"]
        assert timing["request_s"] >= 0 and timing["clock"] == "duration_seconds"
        if durations:
            assert timing["server_total_s"] == 2. and timing["model_load_s"] == .1
            assert timing["prompt_tokens_per_s"] == 200. and timing["generated_tokens_per_s"] == 20.
        else:
            assert not {"server_total_s", "model_load_s", "prompt_processing_s", "generated_tokens_per_s"} & timing.keys()
        assert "private-marker" not in response.model_dump_json()
    finally:
        await adapter.close()


async def test_luna_task_supervision_sets_only_goal_grounded_plan_without_operational_context():
    from types import SimpleNamespace
    from backend.mission import Mission
    from backend.mission_supervisor import establish_task_supervision
    captured = []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            captured.append({"profile": profile, "reasoning": reasoning, "goal": goal, "inputs": inputs})
            return task_supervision_response({"status": "aligned",
                "plan": {"kind": "circuit", "target": "table", "circle_direction": "clockwise"},
                "guidance": "Keep the clockwise circuit objective; reject identify-only completion."})
    mission = Mission("run", 0, 0, 0, 100.)
    state = {"task_supervision": {"requests": 1, "tokens": 100, "input_tokens": 80, "output_tokens": 20}}
    controller = SimpleNamespace(mission=mission, state=state, _check_live=lambda *args: None,
        _trace=lambda *args: None)
    worker = SimpleNamespace(sim=SimpleNamespace(run_id="run"), epoch=0, stop_revision=0, task_revision=0)
    settings = SimpleNamespace(goal="Circle the table clockwise", task_supervisor_reasoning="low")
    review = await establish_task_supervision(controller, worker, settings, Supervisor(), SimpleNamespace(label="Luna"))
    assert mission.plan.kind == "circuit" and mission.plan.circle_direction == "clockwise"
    assert review["operational_authority"] is review["motion_authorized"] is False
    request = json.loads(captured[0]["inputs"][0]["content"][0]["text"])
    assert request == {"contract": "task-supervision-v1", "goal": settings.goal,
        "output_scope": "immutable task plan and short consistency guidance only",
        "forbidden": ["available actions", "coordinates", "routes", "motor commands", "completion claims"]}
    assert "observation" not in request and "available_actions" not in request


async def test_foundry_task_supervision_tool_has_no_operational_action_fields():
    from types import SimpleNamespace
    from backend.agent import FoundryModel
    captured = []
    async def create(**payload):
        captured.append(payload)
        return task_supervision_response({"status": "aligned", "plan": {"kind": "explore"}, "guidance": ""})
    model = FoundryModel.__new__(FoundryModel)
    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    await model.supervise_task(SimpleNamespace(deployment="luna"), "low", "Explore", [{"role":"user","content":[]}])
    payload = captured[0]
    assert payload["tool_choice"] == {"type": "function", "name": "review_task"}
    schema = payload["tools"][0]["parameters"]
    assert set(schema["properties"]) == {"status", "plan", "guidance"}
    assert "available_actions" not in json.dumps(payload) and "motor command" in payload["instructions"]


def test_qwen_payload_labels_luna_guidance_non_authoritative():
    from backend.mission import Mission, MissionPlan
    from backend.mission_supervisor import instructions_for, semantic_payload
    mission = Mission("run", 0, 0, 0, 100.)
    mission.configure(MissionPlan(kind="circuit", target="table", circle_direction="clockwise"))
    observation_payload = observation().model_dump()
    observation_payload["spatial"] = {"frame": "wheel_odometry", "map_id": "map", "localization": {"status": "localized"},
        "places": [], "frontiers": [], "room_observations": []}
    task_review = {"plan": mission.plan.model_dump(), "guidance": "Keep the clockwise circuit objective."}
    payload = semantic_payload(observation_payload, mission, None, None, ["circle", "look"], task_supervision=task_review)
    assert payload["task_supervision"] == {"source": "luna_task_contract", "plan": mission.plan.model_dump(),
        "guidance": task_review["guidance"], "operational_authority": False, "motion_authorized": False,
        "applies_to": "qwen_current_observation_decisions"}
    instructions = instructions_for(payload, structured=True)
    assert "not sensor evidence" in instructions and "Choose every action" in instructions


async def test_luna_task_supervision_rejects_changed_goal_and_action_fields():
    from types import SimpleNamespace
    from backend.mission import Mission, TaskSupervision
    from backend.mission_supervisor import establish_task_supervision
    with pytest.raises(ValueError):
        TaskSupervision.model_validate({"status": "aligned", "plan": {"kind": "room", "target": "Kitchen"},
            "guidance": "Continue", "action": "navigate_frontier"})
    with pytest.raises(ValueError, match="circuit plan"):
        TaskSupervision.model_validate({"status":"aligned","plan":{"kind":"object","target":"table"}},
            context={"goal":"Circle the table clockwise"})
    with pytest.raises(ValueError, match="direction"):
        TaskSupervision.model_validate({"status":"aligned","plan":{"kind":"circuit","target":"table","circle_direction":"clockwise"}},
            context={"goal":"Circle the table counterclockwise"})
    with pytest.raises(ValueError, match="Return Home"):
        TaskSupervision.model_validate({"status":"aligned","plan":{"kind":"place","target":"green bay","return_home":True}},
            context={"goal":"Drive to the green bay and stop"})
    class Supervisor:
        async def respond(self, *args):
            return task_supervision_response({"status": "corrective",
                "plan": {"kind": "object", "target": "television"}, "guidance": "Find another target."})
    mission = Mission("run", 0, 0, 0, 100.)
    controller = SimpleNamespace(mission=mission, state={"task_supervision": {}}, _check_live=lambda *args: None,
        _trace=lambda *args: None)
    worker = SimpleNamespace(sim=SimpleNamespace(run_id="run"), epoch=0, stop_revision=0, task_revision=0)
    settings = SimpleNamespace(goal="Find the yellow cube", task_supervisor_reasoning="low")
    with pytest.raises(ValueError, match="exact noun phrase"):
        await establish_task_supervision(controller, worker, settings, Supervisor(), SimpleNamespace(label="Luna"))
    assert mission.plan is None


@pytest.mark.parametrize("direction", ["clockwise", "counterclockwise"])
@pytest.mark.parametrize("review", [False, True])
def test_full_circuit_goal_preserves_requested_direction(direction, review):
    from backend.challenges import furniture_circuit
    from backend.mission import MissionDecision, TaskSupervision
    goal = furniture_circuit("table", direction).goal
    plan = {"kind": "circuit", "target": "table", "circle_direction": direction}
    validator = TaskSupervision if review else MissionDecision
    payload = {"status": "aligned", "plan": plan} if review else {"action": "plan", "plan": plan}
    assert validator.model_validate(payload, context={"goal": goal}).plan.circle_direction == direction
    plan["circle_direction"] = "counterclockwise" if direction == "clockwise" else "clockwise"
    with pytest.raises(ValueError, match="direction"):
        validator.model_validate(payload, context={"goal": goal})


@pytest.mark.parametrize("goal,direction", [
    ("Keep the object left for counterclockwise travel or right for clockwise travel. Circle the table clockwise.", "clockwise"),
    ("Do not circle the table counterclockwise. Circle the table clockwise.", "clockwise"),
    ("Circle the table clockwise, not counterclockwise.", "clockwise"),
    ("Never circle clockwise; circle the table counterclockwise.", "counterclockwise"),
    ("Circle the table. Do not go clockwise; go counterclockwise around it.", "counterclockwise"),
    ("Complete one anticlockwise lap around the table.", "counterclockwise"),
    ("Drive one anti-clockwise circuit around the table.", "counterclockwise"),
])
def test_circuit_direction_distinguishes_instructions_and_alternatives(goal, direction):
    from backend.mission import MissionPlan, validate_plan_against_goal
    validate_plan_against_goal(MissionPlan(kind="circuit", target="table", circle_direction=direction), goal)
    opposite = "counterclockwise" if direction == "clockwise" else "clockwise"
    with pytest.raises(ValueError, match="direction"):
        validate_plan_against_goal(MissionPlan(kind="circuit", target="table", circle_direction=opposite), goal)


def test_conflicting_circuit_directions_fail_closed():
    from backend.mission import MissionPlan, validate_plan_against_goal
    with pytest.raises(ValueError, match="direction is ambiguous"):
        validate_plan_against_goal(MissionPlan(kind="circuit", target="table"),
            "Circle the table clockwise. Then circle it counterclockwise.")


@pytest.mark.parametrize("ending", ["corrected", "invalid", "budget", "stop"])
async def test_task_contract_correction_is_bounded_and_has_no_operational_context(ending):
    from types import SimpleNamespace
    from backend.mission import Mission
    from backend.mission_supervisor import establish_task_supervision
    requests = []
    state = {"requests": 0, "tokens": 0, "max_requests": 1 if ending == "budget" else 4, "max_tokens": 1000}
    mission = Mission("run", 0, 0, 0, 100.)
    def check(*args):
        if ending == "stop" and len(requests) == 2:
            raise asyncio.CancelledError
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            assert mission.plan is None and not mission.receipts
            requests.append(json.loads(inputs[0]["content"][0]["text"]))
            state["requests"] += 1
            state["tokens"] += 100
            target = "yellow cube" if len(requests) == 2 and ending != "invalid" else "x" * 161
            return task_supervision_response({"status": "aligned", "plan": {"kind": "object", "target": target}})
    controller = SimpleNamespace(mission=mission, state={"task_supervision": state}, _check_live=check, _trace=lambda *args: None)
    settings = SimpleNamespace(goal="Find and approach the yellow cube", task_supervisor_reasoning="low")
    if ending == "corrected":
        result = await establish_task_supervision(controller, object(), settings, Supervisor(), SimpleNamespace(label="Luna"))
        assert mission.plan.target == "yellow cube" and result["motion_authorized"] is False
    else:
        with pytest.raises(asyncio.CancelledError if ending == "stop" else ValueError):
            await establish_task_supervision(controller, object(), settings, Supervisor(), SimpleNamespace(label="Luna"))
        assert mission.plan is None and not mission.receipts
    assert len(requests) == (1 if ending == "budget" else 2)
    assert all("observation" not in request and "available_actions" not in request for request in requests)
    if len(requests) == 2:
        assert "validation_feedback" in requests[1] and requests[1]["goal"] == settings.goal


async def test_luna_task_supervision_uses_separate_bounded_budget_and_closes_client():
    from types import SimpleNamespace
    from backend.agent import BudgetedTaskSupervisor
    class Model:
        closed = False
        async def supervise_task(self, profile, reasoning, goal, inputs):
            return task_supervision_response({"status": "aligned", "plan": {"kind": "explore"}, "guidance": ""})
        async def close(self):
            self.closed = True
    model = Model()
    state = {"session_id": "session", "inference_budget": {"requests": 7, "tokens": 700},
        "task_supervision": {"requests": 0, "tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "max_requests": 1, "max_tokens": 1000, "usage_unknown": False}}
    controller = SimpleNamespace(state=state, active=True, _check_live=lambda *args: None)
    settings = SimpleNamespace(task_supervisor_reasoning="low")
    supervisor = BudgetedTaskSupervisor(model, controller, object(), settings)
    response = await supervisor.respond(SimpleNamespace(), "none", "Explore", [])
    assert response.status == "completed"
    assert state["task_supervision"] == {"requests": 1, "tokens": 100, "input_tokens": 80, "output_tokens": 20,
        "max_requests": 1, "max_tokens": 1000, "usage_unknown": False}
    assert state["inference_budget"] == {"requests": 7, "tokens": 700}
    assert await supervisor.respond(SimpleNamespace(), "none", "Explore", []) is None
    await supervisor.close()
    assert model.closed


async def test_hybrid_controller_passes_separate_models_and_closes_both(monkeypatch):
    from backend import mission_supervisor
    from backend.agent import AgentStart, ModelProfile
    models = []
    class Model:
        closed = False
        async def close(self):
            self.closed = True
    def factory(config):
        model = Model()
        models.append(model)
        return model
    captured = {}
    async def run(controller, worker, settings, model, profile, stop_revision, *, task_supervisor, task_profile):
        captured.update(primary=profile.id, task=task_profile.id, supervisor=task_supervisor,
            state=controller.state["task_supervision"].copy())
    monkeypatch.setattr(mission_supervisor, "run", run)
    config = FoundryConfig(endpoint="https://test.openai.azure.com", models=[
        ModelProfile(id="luna", label="Luna", deployment="luna", provider="foundry", reasoning_efforts=["low"]),
        ModelProfile(id="qwen", label="Qwen", deployment="qwen", provider="ollama", context_window=16384)])
    worker = SimulationWorker(rendering="tiny", pace=False)
    controller = AgentController(config, model_factory=factory)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, goal="Explore",
            execution_mode="luna_continuous", unified_mission=True, images_per_request=2,
            model_id="qwen", reasoning="none", task_supervisor_model_id="luna", task_supervisor_reasoning="low")
        controller.start(worker, settings)
        await controller.task
        assert captured["primary"] == "qwen" and captured["task"] == "luna"
        assert captured["state"]["requests"] == 0 and captured["state"]["max_requests"] == 4
        assert len(models) == 2 and all(model.closed for model in models)
    finally:
        await controller.halt()
        await worker.close()


async def test_qwen_context_is_bounded_independently_of_history_setting():
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=local_response())
    adapter = await adapter_for(handle)
    adapter.context_tokens = 32768
    adapter.images_per_request = 8
    profile = next(entry for entry in FoundryConfig().models if entry.id == "qwen")
    try:
        await adapter.respond(profile, "none", "Inspect", [feedback_message(observation(), b"camera")])
        assert captured[0]["options"]["num_ctx"] == 16384
        assert profile.deployment == "qwen3-vl:4b-instruct-q4_K_M" and profile.reasoning_efforts == ["none"]
        assert FoundryConfig(endpoint="https://test.openai.azure.com").public()["default_model_id"] == "luna"
    finally:
        await adapter.close()


@pytest.mark.parametrize("condition", ["installed", "missing", "text_only", "offline", "invalid", "cloud"])
async def test_local_readiness_is_metadata_only_and_fails_closed(condition):
    requests = []
    def handle(request):
        requests.append(request.url.path)
        if condition == "offline":
            raise httpx.ConnectError("offline")
        if condition == "missing":
            return httpx.Response(404)
        if condition == "invalid":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"capabilities": ["completion", "vision"] if condition == "installed" else ["completion"],
            "details": {"quantization_level": "Q4_K_M"}, "template": "do-not-return-template",
            "remote_host": "https://ollama.com" if condition == "cloud" else None})
    adapter = await adapter_for(handle)
    try:
        result = await adapter.readiness(next(entry for entry in FoundryConfig().models if entry.id == "qwen"))
        assert result["ready"] == (condition == "installed")
        assert requests == ["/api/show"] and "do-not-return-template" not in json.dumps(result)
    finally:
        await adapter.close()


@pytest.mark.parametrize("ending", ["normal", "stop", "invalid", "incomplete", "offline"])
async def test_local_mission_pipeline_preserves_worker_authority(tmp_path, ending):
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    entered = asyncio.Event()
    captured = []
    async def handle(request):
        captured.append(json.loads(request.content))
        entered.set()
        if ending == "stop":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                pass
        if ending == "offline":
            raise httpx.ConnectError("offline")
        decision = ({"action": "plan",
            "plan": {"kind": "room", "target": "Kitchen", "completion": "identify"}}
            if len(captured) == 1 else {"action": "turn", "turn_rad": .2})
        return httpx.Response(200, json=local_response(
            message={"role": "assistant", "content": "[invalid" if ending == "invalid" else json.dumps(decision)},
            done_reason="length" if ending == "incomplete" else "stop"))
    adapter = await adapter_for(handle)
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "map.sqlite3"))
        controller.start(worker, start_settings(worker, model_id="qwen", reasoning="none", execution_mode="luna_continuous",
            unified_mission=True, images_per_request=2, map_context=True, mission_budget_s=45., max_turns=2).model_copy(
                update={"goal": "Find the Kitchen"}))
        await asyncio.wait_for(entered.wait(), 35.)
        if ending == "stop":
            before = worker.sim.ticks
            await controller.halt()
            assert worker.sim.ticks == before
        else:
            await asyncio.wait_for(controller.task, 45.)
        assert worker.latest["stopped"] and not controller.active and adapter.client.is_closed
        assert not worker.sim.proximity_sensors().collisions
        assert all(payload["model"] == "qwen3-vl:4b-instruct-q4_K_M" for payload in captured)
        feedback = next(event for event in controller.trace()["events"] if event["kind"] == "feedback")
        assert captured[0]["messages"][0]["content"] == feedback["payload"]["effective_instructions"]
        assert base64.b64decode(captured[0]["messages"][-1]["images"][0]) == controller.trace_images[feedback["id"]]
        assert captured[0]["format"] == feedback["payload"]["response_format"]
        if ending == "normal":
            assert len(captured) == 2 and worker.sim.odometry[2] > .1
            assert controller.state["error"] is None, controller.state["error"]
        else:
            assert len(captured) == 1 and abs(worker.sim.odometry[2]) < .01
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("condition", ["ready", "missing", "stop", "disconnect", "config", "handoff_stop", "warmup_stop", "warmup_disconnect", "warmup_error"])
def test_local_mission_api_preflight_never_falls_back(monkeypatch, condition):
    from importlib import import_module
    from fastapi.testclient import TestClient
    module = import_module("backend.app")
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(FoundryConfig, "from_environment", classmethod(lambda cls: cls(endpoint="https://test.openai.azure.com")))
    started = []
    async def readiness(config, model_id):
        assert model_id == "qwen"
        if condition == "stop":
            module.lab.worker.stop()
        elif condition == "disconnect":
            module.lab.worker.stop()
            module.lab.connections = 0
        elif condition == "config":
            module.lab.agent.config = FoundryConfig()
        return {"ready": condition != "missing", "message": "Local model missing"}
    monkeypatch.setattr(module, "check_local_supervisor", readiness)
    async def prepare(config, model_id, image):
        assert model_id == "qwen" and image.startswith(b"\x89PNG")
        if condition == "warmup_stop":
            module.lab.worker.stop()
        elif condition == "warmup_disconnect":
            module.lab.worker.stop()
            module.lab.connections = 0
        elif condition == "warmup_error":
            raise ValueError("Model warmup failed")
        return {"duration_s": 1., "motion_authorized": False}
    monkeypatch.setattr(module, "prepare_local_supervisor", prepare)
    with TestClient(module.app) as client, client.websocket_connect("/api/live") as socket:
        initial = socket.receive_json()
        monkeypatch.setattr(module.lab.agent, "start", lambda worker, settings: started.append(settings.model_id))
        if condition == "handoff_stop":
            async def finish(reason):
                module.lab.worker.stop()
            monkeypatch.setattr(module.lab, "finish_home_recording", finish)
        response = client.post("/api/mission/start", json={"run_id": initial["run_id"], "episode_epoch": initial["episode_epoch"],
            "unified_mission": True, "execution_mode": "luna_continuous", "model_id": "qwen", "reasoning": "none",
            "images_per_request": 2, "goal": "Find Kitchen"})
        assert response.status_code == (200 if condition == "ready" else 409), response.text
        assert started == (["qwen"] if condition == "ready" else [])
        assert module.lab.worker.latest["snapshot"] == initial["snapshot"]
        if condition in {"disconnect", "warmup_disconnect"}:
            module.lab.connections = 1


@pytest.mark.parametrize("condition", ["ready", "incomplete", "tool", "malformed", "offline"])
async def test_local_preparation_is_bounded_vision_only_without_actions(monkeypatch, condition):
    from backend.agent import prepare_local_supervisor
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        if condition == "offline":
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"done": True, "done_reason": "length" if condition == "incomplete" else "stop",
            "message": {"content": "invalid" if condition == "malformed" else '{"ready":true}',
                "tool_calls": [{"function": {"name": "drive_base"}}] if condition == "tool" else []},
            "prompt_eval_count": 100, "eval_count": 5})
    client_type = httpx.AsyncClient
    options = []
    def client(**kwargs):
        options.append(kwargs)
        return client_type(**kwargs, transport=httpx.MockTransport(handle))
    monkeypatch.setattr("backend.agent.httpx.AsyncClient", client)
    if condition == "ready":
        result = await prepare_local_supervisor(FoundryConfig(), "qwen", b"paired-camera")
        assert result["motion_authorized"] is False and result["input_tokens"] == 100
    else:
        with pytest.raises(ValueError):
            await prepare_local_supervisor(FoundryConfig(), "qwen", b"paired-camera")
    assert options[0]["timeout"] == 180. and options[0]["trust_env"] is False
    assert captured[0]["options"] == {"num_ctx": 16384, "num_predict": 32, "temperature": 0}
    assert "tools" not in captured[0] and captured[0]["think"] is False
    assert base64.b64decode(captured[0]["messages"][-1]["images"][0]) == b"paired-camera"


def test_local_json_schema_never_offers_irrelevant_action_parameters():
    from backend.mission import MissionDecision
    from backend.mission_supervisor import local_response_schema
    schema = local_response_schema(["identify_target", "explore"])
    identify, explore = schema["anyOf"]
    assert set(identify["properties"]) == {"action", "reason", "object_bounds", "evidence_text"}
    assert identify["additionalProperties"] is False
    assert explore["properties"]["objective_travel_m"]["exclusiveMinimum"] == 0
    with pytest.raises(ValueError, match="greater than 0"):
        MissionDecision(action="explore", objective_travel_m=0)
    with pytest.raises(ValueError, match="circuit plan"):
        MissionDecision.model_validate({"action":"plan","plan":{"kind":"object","target":"table"}},
            context={"goal":"Circle the table clockwise"})
    with pytest.raises(ValueError, match="object plan"):
        MissionDecision.model_validate({"action":"plan","plan":{"kind":"explore"}},
            context={"goal":"Find and approach the yellow cube"})
    with pytest.raises(ValueError, match="room plan"):
        MissionDecision.model_validate({"action":"plan","plan":{"kind":"explore"}},
            context={"goal":"Find and enter the kitchen"})
    with pytest.raises(ValueError, match="place plan"):
        MissionDecision.model_validate({"action":"plan","plan":{"kind":"explore"}},
            context={"goal":"Drive to the green parking bay and stop"})
    initial = local_response_schema(["plan", "identify_target", "look"])
    assert all("plan" in branch["properties"] for branch in initial["anyOf"])
    assert "MissionPlan" in initial["$defs"]
    assert all("$defs" not in branch for branch in initial["anyOf"])
    assert all({"action", "plan"}.issubset(branch["required"]) for branch in initial["anyOf"])
    assert "evidence_text" in identify["required"]
    assert identify["properties"]["evidence_text"]["minLength"] == 1
    assert "evidence_text" in initial["anyOf"][1]["required"]
    plans = {branch["properties"]["kind"]["const"]: branch for branch in initial["$defs"]["MissionPlan"]["anyOf"]}
    assert plans["explore"]["properties"]["target"]["const"] == ""
    assert plans["explore"]["properties"]["completion"]["const"] == "arrive"
    assert plans["room"]["properties"]["completion"]["enum"] == ["identify", "arrive"]
    assert plans["movement"]["properties"]["movements"]["minItems"] == 1
    assert plans["movement"]["properties"]["return_home"]["const"] is False
    assert plans["movement"]["required"] == ["kind", "movements"]
    assert identify["properties"]["object_bounds"]["items"]["minimum"] == 0
    with pytest.raises(ValueError, match="Visual identification"):
        MissionDecision(action="explore", plan={"kind": "explore", "completion": "identify"})


def test_unified_parking_exposes_only_observed_region_ids_and_required_selection():
    from types import SimpleNamespace
    from backend.mission import Mission, MissionDecision, MissionPlan
    from backend.mission_supervisor import available_actions, local_response_schema, semantic_payload
    mission = Mission("run", 0, 0, 0, 100.)
    mission.configure(MissionPlan(kind="place", target="green parking bay"))
    region = {"id": "floor-0", "color": "green", "visible": True, "age_s": .1,
        "source": "head_rgb_depth", "frame": "wheel_odometry", "center_m": [1., 0.], "polygon_m": [[0., 0.]]}
    observed = observation().model_dump()
    observed["spatial"] = {"floor_regions": [region]}
    actions = available_actions(mission, SimpleNamespace(spatial=observed["spatial"]), None)
    assert "park_floor" in actions
    payload = semantic_payload(observed, mission, None, None, actions)
    offered = payload["observation"]["spatial"]["floor_regions"][0]
    assert offered["id"] == "floor-0" and offered["source"] == "head_rgb_depth"
    assert "center_m" not in offered and "polygon_m" not in offered
    schema = local_response_schema(["park_floor"])["anyOf"][0]
    assert schema["required"] == ["action", "floor_target_id", "parking_maneuver_id"]
    with pytest.raises(ValueError, match="floor-region"):
        MissionDecision(action="park_floor")
    region["complete_view"] = True
    ready_actions = available_actions(mission, SimpleNamespace(spatial=observed["spatial"]), None)
    assert ready_actions == ["park_floor", "look", "report_observation", "wait"]
    region["color"] = "cyan"
    assert "navigate_frontier" not in ready_actions and "explore" not in ready_actions
    assert "explore" in available_actions(mission, SimpleNamespace(spatial=observed["spatial"]), None)
    region["color"] = "green"
    region["age_s"] = 16.
    assert "park_floor" not in available_actions(mission, SimpleNamespace(spatial=observed["spatial"]), None)


@pytest.mark.parametrize("offset,fault", [(0., None), (.1, None), (-.1, None),
    ("park_left", None), ("park_right", None), ("park_far", None), (0., "reply_stop"),
    ("park_left", "motion_stop"), ("park_left", "sensor_loss"), ("park_right", "dwell_stop"),
    (0., "motion_stop"), (0., "dwell_stop"), (0., "sensor_loss"), (0., "unknown_region"),
    (0., "expired_target"), (0., "wrong_color"), (0., "wrong_place")])
async def test_unified_observed_parking_completes_through_production_contract(tmp_path, record_property, monkeypatch, offset, fault):
    import time
    from backend import mission_supervisor
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from tests.test_agent import ScriptedModel, controller_for, model_response

    class ParkingModel(ScriptedModel):
        requires_grounded_plan = True

        async def respond(self, profile, reasoning, goal, inputs):
            self.inputs.append(inputs)
            brief = json.loads(inputs[-1]["content"][0]["text"])
            if len(self.inputs) == 1:
                assert brief["available_actions"] == ["plan"]
                decision = {"action": "plan", "plan": {"kind": "place",
                    "target": "cyan parking bay" if fault == "wrong_color" else "green parking bay"}}
            else:
                assert "park_floor" in brief["available_actions"]
                if isinstance(offset, str):
                    parking = brief["observation"]["spatial"]["parking"]
                    if not parking["maneuvers"] and fault == "sensor_loss":
                        decision = {"action": "report_observation", "evidence_text": "Depth unavailable; holding position."}
                    else:
                        assert parking["maneuvers"], brief
                        maneuver = parking["maneuvers"][0]
                        decision = {"action": "park_floor", "floor_target_id": parking["floor_target_id"],
                            "parking_maneuver_id": maneuver["id"]}
                    assert "explore" not in brief["available_actions"] and "navigate_frontier" not in brief["available_actions"]
                else:
                    target = next(region for region in brief["observation"]["spatial"]["floor_regions"]
                        if region["color"] == "green" and region["visible"])
                    decision = {"action": "park_floor", "floor_target_id": target["id"]}
                if fault == "reply_stop":
                    worker.stop()
                elif fault == "unknown_region":
                    decision["floor_target_id"] = "invented-region"
                elif fault == "wrong_place":
                    decision = {"action": "navigate_place", "place_id": brief["mission"]["home_id"]}
            return model_response("guide_mission", json.dumps(decision), call_id=f"parking-{len(self.inputs)}")

    model = ParkingModel([])
    challenge = get_challenge(offset) if isinstance(offset, str) else get_challenge("park").model_copy(update={"initial_xy": [offset, 0.]})
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    controller = controller_for(model)
    original_trace = controller._trace
    def trace(kind, title, payload, **kwargs):
        original_trace(kind, title, payload, **kwargs)
        if fault == "dwell_stop" and title == "Floor parking phase" and payload["phase"] == "verifying":
            worker.stop()
    monkeypatch.setattr(controller, "_trace", trace)
    if fault == "expired_target":
        original_park = mission_supervisor.park_observed_floor
        async def expired(*args):
            return await original_park(*args[:-1], {**args[-1], "captured_at": time.monotonic() - 121.})
        monkeypatch.setattr(mission_supervisor, "park_observed_floor", expired)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "parking.sqlite3"))
        original_tick = worker.home_mission.tick
        injected = []
        contacts = []
        def tick():
            original_tick()
            contacts.extend(worker.sim.proximity_sensors().collisions)
            if not injected and worker.sim.path_length > .15 and fault in {"motion_stop", "sensor_loss"}:
                injected.append(fault)
                if fault == "motion_stop":
                    worker.stop()
                else:
                    monkeypatch.setattr(worker, "_sample_spatial", lambda *args, **kwargs: None)
                    worker.spatial_error = "Injected missing depth"
        monkeypatch.setattr(worker.home_mission, "tick", tick)
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            images_per_request=2, map_context=True, mission_budget_s=120.,
            max_turns=12 if isinstance(offset, str) else 2, max_model_requests=12 if isinstance(offset, str) else 2).model_copy(
                update={"goal": "Drive to the cyan parking bay and stop" if fault == "wrong_color" else worker.challenge.goal}))
        await asyncio.wait_for(controller.task, 135.)
        measured = await worker.call(lambda sim: {"challenge": sim.challenge_status(), "distance_m": sim.path_length,
            "contacts": len(sim.proximity_sensors().collisions), "ticks": sim.ticks, "pose_m_rad": sim.odometry.tolist()})
        record_property("unified_parking", json.dumps({"evidence": "scripted_inference_real_enhanced_physics",
            "offset_m": offset, "fault": fault, "requests": len(model.inputs), "mission": controller.mission.state(),
            "decision_timing": [event["payload"]["timing"] for event in controller.trace()["events"]
                if event["kind"] == "response" and event["payload"].get("timing")],
            "parking_phases": [event["payload"] for event in controller.trace()["events"] if event["title"] == "Floor parking phase"],
            "rejections": [event["payload"]["result"] for event in controller.trace()["events"]
                if event["title"] == "Mission capability feedback" and event["payload"]["result"].get("status") == "rejected"], **measured}))
        if isinstance(offset, str) and fault == "sensor_loss":
            assert controller.state["error"] == "Mission feedback requires fresh observed sensors"
        else:
            assert controller.state["error"] is None, controller.trace()
        if fault:
            assert "target" not in controller.mission.receipts and controller.mission.phase != "completed"
            if fault in {"motion_stop", "sensor_loss"}:
                assert injected and measured["distance_m"] > .15
            elif fault != "dwell_stop":
                assert measured["distance_m"] < .01
        else:
            assert controller.mission.phase == "completed", controller.trace()
            assert measured["challenge"]["status"] == "completed" and measured["distance_m"] > .8
            assert controller.mission.receipts["target"]["status"] == "floor_parking_verified"
        assert measured["contacts"] == 0 and not contacts
        assert 3 <= len(model.inputs) <= 12 if isinstance(offset, str) else len(model.inputs) == 2
        assert worker.latest["stopped"] and not controller.active
        assert worker.navigation is None or not worker.navigation.buffer
        await asyncio.sleep(.2)
        assert await worker.call(lambda sim: sim.ticks) == measured["ticks"]
    finally:
        await controller.halt()
        await worker.close()


def test_circuit_plan_exposes_only_inspection_and_circle_actions():
    from types import SimpleNamespace
    from backend.mission import Mission, MissionPlan
    from backend.mission_supervisor import available_actions
    mission=Mission("run",0,0,0,100.);mission.configure(MissionPlan(kind="circuit",target="table"))
    observation=SimpleNamespace(spatial={"frontiers":[{"frontier_id":"1:0"}]})
    actions=available_actions(mission,observation,None)
    assert "circle" in actions and {"look","turn","wait"}<set(actions)
    assert "explore" not in actions and "navigate_frontier" not in actions


@pytest.mark.parametrize("kind",["object","room","place"])
def test_semantic_search_prefers_exact_frontier_over_generic_explore(kind):
    from types import SimpleNamespace
    from backend.mission import Mission, MissionPlan
    from backend.mission_supervisor import available_actions
    mission=Mission("run",0,0,0,100.);mission.configure(MissionPlan(kind=kind,target="target"))
    with_frontier=SimpleNamespace(spatial={"frontiers":[{"frontier_id":"1:0"}],"places":[]})
    actions=available_actions(mission,with_frontier,None)
    assert "navigate_frontier" in actions and "explore" not in actions
    without_frontier=SimpleNamespace(spatial={"frontiers":[],"places":[]})
    assert "explore" in available_actions(mission,without_frontier,None)


def test_shared_mission_schema_requires_action_evidence_and_preserves_whole_goal():
    from types import SimpleNamespace
    from backend.mission import Mission, MissionPlan
    from backend.mission_supervisor import available_actions, instructions_for, tools
    schema = tools(["plan", "identify_target", "circle", "explore", "report_observation"])[0]["parameters"]
    branches = {branch["properties"]["action"]["enum"][0]: branch for branch in schema["anyOf"]}
    assert {"action", "plan", "object_bounds", "object_label", "circle_direction"} == set(branches["circle"]["required"])
    assert "evidence_text" in branches["identify_target"]["required"]
    assert "search_target" not in branches["explore"]["properties"]
    assert "room_label" not in branches["plan"]["properties"]
    assert branches["identify_target"]["additionalProperties"] is False
    assert "search_target" in branches["report_observation"]["properties"]
    guidance = instructions_for({"mission": {"plan": None}})
    assert "identify then circle is kind=circuit/completion=arrive" in guidance
    assert "Kitchen with cooking appliances" not in guidance
    mission = Mission("run", 1, 0, 0, deadline=60., clock=lambda: 0.)
    mission.configure(MissionPlan(kind="object", target="yellow cube"))
    observation = SimpleNamespace(spatial={"frontiers": [{"frontier_id": "observed-1"}]})
    assert "navigate_frontier" in available_actions(mission, observation, None)
    mission.begin("exploring", mission.authority)
    assert "navigate_frontier" not in available_actions(mission, observation, None)


@pytest.mark.parametrize("target,valid", [("yellow cube", True), ("YELLOW   CUBE", True),
    ("Kitchen with cooking appliances", False), ("red cube", False), ("", False), ("low", False)])
def test_real_model_plan_target_is_grounded_in_user_goal(target, valid):
    from backend.agent import ConfiguredModel, FoundryModel, OllamaModel
    from backend.mission import MissionDecision
    payload = {"action": "explore", "plan": {"kind": "object", "target": target, "completion": "arrive"}}
    assert all(adapter.requires_grounded_plan for adapter in (ConfiguredModel, FoundryModel, OllamaModel))
    if valid:
        assert MissionDecision.model_validate(payload, context={"goal": "Find the yellow cube and approach it."}).plan.target == target
    else:
        with pytest.raises(ValueError, match="original user goal"):
            MissionDecision.model_validate(payload, context={"goal": "Find the yellow cube and approach it."})


async def test_grounded_plan_rejection_does_not_authorize_motion(tmp_path):
    import asyncio
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    model = ScriptedModel([model_response("guide_mission", json.dumps({"action": "explore",
        "plan": {"kind": "room", "target": "Kitchen with cooking appliances", "completion": "identify"}}),
        call_id=f"wrong-target-{index}") for index in range(3)])
    model.requires_grounded_plan = True
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "grounded.sqlite3"))
        controller.start(worker, start_settings(worker,
            execution_mode="luna_continuous", unified_mission=True, images_per_request=2,
            mission_budget_s=60., max_turns=3, max_model_requests=3).model_copy(update={"goal": "Find the yellow cube and approach it."}))
        await asyncio.wait_for(controller.task, 70.)
        assert controller.mission.plan is None and controller.mission.rejections == 3
        assert "original user goal" in controller.state["error"]
        assert worker.home_mission.task is None and abs(worker.sim.odometry[0]) < .005
        assert worker.latest["stopped"] and not controller.mission.receipts
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("fault", [None, "stale", "image", "stop"])
async def test_grounded_find_recovers_with_current_evidence_and_no_approach(tmp_path, monkeypatch, fault):
    import asyncio
    import time
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    wrong = {"action": "explore", "plan": {"kind": "room", "target": "Kitchen", "completion": "identify"}}
    plan = {"action": "plan", "plan": {"kind": "object", "target": "green parking bay", "completion": "identify"}}
    corrected = {"action": "identify_target",
        "object_bounds": [.3, .2, .7, .8], "evidence_text": "Scripted current-camera green floor marking"}
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"grounded-find-{index}")
        for index, reply in enumerate([wrong, plan, corrected, corrected, corrected])])
    model.requires_grounded_plan = True
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    controller = controller_for(model)
    original = worker.identify_mission_target
    async def identify(mission, sensor, image, observation, decision):
        if fault == "stale":
            sensor = sensor.model_copy(update={"captured_at": time.monotonic()-16.})
        elif fault == "image":
            image = b"not-the-paired-camera"
        elif fault == "stop":
            worker.stop()
        return await original(mission, sensor, image, observation, decision)
    monkeypatch.setattr(worker, "identify_mission_target", identify)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "identified.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            images_per_request=2, mission_budget_s=60., max_turns=5, max_model_requests=5).model_copy(
                update={"goal": "Find the green parking bay. Stop without approaching it."}))
        await asyncio.wait_for(controller.task, 70.)
        assert controller.mission.plan.target == "green parking bay"
        assert controller.mission.rejections == (3 if fault in {"stale", "image"} else 0)
        payloads = [json.loads(inputs[-1]["content"][0]["text"]) for inputs in model.inputs]
        assert payloads[0]["available_actions"] == payloads[1]["available_actions"] == ["plan"]
        assert all("plan" not in payload["available_actions"] and "identify_target" in payload["available_actions"]
            for payload in payloads[2:])
        assert worker.home_mission.task is None and abs(worker.sim.odometry[0]) < .005
        assert not worker.sim.proximity_sensors().collisions and worker.latest["stopped"]
        if fault:
            assert not controller.mission.receipts and controller.mission.phase != "completed"
        else:
            receipt = controller.mission.receipts["target"]
            assert controller.mission.phase == "completed" and len(model.inputs) == 3
            assert receipt["motion_authorized"] is False and receipt["arrival_verified"] is False
            assert receipt["identity_verified"] is False and receipt["image_sha256"]
        frozen = await worker.call(lambda sim: (sim.ticks, sim.odometry.tolist()))
        await asyncio.sleep(.2)
        assert await worker.call(lambda sim: (sim.ticks, sim.odometry.tolist())) == frozen
    finally:
        await controller.halt()
        await worker.close()


def test_qwen_variant_records_transport_bounds_and_explicit_fallback():
    from backend.agent import AgentStart
    from backend.experiment_variants import variant_snapshot
    profile = next(entry for entry in FoundryConfig().models if entry.id == "qwen")
    settings = AgentStart(run_id="variant", episode_epoch=0, execution_mode="luna_continuous", unified_mission=True,
        model_id="qwen", reasoning="none", images_per_request=2, goal="Find Kitchen")
    variant = variant_snapshot(settings, profile)["model_variant"]
    assert variant["provider"] == "ollama" and variant["profile_id"] == "qwen"
    assert variant["configuration"]["semantic_review_interval_s"] == 3.
    assert variant["configuration"]["semantic_review_travel_m"] == .75
    assert variant["configuration"]["semantic_reviews_require_stopped_evidence"] is True
    assert variant["configuration"]["local_context_window"] == 16384
    assert variant["configuration"]["local_transport"] == "mission_json_v1"
    assert variant["configuration"]["automatic_fallback"] is False and variant["underlying_identity_verified"] is False


async def test_router_selects_ollama_without_constructing_foundry(monkeypatch):
    adapter = await adapter_for(lambda request: httpx.Response(200, json=local_response()))
    monkeypatch.setattr("backend.agent.OllamaModel", lambda config: adapter)
    def unexpected(config):
        raise AssertionError("Foundry must not be constructed for an Ollama session")
    monkeypatch.setattr("backend.agent.FoundryModel", unexpected)
    routed = ConfiguredModel(FoundryConfig())
    try:
        response = await routed.respond(FoundryConfig().models[2], "none", "Move once", [feedback_message(observation(), b"camera")])
        assert response.status == "completed"
    finally:
        await routed.close()
    assert adapter.client.is_closed


async def test_ollama_multi_image_payload_and_context_window():
    from backend.feedback import camera_batch
    captured = []
    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=local_response("stop"))
    adapter = await adapter_for(handle)
    adapter.context_tokens = 32768
    adapter.images_per_request = 8
    message, _ = camera_batch(feedback_message(observation(2), b"new"), [feedback_message(observation(1), b"old")], 2)
    try:
        await adapter.respond(FoundryConfig().models[2], "none", "Inspect", [message])
        payload = captured[0]
        assert [base64.b64decode(image) for image in payload["messages"][1]["images"]] == [b"new", b"old"]
        assert "historical_camera_observation" in payload["messages"][1]["content"]
        assert payload["options"]["num_ctx"] == 49152 and payload["think"] is False
    finally:
        await adapter.close()


@pytest.mark.parametrize("failure", ["multiple", "unknown", "nonfinite", "incomplete", "missing", "offline"])
async def test_ollama_failed_responses_never_move_robot(failure):
    def handle(request):
        data = local_response()
        if failure == "multiple":
            data["message"]["tool_calls"] *= 2
        elif failure == "unknown":
            data["message"]["tool_calls"][0]["function"]["name"] = "unpublished_tool"
        elif failure == "nonfinite":
            data["message"]["tool_calls"][0]["function"]["arguments"]["linear_mps"] = "NaN"
        elif failure == "incomplete":
            data["done_reason"] = "length"
        elif failure == "missing":
            return httpx.Response(404, json={"error": "model missing"})
        elif failure == "offline":
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json=data)
    adapter = await adapter_for(handle)
    worker = SimulationWorker(pace=False)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, model_id="gemma", reasoning="none", max_turns=1))
        await controller.task
        assert worker.latest["snapshot"]["simulated_time_s"] == 0
        assert worker.latest["stopped"] and not controller.active
        assert adapter.client.is_closed
        assert "private-marker" not in json.dumps(controller.trace())
        if failure in {"missing", "offline"}:
            assert "Ollama" in controller.state["error"] and "Foundry" not in controller.state["error"]
    finally:
        await controller.halt()
        await worker.close()


async def test_ollama_drives_real_worker_and_stop_cancels_pending_http():
    entered = asyncio.Event()
    requests = []
    async def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=local_response())
        entered.set()
        await asyncio.Event().wait()
    adapter = await adapter_for(handle)
    worker = SimulationWorker(pace=False)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, model_id="gemma", reasoning="none", feedback_interval_s=.25))
        await asyncio.wait_for(entered.wait(), 10)
        await controller.halt()
        assert worker.latest["snapshot"]["simulated_time_s"] == .5
        assert worker.latest["observation"]["odometry_m_rad"][0] > 0
        assert controller.state["input_tokens"] == 100
        assert any(message["role"] == "tool" for message in requests[1]["messages"])
        assert adapter.client.is_closed and controller.state["phase"] == "stopped"
        assert not controller.state["auto_wake"] and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


async def test_ollama_repeated_native_call_id_cannot_repeat_motion():
    def handle(request):
        data = local_response()
        data["message"]["tool_calls"][0]["id"] = "repeated-ollama-call"
        return httpx.Response(200, json=data)
    adapter = await adapter_for(handle)
    worker = SimulationWorker(pace=False)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, model_id="gemma", reasoning="none", feedback_interval_s=.25, max_turns=2))
        await controller.task
        assert controller.state["phase"] == "error" and "repeated" in controller.state["error"]
        assert worker.latest["snapshot"]["simulated_time_s"] == .5
        assert worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


def test_ollama_rejects_non_camera_urls():
    with pytest.raises(ValueError, match="camera PNG"):
        ollama_messages([{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com"}]}])