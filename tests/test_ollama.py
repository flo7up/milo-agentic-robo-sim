import asyncio
import base64
import json

import httpx
import pytest

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