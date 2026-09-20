"""Malformed local model replies never authorize motion or bypass inference budgets."""
import asyncio
import json

import httpx
import pytest

from backend.agent import AgentController, FoundryConfig
from backend.challenges import get_challenge
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission
from backend.worker import SimulationWorker
from tests.test_agent import start_settings
from tests.test_ollama import adapter_for, local_response


@pytest.mark.parametrize("content,ending,code", [
    ('{"action":', "stop", "invalid_json"),
    ("", "stop", "invalid_json"),
    (None, "stop", "invalid_json"),
    ('[{"action":"turn","turn_rad":1}]', "stop", "invalid_decision"),
    ('```json\n{"action":"turn","turn_rad":1}\n```', "stop", "invalid_json"),
    ('{"action":"turn","turn_rad":1}', "length", "incomplete_decision"),
    ('{"action":', "length", "incomplete_decision"),
    ("x" * 3000, "stop", "invalid_json"),
], ids=["truncated", "empty", "nontext", "array", "fenced", "incomplete", "truncated_incomplete", "oversized"])
async def test_malformed_reply_retains_usage_and_diagnostics_without_calls(content, ending, code):
    def handle(request):
        return httpx.Response(200, json=local_response(done_reason=ending, message={
            "role": "assistant", "content": content, "thinking": "private-marker"}))
    adapter = await adapter_for(handle)
    adapter.unified_mission = True
    try:
        response = await adapter.respond(FoundryConfig().models[2], "none", "Find Kitchen", [
            {"role": "user", "content": [{"type": "input_text", "text": '{"available_actions":["plan"]}'}]}])
        assert response.usage.total_tokens == 120
        assert not any(item.type == "function_call" for item in response.output)
        error = response.model_extra["local_mission_error"]
        assert error["code"] == code and error["done_reason"] == ending
        assert error["content_chars"] == (len(content) if isinstance(content, str) else None)
        assert len(response.output_text) <= 2000
        assert "private-marker" not in response.model_dump_json()
        assert response.status == ("completed" if ending == "stop" else "incomplete")
    finally:
        await adapter.close()


@pytest.mark.parametrize("ending", ["recovered", "stop", "requests", "tokens", "missing_usage"])
async def test_correction_uses_fresh_evidence_and_preserves_authority_and_budgets(tmp_path, ending):
    captured = []
    retry_started = asyncio.Event()
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)

    async def handle(request):
        payload = json.loads(request.content)
        captured.append(json.JSONDecoder().raw_decode(payload["messages"][-1]["content"])[0])
        assert abs(worker.sim.odometry[2]) < .01 or len(captured) > 3
        if len(captured) == 2:
            retry_started.set()
            if ending == "stop":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass  # Simulate a provider returning a late motion decision after Stop.
        decision = ({"action": "plan", "plan": {"kind": "room", "target": "Kitchen", "completion": "identify"}}
            if len(captured) == 2 and ending != "stop" else {"action": "turn", "turn_rad": .2})
        return httpx.Response(200, json=local_response(
            **({"prompt_eval_count": None, "eval_count": None} if ending == "missing_usage" else {}),
            message={"role": "assistant", "content": '{"action":' if len(captured) == 1 else json.dumps(decision)}))

    adapter = await adapter_for(handle)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "recovery.sqlite3"))
        controller.start(worker, start_settings(worker, model_id="qwen", reasoning="none", execution_mode="luna_continuous",
            unified_mission=True, images_per_request=2, map_context=True, mission_budget_s=45., max_turns=3,
            max_model_requests=1 if ending == "requests" else 3,
            max_model_tokens=120 if ending == "tokens" else 10000).model_copy(update={"goal": "Find Kitchen"}))
        deadline = controller.mission.deadline
        if ending == "stop":
            await asyncio.wait_for(retry_started.wait(), 35.)
            before = worker.sim.ticks
            await controller.halt()
            assert worker.sim.ticks == before
        else:
            await asyncio.wait_for(controller.task, 45.)
        assert worker.latest["stopped"] and not controller.active and adapter.client.is_closed
        assert not worker.sim.proximity_sensors().collisions
        assert controller.mission.deadline == deadline
        budget = controller.state["inference_budget"]
        assert budget["requests"] == len(captured)
        if ending in {"recovered", "stop"}:
            assert captured[1]["last_execution"]["status"] == "rejected"
            assert "valid JSON object" in captured[1]["last_execution"]["reason"]
            assert captured[1]["last_execution"]["motion_authorized"] is False
            assert captured[1]["observation"]["seq"] > captured[0]["observation"]["seq"]
            assert captured[1]["sensing"]["source_sequence"] > captured[0]["sensing"]["source_sequence"]
        if ending == "recovered":
            assert len(captured) == 3 and worker.sim.odometry[2] > .1
            assert controller.state["error"] is None
            assert budget["tokens"] == 360 and not budget["usage_unknown"]
            assert controller.state["input_tokens"] == 300 and controller.state["output_tokens"] == 60
            failures = [event for event in controller.trace()["events"] if event["kind"] == "response"
                and event["payload"].get("format_error")]
            assert len(failures) == 1 and not failures[0]["payload"]["calls"]
            assert failures[0]["payload"]["text"] == '{"action":'
        else:
            assert abs(worker.sim.odometry[2]) < .01
            assert len(captured) == (2 if ending == "stop" else 1)
            if ending != "stop":
                assert budget["tokens"] == (0 if ending == "missing_usage" else 120)
                assert budget["usage_unknown"] == (ending == "missing_usage")
    finally:
        await controller.halt()
        await worker.close()


async def test_malformed_moving_review_closes_objective_before_retry(tmp_path, monkeypatch):
    from backend.mission_supervisor import MappedReviewOperation
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="enhanced", pace=True)
    captured = []
    malformed_at = []
    async def handle(request):
        captured.append(json.JSONDecoder().raw_decode(json.loads(request.content)["messages"][-1]["content"])[0])
        if len(captured) == 1:
            reply = '{"action":"plan","plan":{"kind":"room","target":"Kitchen","completion":"identify"}}'
        elif malformed_at:
            reply = '{"action":"wait","duration_s":0.5}'
        elif captured[-1]["mission"]["operation_id"]:
            malformed_at.append(len(captured)-1)
            reply = '{"action":'
        else:
            # Select fresh evidence again if the initial floor survey invalidated
            # an offer. Inject the bad reply only after measured motion starts.
            frontiers = captured[-1]["observation"]["spatial"]["frontiers"]
            assert frontiers
            reply = json.dumps({"action": "navigate_frontier", "frontier_id": frontiers[0]["frontier_id"]})
        return httpx.Response(200, json=local_response(message={"role": "assistant", "content": reply}))
    adapter = await adapter_for(handle)
    controller = AgentController(FoundryConfig(), lambda config: adapter)
    original_review = MappedReviewOperation.review
    async def review_after_motion(mapped):
        async with asyncio.timeout(25.):
            while mapped.mission.objective.travel_m < .15:
                result = await mapped.poll()
                assert result["task"]["status"] == "running", result
                await asyncio.sleep(.05)
        mapped.reviewed_at = 0.
        return await original_review(mapped)
    monkeypatch.setattr(MappedReviewOperation, "review", review_after_motion)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "moving-recovery.sqlite3"))
        controller.start(worker, start_settings(worker, model_id="qwen", reasoning="none", execution_mode="luna_continuous",
            unified_mission=True, map_context=True, images_per_request=2, mission_budget_s=90.,
            max_turns=6, max_model_requests=6).model_copy(update={"goal": "Find Kitchen"}))
        await asyncio.wait_for(controller.task, 100.)
        assert controller.state["error"] is None, controller.state["error"]
        assert len(malformed_at) == 1
        moving, stopped = captured[malformed_at[0]:malformed_at[0]+2]
        assert moving["mission"]["operation_id"], moving.get("last_execution")
        assert stopped["mission"]["operation_id"] is None
        assert stopped["review_mode"] == "choose_next_observed_destination"
        assert stopped["last_execution"]["motion_authorized"] is False
        assert stopped["sensing"]["source_sequence"] > moving["sensing"]["source_sequence"]
        assert controller.state["inference_budget"]["tokens"] == 120*len(captured)
        assert worker.latest["stopped"] and not controller.active
        assert not worker.sim.proximity_sensors().collisions
    finally:
        await controller.halt()
        await worker.close()
