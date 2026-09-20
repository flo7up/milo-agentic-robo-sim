import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from backend.agent import AgentController, AgentStart, BudgetedModel, BudgetedTaskSupervisor, ModelProfile
from backend.model_call_history import ModelCallHistory


def fixture():
    worker = SimpleNamespace(closed=False, powered=True, latest={"run_id": "run", "episode_epoch": 3, "stopped": False,
        "robot_body_id": 4, "snapshot": {"simulated_time_s": 12.5, "poses": [{"key": "4:-1", "position": [2., 3., .1]}]}})
    controller = AgentController()
    controller.active = True
    controller.state.update(session_id="session", turns=1,
        inference_budget={"requests": 0, "tokens": 0, "max_requests": 5, "max_tokens": 1000, "usage_unknown": False},
        task_supervision={"requests": 0, "tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "max_requests": 4, "max_tokens": 1000, "usage_unknown": False})
    return worker, controller, AgentStart(run_id="run", episode_epoch=3, goal="Explore"), ModelProfile(id="luna", label="Luna", deployment="scripted")


def response():
    return SimpleNamespace(output_text="", usage=SimpleNamespace(input_tokens=10, output_tokens=2), output=[
        SimpleNamespace(model_dump=lambda **_: {"type": "reasoning", "summary": [{"text": "PRIVATE_REASONING"}]}),
        SimpleNamespace(model_dump=lambda **_: {"type": "function_call", "name": "guide_mission",
            "arguments": '{"action":"navigate_frontier","frontier_id":"3:13"}'})])


async def test_call_locations_freeze_at_dispatch_and_correlate_both_models_without_input_changes():
    worker, controller, settings, profile = fixture()
    inputs = [{"role": "user", "content": "sensor-only fixture"}]
    before = deepcopy(inputs)
    entered, release = asyncio.Event(), asyncio.Event()

    class Model:
        async def respond(self, *args):
            assert args[-1] is inputs and inputs == before
            entered.set()
            await release.wait()
            return response()

        supervise_task = respond

    pending = asyncio.create_task(BudgetedModel(Model(), controller, worker, settings).respond(profile, "low", "Explore", inputs))
    await entered.wait()
    call = controller.public()["model_calls"][0]
    assert call["status"] == "thinking" and call["number"] == 1
    worker.latest["snapshot"]["poses"][0]["position"][0] = 8.
    worker.latest["snapshot"]["simulated_time_s"] = 19.
    release.set()
    await pending
    assert call["position_world_m"] == [2., 3.] and call["simulated_time_s"] == 12.5
    assert call["summary"] == "navigate frontier / target 3:13"
    assert "PRIVATE_REASONING" not in str(controller.public()["model_calls"])
    controller.navigation_reply("Investigate this opening.")
    assert controller.public()["run_messages"][-1]["model_call_id"] == call["id"]
    event = controller._trace("response", "Decision", {})
    assert controller.trace_records[event]["model_call_id"] == call["id"]
    assert controller.trace()["model_calls"][0]["position_world_m"] == [2., 3.]
    await BudgetedTaskSupervisor(Model(), controller, worker, settings).respond(profile, "low", "Explore", inputs)
    second = controller.public()["model_calls"][1]
    assert second["number"] == 2 and second["kind"] == "task_supervision"
    assert second["position_world_m"] == [8., 3.] and inputs == before


@pytest.mark.parametrize("failure", [asyncio.CancelledError, ValueError])
async def test_interrupted_calls_retain_the_original_location(failure):
    worker, controller, settings, profile = fixture()

    class Model:
        async def respond(self, *args):
            raise failure()

    with pytest.raises(failure):
        await BudgetedModel(Model(), controller, worker, settings).respond(profile, "low", "Explore", [])
    call = controller.public()["model_calls"][0]
    assert call["position_world_m"] == [2., 3.]
    assert call["status"] == ("cancelled" if failure is asyncio.CancelledError else "failed")


def test_history_is_bounded_and_missing_poses_are_not_guessed():
    worker, controller, _, profile = fixture()
    worker.latest["snapshot"]["poses"] = []
    for turn in range(ModelCallHistory.capacity + 10):
        controller.model_calls.begin(worker, profile, "session", turn, "decision")
    calls = controller.public()["model_calls"]
    assert len(calls) == ModelCallHistory.capacity and calls[0]["number"] == 11
    assert all(call["position_world_m"] is None for call in calls)
    old = calls[-1]
    controller.model_calls = ModelCallHistory()
    fresh = controller.model_calls.begin(worker, profile, "next-session", 1, "decision")
    ModelCallHistory.finish(old, response())
    assert fresh["number"] == 1 and fresh["status"] == "thinking" and fresh["id"] != old["id"]
