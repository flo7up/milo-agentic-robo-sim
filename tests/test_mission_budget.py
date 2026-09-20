"""Longer interactive allowances must not extend a fixed evaluator's deadline."""
import asyncio
import time

import pytest
from pydantic import ValidationError

from backend.agent import NavigationEvaluationBudget
from backend.worker import SimulationWorker
from tests.test_agent import ScriptedModel, controller_for, start_settings


@pytest.mark.parametrize("evaluation_seconds", [None, 30.])
async def test_extended_mission_keeps_explicit_deadline_and_evaluator_cap(evaluation_seconds):
    worker = SimulationWorker(pace=False)
    controller = controller_for(ScriptedModel([]))
    if evaluation_seconds:
        controller.evaluation_budget = NavigationEvaluationBudget(timeout_s=evaluation_seconds, max_turns=3)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            images_per_request=2, mission_budget_s=600., max_turns=200,
            max_model_requests=200, max_model_tokens=1500000)
        before = time.monotonic()
        controller.start(worker, settings)
        after = time.monotonic()
        expected = evaluation_seconds or 600.
        assert before+expected <= controller.session_deadline <= after+expected
        assert controller.mission.deadline == controller.session_deadline
        assert controller.state["max_turns"] == (3 if evaluation_seconds else 200)
        with pytest.raises(ValidationError):
            start_settings(worker, mission_budget_s=601.)
        # Halt before the scheduled inference coroutine gets a turn.
        await controller.halt()
        assert not controller.active and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()
