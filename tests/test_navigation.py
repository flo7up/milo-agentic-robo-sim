import asyncio
from uuid import uuid4

import pytest
import pybullet as bullet
from pydantic import ValidationError

from backend.navigation import CompleteNavigationSkill, NavigationPlan, NavigationRuntime, ReplaceMotionBuffer
from backend.simulation import BulletSimulation, MotionError
from backend.contracts import Command, ManualPlacement
from backend.worker import SimulationWorker


def ticket(runtime, sim):
    observation = sim.observe()
    runtime.observe(sim, observation)
    return observation.seq, runtime.revision


def plan(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "set_navigation_plan", NavigationPlan(expected_revision=revision, steps=[
        {"skill": skill, "goal": goal} for skill, goal in [
            ("inspect_room", "Identify visible room fixtures"), ("locate_doorway", "Find a visible opening"),
            ("approach", "Approach the observed opening"), ("cross", "Cross and inspect the next room")]]), sequence)


def scan(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
        {"kind": "head", "yaw_rad": -.6, "pitch_rad": .2, "duration_s": 1},
        {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": 1}]), sequence)
    while runtime.buffer:
        runtime.tick(sim)


def complete(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "complete_navigation_skill", CompleteNavigationSkill(
        expected_revision=revision, evidence="Fixture observation reviewed in the camera"), sequence)


def test_navigation_skill_order_limits_and_feedback_checkpoints():
    with pytest.raises(ValidationError):
        NavigationPlan(expected_revision=0, steps=[{"skill": "cross", "goal": "Skip inspection"}])
    with pytest.raises(ValidationError):
        ReplaceMotionBuffer(expected_revision=0, segments=[
            {"kind": "drive", "linear_mps": .15, "angular_radps": 0, "duration_s": 1}] * 3)
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime()
    try:
        plan(runtime, sim)
        with pytest.raises(MotionError, match="Inspect more"):
            complete(runtime, sim)
        scan(runtime, sim)
        assert runtime.status == "awaiting_feedback" and runtime.scan_max - runtime.scan_min > .4
        complete(runtime, sim)
        assert runtime.index == 1 and runtime.steps[0]["status"] == "completed"
        sequence, revision = ticket(runtime, sim)
        with pytest.raises(MotionError, match="Locate the opening"):
            runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0, "duration_s": 1}]), sequence)
    finally:
        sim.close()


def test_navigation_expiry_and_stop_invalidate_late_replacements():
    now = [10.0]
    runtime = NavigationRuntime(clock=lambda: now[0])
    sim = BulletSimulation(width=160, height=120)
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        buffer = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": 1}])
        runtime.apply(sim, "replace_motion_buffer", buffer, sequence)
        sequence, revision = ticket(runtime, sim)
        now[0] += 2
        runtime.tick(sim)
        assert runtime.status == "failed" and not runtime.buffer and sim.cancel.is_set()
        assert sim.ticks == 0 and runtime.revision > revision
        with pytest.raises(MotionError):
            runtime.apply(sim, "replace_motion_buffer", buffer, sequence)
    finally:
        sim.close()


async def submit(worker, tool, arguments):
    observation, _ = await worker.feedback()
    return await worker.execute_navigation(Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
        observation_seq=observation.seq, action_id=str(uuid4()), tool=tool,
        arguments={"expected_revision": observation.navigation.revision, **arguments}))


def test_slow_simulation_expires_buffer_without_failing_plan():
    now = [10.0]
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: now[0])
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        update = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": .5}])
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        update.expected_revision = revision
        for _ in range(5):
            now[0] += .21
            runtime.tick(sim)
        assert sim.ticks < 120
        assert runtime.status == "awaiting_feedback", runtime.reason
        assert "expired" in runtime.reason.lower()
        assert not runtime.buffer and runtime.velocity.tolist() == [0, 0]
        assert not sim.cancel.is_set()
        assert runtime.steps[0]["status"] == "running"
        with pytest.raises(MotionError, match="revision changed"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, update.expected_revision = ticket(runtime, sim)
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        runtime.tick(sim)
        assert runtime.status == "running" and runtime.buffer
    finally:
        sim.close()


async def test_worker_accepts_replacements_while_moving_and_stop_invalidates_them():
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.begin_navigation(worker.stop_revision)
        result = await submit(worker, "set_navigation_plan", {"steps": [
            {"skill": "inspect_room", "goal": "Inspect fixtures"}]})
        assert result.status == "ok"
        result = await submit(worker, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": -.6, "pitch_rad": .2, "duration_s": 1},
            {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": 1}]})
        async with asyncio.timeout(5):
            while worker.latest["snapshot"]["simulated_time_s"] < .2:
                await asyncio.sleep(.01)
        observation, _ = await worker.feedback()
        assert observation.navigation.remaining_s > 0 and worker.latest["busy"]
        revision = observation.navigation.revision
        replacement = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
            observation_seq=observation.seq, action_id="replace-active", tool="replace_motion_buffer", arguments={
                "expected_revision": revision, "segments": [{"kind": "head", "yaw_rad": .8, "pitch_rad": .2, "duration_s": 1}]})
        assert (await worker.execute_navigation(replacement)).status == "ok"
        assert worker.latest["navigation"]["revision"] > revision
        worker.stop()
        replacement.action_id = "late-after-stop"
        result = await worker.execute_navigation(replacement)
        assert result.status == "error" and result.error == "CANCELLED"
        assert not worker.latest["navigation"]["remaining_s"] and worker.latest["stopped"]
        assert not worker.latest["busy"]
    finally:
        await worker.close()


async def test_controller_refills_before_empty_and_brakes_on_inference_failure():
    import json
    from backend.agent import AgentController, FoundryConfig, robot_tools
    from tests.test_agent import model_response, start_settings
    seen = []

    class Planner:
        async def respond(self, profile, reasoning, goal, inputs):
            observation = json.loads(inputs[-1]["content"][0]["text"])
            state = observation["navigation"]
            seen.append(state)
            if len(seen) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Look around"}]}), call_id="plan")
            if len(seen) <= 3:
                if len(seen) == 3:
                    assert state["remaining_s"] > 0
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                    "segments": [{"kind": "head", "yaw_rad": -.6 if len(seen) == 2 else .6, "pitch_rad": .2,
                                  "duration_s": 1}] * 2}), call_id=f"buffer-{len(seen)}")
            raise ValueError("Scripted inference failure")

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Planner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25, max_turns=8))
        await controller.task
        assert len(seen) == 4
        assert controller.state["phase"] == "error" and "Scripted inference failure" in controller.state["error"]
        assert worker.latest["navigation"]["remaining_s"] == 0 and worker.latest["stopped"]
        assert all(step["status"] == "cancelled" for step in worker.latest["navigation"]["steps"])
        assert not controller.state["auto_wake"]
        assert "replace_motion_buffer" in {tool["name"] for tool in robot_tools("navigation_plan")}
        assert "replace_motion_buffer" not in {tool["name"] for tool in robot_tools()}
    finally:
        await controller.halt()
        await worker.close()


def drive_buffer(runtime, sim, linear=.15, angular=0):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
        {"kind": "drive", "linear_mps": linear, "angular_radps": angular, "duration_s": 1}] * 2), sequence)


@pytest.mark.parametrize("expire", [False, True])
async def test_buffer_pause_cancels_outdated_inference_and_replans_from_rest(expire):
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    entered, cancelled = asyncio.Event(), asyncio.Event()
    seen = []
    in_flight = 0

    class Planner:
        async def respond(self, profile, reasoning, goal, inputs):
            nonlocal in_flight
            in_flight += 1
            assert in_flight == 1
            try:
                state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
                seen.append(state)
                if len(seen) == 1:
                    return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                        "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
                if len(seen) == 3:
                    assert state["remaining_s"] > 0
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                if len(seen) == 4:
                    assert cancelled.is_set() and state["remaining_s"] == 0
                    assert state["status"] == "awaiting_feedback" and state["revision"] > seen[2]["revision"]
                if len(seen) in {2, 4}:
                    return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                        "segments": [{"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": 1}] * 2}),
                        call_id=f"buffer-{len(seen)}")
                return model_response("stop", "{}", call_id="stop")
            finally:
                in_flight -= 1

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Planner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25, max_turns=8))
        await asyncio.wait_for(entered.wait(), 5)
        if expire:
            await worker.call(lambda sim: setattr(worker.navigation, "expires_at", 0))
        await asyncio.wait_for(controller.task, 12)
        assert controller.state["phase"] == "completed" and controller.state["error"] is None
        assert len(seen) == 5 and in_flight == 0
        assert all(event["status"] == "ok" for event in controller.state["events"])
        assert any(entry["title"] == "Navigation feedback refreshed" for entry in controller.trace()["events"])
        assert worker.latest["stopped"] and not worker.latest["navigation"]["remaining_s"]
    finally:
        await controller.halt()
        await worker.close()


async def test_slow_model_waits_for_rest_before_requesting_next_buffer():
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    seen = []

    class SlowPlanner:
        async def respond(self, profile, reasoning, goal, inputs):
            state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
            seen.append(state)
            if len(seen) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
            if len(seen) == 2:
                await asyncio.sleep(.6)
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                    "segments": [{"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": .3}]}), call_id="buffer")
            assert state["remaining_s"] == 0 and state["status"] == "awaiting_feedback"
            return model_response("stop", "{}", call_id="stop")

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: SlowPlanner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 8)
        assert controller.state["phase"] == "completed" and controller.state["error"] is None
        assert len(seen) == 3
    finally:
        await controller.halt()
        await worker.close()


def test_full_skill_plan_crosses_physical_doorway_and_blends_replacements():
    scene = [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "wall_left", "size": [.1, 1, 1], "position": [.8, 1.15, .5], "color": [.4, .4, .4, 1]},
        {"name": "wall_right", "size": [.1, 1, 1], "position": [.8, -1.15, .5], "color": [.4, .4, .4, 1]}]
    sim = BulletSimulation(scene=scene, width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: 10)
    try:
        plan(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        assert runtime.steps[runtime.index]["skill"] == "approach"
        with pytest.raises(MotionError, match="0.1 m"):
            complete(runtime, sim)
        drive_buffer(runtime, sim)
        for _ in range(15):
            runtime.tick(sim)
        speed_before = runtime.velocity[0]
        assert speed_before == pytest.approx(.15)
        drive_buffer(runtime, sim)
        assert runtime.velocity[0] == speed_before
        runtime.tick(sim)
        assert runtime.velocity[0] == pytest.approx(speed_before)
        while runtime.buffer:
            runtime.tick(sim)
        assert runtime.status == "awaiting_feedback", runtime.reason
        complete(runtime, sim)
        for _ in range(4):
            drive_buffer(runtime, sim)
            while runtime.buffer:
                runtime.tick(sim)
            assert runtime.status == "awaiting_feedback", runtime.reason
        complete(runtime, sim)
        assert runtime.status == "completed"
        assert all(step["status"] == "completed" for step in runtime.steps)
        assert sim.odometry[0] > 1.1
        assert not sim.proximity_sensors().collisions
    finally:
        sim.close()


@pytest.mark.parametrize("fault", ["clearance", "buffer_expiry", "feedback_expiry", "skill_timeout"])
def test_navigation_independent_watchdogs_brake(fault):
    now = [10.0]
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: now[0])
    try:
        plan(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        drive_buffer(runtime, sim, linear=0, angular=.4)
        if fault == "clearance":
            edge = max(bullet.getAABB(sim.robot, index, physicsClientId=sim.client)[1][1]
                       for index in [-1, *sim.joints.values()])
            sim.reposition(ManualPlacement(run_id=sim.run_id, episode_epoch=sim.epoch,
                observation_seq=sim.seq, xy_m=[0, 3 - edge - .005]))
        elif fault == "buffer_expiry":
            runtime.expires_at = now[0] - .01
        elif fault == "feedback_expiry":
            now[0] += .8
        else:
            runtime.skill_deadline = now[0] - .01
        runtime.tick(sim)
        assert not runtime.buffer
        assert runtime.velocity.tolist() == [0, 0]
        if fault == "buffer_expiry":
            assert runtime.status == "awaiting_feedback" and not sim.cancel.is_set(), runtime.reason
            assert runtime.steps[1]["status"] == "running" and runtime.steps[2]["status"] == "pending"
        else:
            assert runtime.status == "failed" and sim.cancel.is_set(), runtime.reason
            assert runtime.steps[1]["status"] == "failed" and runtime.steps[2]["status"] == "cancelled"
    finally:
        sim.close()


def test_exhausted_or_moved_observations_cannot_replenish_old_motion():
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: 10)
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        update = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": .5}])
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        update.expected_revision = revision
        while runtime.buffer:
            runtime.tick(sim)
        with pytest.raises(MotionError, match="revision changed"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        sim.odometry[2] += .4
        update.expected_revision = revision
        with pytest.raises(MotionError, match="moved too far"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
    finally:
        sim.close()


async def test_safety_failure_cancels_inference_without_waiting_for_model_timeout():
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    entered, cancelled = asyncio.Event(), asyncio.Event()
    requests = []

    class SlowPlanner:
        async def respond(self, profile, reasoning, goal, inputs):
            state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
            requests.append(state)
            if len(requests) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: SlowPlanner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25))
        await asyncio.wait_for(entered.wait(), 5)
        await worker.call(lambda sim: setattr(worker.navigation, "skill_deadline", 0))
        await asyncio.wait_for(controller.task, 2)
        assert cancelled.is_set() and controller.state["phase"] == "error"
        assert "SKILL_TIMEOUT" in controller.state["error"]
        assert not worker.latest["navigation"]["remaining_s"] and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()