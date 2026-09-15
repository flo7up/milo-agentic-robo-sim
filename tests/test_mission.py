import pytest

from backend.mission import Mission, MissionPlan


def test_mission_has_one_operation_original_deadline_and_required_receipts():
    authority = ("run", 1, 2, 3)
    mission = Mission(*authority, deadline=60., clock=lambda: 10.)
    mission.configure(MissionPlan(kind="object", target="yellow cube", return_home=True))
    operation = mission.begin("inspecting", authority)
    with pytest.raises(ValueError, match="already active"):
        mission.begin("returning", authority)
    mission.end_operation(operation, authority, "target")
    with pytest.raises(ValueError, match="not verified"):
        mission.complete(authority)
    operation = mission.begin("returning", authority)
    mission.end_operation(operation, authority, "return")
    mission.complete(authority)
    assert mission.state()["phase"] == "completed" and mission.deadline == 60.


@pytest.mark.parametrize("fault", ["stop", "task", "epoch", "run", "cancel", "deadline", "operation"])
def test_mission_rejects_late_results(fault):
    now = [0.]
    authority = ("run", 1, 2, 3)
    mission = Mission(*authority, deadline=10., clock=lambda: now[0])
    mission.configure(MissionPlan(kind="explore"))
    operation = mission.begin("exploring", authority)
    changed = list(authority)
    if fault in {"run", "epoch", "stop", "task"}:
        index = {"run": 0, "epoch": 1, "stop": 2, "task": 3}[fault]
        changed[index] = "another" if index == 0 else changed[index] + 1
    elif fault == "cancel":
        mission.finish("cancelled", "Stop")
    elif fault == "deadline":
        now[0] = 11.
    else:
        operation = "late"
    with pytest.raises((ValueError, TimeoutError)):
        mission.end_operation(operation, changed, "exploration")
    assert not mission.receipts


def test_observed_map_crop_raster_and_unknown_preservation():
    import numpy as np
    from backend.map_context import observed_context, render_observed_map
    grid = np.zeros((200, 200), dtype=int)
    grid[0, 0] = -1
    grid[0, 2] = 100
    options = dict(origin=[-10., -10.], resolution=.1, pose=[0., 0., 0.], identity="observed", run_id="run",
        epoch=1, sequence=4, captured_at=2., now=2.1, frame="map", localization="localized", camera_yaw=0.)
    context = observed_context(grid, overview=True, **options)
    assert context.width <= 96 and context.height <= 96
    assert context.cells[0] == 100
    grid[0, 2] = 0
    context = observed_context(grid, overview=True, **options)
    assert context.cells[0] == -1
    assert render_observed_map(context) == render_observed_map(context.model_copy(deep=True))
    local = observed_context(grid, **options)
    assert local.origin_m == [-4., -4.] and local.width == local.height == 80
    for update in ({"now": 4.}, {"localization": "unlocalized"}):
        with pytest.raises(ValueError):
            observed_context(grid, **{**options, **update})


def test_mission_settings_require_supported_backend_and_explicit_image_budget():
    from backend.agent import AgentStart
    options = dict(run_id="run", episode_epoch=0, goal="Find an object", execution_mode="luna_continuous", unified_mission=True)
    with pytest.raises(ValueError, match="two image"):
        AgentStart(**options)
    assert AgentStart(**options, images_per_request=2).unified_mission
    with pytest.raises(ValueError, match="Built-in"):
        AgentStart(**options, images_per_request=2, navigation_backend="nav2")


@pytest.mark.parametrize("decision", [{"action":"plan"}, {"action":"select_object","object_label":"cube","object_bounds":[0,0,2,2]},
    {"action":"approach_object"}, {"action":"observe_room","place_id":"room"}, {"action":"navigate_place"}])
def test_mission_schema_rejects_missing_or_invalid_evidence(decision):
    from backend.mission import MissionDecision
    with pytest.raises(ValueError):
        MissionDecision.model_validate(decision)


def test_mission_capability_endpoint_requires_new_contract_and_same_origin():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        capabilities = client.get("/api/mission/capabilities")
        assert capabilities.status_code == 200 and capabilities.json()["observed_map_version"] == 1
        assert capabilities.json()["backends"] == ["builtin"]
        state = client.get("/api/state").json()
        request = {"run_id":state["run_id"],"episode_epoch":state["episode_epoch"],"goal":"Explore"}
        assert client.post("/api/mission/start", json=request).status_code == 422
        assert client.post("/api/mission/start", json=request, headers={"Origin":"https://untrusted.example"}).status_code == 403
        assert client.get("/api/state").json()["run_id"] == state["run_id"]


async def test_local_mission_prepares_fresh_observed_map_and_stops_without_model(tmp_path, monkeypatch):
    import asyncio
    from backend.agent import AgentController, AgentStart
    from backend.worker import SimulationWorker
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.challenges import get_challenge
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    controller = AgentController(model_factory=lambda config: pytest.fail("Local mission must not create a model"))
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, goal="Explore locally",
            execution_mode="luna_continuous", unified_mission=True, mission_local_only=True, mission_budget_s=20., max_turns=1))
        await asyncio.wait_for(controller.task, 35.)
        assert controller.state["error"] is None, {"state": controller.state, "trace": controller.trace()}
        assert controller.state["input_tokens"] == 0
        assert controller.mission.home_id
        assert not controller.active and worker.sim.cancel.is_set()
        assert worker.home_mission.mission_owner is None
        assert not worker.home_mission.home.saved
    finally:
        await controller.halt()
        await worker.close()


async def test_exploration_plan_continues_locally_without_repeated_luna_calls(tmp_path):
    import asyncio
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    model = ScriptedModel([model_response("guide_mission", '{"action":"plan","plan":{"kind":"explore","return_home":false}}')])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=25., max_turns=2, max_model_requests=1))
        await asyncio.wait_for(controller.task, 35.)
        assert controller.state["error"] is None
        assert len(model.inputs) == controller.state["inference_budget"]["requests"] == 1
        assert any(event["title"] == "Mission operation finished" for event in controller.trace()["events"])
        assert worker.latest["stopped"] and not controller.active
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("stop_phase", [None, "reply", "return", pytest.param("redirect", id="task-change-cabinet-to-home")])
async def test_unified_visual_object_approach_and_automatic_return(tmp_path, stop_phase, record_property):
    import asyncio
    import base64
    from io import BytesIO
    import json
    import numpy as np
    from PIL import Image
    from backend.agent import AgentController, AgentStart, FoundryConfig, RunInstruction
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission, HomeRequest
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    replies = []
    goals = []
    entered = asyncio.Event()
    replacement_goal = "Cancel the cabinet visit. Go to the saved Home place instead and stop there."
    home_survey = [{"action": "look", "pitch_rad": .85},
        *[{"action": "turn", "turn_rad": 1.3} for heading in range(5)], {"action": "look", "pitch_rad": .45}]

    class VisualModel:
        async def respond(self, profile, reasoning, goal, inputs):
            goals.append(goal)
            if stop_phase == "reply" and not replies:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            content = inputs[-1]["content"]
            payload = json.loads(content[0]["text"])
            frames = [part["image_url"] for part in content if part["type"] == "input_image"]
            assert len(frames) == 2 and payload["map_context"]["status"] == "available", payload["map_context"]
            assert "cells" not in payload["observation"]["observed_map"]
            assert "evaluator" not in json.dumps(payload)
            if goal == replacement_goal:
                if payload["mission"]["plan"] is None:
                    decision = {"action": "plan", "plan": {"kind": "place", "target": "Home", "return_home": False}}
                else:
                    destination = next(place for place in payload["observation"]["spatial"]["places"]
                        if place["name"] == "Home" and place["reachable"])
                    decision = {"action": "navigate_place", "place_id": destination["place_id"]}
                replies.append(decision)
                return model_response("guide_mission", json.dumps(decision), call_id=f"mission-{len(replies)}")
            if stop_phase == "redirect" and payload["mission"]["plan"] is not None and len(replies) <= len(home_survey):
                decision = home_survey[len(replies) - 1]
                replies.append(decision)
                return model_response("guide_mission", json.dumps(decision), call_id=f"mission-{len(replies)}")
            pixels = np.asarray(Image.open(BytesIO(base64.b64decode(frames[0].split(",", 1)[1])))).astype(float)
            rows, columns = np.where((pixels[:, :, 0] > 100) & (pixels[:, :, 0] > pixels[:, :, 1]*1.8)
                & (pixels[:, :, 0] > pixels[:, :, 2]*1.8))
            assert len(rows) > 8
            height, width = pixels.shape[:2]
            bounds = [float(columns.min()/width), float(rows.min()/height), float((columns.max()+1)/width), float((rows.max()+1)/height)]
            selected = payload["object_goal"]
            if payload["mission"]["plan"] is None:
                decision = {"action": "plan", "plan": {"kind": "object", "target": "red cabinet", "return_home": stop_phase != "redirect"}}
            elif selected is None:
                decision = {"action": "select_object", "object_label": "red cabinet", "object_bounds": bounds}
            elif selected["selected_approach"] is None:
                decision = {"action": "approach_object", "object_goal_id": selected["goal_id"]}
            else:
                decision = {"action": "verify_object", "object_goal_id": selected["goal_id"], "object_bounds": bounds}
            replies.append(decision)
            return model_response("guide_mission", json.dumps(decision), call_id=f"mission-{len(replies)}")

        async def close(self):
            pass

    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "red_cabinet", "size": [.4, .5, .7], "position": [1.65, 0., .35], "color": [.9, .03, .03, 1.]},
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]}]})
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: VisualModel())
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        if stop_phase == "redirect":
            request = {"run_id": worker.sim.run_id, "episode_epoch": worker.epoch}
            await worker.scan_continuous(ContinuousScan(**request, compact_arms=True))
            await worker.home_command(HomeRequest(**request, action="start_mapping"))
            await worker.home_command(HomeRequest(**request, action="add_place", name="Home"))
            saved = await worker.home_command(HomeRequest(**request, action="save_map", name="Task change test"))
            destination = next(place for place in saved["places"] if place["name"] == "Home")
            start_pose = worker.sim.odometry.copy()
        initial_goal = "Visit the red cabinet and stop beside it." if stop_phase == "redirect" else "Visit the red cabinet then return to the start"
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, goal=initial_goal,
            execution_mode="luna_continuous", unified_mission=True, images_per_request=2, mission_budget_s=150.,
            max_turns=12 if stop_phase == "redirect" else 5))
        if stop_phase == "redirect":
            async with asyncio.timeout(160.):
                while not (controller.mission.phase == "navigating" and worker.continuous and worker.continuous.active
                        and np.linalg.norm(worker.sim.odometry[:2] - start_pose[:2]) >= .3):
                    assert not controller.task.done(), {"state": controller.state, "mission": controller.mission.state()}
                    await asyncio.sleep(.02)
                departure_m = float(np.linalg.norm(worker.sim.odometry[:2] - start_pose[:2]))
                previous_mission, previous_task = controller.mission, controller.task
                previous_operation = previous_mission.operation
                previous_session, deadline = controller.state["session_id"], controller.session_deadline
                remaining_turns = controller.state["max_turns"] - controller.state["turns"]
                previous_reply_count = len(replies)
                instruction = RunInstruction(**request, session_id=previous_session, message=replacement_goal)
                redirected = await controller.redirect(worker, instruction)
                assert previous_task.done() and previous_mission.phase == "cancelled"
                assert not previous_mission.receipts and previous_mission.operation is None
                assert redirected["session_id"] != previous_session and redirected["goal"] == replacement_goal
                assert controller.session_deadline == deadline == controller.mission.deadline
                assert controller.state["max_turns"] == remaining_turns
                assert worker.sim.run_id == request["run_id"] and worker.epoch == request["episode_epoch"]
                with pytest.raises(RuntimeError, match="inactive or changed"):
                    await controller.redirect(worker, instruction)
                with pytest.raises(ValueError):
                    previous_mission.end_operation(previous_operation, previous_mission.authority, "target")
                await controller.task
            assert controller.state["error"] is None, {"state": controller.state, "trace": controller.trace()}
            assert controller.mission.phase == "completed", {"mission": controller.mission.state(), "feedback": [
                event["payload"]["result"] for event in controller.trace()["events"] if event["title"] == "Mission capability feedback"]}
            assert controller.mission.plan.kind == "place" and controller.mission.plan.target == "Home"
            assert controller.mission.receipts["target"]["place_id"] == destination["place_id"]
            assert not previous_mission.receipts
            assert goals[:previous_reply_count] == [initial_goal] * previous_reply_count
            assert goals[previous_reply_count:] == [replacement_goal, replacement_goal]
            assert [reply["action"] for reply in replies] == ["plan", *[view["action"] for view in home_survey],
                "select_object", "approach_object", "plan", "navigate_place"]
            final_error_m = float(np.linalg.norm(worker.sim.odometry[:2] - start_pose[:2]))
            assert final_error_m <= .15 and departure_m - final_error_m >= .15
            assert not worker.sim.proximity_sensors().collisions
            assert not controller.active and not (worker.navigation and worker.navigation.buffer)
            assert worker.home_mission.mission_owner is None
            ticks = worker.sim.ticks
            await worker.hold_stopped()
            assert worker.sim.ticks == ticks
            record_property("task_change", json.dumps({"evidence": "scripted_test", "initial_goal": initial_goal,
                "replacement_goal": replacement_goal, "departure_m": departure_m, "final_home_error_m": final_error_m,
                "original_deadline_preserved": True, "old_mission_cancelled": True,
                "actions": [reply["action"] for reply in replies]}))
            return
        if stop_phase:
            async with asyncio.timeout(160.):
                while not entered.is_set():
                    if stop_phase == "return" and controller.mission.phase == "returning" and worker.continuous and worker.continuous.active:
                        break
                    assert not controller.task.done(), controller.state
                    await asyncio.sleep(.02)
            await controller.halt("Mission Stop test")
            ticks = worker.sim.ticks
            await worker.hold_stopped()
            assert worker.sim.ticks == ticks and controller.mission.phase == "cancelled"
            assert "return" not in controller.mission.receipts
            assert not controller.active and not (worker.navigation and worker.navigation.buffer)
            return
        await asyncio.wait_for(controller.task, 175.)
        assert controller.state["error"] is None, {"error": controller.state["error"], "events": [
            {"title": event["title"], "payload": event["payload"]} for event in controller.trace()["events"]
            if event["title"] in {"Mission capability feedback", "Internal controller failure"}]}
        assert controller.mission.phase == "completed", controller.trace()
        assert set(controller.mission.receipts) == {"target", "return"}
        assert np.linalg.norm(worker.sim.odometry[:2]) <= .15
        assert not worker.sim.proximity_sensors().collisions
        assert len(replies) == 4
        assert worker.home_mission.mission_owner is None
        assert not any(place.get("mission_id") for place in worker.home_mission.home.places)
    finally:
        await controller.halt()
        await worker.close()


async def test_unified_restart_rejects_stop_before_resume(tmp_path, monkeypatch):
    import asyncio
    from backend.agent import AgentController, AgentStart
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny")
    controller = AgentController(model_factory=lambda config: pytest.fail("Local-only restart must not create a model"))
    entered, release = asyncio.Event(), asyncio.Event()
    resume = worker.resume_manual

    async def pending_resume(expected_stop_revision=None):
        entered.set()
        await release.wait()
        return await resume(expected_stop_revision=expected_stop_revision)

    async def unexpected_scan(request):
        pytest.fail("A newer Stop must prevent the pending scan")

    monkeypatch.setattr(worker, "resume_manual", pending_resume)
    monkeypatch.setattr(worker, "scan_continuous", unexpected_scan)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        worker.stop()
        await worker.hold_stopped()
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            goal="Explore after Stop", execution_mode="luna_continuous", unified_mission=True,
            mission_local_only=True, mission_budget_s=10., max_turns=1))
        await asyncio.wait_for(entered.wait(), 5.)
        worker.stop()
        await worker.hold_stopped()
        ticks = worker.sim.ticks
        release.set()
        await asyncio.wait_for(controller.task, 5.)
        assert controller.mission.phase == "cancelled" and not controller.active
        assert worker.sim.cancel.is_set() and worker.sim.ticks == ticks
        assert worker.home_mission.mission_owner is None
    finally:
        release.set()
        await controller.halt()
        await worker.close()


def test_map_context_shape_and_render_are_independent_of_spectator_data():
    import numpy as np
    from backend.map_context import observed_context, render_observed_map
    from backend.contracts import ObservedMapContext
    context = observed_context(np.zeros((8, 8)), [0., 0.], .1, [.4, .4, 0.], identity="sensor-map",
        run_id="run", epoch=0, sequence=1, captured_at=1., now=1.1, frame="wheel_odometry", localization="local_odometry", camera_yaw=0.)
    original = render_observed_map(context)
    with pytest.raises(ValueError):
        ObservedMapContext.model_validate({**context.model_dump(), "cells": [0]})
    with pytest.raises(ValueError):
        ObservedMapContext.model_validate({**context.model_dump(), "spectator_geometry": [{"target": [100, 100]}]})
    assert render_observed_map(ObservedMapContext.model_validate(context.model_dump())) == original