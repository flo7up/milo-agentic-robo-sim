import pytest

from backend.mission import Mission, MissionPlan


@pytest.mark.parametrize("fault", [None, "mission", "run", "epoch", "map", "expired", "future", "moved", "turned", "nonfinite"])
def test_mission_frontier_selection_requires_same_fresh_stopped_evidence(fault):
    from backend.mission import MissionDecision, MissionFrontierSelection
    selection = MissionFrontierSelection(mission_id="mission", run_id="run", episode_epoch=1, map_id="map", frontier_id="1:0",
        position_m=[1., 0.], odometry_m_rad=[0., 0., 0.], captured_at=10., source_sequence=2)
    values = ["mission", "run", 1, "map", [0., 0., 0.], 11.]
    if fault in {"mission", "run", "epoch", "map"}:
        index = ("mission", "run", "epoch", "map").index(fault)
        values[index] = 2 if fault == "epoch" else "other"
    elif fault in {"expired", "future"}:
        values[-1] = 25.001 if fault == "expired" else 9.999
    elif fault:
        values[4] = [.051, 0., 0.] if fault == "moved" else [0., 0., .081] if fault == "turned" else [float("nan"), 0., 0.]
    if fault:
        with pytest.raises(ValueError, match="FRONTIER_SELECTION"):
            selection.check(*values)
    else:
        selection.check(*values)
        selection.check("mission", "run", 1, "map", [0., 0., 2 * 3.141592653589793], 25.)
        assert MissionDecision(action="navigate_frontier", frontier_id="1:0").frontier_id == "1:0"
        with pytest.raises(ValueError, match="frontier identity"):
            MissionDecision(action="navigate_frontier")


@pytest.mark.parametrize("current", [False, True])
def test_semantic_payload_separates_current_task_and_bounded_history(current):
    import copy
    import json
    from backend.mission_supervisor import semantic_payload
    mission = Mission("run", 1, 2, 3, deadline=60., clock=lambda: 10.)
    mission.configure(MissionPlan(kind="room", target="Kitchen"))
    task = {"mission_id": mission.identity if current else "previous-mission", "task_id": "task", "kind": "explore",
        "status": "running" if current else "failed", "reason": "Replanning after OBSERVED_PATH_BLOCKED",
        "route_failures": [{"reason": "OBSERVED_PATH_BLOCKED", "segment": index,
            "motion_diagnostics": {"recent_events": ["debug-only"] * 200}} for index in range(5)]}
    observation = {"run_id": "run", "episode_epoch": 1, "seq": 4, "frame_ref": "1-4.png", "head_rad": [.123456789, .2],
        "odometry_m_rad": [1.23456789, 0., 0.], "joints": ["unused-joints"] * 100, "grippers": {"left": "unused-grippers"},
        "navigation": {"status": "running", "reason": "Old reason", "diagnostics": {"recent_events": ["debug-only"] * 200}},
        "proximity": {"max_range_m": 2., "distances": [{"direction": "front", "distance_m": .654321, "status": "hit"}]},
        "spatial": {"task": task, "environment_id": "private-scene-name", "localization": {"status": "localized", "age_s": .2}},
        "observed_map": {"frame": "map", "revision": "revision", "age_s": .2, "geometry_source": "accumulated_sensor_map",
            "cells": [0] * 6400, "trail_m": [[0., 0.]] * 128, "route_m": [[1., 1.]] * 128}}
    original = copy.deepcopy(observation)
    payload = semantic_payload(observation, mission, None, {"task": task}, ["explore"], {"tokens": 10, "max_tokens": 100})
    assert observation == original
    assert payload["observation"]["frame_ref"] == "1-4.png"
    assert payload["observation"]["odometry_m_rad"] == [1.235, 0, 0]
    assert payload["observation"]["proximity"]["distances"][0]["distance_m"] == .654
    assert payload["observation"]["observed_map"]["age_s"] == .2
    assert payload["budget"]["max_tokens"] == 100
    text = json.dumps(payload)
    assert len(text) < 2500
    assert all(forbidden not in text for forbidden in ("unused-joints", "unused-grippers", "debug-only", "private-scene-name", "cells", "trail_m", "route_m"))
    if current:
        assert payload["observation"]["spatial"]["task"]["reason"] == "Local task active"
        assert len(payload["observation"]["spatial"]["task"]["recent_failures"]) == 2
        assert payload["observation"]["navigation"]["reason"] == "Executing bounded motion"
        assert payload["last_execution"]["task"]["task_id"] == "task"
    else:
        assert payload["observation"]["spatial"]["task"] is None
        assert payload["observation"]["navigation"] is None
        assert payload["last_execution"] is None
        assert "OBSERVED_PATH_BLOCKED" not in text


@pytest.mark.parametrize("fault", [None, "unknown", "stale", "moved", "turned", "episode", "stop", "deadline", "target_age", "steps", "rescans"])
async def test_parking_maneuver_requires_current_offer_and_original_authority(monkeypatch, fault):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import backend.mission_supervisor as supervisor
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: 10.))
    mission = Mission("run", 1, 2, 3, 100., clock=lambda: 10.)
    mission.configure(MissionPlan(kind="place", target="green parking bay"))
    sensor = SimpleNamespace(run_id="run", episode_epoch=2 if fault == "episode" else 1,
        captured_at=-6. if fault == "stale" else 9., odometry_m_rad=[0., 0., 0.], sequence=5)
    offer = {"sensor": sensor, "target": {"id": "floor-0", "captured_at": -111. if fault == "target_age" else 9.},
        "maneuvers": [{"id": "floor-0:5:0", "kind": "inspect"}]}
    progress = {"deadline": 10. if fault == "deadline" else 60., "steps": 8 if fault == "steps" else 0,
        "rescans": 2 if fault == "rescans" else 0, "travel_m": 0.}
    pose = [.06, 0., 0.] if fault == "moved" else [0., 0., .09] if fault == "turned" else [0., 0., 0.]
    worker = SimpleNamespace(sim=SimpleNamespace(run_id="run"), epoch=1, stop_revision=2, task_revision=3,
        call=AsyncMock(return_value=pose), scan_continuous=AsyncMock())
    def check(*args):
        if fault == "stop":
            raise asyncio.CancelledError
    controller = SimpleNamespace(mission=mission, _check_live=check, _trace=lambda *args: None)
    settings = SimpleNamespace(run_id="run", episode_epoch=1)
    if fault:
        with pytest.raises(asyncio.CancelledError if fault == "stop" else ValueError):
            await supervisor.execute_parking_maneuver(controller, worker, settings, offer,
                "old-id" if fault == "unknown" else "floor-0:5:0", progress)
        worker.scan_continuous.assert_not_awaited()
    else:
        result = await supervisor.execute_parking_maneuver(controller, worker, settings, offer, "floor-0:5:0", progress)
        assert result["status"] == "parking_maneuver_completed" and not result["arrival_verified"]
        assert progress["steps"] == progress["rescans"] == 1
        worker.scan_continuous.assert_awaited_once()
    assert not mission.receipts


def test_retained_parking_target_disables_generic_exploration_when_not_visible():
    from types import SimpleNamespace
    from backend.mission_supervisor import available_actions, semantic_payload
    mission = Mission("run", 1, 2, 3, 100.)
    mission.configure(MissionPlan(kind="place", target="green parking bay"))
    spatial = {"parking": {"retained_target_id": "floor-0", "maneuvers": []},
        "frontiers": [{"frontier_id": "unrelated"}], "floor_regions": []}
    assert available_actions(mission, SimpleNamespace(spatial=spatial), None) == ["park_floor", "look", "report_observation", "wait"]
    from tests.test_agent import observation
    observed = observation().model_dump()
    observed["spatial"] = spatial
    assert semantic_payload(observed, mission, None, None, ["look"])["observation"]["spatial"]["parking"] == spatial["parking"]


def test_observed_mission_trail_keeps_inter_review_turns_and_excludes_future_positions():
    from backend.mission import ObservedMotionTrail
    trail = ObservedMotionTrail()
    points = [[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]]
    for timestamp, point in enumerate(points):
        trail.record([*point, 0.], float(timestamp))
    assert trail.at(4., [0., 0., 0.]) == points
    assert trail.at(2.5, [.5, 1., 0.]) == points[:3] + [[.5, 1.]]
    trail.record([99., 99., 0.], 1.)
    assert trail.at(4., [0., 0., 0.]) == points
    for index in range(140):
        trail.record([float(index), 0., 0.], 10. + index)
    assert len(trail.samples) == 128 and trail.dropped > 0
    assert len(trail.at(149., [139., 0., 0.])) == 128


@pytest.mark.parametrize("invented", [False, True])
async def test_mission_executes_only_a_supplied_observed_frontier(tmp_path, invented):
    import asyncio
    import json
    import math
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings

    class ChoosingModel(ScriptedModel):
        async def respond(self, profile, reasoning, goal, inputs):
            self.inputs.append(inputs)
            brief = json.loads(inputs[-1]["content"][0]["text"])
            if len(self.inputs) == 1:
                decision = {"action": "plan", "plan": {"kind": "room", "target": "Kitchen"}}
            elif len(self.inputs) == 2:
                choices = brief["observation"]["spatial"]["frontiers"]
                assert choices and "navigate_frontier" in brief["available_actions"]
                self.selected = "invented-choice" if invented else choices[0]["frontier_id"]
                decision = {"action": "navigate_frontier", "frontier_id": self.selected}
            elif not invented and len(self.inputs) == 3:
                choices = brief["observation"]["spatial"]["frontiers"]
                assert choices and "navigate_frontier" in brief["available_actions"]
                self.selected = choices[0]["frontier_id"]
                decision = {"action": "navigate_frontier", "frontier_id": self.selected}
            else:
                decision = {"action": "look"}
            return model_response("guide_mission", json.dumps(decision), call_id=f"choice-{len(self.inputs)}")

    model = ChoosingModel([])
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=True, images_per_request=2, mission_budget_s=60., max_turns=4, max_model_requests=4))
        await asyncio.wait_for(controller.task, 70.)
        assert controller.state["error"] is None, controller.state["error"]
        assert not worker.sim.proximity_sensors().collisions and worker.latest["stopped"]
        if invented:
            assert worker.home_mission.task is None
            assert any("UNKNOWN_FRONTIER" in str(event["payload"]) for event in controller.trace()["events"])
        else:
            operations = [event["payload"]["task"] for event in controller.trace()["events"]
                if event["title"] == "Mission operation finished"]
            assert operations
            results = [event["payload"]["result"] for event in controller.trace()["events"]
                if event["title"] == "Mission capability feedback"]
            summary = [(task.get("status"), task.get("reason"), task.get("frontier_id"), task.get("segments"))
                for task in operations]
            assert math.hypot(*worker.sim.odometry[:2]) > .15, {"operations": summary,
                "results": [(result.get("status"), result.get("reason")) for result in results]}
            snapshots = [event["payload"]["observed_map_snapshot"] for event in controller.trace()["events"]
                if event["kind"] == "feedback" and event["payload"]["observed_map_snapshot"]]
            assert len(snapshots[-1]["trail_m"]) > len(model.inputs)
    finally:
        await controller.halt()
        await worker.close()


def test_mission_tools_only_expose_current_capabilities_and_observed_ids():
    from types import SimpleNamespace
    from backend.mission_supervisor import available_actions, instructions_for, tools
    planning = tools(["plan"])[0]["parameters"]
    assert set(planning["properties"]) == {"action", "reason", "plan"}
    navigation = tools(["explore", "navigate_frontier", "look"], ["1:0", "2:3"])[0]["parameters"]
    assert navigation["properties"]["action"]["enum"] == ["explore", "navigate_frontier", "look"]
    assert navigation["properties"]["frontier_id"]["enum"] == ["1:0", "2:3"]
    assert "object_bounds" not in navigation["properties"] and "$defs" not in navigation
    mission = Mission("run", 1, 0, 0, deadline=60., clock=lambda: 0.)
    mission.configure(MissionPlan(kind="room", target="Kitchen"))
    assert "navigate_frontier" in available_actions(mission, SimpleNamespace(spatial={"frontiers": [{"frontier_id": "1:0"}]}), None)
    operation = mission.begin("exploring", mission.authority)
    running_actions = available_actions(mission, SimpleNamespace(spatial={"frontiers": [{"frontier_id": "1:0"}]}), None)
    assert "navigate_frontier" not in running_actions
    assert {"explore", "look", "report_observation"}.issubset(running_actions)
    mission.end_operation(operation, mission.authority)
    guidance = instructions_for({"mission": mission.state()})
    assert "observe_room" in guidance and "object_bounds" not in guidance and "circle_direction" not in guidance
    assert len(guidance) < 3800
    assert "Stop" in guidance and "unknown" in guidance and "Never bypass" in guidance


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


@pytest.mark.parametrize("direction", ["clockwise", "counterclockwise"])
def test_circuit_mission_exposes_existing_capability_and_requires_measured_receipt(direction):
    from types import SimpleNamespace
    from backend.mission import MissionDecision
    from backend.mission_supervisor import available_actions, tools
    mission = Mission("run", 0, 0, 0, deadline=10., clock=lambda: 0.)
    mission.configure(MissionPlan(kind="circuit", target="Table", circle_direction=direction))
    assert "circle" in available_actions(mission, SimpleNamespace(spatial={}), None)
    assert "circle" in tools()[0]["parameters"]["properties"]["action"]["enum"]
    for bounds, label in [(None, "Table"), ([0., 0., 2., 1.], "Table"), ([.3, .1, .7, .6], "")]:
        with pytest.raises(ValueError):
            MissionDecision(action="circle", object_bounds=bounds, object_label=label, circle_direction=direction)
    decision = MissionDecision(action="circle", object_bounds=[.3, .1, .7, .6], object_label="Table", circle_direction=direction)
    assert decision.circle_direction == mission.plan.circle_direction
    with pytest.raises(ValueError, match="not verified"):
        mission.complete(mission.authority)
    operation = mission.begin("navigating", mission.authority)
    mission.end_operation(operation, mission.authority, "target")
    mission.complete(mission.authority)


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


@pytest.mark.parametrize("target_length", [416, 317, 349])
def test_mission_validation_feedback_reports_target_limit_without_echoing_input(target_length):
    from pydantic import ValidationError
    from backend.mission import MissionDecision
    from backend.mission_supervisor import decision_validation_feedback, tools
    with pytest.raises(ValidationError) as rejected:
        MissionDecision.model_validate({"action": "plan", "plan": {"kind": "room", "target": "x" * target_length}})
    feedback = decision_validation_feedback(rejected.value)
    assert feedback["reason"] == "plan.target: String should have at most 160 characters"
    assert feedback["validation_errors"] == [{"field": "plan.target", "type": "string_too_long",
        "message": "String should have at most 160 characters"}]
    assert "x" * target_length not in str(feedback)
    target = tools()[0]["parameters"]["$defs"]["MissionPlan"]["properties"]["target"]
    assert target["maxLength"] == 160 and "Short target label" in target["description"]
    decision = MissionDecision.model_validate({"action": "plan", "plan": {"kind": "room", "target": "Bathroom with toilet and sink"}})
    assert decision.plan.target == "Bathroom with toilet and sink"


def test_mission_capability_endpoint_requires_new_contract_and_same_origin():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        capabilities = client.get("/api/mission/capabilities")
        assert capabilities.status_code == 200 and capabilities.json()["observed_map_version"] == 1
        assert capabilities.json()["backends"] == ["builtin"]
        assert capabilities.json()["task_kinds"] == MissionPlan.model_json_schema()["properties"]["kind"]["enum"]
        assert "circuit" in capabilities.json()["task_kinds"]
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


@pytest.mark.parametrize("fault", [None, "stop", "sensors", "unknown", "deadline"])
async def test_cold_kitchen_exploration_entry_is_observed_bounded_and_cancellable(tmp_path, monkeypatch, record_property, fault):
    import asyncio
    import json
    from backend.agent import AgentController, AgentStart
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from scripts.benchmark_household import PhysicsMeasurements

    challenge = get_challenge("flat_kitchen")
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    controller = AgentController(model_factory=lambda config: pytest.fail("Exploration diagnostic must stay model-free"))
    physical = PhysicsMeasurements(challenge.initial_xy)
    preparing = False
    entry_starts = 0
    entry_travel = []
    original_entry, original_sample, original_start = worker.prepare_exploration_entry, worker._sample_spatial, worker._start_navigation_path

    async def entry(mission, duration_s=8., travel_m=.5):
        nonlocal preparing
        preparing = True
        before = physical.travel
        try:
            return await original_entry(mission, 0. if fault == "deadline" else duration_s, travel_m)
        finally:
            entry_travel.append(physical.travel-before)
            preparing = False

    def sample(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        if preparing and fault == "sensors":
            worker.spatial_error = "Injected unavailable depth"
        if preparing and fault == "unknown":
            worker.spatial_map.cells[:] = -1
        return result

    def start(*args, **kwargs):
        nonlocal entry_starts
        if preparing:
            entry_starts += 1
        result = original_start(*args, **kwargs)
        if preparing and fault == "stop":
            worker.stop()
        return result

    monkeypatch.setattr(worker, "prepare_exploration_entry", entry)
    monkeypatch.setattr(worker, "_sample_spatial", sample)
    monkeypatch.setattr(worker, "_start_navigation_path", start)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        def attach(sim):
            previous = sim.on_tick
            def tick():
                previous()
                physical.sample(worker)
            sim.on_tick = tick
            physical.sample(worker)
        await worker.call(attach)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            goal="Explore the observed environment", execution_mode="luna_continuous", unified_mission=True,
            mission_local_only=True, mission_budget_s=60., max_turns=12, feedback_interval_s=.25))
        if fault is None:
            async with asyncio.timeout(75.):
                while not controller.task.done():
                    if physical.travel >= 1.5:
                        worker.stop()
                        break
                    await asyncio.sleep(.05)
        await asyncio.wait_for(controller.task, 90.)
        entries = [event["payload"] for event in controller.trace()["events"] if event["title"] == "Observed exploration entry"]
        record_property("exploration_entry", json.dumps({"fault": fault, "travel_m": physical.travel,
            "contacts": physical.contacts, "entries": entries, "error": controller.state["error"]}))
        assert physical.contacts == 0
        assert controller.state["input_tokens"] == 0 and not controller.active
        assert worker.sim.cancel.is_set() and not worker.home_mission.active
        assert worker.latest["challenge"]["status"] != "completed"
        if fault is None:
            assert entries and all(0.15 <= item["travel_m"] <= .5 and item["elapsed_s"] <= 8. for item in entries)
            assert physical.travel >= 1.5
            assert controller.state["outcome"]["kind"] == "interrupted"
            assert (await worker.home_state(compact=True))["coverage"]["free_m2"] >= 35.
        else:
            assert not entries and sum(entry_travel) < .03
            if fault != "unknown":
                assert physical.travel < .03
            assert entry_starts == (1 if fault == "stop" else 0)
            assert controller.state["error"] is not None or controller.state["outcome"]["kind"] == "interrupted"
        ticks = await worker.call(lambda sim: sim.ticks)
        await asyncio.sleep(.2)
        assert await worker.call(lambda sim: sim.ticks) == ticks
    finally:
        await controller.halt()
        await worker.close()


def test_targeted_exploration_plan_is_rejected_before_it_can_disable_room_supervision():
    from backend.mission import MissionDecision
    with pytest.raises(ValueError, match="kind=room"):
        MissionDecision(action="plan", plan=MissionPlan(kind="explore",
            target="Kitchen identified by visible cooking appliances; enter fully and stop in its clear interior"))
    assert MissionDecision(action="plan", plan=MissionPlan(kind="explore")).plan.target == ""
    assert MissionDecision(action="plan", plan=MissionPlan(kind="room", target="Kitchen with cooking appliances")).plan.kind == "room"


@pytest.mark.parametrize("bad_target", [False, True])
async def test_exploration_plan_cannot_disable_requested_luna_supervision(tmp_path, bad_target):
    import asyncio
    import json
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    worker = SimulationWorker(challenge=get_challenge("park"), rendering="tiny", pace=True)
    replies = ([model_response("guide_mission", '{"action":"plan","plan":{"kind":"explore","target":"Kitchen identified by cooking appliances"}}',
        call_id="bad-kitchen-plan")] if bad_target else [])
    replies.extend([
        model_response("guide_mission", json.dumps({"action": "plan", "plan": {"kind": "room" if bad_target else "explore",
            "target": "Kitchen" if bad_target else "", "return_home": False}}), call_id="plan"),
        model_response("guide_mission", '{"action":"wait","duration_s":0.5}', call_id="supervised-wait")])
    model = ScriptedModel(replies)
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=25., max_turns=len(replies), max_model_requests=len(replies)))
        await asyncio.wait_for(controller.task, 35.)
        assert controller.state["error"] is None
        assert len(model.inputs) == controller.state["inference_budget"]["requests"] == len(replies)
        assert json.loads(model.inputs[-1][-1]["content"][0]["text"])["mission"]["plan"]["kind"] == ("room" if bad_target else "explore")
        if bad_target:
            correction = json.loads(model.inputs[1][-1]["content"][0]["text"])
            assert correction["mission"]["plan"] is None
            assert {"plan", "execute_movement"} <= set(correction["available_actions"])
            assert "kind=room" in correction["last_execution"]["reason"]
        assert worker.home_mission.task is None
        assert worker.latest["stopped"] and not controller.active
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("direction,offset,ending", [
    (direction, offset, "complete") for direction in ("clockwise", "counterclockwise") for offset in (0., .1, -.1)] + [
    (direction, 0., ending) for direction in ("clockwise", "counterclockwise")
    for ending in ("stop", "wrong_kind", "wrong_direction", "timeout")])
async def test_unified_table_circuit_drives_and_preserves_authority(tmp_path, monkeypatch, record_property, direction, offset, ending):
    import asyncio
    import json
    import math
    import time
    from backend.challenges import furniture_circuit
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    plan = {"kind": "object" if ending == "wrong_kind" else "circuit", "target": "table", "circle_direction": direction}
    opposite = "counterclockwise" if direction == "clockwise" else "clockwise"
    decision = {"action": "circle", "object_label": "Table", "object_bounds": [.32, 0., .68, .58],
        "circle_direction": opposite if ending == "wrong_direction" else direction}
    model = ScriptedModel(["wait"] if ending == "timeout" else [
        model_response("guide_mission", json.dumps({"action": "plan", "plan": plan}), call_id="circuit-plan"),
        model_response("guide_mission", json.dumps(decision), call_id="circuit-drive")])
    model.requires_grounded_plan = True
    controller = controller_for(model, request_timeout_s=.03 if ending == "timeout" else 45)
    challenge = furniture_circuit("table", direction)
    challenge = challenge.model_copy(update={"initial_xy": [challenge.initial_xy[0] + offset, challenge.initial_xy[1]]})
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    progress = []
    original_trace = controller._trace

    def trace(kind, title, payload, **kwargs):
        original_trace(kind, title, payload, **kwargs)
        if title == "Observed furniture circuit":
            progress.append(payload)
            if ending == "stop" and payload["swept_degrees"] > 20.:
                worker.stop()

    monkeypatch.setattr(controller, "_trace", trace)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        started = time.monotonic()
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=True, images_per_request=2,
            mission_budget_s=180., max_turns=2, max_model_requests=2).model_copy(update={"goal": challenge.goal}))
        deadline = controller.mission.deadline
        await asyncio.wait_for(controller.task, 195.)
        measured = await worker.call(lambda sim: {"pose": sim.odometry.tolist(), "travel_m": sim.path_length,
            "contacts": len(sim.proximity_sensors().collisions), "challenge": sim.challenge_status(), "ticks": sim.ticks})
        record_property("unified_circuit", json.dumps({"evidence": "scripted_image_box_real_enhanced_physics",
            "direction": direction, "offset_m": offset, "ending": ending, "seconds": time.monotonic()-started, "requests": len(model.inputs),
            "progress": progress[-1] if progress else None, "mission": controller.mission.state(), **measured}))
        assert worker.latest["stopped"] and not controller.active and not worker.home_mission.active
        assert controller.mission.deadline == deadline and measured["contacts"] == 0
        if ending == "complete":
            assert controller.state["error"] is None, controller.trace()
            assert controller.mission.phase == "completed" and "target" in controller.mission.receipts
            assert measured["challenge"]["status"] == "completed", measured
            assert measured["travel_m"] > 6. and progress[-1]["complete"]
        else:
            assert "target" not in controller.mission.receipts
            if ending == "stop":
                assert progress[-1]["swept_degrees"] > 20. and measured["travel_m"] > .3
            else:
                assert not progress and math.hypot(*measured["pose"][:2]) < .01
            if ending == "timeout":
                assert "response timed out" in controller.state["error"] and "no objective was authorized" in controller.state["error"]
                assert controller.state["inference_budget"]["usage_unknown"] and len(model.inputs) == 1
        await asyncio.sleep(.2)
        assert await worker.call(lambda sim: sim.ticks) == measured["ticks"]
    finally:
        model.release.set()
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("target,direction,lateral,fault,preset", [
    ("table", "clockwise", 0., None, None), ("table", "counterclockwise", .25, None, None), ("chair", "clockwise", 0., None, None),
    *[("table", "clockwise", 0., fault, None) for fault in ("unknown", "wrong_direction", "pending_stop")],
    pytest.param("chair", "clockwise", 0., None, "chair_circuit_far", id="chair-far")])
async def test_circle_candidate_makes_directed_arc_and_obeys_stop(tmp_path, monkeypatch, record_property, target, direction, lateral, fault, preset):
    import asyncio
    import json
    import math
    from backend.challenges import furniture_circuit, get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    challenge = get_challenge(preset) if preset else furniture_circuit(target, direction)
    center = challenge.orbit.center_m
    if not preset:
        challenge = challenge.model_copy(update={"initial_xy": [center[0]-2.4, center[1]+lateral]})
    selections, progress = [], []

    class CandidateModel(ScriptedModel):
        async def respond(self, profile, reasoning, goal, inputs):
            self.inputs.append(inputs)
            brief = json.loads(inputs[-1]["content"][0]["text"])
            if brief["mission"]["plan"] is None:
                decision = {"action": "plan", "plan": {"kind": "circuit", "target": target, "circle_direction": direction}}
            else:
                candidates = brief["observation"]["spatial"]["object_candidates"]
                assert candidates
                selected = candidates[0]
                selections.append(selected)
                decision = {"action": "circle", "object_label": target, "object_candidate_id": selected["id"],
                    "evidence_text": "Scripted selection of the largest visible measured surface", "circle_direction": direction}
                if fault == "unknown":
                    decision["object_candidate_id"] = "previous-run:object-0"
                elif fault == "wrong_direction":
                    decision["circle_direction"] = "counterclockwise"
                elif fault == "pending_stop":
                    worker.stop()
            return model_response("guide_mission", json.dumps(decision), call_id=f"candidate-{len(self.inputs)}")

    model = CandidateModel([])
    model.requires_grounded_plan = True
    controller = controller_for(model)
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    original_trace = controller._trace

    def trace(kind, title, payload, **kwargs):
        original_trace(kind, title, payload, **kwargs)
        if title == "Observed furniture circuit":
            progress.append(payload)
            if payload["swept_degrees"] >= 35.:
                worker.stop()

    monkeypatch.setattr(controller, "_trace", trace)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=True, images_per_request=2, mission_budget_s=120., max_turns=2, max_model_requests=2
            ).model_copy(update={"goal": challenge.goal}))
        await asyncio.wait_for(controller.task, 135.)
        measured = await worker.call(lambda sim: {"travel_m": sim.path_length, "contacts": len(sim.proximity_sensors().collisions),
            "challenge": sim.challenge_status(), "ticks": sim.ticks})
        record_property("circle_attempt", json.dumps({"evidence": "scripted_candidate_selection_real_enhanced_physics",
            "target": target, "direction": direction, "lateral_m": lateral, "fault": fault, "selection": selections,
            "preset": preset, "initial_target_distance_m": math.dist(challenge.initial_xy, center),
            "progress": progress, "error": controller.state["error"], **measured}))
        if fault:
            assert not progress and measured["travel_m"] < .01, controller.trace()
        else:
            assert progress and progress[-1]["swept_degrees"] >= 35., controller.trace()
            assert measured["travel_m"] > (1.5 if preset else .5)
            assert all(item["direction"] == direction for item in progress)
            expected_center = [center[0]-challenge.initial_xy[0], center[1]-challenge.initial_xy[1]]
            assert math.dist(progress[0]["observed_center_m"], expected_center) < .7
        assert measured["contacts"] == 0
        assert worker.latest["stopped"] and not controller.active
        assert "target" not in controller.mission.receipts and measured["challenge"]["status"] != "completed"
        await asyncio.sleep(.2)
        assert await worker.call(lambda sim: sim.ticks) == measured["ticks"]
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("supervised", [False, True])
async def test_table_scene_drives_from_simple_local_to_supervised_exploration(tmp_path, record_property, supervised):
    import asyncio
    import json
    import math
    import time
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    model = ScriptedModel([
        model_response("guide_mission", '{"action":"plan","plan":{"kind":"explore"}}', call_id="startup-plan"),
        model_response("guide_mission", '{"action":"explore"}', call_id="startup-explore")])
    controller = controller_for(model)
    worker = SimulationWorker(challenge=get_challenge("furniture_circuit"), rendering="enhanced", pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        started = time.monotonic()
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            mission_local_only=not supervised, map_context=True, images_per_request=2,
            mission_budget_s=60., max_turns=3, max_model_requests=2))
        async with asyncio.timeout(65.):
            while math.hypot(*worker.sim.odometry[:2]) < .3:
                assert not controller.task.done(), {"state": controller.state, "task": worker.home_mission.task}
                await asyncio.sleep(.05)
        measured = await worker.call(lambda sim: {"pose": sim.odometry.tolist(),
            "contacts": len(sim.proximity_sensors().collisions), "ticks": sim.ticks})
        record_property("startup_drive", json.dumps({"evidence": "scripted_enhanced_physics",
            "supervised": supervised, "time_to_30cm_s": time.monotonic()-started,
            "model_requests": len(model.inputs), **measured}))
        worker.stop()
        await controller.halt()
        frozen = await worker.call(lambda sim: (sim.odometry.copy(), sim.ticks))
        await asyncio.sleep(.2)
        after = await worker.call(lambda sim: (sim.odometry.copy(), sim.ticks))
        assert math.dist(frozen[0], after[0]) == 0. and frozen[1] == after[1]
        assert measured["contacts"] == 0 and len(model.inputs) == 2 * int(supervised)
        assert worker.latest["stopped"] and not controller.active
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("challenge_id", ["park", "kitchen_bathroom"])
async def test_room_search_delegates_frontiers_to_local_worker(tmp_path, monkeypatch, record_property, interrupted, challenge_id):
    import asyncio
    import json
    import math
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.mission_supervisor import tools
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    replies = [{"action": "plan", "plan": {"kind": "room", "target": "Bathroom with toilet and sink"}},
        {"action": "explore", "frontier_id": "old-model-frontier"}, {"action": "look", "pitch_rad": .2}]
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"delegated-{index}")
        for index, reply in enumerate(replies)])
    worker = SimulationWorker(challenge=get_challenge(challenge_id), rendering="tiny", pace=True)
    controller = controller_for(model)
    requests = []
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        original_command = worker.home_mission.command

        def tracked_command(request, *args, **kwargs):
            requests.append(request)
            return original_command(request, *args, **kwargs)

        monkeypatch.setattr(worker.home_mission, "command", tracked_command)
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=60., max_turns=3, max_model_requests=3))
        deadline = controller.mission.deadline
        if interrupted:
            async with asyncio.timeout(70.):
                while (not worker.home_mission.active or not worker.home_mission.task["segments"]
                        or math.hypot(*worker.sim.odometry[:2]) < .15):
                    assert not controller.task.done(), {"state": controller.state, "task": worker.home_mission.task}
                    await asyncio.sleep(.02)
            worker.stop()
            await controller.halt()
        else:
            await asyncio.wait_for(controller.task, 70.)
        assert "frontier_id" not in tools(["explore"])[0]["parameters"]["properties"]
        surveys = [event for event in controller.trace()["events"] if event["title"] == "Local exploration survey finished"]
        assert len(surveys) <= 1 and len(requests) == 1 + len(surveys)
        assert all(request.action == "explore" and request.frontier_id is None for request in requests)
        assert controller.mission.deadline == deadline
        assert controller.mission.plan.target == "Bathroom with toilet and sink" and "target" not in controller.mission.receipts
        task = worker.home_mission.task
        assert task["local_exploration"] and not task["single_frontier"]
        displacement = math.hypot(*worker.sim.odometry[:2])
        record_property("local_supervision", json.dumps({"evidence": "scripted_real_physics", "interrupted": interrupted,
            "challenge": challenge_id, "displacement_m": displacement, "segments": task["segments"], "visited_frontiers": task["visited_frontiers"],
            "requests": len(model.inputs), "local_surveys": len(surveys), "reason": task["reason"]}))
        assert displacement >= .15 and task["segments"] >= 1
        assert not worker.sim.proximity_sensors().collisions and worker.latest["stopped"] and not controller.active
        if interrupted:
            assert len(model.inputs) == 2
        else:
            assert controller.state["error"] is None, {"state": controller.state, "task": task}
            assert len(model.inputs) == 3
            review = json.loads(model.inputs[2][-1]["content"][0]["text"])
            assert review["last_execution"]["task"]["status"] in {"completed", "running", "failed", "limited"}
            if review["last_execution"]["task"]["status"] in {"failed", "limited"}:
                assert review["mission"]["operation_id"] is None
                assert review["last_execution"]["task"]["reason"]
                assert not review["last_execution"]["task"]["completion_verified"]
            assert not any(event["payload"].get("paused_for_review") for event in controller.trace()["events"]
                if event["title"] == "Mission review checkpoint")
    finally:
        await controller.halt()
        await worker.close()


async def test_kitchen_roaming_continues_across_supervision_checkpoints(tmp_path, monkeypatch, record_property):
    import asyncio
    import json
    import math
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    replies = [{"action": "plan", "plan": {"kind": "room", "target": "Bathroom with toilet and sink"}},
        *[{"action": "explore"} for index in range(3)], {"action": "wait", "duration_s": .5}]
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"sustained-{index}")
        for index, reply in enumerate(replies)])
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="enhanced", pace=True)
    controller = controller_for(model)
    samples = []
    try:
        await asyncio.wrap_future(worker.ready)
        home = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        worker.home_mission = home
        original_tick = home.tick

        def tracked_tick():
            original_tick()
            samples.append({"pose": worker.sim.odometry.copy(), "contacts": list(worker.sim.proximity_sensors().collisions)})

        monkeypatch.setattr(home, "tick", tracked_tick)
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            reasoning="low", map_context=True, images_per_request=2, mission_budget_s=90., max_turns=5, max_model_requests=5))
        await asyncio.wait_for(controller.task, 100.)
        results = [event["payload"]["result"] for event in controller.trace()["events"] if event["title"] == "Mission capability feedback"]
        operations = [event["payload"]["task"] for event in controller.trace()["events"] if event["title"] == "Mission operation finished"]
        distance = sum(math.dist(first["pose"][:2], second["pose"][:2]) for first, second in zip(samples, samples[1:]))
        record_property("sustained_roaming", json.dumps({"evidence": "scripted_enhanced_physics", "distance_m": distance,
            "requests": len(model.inputs), "operations": len(operations), "results": results,
            "sampled_contacts": sum(len(sample["contacts"]) for sample in samples)}))
        assert controller.state["error"] is None, controller.state
        assert not any(result.get("status") == "rejected" for result in results), results
        renewals = [event["payload"] for event in controller.trace()["events"] if event["title"] == "Mission objective renewed"]
        assert len(operations) == 1 and len(renewals) == 2 and len(model.inputs) == 5
        assert all(renewal["task"]["task_id"] == operations[0]["task_id"] for renewal in renewals)
        assert distance >= 1.
        assert not any(sample["contacts"] for sample in samples)
        assert all(task["local_exploration"] and not task["single_frontier"] for task in operations)
        assert worker.latest["stopped"] and not controller.active and "target" not in controller.mission.receipts
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("corrected", [True, False])
async def test_mission_target_correction_receives_exact_error_and_preserves_limits(tmp_path, corrected):
    import asyncio
    import json
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    invalid = {"action": "plan", "plan": {"kind": "room", "target": "x" * 416, "return_home": False}}
    valid = {"action": "plan", "plan": {"kind": "room", "target": "Bathroom with toilet and sink", "return_home": False}}
    replies = [invalid, valid] if corrected else [invalid] * 3
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="tiny", pace=False)
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"plan-{index}")
        for index, reply in enumerate(replies)])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=60., max_turns=len(replies), max_model_requests=len(replies)))
        await asyncio.wait_for(controller.task, 70.)
        retry = json.loads(model.inputs[1][-1]["content"][0]["text"])["last_execution"]
        assert retry["reason"] == "plan.target: String should have at most 160 characters"
        assert retry["validation_errors"][0]["field"] == "plan.target"
        assert len(model.inputs) == controller.state["inference_budget"]["requests"] == len(replies)
        rejections = [event for event in controller.trace()["events"] if event["title"] == "Mission decision rejected"]
        assert len(rejections) == (1 if corrected else 3)
        if corrected:
            assert controller.state["error"] is None
            assert controller.mission.plan.target == valid["plan"]["target"]
            assert controller.state["outcome"]["kind"] == "limited"
        else:
            assert controller.mission.plan is None
            assert controller.state["error"] == "Mission decision correction limit reached: plan.target: String should have at most 160 characters"
        assert not controller.mission.receipts and worker.latest["stopped"] and not controller.active
        assert not any(event["title"] == "Mission operation finished" for event in controller.trace()["events"])
    finally:
        await controller.halt()
        await worker.close()


def test_mission_reports_current_fixtures_without_unlocking_arrival_or_replanning():
    from types import SimpleNamespace
    from backend.mission import MissionDecision
    from backend.mission_supervisor import available_actions
    mission = Mission("run", 0, 0, 0, deadline=60., clock=lambda: 0.)
    observation = SimpleNamespace(spatial={"places": [{"place_id": "home", "kind": "place", "reachable": True}]})
    assert {"plan", "execute_movement"} <= set(available_actions(mission, observation, None))
    mission.configure(MissionPlan(kind="room", target="Bathroom with toilet and sink"))
    assert available_actions(mission, observation, None) == ["report_observation", "look", "turn", "wait", "explore"]
    report = MissionDecision(action="report_observation", evidence_text="Visible oven and worktop suggest a kitchen")
    assert report.place_id is None
    with pytest.raises(ValueError, match="current visual evidence"):
        MissionDecision(action="report_observation")
    with pytest.raises(ValueError, match="report_observation"):
        MissionDecision(action="observe_room", room_matches=False, evidence_text=report.evidence_text)
    with pytest.raises(ValueError, match="already established"):
        mission.configure(MissionPlan(kind="room", target="Kitchen"))
    assert mission.plan.target == "Bathroom with toilet and sink" and not mission.receipts


@pytest.mark.parametrize("repeat_plan", [False, True])
async def test_mission_recovers_starting_room_report_then_executes_next_action(tmp_path, repeat_plan):
    import asyncio
    import json
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    evidence = "Starting room appears to be a kitchen: visible oven, cabinets and worktop."
    plan = {"kind": "room", "target": "Bathroom with toilet and sink", "return_home": False, "circle_direction": "clockwise",
        "completion": "arrive", "movements": []}
    replies = [{"action": "plan", "plan": plan},
        {"action": "observe_room", "room_matches": False, "evidence_text": evidence}]
    if repeat_plan:
        replies.extend([{"action": "observe_room", "place_id": None, "room_matches": False, "evidence_text": evidence},
            {"action": "plan", "plan": plan}])
    replies.extend([{"action": "report_observation", "evidence_text": evidence, "room_label": "Kitchen", "room_confidence": .7},
        {"action": "look", "yaw_rad": .2, "pitch_rad": .2}])
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="tiny", pace=False)
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"report-{index}")
        for index, reply in enumerate(replies)])
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=60., max_turns=len(replies), max_model_requests=len(replies)))
        deadline = controller.mission.deadline
        await asyncio.wait_for(controller.task, 70.)
        payloads = [json.loads(inputs[-1]["content"][0]["text"]) for inputs in model.inputs]
        assert {"plan", "execute_movement"} <= set(payloads[0]["available_actions"])
        assert all("plan" not in payload["available_actions"] for payload in payloads[1:])
        assert "report_observation" in payloads[2]["last_execution"]["reason"]
        assert all(payload["mission"]["plan"] == plan for payload in payloads[1:])
        if repeat_plan:
            assert payloads[-2]["last_execution"]["status"] == "plan_already_accepted"
            assert "plan" not in payloads[-2]["last_execution"]["available_actions"]
            assert payloads[-2]["mission"]["home_id"] == payloads[1]["mission"]["home_id"]
        reported = payloads[-1]["last_execution"]
        assert reported["status"] == "observation_reported" and reported["evidence_text"] == evidence
        record = reported["room_observation"]
        assert record["label"] == "Kitchen" and record["confidence"] == .7
        assert record["review_status"] == "tentative" and not record["identity_verified"]
        from backend.feedback import compact_numbers
        stored_record = worker.home_mission.store.room_observations(record["map_id"])[0]
        assert compact_numbers(stored_record) == record
        full_report = next(event["payload"]["result"] for event in controller.trace()["events"]
            if event["title"] == "Mission capability feedback" and event["payload"]["result"].get("status") == "observation_reported")
        assert full_report["room_observation"] == stored_record
        assert worker.home_mission.store.room_image(record["map_id"], record["observation_id"]).startswith(b"\x89PNG")
        assert not reported["arrival_verified"] and not reported["identity_verified"]
        assert reported["observation_seq"] == payloads[-2]["observation"]["seq"]
        assert controller.state["error"] is None and controller.mission.rejections == 0
        assert controller.mission.deadline == deadline
        assert controller.mission.plan.model_dump() == plan and not controller.mission.receipts
        assert len(model.inputs) == controller.state["inference_budget"]["requests"] == len(replies)
        assert worker.latest["stopped"] and not controller.active
        results = [event["payload"]["result"] for event in controller.trace()["events"] if event["title"] == "Mission capability feedback"]
        assert results[-1] == {"status": "observed", "action": "look"}
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("angle, interrupted", [(3.14159, False), (-3.14159, False), (3.14159, True)])
async def test_mission_half_turn_stays_in_place_refreshes_view_and_obeys_stop(tmp_path, monkeypatch, record_property, angle, interrupted):
    import asyncio
    import json
    import math
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from scripts import navigation_policy
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    replies = [{"action": "plan", "plan": {"kind": "room", "target": "Bathroom with toilet and sink"}},
        {"action": "turn", "turn_rad": angle, "reason": "Turn around at the observed dead end"},
        {"action": "wait", "duration_s": .5}]
    model = ScriptedModel([model_response("guide_mission", json.dumps(reply), call_id=f"half-turn-{index}")
        for index, reply in enumerate(replies)])
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="tiny", pace=True)
    controller = controller_for(model)
    segments = []
    contacts = []
    original_apply = navigation_policy.apply

    def tracked_apply(navigation, sim, tool, arguments):
        if tool == "replace_motion_buffer":
            segments.extend(arguments["segments"])
            contacts.extend(sim.proximity_sensors().collisions)
        return original_apply(navigation, sim, tool, arguments)

    monkeypatch.setattr(navigation_policy, "apply", tracked_apply)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=False, mission_budget_s=60., max_turns=3, max_model_requests=3))
        if interrupted:
            async with asyncio.timeout(40.):
                while len(model.inputs) < 2 or abs(worker.sim.odometry[2]) < .3:
                    assert not controller.task.done(), controller.state
                    await asyncio.sleep(.02)
            worker.stop()
            await controller.halt()
        else:
            await asyncio.wait_for(controller.task, 70.)
        before = json.loads(model.inputs[1][-1]["content"][0]["text"])["observation"]
        initial_pose = before["odometry_m_rad"]
        final_pose = worker.sim.odometry.copy()
        delta_yaw = float(final_pose[2] - initial_pose[2])
        displacement = math.dist(initial_pose[:2], final_pose[:2])
        record_property("half_turn", json.dumps({"evidence": "scripted_real_physics", "requested_rad": angle,
            "actual_rad": delta_yaw, "displacement_m": displacement, "interrupted": interrupted}))
        assert segments and all(segment["linear_mps"] == 0. for segment in segments)
        assert not contacts and not worker.sim.proximity_sensors().collisions
        assert displacement < .03
        assert worker.latest["stopped"] and not controller.active
        if interrupted:
            assert .2 < abs(delta_yaw) < 1.
            assert len(model.inputs) == 2
        else:
            assert controller.state["error"] is None, controller.state
            assert controller.mission.rejections == 0
            assert delta_yaw == pytest.approx(angle, abs=.08)
            after = json.loads(model.inputs[2][-1]["content"][0]["text"])
            assert after["last_execution"] == {"status": "observed", "action": "turn"}
            assert after["observation"]["seq"] > before["seq"]
            assert after["observation"]["odometry_m_rad"][2] - initial_pose[2] == pytest.approx(angle, abs=.08)
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("stop_phase,with_memory,use_candidates", [(None, False, False), ("reply", False, False), ("return", False, False),
    pytest.param("redirect", False, False, id="task-change-cabinet-to-home"), pytest.param(None, True, False, id="persistent-memory"),
    pytest.param(None, False, True, id="observed-candidate-ids")])
async def test_unified_visual_object_approach_and_automatic_return(tmp_path, stop_phase, record_property, with_memory, use_candidates):
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
        requires_grounded_plan = use_candidates

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
            if use_candidates and decision["action"] in {"select_object", "verify_object"}:
                candidates = payload["observation"]["spatial"]["object_candidates"]
                candidate = min(candidates, key=lambda item: sum(abs(actual-expected)
                    for actual, expected in zip(item["bounds"], bounds)))
                assert candidate["bounds"][0] <= float(columns.mean()/width) <= candidate["bounds"][2]
                assert candidate["bounds"][1] <= float(rows.mean()/height) <= candidate["bounds"][3]
                decision.pop("object_bounds")
                decision.update(object_candidate_id=candidate["id"], evidence_text="Visible red cabinet body")
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
        if with_memory:
            from backend.memory_session import configure_memory
            await configure_memory(worker, instance_id="object-integration-fixture")
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
        if with_memory:
            await worker.memory.flush()
            assert worker.memory.error is None, worker.memory.error
            sightings = worker.memory.store.object_observations(worker.memory.scope.map_id)
            assert len(sightings) == 2, sightings
            assert len({item["entity_id"] for item in sightings}) == 1
            assert all(item["image_sha256"] and item["position_m"] and not item["identity_verified"] for item in sightings)
            assert sightings[0]["spatial_sequence"] > sightings[1]["spatial_sequence"]
            assert worker.memory.summary["objects"][0]["observation_count"] == 2
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