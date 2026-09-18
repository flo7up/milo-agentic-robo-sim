import asyncio
import json
import math

import pytest

from backend.mission import MissionPlan
from backend.movement import MovementStep


def test_movement_contract_preserves_distance_direction_and_turns():
    from types import SimpleNamespace
    from backend.mission import Mission
    from backend.mission_supervisor import available_actions, tools
    steps = [MovementStep(kind="drive", distance_m=1.), MovementStep(kind="drive", distance_m=-1.),
        MovementStep(kind="turn", angle_rad=2*math.pi)]
    plan = MissionPlan(kind="movement", target="Forward, backward, spin", movements=steps)
    assert plan.movements[1].distance_m == -1.
    assert plan.movements[2].angle_rad == 2*math.pi
    mission = Mission("run", 0, 0, 0, 60., clock=lambda: 0.)
    assert "execute_movement" in available_actions(mission, SimpleNamespace(spatial={}), None)
    mission.configure(plan)
    assert available_actions(mission, SimpleNamespace(spatial={}), None) == ["execute_movement"]
    assert tools(["plan", "execute_movement"])[0]["parameters"]["$defs"]["MovementStep"]
    for step in ({"kind": "drive", "distance_m": 2.}, {"kind": "drive", "distance_m": 0.},
            {"kind": "turn", "angle_rad": 20.}, {"kind": "drive", "distance_m": 1., "angle_rad": .5}):
        with pytest.raises(ValueError):
            MovementStep(**step)
    with pytest.raises(ValueError):
        MissionPlan(kind="movement", target="Empty")
    with pytest.raises(ValueError):
        MissionPlan(kind="movement", target="Too far", movements=[steps[0]]*5)


def test_movement_program_preserves_existing_challenge_serialization():
    from backend.challenges import get_challenge, shared_apartment
    assert "movement_program" not in shared_apartment("flat_kitchen").model_dump()
    practice = get_challenge("movement_practice")
    assert practice.model_dump()["movement_program"] == [step.model_dump() for step in practice.movement_program]
    assert "movement_program" not in practice.public()


@pytest.mark.parametrize("program", ["out-and-back", "spin", "default", "triple-spin", "clockwise"])
async def test_requested_movement_executes_measured_sequence(tmp_path, record_property, program):
    import numpy as np
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    steps = ([{"kind": "drive", "distance_m": 1.}, {"kind": "drive", "distance_m": -1.}] if program in {"out-and-back", "default"} else [])
    if program != "out-and-back":
        steps.append({"kind": "turn", "angle_rad": (6 if program == "triple-spin" else -2 if program == "clockwise" else 2)*math.pi})
    model = ScriptedModel([model_response("guide_mission", json.dumps({"action": "execute_movement",
        "plan": {"kind": "movement", "target": program, "movements": steps}}), call_id="movement")])
    worker = SimulationWorker(challenge=get_challenge("movement_practice"), rendering="enhanced", pace=True)
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "movement.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            images_per_request=2, mission_budget_s=120., max_turns=1, max_model_requests=1))
        await asyncio.wait_for(controller.task, 135.)
        def diagnostic(sim):
            observed = worker.spatial_map
            radius = sim.robot_footprint()["radius_m"]
            allowed = observed.traversable(sim.odometry[:2], radius)
            points = [[distance, 0.] for distance in np.linspace(0., 1., 11)]
            return {"odometry": sim.odometry.tolist(), "radius": radius,
                "corridor": [{"point": point, "cell": int(observed.cells[observed.cell_index(point)]),
                    "allowed": observed.contains_path(allowed, [point])} for point in points],
                "completed_steps": [event for event in controller.trace()["events"]
                    if event.get("message") == "Requested movement step completed"]}
        if controller.state["error"]:
            details = await worker.call(diagnostic)
            record_property("clearance_failure", json.dumps(details))
            pytest.fail(str({"error": controller.state["error"], **details}))
        assert controller.mission.phase == "completed", controller.trace()
        receipts = controller.mission.receipts["target"]["steps"]
        assert len(receipts) == len(steps) and len(model.inputs) == 1
        for result, step in zip(receipts, steps):
            assert result["measured"] == pytest.approx(step.get("distance_m", step.get("angle_rad")), abs=.04 if step["kind"] == "drive" else .08)
        assert np.linalg.norm(worker.sim.odometry[:2]) <= .06
        assert not worker.sim.proximity_sensors().collisions and worker.latest["stopped"]
        if program == "default":
            assert worker.latest["challenge"]["status"] == "completed", worker.latest["challenge"]
        record_property("movement", json.dumps({"program": program, "steps": receipts, "evidence": "scripted_real_physics",
            "final_odometry": worker.sim.odometry.tolist(), "challenge": worker.latest["challenge"]["status"]}))
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("interrupt", ["stop", "sensor", "obstacle", "authority", "deadline", "unknown-reverse"])
async def test_movement_safety_stops_without_completion(tmp_path, interrupt):
    import numpy as np
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.worker import SimulationWorker
    from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings
    step = {"kind": "drive", "distance_m": -1.} if interrupt == "unknown-reverse" else {"kind": "turn", "angle_rad": 6*math.pi}
    model = ScriptedModel([model_response("guide_mission", json.dumps({"action": "execute_movement",
        "plan": {"kind": "movement", "target": "Safety fixture", "movements": [step]}}), call_id="safety")])
    worker = SimulationWorker(challenge=get_challenge("movement_practice"), rendering="enhanced", pace=True)
    controller = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "movement.sqlite3"))
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            images_per_request=2, mission_budget_s=90., max_turns=1, max_model_requests=1))
        if interrupt != "unknown-reverse":
            async def until_moving():
                while abs((await worker.call(lambda sim: sim.odometry.copy()))[2]) < .15:
                    assert not controller.task.done(), controller.state
                    await asyncio.sleep(.05)
            await asyncio.wait_for(until_moving(), 45.)
            def invalidate(sim):
                if interrupt == "stop":
                    worker.stop()
                elif interrupt == "sensor":
                    worker.spatial_enabled = False
                elif interrupt == "obstacle":
                    worker.spatial_map.cells[worker.spatial_map.cell_index(sim.odometry[:2])] = 100
                elif interrupt == "authority":
                    worker.task_revision += 1
                else:
                    controller.mission.deadline = 0.
            await worker.call(invalidate)
        await asyncio.wait_for(controller.task, 45.)
        assert controller.mission.phase != "completed"
        assert "target" not in controller.mission.receipts
        assert len(model.inputs) == 1
        frozen = await worker.call(lambda sim: (sim.odometry.copy(), sim.ticks))
        await worker.call(lambda sim: None)
        assert np.allclose(worker.sim.odometry, frozen[0]) and worker.sim.ticks == frozen[1]
        assert worker.latest["stopped"] and worker.movement_guard is None
        if interrupt == "unknown-reverse":
            assert np.linalg.norm(frozen[0][:2]) < .005
            assert "OBSERVED_PATH_BLOCKED" in controller.state["error"]
    finally:
        await controller.halt()
        await worker.close()


def test_movement_scorer_rejects_wrong_direction_and_teleports():
    from backend.challenges import ChallengeProgress, get_challenge
    for distance, heading, contact in [(-.05, 0., False), (1., 0., False), (0., .1, False), (0., 0., True)]:
        progress = ChallengeProgress(get_challenge("movement_practice"))
        measurement = {"position_xy": [0., 0.], "heading_rad": 0., "contact": False,
            "grounded": True, "speed": 0., "angular_speed": 0.}
        progress.update({"robot": measurement}, 0.)
        measurement.update(position_xy=[distance, 0.], heading_rad=heading, contact=contact)
        assert progress.update({"robot": measurement}, .1)["status"] == "failed"
        assert progress.update({"robot": measurement}, 1.)["completed_objectives"] == 0