from types import SimpleNamespace

import numpy as np
import pytest

from backend.object_navigation import ObservedObjectGoal
from backend.spatial import calibration


def object_sensor(**changes):
    values = dict(calibration=calibration(24, 16), depth_m=[1.5] * (24 * 16),
        head_rad=[0., 0.], odometry_m_rad=[0., 0., 0.], sequence=1, captured_at=10., run_id="run", episode_epoch=1)
    return SimpleNamespace(**(values | changes))


def test_object_options_keep_intention_and_reject_blocked_corridor():
    goal = ObservedObjectGoal(object_sensor(), [.4, .4, .6, .6], "table", .8, 2, 3)
    options = goal.options([0., 0., 0.], lambda start, end: [start, end], lambda points: np.all(points[:, 1] >= -.01))
    assert [option["status"] for option in options] == ["reachable", "reachable", "blocked"]
    assert goal.pose("left")[1] > 0 and goal.pose("right")[1] < 0
    assert goal.pose("front")[0] == pytest.approx(.755)
    target = goal.pose("left")
    goal.options([.2, .1, .1], lambda start, end: [start, end], lambda points: True)
    assert goal.pose("left") == target
    assert not goal.verified


def test_object_goal_rejects_revoked_and_expired_evidence():
    goal = ObservedObjectGoal(object_sensor(), [.4, .4, .6, .6], "table", .8, 2, 3)
    goal.check("run", 1, 2, 3, 11.)
    for authority in (("other", 1, 2, 3), ("run", 2, 2, 3), ("run", 1, 3, 3), ("run", 1, 2, 4)):
        with pytest.raises(ValueError, match="authority"):
            goal.check(*authority, 11.)
    with pytest.raises(ValueError, match="expired"):
        goal.check("run", 1, 2, 3, 131.)
    with pytest.raises(ValueError, match="insufficient"):
        ObservedObjectGoal(object_sensor(depth_m=[None] * 384), [.4, .4, .6, .6], "table", .8, 2, 3)


def test_object_arrival_needs_fresh_associated_surface_and_actual_pose():
    goal = ObservedObjectGoal(object_sensor(), [.4, .4, .6, .6], "table", .8, 2, 3)
    goal.selected = "front"
    pose = goal.pose("front")
    sensor = object_sensor(sequence=2, captured_at=12., odometry_m_rad=pose, depth_m=[.745] * 384)
    assert goal.arrival(sensor, [.4, .4, .6, .6], pose, 12.1)["pose_ok"]
    with pytest.raises(ValueError, match="fresh"):
        goal.arrival(sensor, [.4, .4, .6, .6], pose, 14.)
    with pytest.raises(ValueError, match="association"):
        goal.arrival(object_sensor(sequence=3, captured_at=12., odometry_m_rad=pose), [.4, .4, .6, .6], pose, 12.1)
    sensor.odometry_m_rad = [pose[0] - .06, pose[1], pose[2]]
    sensor.depth_m = [.805] * 384
    assert not goal.arrival(sensor, [.4, .4, .6, .6], sensor.odometry_m_rad, 12.1)["pose_ok"]


@pytest.mark.parametrize("fault", ["stop", "task", "view", "age"])
async def test_worker_object_selection_rejects_stale_authority(fault):
    import time
    from threading import Event
    from backend.worker import SimulationWorker
    from backend.simulation import MotionError
    worker = SimulationWorker.__new__(SimulationWorker)
    worker.home_mission = worker.renderer = worker.skill = worker.navigation = worker.object_goal = None
    worker.stop_revision, worker.task_revision = 2, 3
    sim = SimpleNamespace(run_id="run", epoch=1, cancel=Event(), odometry=np.zeros(3),
        observe=lambda **kwargs: SimpleNamespace(head_rad=[0., 0.]))
    async def call(operation):
        return operation(sim)
    worker.call = call
    sensor = object_sensor(captured_at=time.monotonic())
    if fault == "stop":
        sim.cancel.set()
    elif fault == "task":
        worker.task_revision += 1
    elif fault == "view":
        sensor.head_rad = [.1, 0.]
    else:
        sensor.captured_at -= 16.
    with pytest.raises((ValueError, MotionError)):
        await worker.object_command(sensor, "select", 2, 3, bounds=[.4, .4, .6, .6], label="table")
    assert worker.object_goal is None


def test_object_tool_contract_requires_evidence_and_explicit_completion():
    from backend.continuous_supervisor import GuideContinuous, tools
    for payload in ({"action": "select_object"}, {"action": "verify_object"}, {"action": "approach_object"},
                    {"action": "finish", "completion_mode": "object"}, {"action": "finish", "completion_mode": "saved_place"}):
        with pytest.raises(ValueError):
            GuideContinuous(**payload)
    assert GuideContinuous(action="finish").completion_mode == "parking"
    assert GuideContinuous(action="finish", completion_mode="saved_place", place_id="Home").place_id == "Home"
    assert "approach_object" in tools()[0]["parameters"]["properties"]["action"]["enum"]


@pytest.mark.parametrize("stop_phase,rendering", [(None, "tiny"), (None, "enhanced"),
    ("reply", "tiny"), ("motion", "tiny"), ("verification", "tiny")])
async def test_object_supervisor_real_camera_approach_and_confirmation(monkeypatch, stop_phase, rendering):
    import asyncio
    import base64
    import json
    from io import BytesIO
    from PIL import Image
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    replies = []
    entered = asyncio.Event()

    class ScriptedVisualModel:
        async def respond(self, profile, reasoning, goal, inputs):
            if stop_phase == "reply" and not replies:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            message = inputs[-1]
            context = next(json.loads(part["text"]) for part in message["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            data = next(part["image_url"] for part in message["content"] if part["type"] == "input_image")
            pixels = np.asarray(Image.open(BytesIO(base64.b64decode(data.split(",", 1)[1])))).astype(float)
            rows, columns = np.where((pixels[:, :, 0] > 100) & (pixels[:, :, 0] > pixels[:, :, 1] * 1.8)
                & (pixels[:, :, 0] > pixels[:, :, 2] * 1.8))
            assert len(rows) >= 8
            height, width = pixels.shape[:2]
            bounds = [float(columns.min() / width), float(rows.min() / height),
                float((columns.max() + 1) / width), float((rows.max() + 1) / height)]
            selected = context["object_goal"]
            if selected is None:
                decision = {"action": "select_object", "object_label": "red cabinet", "object_bounds": bounds}
            elif selected["selected_approach"] is None:
                assert next(option for option in selected["approaches"] if option["id"] == "front")["status"] == "reachable", selected
                decision = {"action": "approach_object", "object_goal_id": selected["goal_id"]}
            elif not selected["verified"]:
                assert context["local_execution"]["status"] == "awaiting_object_verification", context["local_execution"]
                decision = {"action": "verify_object", "object_goal_id": selected["goal_id"], "object_bounds": bounds}
            else:
                decision = {"action": "finish", "completion_mode": "object", "object_goal_id": selected["goal_id"], "entry_confirmed": True}
            replies.append(decision)
            return model_response("guide_continuous", json.dumps({**decision, "reason": "Scripted RGB detection and object contract check"}), call_id=f"object-{len(replies)}")

        async def close(self):
            pass

    challenge = get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "red_cabinet", "size": [.4, .5, .7], "position": [1.65, 0., .35], "color": [.9, .03, .03, 1.]},
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]}]})
    worker = SimulationWorker(challenge=challenge, pace=True, rendering=rendering)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: ScriptedVisualModel())
    try:
        await asyncio.wrap_future(worker.ready)
        if stop_phase == "verification":
            original = worker._verify_navigation_rest
            def stop_during_dwell(stop_revision, task_revision, validate):
                ticks = worker.sim._ticks
                def interrupt(count):
                    ticks(count)
                    worker.stop()
                    entered.set()
                monkeypatch.setattr(worker.sim, "_ticks", interrupt)
                try:
                    return original(stop_revision, task_revision, validate)
                finally:
                    monkeypatch.setattr(worker.sim, "_ticks", ticks)
            monkeypatch.setattr(worker, "_verify_navigation_rest", stop_during_dwell)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            goal="Stop beside the red cabinet and face it", max_turns=4, adaptive_navigation=True))
        if stop_phase is not None:
            async with asyncio.timeout(60.):
                while not entered.is_set():
                    if stop_phase == "motion" and worker.continuous and worker.continuous.active and worker.sim.odometry[0] > .1:
                        break
                    assert not controller.task.done(), controller.state
                    await asyncio.sleep(.02)
            await controller.halt("Scripted object Stop")
            ticks = worker.sim.ticks
            await worker.hold_stopped()
            assert worker.sim.ticks == ticks and not controller.active
            assert not worker.navigation or not worker.navigation.buffer
            assert worker.object_goal is None or not worker.object_goal.verified
            return
        await asyncio.wait_for(controller.task, 60.)
        assert controller.state["error"] is None, {"state": controller.state, "replies": replies,
            "object_results": [event["payload"] for event in controller.trace()["events"] if event["title"] == "Object approach feedback"]}
        assert controller.state["outcome"]["kind"] == "completed", controller.trace()
        assert worker.object_goal.verified
        assert worker.sim.odometry[0] > .5
        assert not worker.sim.proximity_sensors().collisions
        assert [item["action"] for item in replies] == ["select_object", "approach_object", "verify_object", "finish"]
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("fault", [None, "stop", "task", "place", "incomplete", "outside", "unverified_object"])
async def test_saved_place_completion_keeps_task_and_pose_checks(fault):
    from threading import Event
    from backend.worker import SimulationWorker
    worker = SimulationWorker.__new__(SimulationWorker)
    worker.stop_revision, worker.task_revision = 2, 3
    worker.object_goal = None
    mission = SimpleNamespace(active=False, stop_revision=2, task_revision=3, pose=np.zeros(3),
        sample=lambda **kwargs: None, require_localized=lambda: None,
        task={"kind": "navigate", "place_id": "Home", "status": "completed", "completion_verified": True, "target_m": [0., 0.]})
    worker.home_mission = mission
    sim = SimpleNamespace(cancel=Event(), proximity_sensors=lambda: SimpleNamespace(collisions=0))
    checked = []
    def rest(stop_revision, task_revision, validate):
        checked.append((stop_revision, task_revision))
        validate()
    worker._verify_navigation_rest = rest
    async def call(operation):
        return operation(sim)
    worker.call = call
    if fault == "stop":
        sim.cancel.set()
    elif fault == "task":
        worker.task_revision += 1
    elif fault == "place":
        mission.task["place_id"] = "Other"
    elif fault == "incomplete":
        mission.task["completion_verified"] = False
    elif fault == "outside":
        mission.pose = [.151, 0., 0.]
    elif fault == "unverified_object":
        worker.object_goal = SimpleNamespace(stop_revision=2, task_revision=3, verified=False)
    if fault:
        with pytest.raises(ValueError):
            await worker.verify_saved_place("Home", 2, 3)
        assert not checked
    else:
        result = await worker.verify_saved_place("Home", 2, 3)
        assert result["status"] == "saved_place_verified" and not result["mission_success_verified"]
        assert checked == [(2, 3)]


@pytest.mark.parametrize("change", [None, "pose", "head", "stop", "expired"])
async def test_object_arrival_receipt_is_current_and_json_serializable(change):
    import json
    import time
    from threading import Event
    from backend.worker import SimulationWorker
    goal = ObservedObjectGoal(object_sensor(captured_at=time.monotonic()), [.4, .4, .6, .6], "table", .8, 2, 3)
    goal.verified, goal.verified_at = True, time.monotonic()
    goal.verified_pose, goal.verified_head = np.zeros(3), np.zeros(2)
    worker = SimulationWorker.__new__(SimulationWorker)
    worker.object_goal, worker.stop_revision, worker.task_revision, worker.continuous = goal, 2, 3, None
    head = np.zeros(2)
    sim = SimpleNamespace(run_id="run", epoch=1, cancel=Event(), odometry=np.zeros(3),
        observe=lambda **kwargs: SimpleNamespace(head_rad=head))
    if change == "pose":
        sim.odometry[0] = .03
    elif change == "head":
        head[0] = .03
    elif change == "stop":
        worker.stop_revision += 1
    elif change == "expired":
        goal.verified_at -= 16.
    worker._object_options = lambda: []
    async def call(operation):
        return operation(sim)
    worker.call = call
    state = json.loads(json.dumps(await worker.object_state()))
    assert state["arrival_valid"] == (change is None)


@pytest.mark.parametrize("fault", [None, "free", "unknown_margin", "unknown_all", "occupied_inside",
    "occupied_margin", "stale", "sensor_error", "stop", "task", "stop_dwell", "task_dwell", "shield"])
def test_navigation_rest_masks_only_own_unknown_footprint(monkeypatch, fault):
    import time
    from threading import Event
    from backend.navigation import NavigationRuntime
    from backend.simulation import MotionError
    from backend.spatial import ObservedMap
    from backend.worker import SimulationWorker
    mapped = ObservedMap("run", 1)
    mapped.captured_at = time.monotonic()
    mapped.cells[:] = 0
    if fault != "free":
        mapped.cells[77:84, 77:84] = -1
    if fault == "unknown_margin":
        mapped.cells[80, 89] = -1
    elif fault == "unknown_all":
        mapped.cells[:] = -1
    elif fault == "occupied_inside":
        mapped.cells[80, 80] = 100
    elif fault == "occupied_margin":
        mapped.cells[80, 89] = 100
    elif fault == "stale":
        mapped.captured_at -= 2.
    worker = SimulationWorker.__new__(SimulationWorker)
    worker.spatial_map, worker.spatial_error = mapped, "unavailable" if fault == "sensor_error" else None
    worker.stop_revision, worker.task_revision = 2, 3
    worker._sample_spatial = lambda: None
    checks = []
    sim = SimpleNamespace(odometry=np.array([.025, .025, 0.]), cancel=Event(), held=False, ticks=0,
        hold_current=lambda: None, robot_footprint=lambda: {"radius_m": .4},
        observe=lambda **kwargs: SimpleNamespace(grippers={}),
        proximity_sensors=lambda: SimpleNamespace(collisions=[]))
    worker.sim = sim
    def tick(count):
        sim.ticks += count
        if fault == "stop_dwell":
            sim.cancel.set()
            worker.stop_revision += 1
        elif fault == "task_dwell":
            worker.task_revision += 1
    sim._ticks = tick
    def shield(runtime, current, linear, angular):
        assert current is sim and linear == angular == 0.
        checks.append("shield")
        if fault == "shield":
            raise MotionError("CLEARANCE_STOP", "Observed parking still requires the safety shield")
    monkeypatch.setattr(NavigationRuntime, "check_clearance", shield)
    if fault == "stop":
        sim.cancel.set()
    elif fault == "task":
        worker.task_revision += 1
    original_cells = mapped.cells.copy()
    if fault in {None, "free"}:
        worker._verify_navigation_rest(2, 3, lambda: checks.append("validate"))
        assert sim.ticks == 120 and checks.count("shield") == 10 and checks.count("validate") == 11
    else:
        message = "authority" if fault in {"stop", "task", "stop_dwell", "task_dwell"} else (
            "fresh observed" if fault in {"stale", "sensor_error"} else "safety shield" if fault == "shield" else "parking margin")
        with pytest.raises((ValueError, MotionError), match=message):
            worker._verify_navigation_rest(2, 3, lambda: checks.append("validate"))
        assert sim.ticks == (12 if fault in {"stop_dwell", "task_dwell"} else 0)
    np.testing.assert_array_equal(mapped.cells, original_cells)


async def test_home_parking_redirect_clearance_evidence(tmp_path, monkeypatch, record_property):
    import json
    import time
    from scipy.ndimage import distance_transform_edt
    from backend.worker import SimulationWorker
    from tests.test_mission import test_unified_visual_object_approach_and_automatic_return
    original = SimulationWorker._verify_navigation_rest
    snapshots = []

    def inspect_rest(worker, stop_revision, task_revision, validate):
        try:
            return original(worker, stop_revision, task_revision, validate)
        finally:
            mapped, sim, home = worker.spatial_map, worker.sim, worker.home_mission
            now = time.monotonic()
            radius = sim.robot_footprint()["radius_m"]
            rows, columns = np.indices(mapped.cells.shape)
            positions = (np.stack((columns, rows), axis=-1) + mapped.origin + .5) * mapped.resolution_m
            distances = np.linalg.norm(positions - sim.odometry[:2], axis=-1)
            supported = (mapped.cells == 0) | (distances <= radius)
            supported[mapped.cells == 100] = False
            local_cell = mapped.cell_index(sim.odometry[:2])
            home_cell = home.home.indices(home.pose[:2])
            home_known = home.home.cells == 0
            for obstacle in home.obstacles():
                cell = home.home.indices(obstacle)
                if home.home.inside(cell):
                    home_known[cell[1], cell[0]] = False
            def clearance(known, resolution, cell):
                return float(distance_transform_edt(np.pad(known, 1))[1:-1, 1:-1][cell] * resolution)
            snapshots.append({"evidence": "scripted_test", "pose": sim.odometry.tolist(),
                "footprint": sim.robot_footprint(), "required_margin_m": radius + .08,
                "rolling_age_s": now - mapped.captured_at, "rolling_sequence": mapped.sequence,
                "rolling_clearance_m": clearance(mapped.cells == 0, mapped.resolution_m, local_cell),
                "footprint_supported_clearance_m": clearance(supported, mapped.resolution_m, local_cell),
                "home_clearance_m": clearance(home_known, home.home.resolution_m, (home_cell[1], home_cell[0])),
                "home_pose": home.pose, "home_sample_age_s": now - home.sampled_at,
                "home_validation_age_s": now - home.validated_at, "home_geometry_updated_at": home.home.updated_at,
                "home_localization": home.localization, "proximity": sim.proximity_sensors().model_dump(),
                "nearby_cells": [{"position_m": position.tolist(), "value": int(value)}
                    for position, value in zip(positions[distances <= radius + .15], mapped.cells[distances <= radius + .15])]})

    monkeypatch.setattr(SimulationWorker, "_verify_navigation_rest", inspect_rest)
    try:
        await test_unified_visual_object_approach_and_automatic_return(tmp_path, "redirect", record_property)
    finally:
        record_property("home_parking_clearance", json.dumps(snapshots))
    assert snapshots