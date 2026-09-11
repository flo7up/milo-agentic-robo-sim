import asyncio
from io import BytesIO
import json
import math

import numpy as np
from PIL import Image
import pybullet as bullet
import pytest

from backend.challenges import ChallengeProgress, PRESETS, get_challenge
from backend.simulation import BulletSimulation
from backend.agent import AgentStart, FoundryConfig
from backend.realtime import RealtimeConfig, VoiceController, VoiceStart
from backend.worker import SimulationWorker
from tests.test_agent import ScriptedModel, controller_for, model_response
from tests.test_realtime import RealtimeModel, VoiceBrowser
from tests.test_simulation import command


def test_presets_have_distinct_goals_and_private_geometry():
    assert set(PRESETS) == {"park", "tidy", "sort", "recharge", "apartment", "kitchen_bathroom", "clinic_delivery", "warehouse", "inspection", "workshop"}
    for preset in PRESETS.values():
        assert preset.goal and preset.objectives
        assert "objects" not in preset.public()
        assert "floor_size_m" not in preset.public()
        assert preset.category in {"Navigation", "Perception", "Manipulation"}
        assert all(isinstance(label, str) for label in preset.public()["objectives"])
        assert any(item.get("marker") for item in preset.scene())
    assert get_challenge("bench") is None


def test_kitchen_bathroom_starts_in_kitchen_without_room_labels_or_target_markers():
    preset = get_challenge("kitchen_bathroom")
    sim = BulletSimulation(challenge=preset, width=320, height=240)
    try:
        observation = sim.observe()
        position = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0]
        assert position[:2] == pytest.approx([1.1, -1.5], abs=.01)
        assert observation.odometry_m_rad == [0, 0, 0]
        assert not observation.proximity.collisions
        objects = {item["name"]: item for item in sim.objects}
        assert {"fridge_body", "stove_hob", "kitchen_sink_basin", "toilet_bowl", "toilet_cistern", "bathroom_basin", "tub_interior"} <= objects.keys()
        assert not any(name.startswith("zone_") for name in objects)
        assert "initial_xy" not in preset.public()
        assert "current_room" not in observation.model_dump()
        assert all(name not in observation.model_dump_json() for name in objects)
        assert sim.challenge_status()["status"] == "in_progress"
        assert bullet.rayTest([1.1, -1.5, .55], [1.1, 1.5, .55], physicsClientId=sim.client)[0][0] == objects["room_divider"]["id"]
        for doorway in (-1.5, 1.5):
            assert bullet.rayTest([.05, doorway, .55], [.8, doorway, .55], physicsClientId=sim.client)[0][0] == -1
    finally:
        sim.close()


def test_apartment_has_open_doors_and_target_hidden_from_entry():
    preset = get_challenge("apartment")
    sim = BulletSimulation(challenge=preset, width=160, height=120)
    try:
        objects = {item["name"]: item for item in sim.objects}
        assert "search_target" not in preset.public()
        assert not any(item["name"].startswith("zone_") for item in sim.objects)
        assert bullet.rayTest([.2, 0, .55], objects["yellow_target"]["position"], physicsClientId=sim.client)[0][0] == objects["hall_partition"]["id"]
        for doorway in (-1.5, 1.5):
            assert bullet.rayTest([0, doorway, .5], [1.2, doorway, .5], physicsClientId=sim.client)[0][0] == -1
        assert sim.challenge_status()["status"] == "in_progress"
        target = objects["yellow_target"]
        height = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0][2]
        bullet.resetBasePositionAndOrientation(sim.robot, [2.75, 1.7, height], bullet.getQuaternionFromEuler([0, 0, math.pi]), physicsClientId=sim.client)
        assert not sim._target_in_camera(target["id"], np.array(target["position"]))
        bullet.resetBasePositionAndOrientation(sim.robot, [1.2, 1.7, height], [0, 0, 0, 1], physicsClientId=sim.client)
        assert sim._target_in_camera(target["id"], np.array(target["position"]))
    finally:
        sim.close()


def robot_measurement(center, speed=0):
    return {"bounds": [[center[0] - .17, center[1] - .22, 0], [center[0] + .17, center[1] + .22, .25]],
            "grounded": True, "speed": speed, "angular_speed": 0}


def test_apartment_inspection_requires_visible_nearby_target_and_stationary_dwell():
    progress = ChallengeProgress(get_challenge("apartment"))
    measurement = {**robot_measurement([1.1, 1.5]), "target_nearby": True, "target_visible": True, "head_stationary": True}
    elapsed = 0
    for invalid in ({"target_nearby": False}, {"target_visible": False}, {"speed": .1},
                    {"angular_speed": .2}, {"head_stationary": False}, {"grounded": False}):
        elapsed += 2
        assert progress.update({"robot": {**measurement, **invalid}}, set(), elapsed)["status"] == "in_progress"
        assert progress.search_dwell_s == 0
    assert progress.update({"robot": measurement}, set(), elapsed + .5)["status"] == "in_progress"
    assert progress.update({"robot": measurement}, set(), elapsed + .5)["status"] == "in_progress"
    assert progress.update({"robot": measurement}, set(), elapsed + 1)["status"] == "completed"
    assert progress.update({"robot": {**measurement, "target_visible": False}}, set(), elapsed + 1.1)["status"] == "in_progress"
    assert progress.search_dwell_s == 0


def test_recharge_requires_departure_scan_low_battery_return_and_charging():
    progress = ChallengeProgress(get_challenge("recharge"))
    dock = {"robot": robot_measurement([0, 0])}
    survey = {"robot": robot_measurement([1.45, 0])}
    assert progress.update(dock, set(), 5, 0)["completed_objectives"] == 0
    assert progress.battery.charge_pct == 100
    assert progress.update(survey, set(), 5.5, 3)["completed_objectives"] == 0
    assert progress.update(survey, set(), 6, 3)["completed_objectives"] == 1
    assert progress.battery.low and not progress.battery.charging
    assert progress.update(dock, set(), 6.1, 6)["completed_objectives"] == 2
    assert progress.battery.charging and progress.battery.charge_pct < 90
    assert progress.update(dock, set(), 10.1, 6)["status"] == "completed"
    assert progress.battery.charge_pct >= 90
    assert set(progress.battery.model_dump()) == {"charge_pct", "low", "charging"}


def test_recharge_does_not_charge_while_moving_outside_or_partly_on_dock():
    progress = ChallengeProgress(get_challenge("recharge"))
    for time, center, speed in [(1, [1.45, 0], 0), (2, [0, 0], .1), (3, [.35, 0], 0)]:
        progress.update({"robot": robot_measurement(center, speed)}, set(), time, time)
        assert not progress.battery.charging
    progress.update({"robot": robot_measurement([1.45, 0])}, set(), 4, 4)
    assert progress.status["status"] != "completed"
    progress.update({"robot": robot_measurement([.5, 1])}, set(), 1000, 100)
    assert progress.battery.charge_pct == 0 and progress.status["status"] == "failed"


def drive_to(sim, target):
    for _ in range(20):
        position, rotation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        delta = [target[axis] - position[axis] for axis in (0, 1)]
        distance = math.hypot(*delta)
        if distance < .06:
            return
        heading = bullet.getEulerFromQuaternion(rotation)[2]
        angle = math.atan2(math.sin(math.atan2(delta[1], delta[0]) - heading), math.cos(math.atan2(delta[1], delta[0]) - heading))
        if abs(angle) > .08:
            duration = min(2, max(.5, abs(angle) / .8 + .3))
            result = command(sim, "drive_base", linear_mps=0,
                             angular_radps=max(-.8, min(.8, angle / (duration - .3))), duration_s=duration)
        else:
            result = command(sim, "drive_base", linear_mps=.3, angular_radps=0, duration_s=min(2, max(.5, distance / .3 + .3)))
        assert result.status == "ok", (result.error, sim.battery_sensor())
    final_position = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0]
    assert math.hypot(target[0] - final_position[0], target[1] - final_position[1]) < .06, f"Robot did not reach the scripted feasibility waypoint {target}"


def test_kitchen_to_bathroom_route_is_physically_feasible():
    sim = BulletSimulation(challenge=get_challenge("kitchen_bathroom"), width=320, height=240)
    try:
        kitchen = sim.observe()
        kitchen_image = sim.frame(kitchen.frame_ref)
        assert command(sim, "wait", duration_s=1).status == "ok"
        assert sim.challenge_status()["status"] == "in_progress"
        for point in ([0, -1.5], [0, 1.5]):
            drive_to(sim, point)
            assert sim.challenge_status()["status"] == "in_progress"
        drive_to(sim, [1.2, 1.4])
        result = command(sim, "wait", duration_s=.5)
        assert result.status == "ok"
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        assert sim.path_length > 4
        assert sim.frame(result.observation.frame_ref) != kitchen_image
        objects = {item["name"]: item for item in sim.objects}
        assert sim._target_in_camera(objects["toilet_lid"]["id"], np.array(objects["toilet_lid"]["position"]))
        assert not any(name in result.observation.model_dump_json() for name in ("current_room", "bathroom_floor", "toilet_cistern", "initial_xy"))
        drive_to(sim, [0, 1.4])
        assert sim.challenge_status()["status"] == "in_progress"
    finally:
        sim.close()


def test_clinic_delivery_has_detailed_private_geometry_and_a_blocked_direct_corridor():
    preset = get_challenge("clinic_delivery")
    sim = BulletSimulation(challenge=preset, width=320, height=240)
    try:
        observation = sim.observe()
        objects = {item["name"]: item for item in sim.objects}
        assert len(objects) >= 45
        assert {"reception_desk", "maintenance_barrier", "sanitizer_dispenser", "diagnostics_scanner",
                "red_isolation_sign", "treatment_bed", "delivery_bay"} <= objects.keys()
        assert sim.challenge_status()["completed_objectives"] == 0
        assert not any(name.startswith("zone_") for name in objects)
        assert bullet.rayTest([-.5, 0, .45], [1.75, 0, .45], physicsClientId=sim.client)[0][0] == objects["maintenance_barrier"]["id"]
        payload = observation.model_dump_json()
        assert all(value not in payload for value in ("maintenance_barrier", "delivery_bay", "ordered_objectives", "initial_xy"))
        assert "objects" not in preset.public()
    finally:
        sim.close()


def test_clinic_delivery_requires_ordered_route_gates():
    progress = ChallengeProgress(get_challenge("clinic_delivery"))
    assert progress.update({"robot": robot_measurement([1.2, 2.0])}, set())["completed_objectives"] == 0
    airborne = {**robot_measurement([-1.15, 1.95]), "grounded": False}
    assert progress.update({"robot": airborne}, set())["completed_objectives"] == 0
    assert progress.update({"robot": robot_measurement([-1.15, 1.95])}, set())["completed_objectives"] == 1
    assert progress.update({"robot": robot_measurement([1.2, 2.0])}, set())["completed_objectives"] == 1
    assert progress.update({"robot": robot_measurement([.9, 2.0])}, set())["completed_objectives"] == 2
    moving = robot_measurement([1.2, 2.0], speed=.1)
    assert progress.update({"robot": moving}, set())["status"] == "in_progress"
    assert progress.update({"robot": robot_measurement([1.2, 2.0])}, set())["status"] == "completed"
    assert progress.update({"robot": robot_measurement([1.2, 2.0], speed=.1)}, set())["status"] == "in_progress"
    assert progress.update({"robot": robot_measurement([0, 0])}, set())["completed_objectives"] == 2


def test_ordered_destination_dwell_uses_simulation_time_and_resets_on_departure():
    challenge = get_challenge("clinic_delivery")
    challenge.objectives = [challenge.objectives[-1].model_copy(update={"dwell_s": 1})]
    progress = ChallengeProgress(challenge)
    parked = {"robot": robot_measurement([1.2, 2.0])}
    assert progress.update(parked, set(), .5)["status"] == "in_progress"
    assert progress.update(parked, set(), .5)["status"] == "in_progress"
    assert progress.update({"robot": robot_measurement([0, 0])}, set(), .6)["status"] == "in_progress"
    assert progress.update(parked, set(), 1.1)["status"] == "in_progress"
    assert progress.update(parked, set(), 1.6)["status"] == "completed"


def test_clinic_delivery_detour_is_physically_feasible():
    sim = BulletSimulation(challenge=get_challenge("clinic_delivery"), width=160, height=120)
    try:
        assert not sim.proximity_sensors().collisions
        route = ([-2.15, -.30], [-1.35, -.30], [-1.35, .35], [-1.35, 1.20], [-1.02, 1.20],
                 [-.98, 2.05], [-.20, 2.05], [.78, 2.05], [1.20, 2.00])
        expected_progress = (0, 0, 0, 0, 0, 1, 1, 2, 3)
        for waypoint, completed in zip(route, expected_progress):
            drive_to(sim, waypoint)
            assert sim.challenge_status()["completed_objectives"] == completed
        assert sim.challenge_status()["status"] == "completed"
        assert sim.path_length > 6
        assert not any(name in sim.observe().model_dump_json() for name in ("delivery_bay", "diagnostics_scanner", "maintenance_barrier"))
    finally:
        sim.close()


def test_warehouse_circuit_is_feasible_with_real_wheels_and_requires_final_dwell():
    sim = BulletSimulation(challenge=get_challenge("warehouse"), width=160, height=120)
    try:
        assert not sim.proximity_sensors().collisions
        assert next(item for item in sim.objects if item["name"] == "floor")["size"][:2] == [10, 8]
        for waypoint in ([-3.6, 2.8], [0, 2.8], [0, -2.8], [3.5, -2.8], [3.5, 2.7]):
            drive_to(sim, waypoint)
        assert command(sim, "wait", duration_s=1.2).status == "ok"
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        assert sim.path_length > 20
        drive_to(sim, [3.5, 1.7])
        assert sim.challenge_status()["completed_objectives"] == 2
    finally:
        sim.close()


def test_inspection_gallery_route_rejects_decoy_and_reaches_occluded_target():
    sim = BulletSimulation(challenge=get_challenge("inspection"), width=160, height=120)
    try:
        objects = {item["name"]: item for item in sim.objects}
        assert not sim.proximity_sensors().collisions
        assert bullet.rayTest([-2.5, -2.7, .55], [2.8, -2.3, .43], physicsClientId=sim.client)[0][0] == objects["service_partition"]["id"]
        for point in ([-2.7, 2.6], [2.0, 2.6]):
            drive_to(sim, point)
        assert command(sim, "wait", duration_s=1.2).status == "ok"
        assert sim.challenge_status()["status"] == "in_progress"
        for point in ([.55, 2.6], [.55, -2.3], [2.0, -2.3]):
            drive_to(sim, point)
        result = command(sim, "wait", duration_s=1.2)
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        assert sim.path_length > 15
        pixels = np.array(Image.open(BytesIO(sim.frame(result.observation.frame_ref))))
        assert np.count_nonzero((pixels[:, :, 0] > 140) & (pixels[:, :, 1] > 110) & (pixels[:, :, 2] < 80)) > 25
        assert "inspection_target" not in result.observation.model_dump_json()
    finally:
        sim.close()


def test_workshop_both_cubes_reachable_without_displacing_static_fixtures():
    sim = BulletSimulation(challenge=get_challenge("workshop"), width=160, height=120)
    try:
        assert not sim.proximity_sensors().collisions
        fixture_poses = {item["id"]: bullet.getBasePositionAndOrientation(item["id"], physicsClientId=sim.client)
                         for item in sim.objects if item["name"].startswith("fixture_")}
        place_cube(sim, "left", 1)
        place_cube(sim, "right", -1)
        assert sim.challenge_status()["status"] == "completed"
        assert all(bullet.getBasePositionAndOrientation(body, physicsClientId=sim.client) == pose for body, pose in fixture_poses.items())
    finally:
        sim.close()


def test_bathroom_arrival_rejects_wrong_room_doorway_motion_and_ungrounded_pose():
    progress = ChallengeProgress(get_challenge("kitchen_bathroom"))
    for center in ([1.1, -1.5], [0, 1.5], [.75, 1.4]):
        assert progress.update({"robot": robot_measurement(center)}, set())["status"] == "in_progress"
        assert progress.status["progress"][0]["detail"] == "Move fully inside the bathroom"
    assert progress.update({"robot": robot_measurement([1.2, 1.4], speed=.1)}, set())["status"] == "in_progress"
    floating = {**robot_measurement([1.2, 1.4]), "grounded": False}
    assert progress.update({"robot": floating}, set())["status"] == "in_progress"
    assert progress.update({"robot": robot_measurement([1.2, 1.4])}, set())["status"] == "completed"


def test_apartment_search_route_is_physically_feasible_and_rejects_decoy():
    sim = BulletSimulation(challenge=get_challenge("apartment"), width=160, height=120)
    try:
        for point in ([0, -1.5], [1.1, -1.5]):
            drive_to(sim, point)
        command(sim, "wait", duration_s=1.2)
        assert sim.challenge_status()["status"] == "in_progress"
        for point in ([0, -1.5], [0, 1.5], [1.2, 1.5]):
            drive_to(sim, point)
        result = command(sim, "wait", duration_s=1.2)
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        pixels = np.array(Image.open(BytesIO(sim.frame(result.observation.frame_ref))))
        yellow = (pixels[:, :, 0] > 140) & (pixels[:, :, 1] > 110) & (pixels[:, :, 2] < 80)
        assert np.count_nonzero(yellow) > 25
        assert not any(name in json.dumps(result.observation.model_dump()) for name in ("yellow_target", "target_visible", "target_nearby", "search_target"))
        command(sim, "set_head", yaw_rad=-1.5, pitch_rad=.15, duration_s=1)
        assert sim.challenge_status()["status"] == "in_progress"
        assert sim.challenge_progress.search_dwell_s == 0
    finally:
        sim.close()


def test_recharge_round_trip_is_physically_feasible_and_needs_waiting():
    sim = BulletSimulation(challenge=get_challenge("recharge"), width=160, height=120)
    try:
        initial = sim.observe()
        assert initial.battery.charge_pct == 100
        assert initial.head_rad[1] > .85
        assert sim.challenge_status()["completed_objectives"] == 0
        screen = next(item["id"] for item in sim.objects if item["name"] == "screen")
        assert bullet.rayTest([1.45, 0, .4], [0, 0, .4], physicsClientId=sim.client)[0][0] == screen
        for point in ([0, .85], [1.45, .85], [1.45, 0]):
            drive_to(sim, point)
        command(sim, "wait", duration_s=1.2)
        assert sim.challenge_progress.survey_complete
        assert sim.battery_sensor().low and not sim.battery_sensor().charging
        assert sim.challenge_status()["completed_objectives"] == 1
        for point in ([1.45, .85], [0, .85], [0, 0]):
            drive_to(sim, point)
        command(sim, "wait", duration_s=.5)
        assert sim.battery_sensor().charging and sim.battery_sensor().charge_pct < 90
        before = sim.battery_sensor().charge_pct
        command(sim, "observe")
        assert sim.battery_sensor().charge_pct == before
        for _ in range(3):
            command(sim, "wait", duration_s=2)
        assert sim.challenge_status()["status"] == "completed"
        assert sim.battery_sensor().charge_pct >= 90
        assert set(sim.observe().battery.model_dump()) == {"charge_pct", "low", "charging"}
    finally:
        sim.close()


def test_depleted_battery_blocks_motion_but_reset_restores_it():
    sim = BulletSimulation(challenge=get_challenge("recharge"), width=160, height=120)
    try:
        drive_to(sim, [0, .85])
        sim.challenge_progress.battery.charge_pct = .01
        result = command(sim, "drive_base", linear_mps=.3, angular_radps=0, duration_s=2)
        assert result.error == "BATTERY_EMPTY" and result.actual_duration_s < 2
        ticks = sim.ticks
        assert command(sim, "drive_base", linear_mps=.1, angular_radps=0, duration_s=1).error == "BATTERY_EMPTY"
        assert sim.ticks == ticks
        assert sim.challenge_status()["status"] == "failed"
    finally:
        sim.close()
    replacement = BulletSimulation(challenge=get_challenge("recharge"), width=160, height=120)
    try:
        assert replacement.battery_sensor().charge_pct == 100
        assert replacement.challenge_status()["completed_objectives"] == 0
    finally:
        replacement.close()


def test_wait_is_bounded_deduplicated_and_respects_stop():
    from backend.contracts import Command
    sim = BulletSimulation(width=160, height=120)
    try:
        command(sim, "observe")
        envelope = Command(run_id=sim.run_id, episode_epoch=sim.epoch, action_id="wait-once", observation_seq=sim.seq,
                           tool="wait", arguments={"duration_s": .5})
        result = sim.execute(envelope)
        assert result.status == "ok" and result.actual_duration_s == .5
        ticks = sim.ticks
        assert sim.execute(envelope) is result and sim.ticks == ticks
        assert command(sim, "wait", duration_s=3).error == "INVALID_ARGUMENT"
        assert sim.ticks == ticks
        command(sim, "stop")
        assert command(sim, "wait", duration_s=1).status == "cancelled"
        assert sim.ticks == ticks
    finally:
        sim.close()


def place_cube(sim, side, sign):
    def move(position):
        result = command(sim, "move_end_effector", arm=side, position_m=position,
                         orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
        assert result.status == "ok", (result.error, result.message)

    move([.36, sign * .25, -.12])
    command(sim, "set_gripper", arm=side, opening_m=0, max_force_n=20)
    assert side in sim.held
    move([.36, sign * .25, .05])
    move([.32, sign * .50, .05])
    move([.32, sign * .50, -.08])
    result = command(sim, "set_gripper", arm=side, opening_m=.11, max_force_n=20)
    assert result.status == "ok" and side not in sim.held
    return result


@pytest.mark.parametrize("identifier,side,sign", [("tidy", "left", 1), ("sort", "left", 1), ("sort", "right", -1)])
def test_preset_objects_can_be_picked_up_and_moved_to_their_zones(identifier, side, sign):
    preset = get_challenge(identifier)
    sim = BulletSimulation(challenge=preset, width=160, height=120)
    try:
        assert sim.challenge_status()["status"] == "in_progress"
        result = place_cube(sim, side, sign)
        target = next(item for item in sim.evaluator_state().bodies if item["name"] == ("red_cube" if sign == 1 else "blue_cube"))
        assert abs(target["position"][0] - .32) < .05
        assert abs(target["position"][1] - sign * .50) < .05
        assert target["position"][2] < .05
        status = sim.challenge_status()
        assert status["progress"][0 if sign == 1 else 1]["complete"], status["progress"]
        assert status["completed_objectives"] == 1
        assert status["status"] == ("completed" if identifier == "tidy" else "in_progress")
        assert "challenge" not in result.observation.model_dump()
    finally:
        sim.close()


def test_color_sort_completes_both_cubes_in_one_physics_episode():
    sim = BulletSimulation(challenge=get_challenge("sort"), width=160, height=120)
    try:
        place_cube(sim, "left", 1)
        assert sim.challenge_status()["completed_objectives"] == 1
        place_cube(sim, "right", -1)
        assert sim.challenge_status()["completed_objectives"] == 2
        assert sim.challenge_status()["status"] == "completed"
    finally:
        sim.close()


def test_parking_preset_is_reachable_with_real_wheel_commands():
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    try:
        assert sim.challenge_status()["status"] == "in_progress"
        for _ in range(2):
            result = command(sim, "drive_base", linear_mps=.3, angular_radps=0, duration_s=2)
            assert result.status == "ok"
        command(sim, "drive_base", linear_mps=0, angular_radps=0, duration_s=.5)
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()["progress"]
        assert sim.evaluator_state().robot_position[0] > .9
    finally:
        sim.close()


def test_placement_requires_lift_release_full_containment_and_settling():
    progress = ChallengeProgress(get_challenge("tidy"))
    measurement = {"bounds": [[.29, .47, 0], [.35, .53, .06]], "speed": 0, "angular_speed": 0, "grounded": True}
    def update(held=()):
        return progress.update({"red_cube": measurement}, set(held))["status"]
    assert update() == "in_progress"
    measurement["bounds"] = [[.29, .47, .1], [.35, .53, .16]]
    measurement["grounded"] = False
    assert update(["red_cube"]) == "in_progress"
    assert update() == "in_progress"
    measurement["bounds"] = [[.29, .47, 0], [.35, .53, .06]]
    measurement["grounded"] = True
    measurement["speed"] = .2
    assert update() == "in_progress"
    measurement["speed"] = 0
    assert update() == "completed"
    measurement["bounds"] = [[.39, .47, 0], [.45, .53, .06]]
    assert update() == "in_progress"


def test_sort_requires_both_cubes_in_the_correct_zones():
    progress = ChallengeProgress(get_challenge("sort"))
    def measurement(sign, height=0):
        return {"bounds": [[.29, sign * .50 - .03, height], [.35, sign * .50 + .03, height + .06]],
            "speed": 0, "angular_speed": 0, "grounded": height == 0}
    progress.update({"red_cube": measurement(1, .1), "blue_cube": measurement(-1, .1)}, {"red_cube", "blue_cube"})
    assert progress.update({"red_cube": measurement(1), "blue_cube": measurement(-1)}, set())["status"] == "completed"
    assert progress.update({"red_cube": measurement(-1), "blue_cube": measurement(1)}, set())["status"] == "in_progress"


@pytest.mark.parametrize("identifier", ["sort", "kitchen_bathroom", "warehouse", "inspection", "workshop"])
async def test_challenge_goal_reaches_voice_without_evaluator_state(identifier):
    preset = get_challenge(identifier)
    worker = SimulationWorker(challenge=preset, pace=False)
    model = RealtimeModel([])
    browser = VoiceBrowser()
    voice = VoiceController(FoundryConfig(), RealtimeConfig(endpoint="https://test.openai.azure.com", deployment="test-voice"), lambda config: model)
    try:
        await asyncio.wrap_future(worker.ready)
        voice.start_voice(worker, VoiceStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch), browser)
        await browser.wait("ready")
        session = model.connection.sent[0]["session"]
        assert preset.goal in session["instructions"]
        assert voice.state["goal"] == preset.goal
        assert "red_cube" not in json.dumps(session) and "completed_objectives" not in json.dumps(session)
        assert all(value not in json.dumps(session) for value in ("initial_xy", "toilet_cistern", "visible_zone", "bathroom_floor"))
        assert worker.latest["challenge"]["id"] == identifier
    finally:
        await voice.halt()
        await worker.close()


@pytest.mark.parametrize("identifier", ["tidy", "kitchen_bathroom", "clinic_delivery", "warehouse", "inspection", "workshop"])
async def test_challenge_state_never_leaks_into_text_model_feedback(identifier):
    preset = get_challenge(identifier)
    worker = SimulationWorker(challenge=preset, pace=False)
    model = ScriptedModel([model_response("stop", "{}")])
    agent = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        agent.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch, goal=preset.goal))
        await agent.task
        payload = json.dumps(model.inputs)
        assert "red_cube" not in payload and "progress" not in payload and "completed_objectives" not in payload
        assert all(value not in payload for value in ("current_room", "initial_xy", "toilet_cistern", "bathroom_floor", "visible_zone"))
        assert all(value not in payload for value in ("floor_size_m", "ordered_objectives", "inspection_target", "stock_0_0_0", "workstation_divider"))
    finally:
        await agent.halt()
        await worker.close()


async def test_charging_memory_keeps_initial_observation_without_target_coordinates():
    preset = get_challenge("recharge")
    worker = SimulationWorker(challenge=preset, pace=False)
    model = ScriptedModel([model_response("set_head", '{"yaw_rad":0.5,"pitch_rad":0.2,"duration_s":0.5}', "head-memory")] +
                         [model_response("observe", "{}", f"memory-{index}") for index in range(6)] +
                          [model_response("stop", "{}", "memory-stop")])
    agent = controller_for(model)
    try:
        await asyncio.wrap_future(worker.ready)
        agent.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                                     goal=preset.goal, feedback_interval_s=.25, max_turns=8, images_per_request=2))
        await agent.task
        first = model.inputs[0][0]
        last = model.inputs[-1]
        assert last[-1]["role"] == "user"
        assert json.loads(last[-1]["content"][0]["text"])["seq"] > json.loads(first["content"][0]["text"])["seq"]
        historical = [part for part in last[-1]["content"] if part["type"] == "input_text" and "historical_camera_observation" in part["text"]]
        assert json.loads(historical[0]["text"])["historical_camera_observation"] == json.loads(first["content"][0]["text"])
        images = [part for item in last for part in item.get("content", []) if isinstance(part, dict) and part["type"] == "input_image"]
        assert len(images) == 2 and images[-1] == first["content"][1]
        assert set(json.loads(first["content"][0]["text"])["battery"]) == {"charge_pct", "low", "charging"}
        serialized = json.dumps(last)
        assert all(secret not in serialized for secret in ("charger_beacon", "survey_beacon", "completed_objectives", "center", "robot_position"))
        feedback = [entry for entry in agent.trace()["events"] if entry["kind"] == "feedback"]
        assert feedback[-1]["payload"]["memory_frame_seq"] == json.loads(first["content"][0]["text"])["seq"]
    finally:
        await agent.halt()
        await worker.close()