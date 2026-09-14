import asyncio
from io import BytesIO
import json
import math
import sys

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


@pytest.fixture(scope="session")
def recorded_physics_tests():
    from scripts.record_scenario_tests import PhysicsTestBatch
    batch = PhysicsTestBatch(label="scripted-scenario-replays-v1")
    try:
        yield batch
    finally:
        batch.finish()


def test_shared_apartment_uses_one_scene_and_rejects_unsupported_options():
    from backend.challenges import ChallengeLoad, shared_apartment
    from backend.materials import scene_material
    standalone = get_challenge("furniture_circuit").model_dump()
    variants = [shared_apartment(name) for name in ("furniture_circuit", "apartment", "flat_kitchen", "recharge")]
    assert all(challenge.scene() == variants[0].scene() for challenge in variants)
    materials = [[scene_material(item, challenge.id == "apartment") for item in challenge.scene()] for challenge in variants]
    assert all(values == materials[0] for values in materials)
    assert get_challenge("furniture_circuit").model_dump() == standalone
    assert variants[0].orbit.center_m == [-4.4, 1.8]
    assert variants[0].orbit.minimum_radius_m == standalone["orbit"]["minimum_radius_m"]
    assert variants[1].search_target == "yellow_target"
    for challenge in variants:
        assert challenge.public()["environment"] == "shared_apartment_v1"
        assert not {"objects", "initial_xy", "center_m", "search_target", "floor_size_m"} & challenge.public().keys()
        assert not any(item["name"].startswith("zone_") for item in challenge.scene())
    with pytest.raises(ValueError, match="supports"):
        ChallengeLoad(challenge_id="park", environment="shared_apartment_v1")
    with pytest.raises(ValueError, match="table"):
        ChallengeLoad(challenge_id="furniture_circuit", environment="shared_apartment_v1", orbit_target="sofa")


@pytest.mark.parametrize("identifier", ["furniture_circuit", "apartment", "flat_kitchen", "recharge"])
def test_shared_apartment_starts_clear_and_has_connected_doorways(identifier):
    from backend.challenges import shared_apartment
    challenge = shared_apartment(identifier)
    sim = BulletSimulation(challenge=challenge, width=160, height=120)
    try:
        observation = sim.observe()
        assert not observation.proximity.collisions
        assert observation.odometry_m_rad == [0, 0, 0]
        assert sim.challenge_status()["status"] == "in_progress"
        assert "yellow_target" not in observation.model_dump_json()
        objects = {item["name"]: item for item in sim.objects}
        assert {"fridge_body", "stove_hob", "toilet_bowl", "bedroom_mattress", "circuit_table_top", "yellow_target", "red_decoy", "charger_pad"} <= objects.keys()
        for start, end in (([-2, 0, .55], [0, 0, .55]), *[([.2, doorway, .55], [2, doorway, .55]) for doorway in (-3.8, 0, 3.8)]):
            assert bullet.rayTest(start, end, physicsClientId=sim.client)[0][0] == -1
        target = objects["yellow_target"]
        assert bullet.rayTest([-6.8, 1.8, .6], target["position"], physicsClientId=sim.client)[0][0] != target["id"]
    finally:
        sim.close()


def test_shared_apartment_api_selection_and_reset_are_isolated():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        catalogue = client.get("/api/challenges?environment=shared_apartment_v1").json()
        assert {item["id"] for item in catalogue} == {"furniture_circuit", "apartment", "flat_kitchen", "recharge"}
        assert all(item["environment"] == "shared_apartment_v1" and "objects" not in item for item in catalogue)
        for identifier in ("furniture_circuit", "apartment", "flat_kitchen", "recharge"):
            response = client.post("/api/challenges/load", json={"challenge_id": identifier, "environment": "shared_apartment_v1"})
            assert response.status_code == 200
            state = response.json()
            reset = client.post("/api/reset").json()
            assert reset["challenge"]["environment"] == "shared_apartment_v1"
            assert reset["challenge"]["id"] == identifier
            assert reset["run_id"] != state["run_id"]
            assert reset["snapshot"]["simulated_time_s"] == 0
            assert not reset["agent"]["active"]
        rejected = client.post("/api/challenges/load", json={"challenge_id": "park", "environment": "shared_apartment_v1"})
        assert rejected.status_code == 422
        assert client.get("/api/state").json()["run_id"] == reset["run_id"]
        state = client.post("/api/challenges/load", json={"challenge_id": "furniture_circuit"}).json()
        assert state["challenge"]["environment"] == "standalone"
        assert state["challenge"]["orbit"] == {"target": "table", "direction": "clockwise"}


def test_presets_have_distinct_goals_and_private_geometry():
    assert set(PRESETS) == {"park", "tidy", "sort", "recharge", "apartment", "kitchen_bathroom", "clinic_delivery", "warehouse", "inspection", "workshop", "local_park", "pedestrian_crossing", "flat_kitchen", "furniture_circuit"}
    for preset in PRESETS.values():
        assert preset.goal and preset.objectives
        assert "objects" not in preset.public()
        assert "floor_size_m" not in preset.public()
        assert preset.category in {"Navigation", "Perception", "Manipulation"}
        assert all(isinstance(label, str) for label in preset.public()["objectives"])
        assert any(item.get("marker") for item in preset.scene())
    assert get_challenge("bench") is None


@pytest.mark.parametrize("direction", ["clockwise", "counterclockwise"])
def test_furniture_circuit_requires_correct_translation_direction_and_rest(direction):
    from backend.challenges import furniture_circuit
    challenge = furniture_circuit("table", direction)
    progress = ChallengeProgress(challenge)
    center = challenge.orbit.center_m
    now = 0.
    def sample(angle, speed=.3, contact=False):
        nonlocal now
        now += .05
        position = [center[0] + 1.7 * math.cos(angle), center[1] + 1.7 * math.sin(angle)]
        return progress.update({"robot": {**robot_measurement(position, speed=speed), "position_xy": position, "contact": contact}}, set(), now)
    sign = -1 if direction == "clockwise" else 1
    for angle in np.linspace(0., -sign * 2 * math.pi, 720):
        assert sample(angle)["status"] == "in_progress"
    assert progress.orbit_angle < .01
    for angle in np.linspace(0., sign * 2 * math.pi, 720):
        assert sample(angle)["status"] == "in_progress"
    for _ in range(12):
        status = sample(sign * 2 * math.pi, speed=0.)
    assert status["status"] == "completed"
    assert "center_m" not in json.dumps(challenge.public())
    assert sample(sign * 2 * math.pi, speed=0., contact=True)["status"] == "failed"


def test_furniture_circuit_ignores_spin_wrong_object_and_teleports():
    from backend.challenges import furniture_circuit
    challenge = furniture_circuit()
    progress = ChallengeProgress(challenge)
    for index in range(30):
        position = [2.6 + math.cos(index), -2.5 + math.sin(index)]
        progress.update({"robot": {**robot_measurement(position), "position_xy": position, "contact": False}}, set(), index * .05)
    assert progress.orbit_angle == 0.
    for index in range(30):
        position = [-2.4 + 1.7 * math.cos(index), 1.8 + 1.7 * math.sin(index)]
        progress.update({"robot": {**robot_measurement(position), "position_xy": position, "contact": False}}, set(), 2 + index * .05)
    assert progress.status["status"] == "in_progress" and progress.orbit_angle < .1
    progress = ChallengeProgress(challenge)
    for angle in np.linspace(0., -2 * math.pi, 720):
        position = [-2.4 + 1.7 * math.cos(angle), 1.8 + 1.7 * math.sin(angle)]
        progress.update({"robot": {**robot_measurement(position), "position_xy": position, "contact": False}}, set(), 0.)
    assert progress.orbit_angle == 0. and not progress.orbit_lap


def test_furniture_circuit_selection_persists_reset_and_keeps_geometry_private():
    from fastapi.testclient import TestClient
    from backend.app import app
    with TestClient(app) as client:
        reply = client.post("/api/challenges/load", json={"challenge_id": "furniture_circuit", "orbit_target": "sofa", "orbit_direction": "counterclockwise"})
        assert reply.status_code == 200
        state = reply.json()
        assert state["challenge"]["orbit"] == {"target": "sofa", "direction": "counterclockwise"}
        assert "counterclockwise circuit around that sofa" in state["challenge"]["goal"]
        assert "center_m" not in json.dumps(state["challenge"])
        assert "circuit_" not in json.dumps(state["observation"])
        reset = client.post("/api/reset", json={}).json()
        assert reset["run_id"] != state["run_id"] and reset["challenge"]["orbit"] == state["challenge"]["orbit"]
        assert client.post("/api/challenges/load", json={"challenge_id": "park", "orbit_target": "chair"}).status_code == 422


@pytest.mark.parametrize("target,direction,environment", [
    (target, direction, "standalone") for target in ("table", "sofa", "chair", "floor lamp")
    for direction in ("clockwise", "counterclockwise")
] + [("table", direction, "shared_apartment_v1") for direction in ("clockwise", "counterclockwise")])
def test_furniture_circuit_is_feasible_with_real_wheels(target, direction, environment, recorded_physics_tests):
    from backend.challenges import furniture_circuit, shared_apartment
    from backend.robot import TRAVEL
    challenge = shared_apartment("furniture_circuit", direction) if environment == "shared_apartment_v1" else furniture_circuit(target, direction)
    radius = (challenge.orbit.minimum_radius_m + challenge.orbit.maximum_radius_m) / 2
    center = np.array(challenge.orbit.center_m)
    challenge.initial_xy = (center + [radius, 0.]).tolist()
    sim = BulletSimulation(challenge=challenge, width=160, height=120)
    sign = -1 if direction == "clockwise" else 1
    recording = recorded_physics_tests.record(sim, f"{environment}-circle-{target.replace(' ', '-')}-{direction}",
        "Known-geometry wheel steering. Test starts on the permitted annulus, stows arms and sets tangent orientation before the lap.")
    try:
        for side in ("left", "right"):
            result = command(sim, "set_arm_joints", arm=side, joint_positions_rad=TRAVEL, duration_s=2.)
            assert result.status == "ok", result.message
        position = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0]
        bullet.resetBasePositionAndOrientation(sim.robot, position, bullet.getQuaternionFromEuler([0, 0, sign * math.pi / 2]), physicsClientId=sim.client)
        sim.challenge_progress = ChallengeProgress(challenge)
        for _ in range(240):
            position, quaternion = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
            heading = bullet.getEulerFromQuaternion(quaternion)[2]
            relative = np.array(position[:2]) - center
            angle = math.atan2(relative[1], relative[0])
            desired_heading = angle + sign * math.pi / 2 + sign * .8 * (np.linalg.norm(relative) - radius)
            error = math.atan2(math.sin(desired_heading - heading), math.cos(desired_heading - heading))
            result = command(sim, "drive_base", linear_mps=.3, angular_radps=float(np.clip(sign * .3 / radius + error, -.5, .5)), duration_s=.5)
            assert result.status == "ok", result.message
            assert not sim.proximity_sensors().collisions
            if sim.challenge_progress.orbit_lap:
                break
        assert sim.challenge_progress.orbit_lap, sim.challenge_status()
        result = command(sim, "wait", duration_s=.6)
        assert result.status == "ok" and sim.challenge_status()["status"] == "completed"
        assert sim.path_length > 2 * math.pi * challenge.orbit.minimum_radius_m
    finally:
        try:
            recording.finish(sys.exc_info()[0])
        finally:
            sim.close()


def test_pedestrian_contact_latches_failure_and_fresh_episode_resets_without_sensor_leak():
    from backend.simulation import MotionError
    for collision in (True, False):
        sim = BulletSimulation(challenge=get_challenge("pedestrian_crossing"), width=160, height=120)
        try:
            assert not sim.pedestrian_crossing["contact"]
            assert sim.pedestrian_crossing["started_at"] is None
            assert sim.challenge_status()["completed_objectives"] == 0
            observation = sim.observe()
            assert "pedestrian" not in observation.model_dump_json()
            sim._ticks(120)
            assert sim.pedestrian_crossing["started_at"] is None
            if collision:
                leg = next(item for item in sim.objects if item["name"] == "pedestrian_leg_left")
                bullet.resetBasePositionAndOrientation(leg["id"], [.15, .2, .38], [0, 0, 0, 1], physicsClientId=sim.client)
                sim._ticks(12)
                assert sim.pedestrian_crossing["contact"]
                assert sim.challenge_status()["status"] == "failed"
                bullet.resetBasePositionAndOrientation(leg["id"], leg["position"], [0, 0, 0, 1], physicsClientId=sim.client)
                sim._ticks(120)
                assert sim.challenge_status()["status"] == "failed"
            before = sim.snapshot()
            sim.stop()
            with pytest.raises(MotionError, match="interrupted"):
                sim._ticks(12)
            assert sim.snapshot() == before
        finally:
            sim.close()


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


@pytest.mark.parametrize("environment", ["standalone", "shared_apartment_v1"])
def test_apartment_inspection_requires_visible_nearby_target_and_stationary_dwell(environment):
    from backend.challenges import shared_apartment
    progress = ChallengeProgress(shared_apartment("apartment") if environment == "shared_apartment_v1" else get_challenge("apartment"))
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


@pytest.mark.parametrize("environment", ["standalone", "shared_apartment_v1"])
def test_recharge_requires_departure_scan_low_battery_return_and_charging(environment):
    from backend.challenges import shared_apartment
    challenge = shared_apartment("recharge") if environment == "shared_apartment_v1" else get_challenge("recharge")
    progress = ChallengeProgress(challenge)
    dock = {"robot": robot_measurement(challenge.objectives[1].center[:2])}
    survey = {"robot": robot_measurement(challenge.objectives[0].center[:2])}
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


def test_flat_kitchen_has_five_detailed_rooms_open_doors_and_private_destination():
    preset = get_challenge("flat_kitchen")
    sim = BulletSimulation(challenge=preset, width=160, height=120)
    try:
        objects = {item["name"]: item for item in sim.objects}
        assert len(objects) == len(sim.objects)
        assert len(objects) >= 120
        assert {"living_floor", "bedroom_floor", "study_floor", "bathroom_floor", "kitchen_floor",
                "living_tv_screen", "bedroom_pillow_0", "study_keyboard", "toilet_bowl", "stove_hob"} <= objects.keys()
        assert objects["floor"]["size"][:2] == [12, 10]
        assert not any(name.startswith("zone_") for name in objects)
        assert not sim.proximity_sensors().collisions
        for center_x, center_y in ((-4, 1.05), (0, 1.05), (4, 1.05), (-3, -1.05), (3, -1.05)):
            assert bullet.rayTest([center_x, center_y - .3, .55], [center_x, center_y + .3, .55], physicsClientId=sim.client)[0][0] == -1
        assert bullet.rayTest([-.3, -2.5, .55], [.3, -2.5, .55], physicsClientId=sim.client)[0][0] == -1
        assert not sim._target_in_camera(objects["stove_hob"]["id"], np.array(objects["stove_hob"]["position"]))
        observation = sim.observe()
        assert observation.odometry_m_rad == [0, 0, 0]
        assert sim.challenge_status()["status"] == "in_progress"
        assert all(name not in observation.model_dump_json() for name in objects)
        assert not {"initial_xy", "objects", "floor_size_m", "ordered_objectives"} & preset.public().keys()
    finally:
        sim.close()


def test_flat_kitchen_requires_full_entry_grounding_rest_and_final_dwell():
    progress = ChallengeProgress(get_challenge("flat_kitchen"))
    elapsed = 0
    for center, invalid in (([-3, -2], {}), ([0, 2], {}), ([4, 1.3], {}), ([3.4, 2.65], {"grounded": False}),
                            ([3.4, 2.65], {"speed": .1}), ([3.4, 2.65], {"angular_speed": .2})):
        elapsed += 1
        assert progress.update({"robot": {**robot_measurement(center), **invalid}}, set(), elapsed)["status"] == "in_progress"
    parked = {"robot": robot_measurement([3.4, 2.65])}
    assert progress.update(parked, set(), elapsed + .25)["status"] == "in_progress"
    assert progress.update(parked, set(), elapsed + .25)["status"] == "in_progress"
    assert progress.update(parked, set(), elapsed + .5)["status"] == "completed"
    assert progress.update({"robot": robot_measurement([4, 0])}, set(), elapsed + .6)["status"] == "in_progress"


def test_flat_kitchen_all_rooms_and_alternate_living_route_are_physically_accessible():
    sim = BulletSimulation(challenge=get_challenge("flat_kitchen"), width=160, height=120)
    try:
        for waypoint in ([-3, -2.2], [-3, 0], [-4, 0], [-4, 1.8], [-4, 0],
                         [0, 0], [0, 2], [0, 0], [3, 0], [3, -1.7], [4.4, -1.7],
                         [4.4, -2.1], [.8, -2.1], [-.8, -2.1], [-3, -2.1], [-3, 0], [4, 0]):
            drive_to(sim, waypoint)
            assert sim.challenge_status()["status"] == "in_progress"
            assert not sim.proximity_sensors().collisions
        drive_to(sim, [4, 2.6])
        assert command(sim, "wait", duration_s=.6).status == "ok"
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        assert sim.path_length > 25
        assert not sim.proximity_sensors().collisions
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


@pytest.mark.parametrize("identifier", ["flat_kitchen", "apartment", "recharge"])
def test_shared_apartment_tasks_are_reachable_with_real_wheels(identifier, recorded_physics_tests):
    from backend.challenges import shared_apartment
    from backend.robot import TRAVEL
    sim = BulletSimulation(challenge=shared_apartment(identifier), width=160, height=120)
    recording = recorded_physics_tests.record(sim, f"shared_apartment_v1-{identifier}",
        "Known-geometry waypoint steering from the scenario start; real wheels, sensing and task scorer. No model inference.")
    try:
        for side in ("left", "right"):
            assert command(sim, "set_arm_joints", arm=side, joint_positions_rad=TRAVEL, duration_s=2.).status == "ok"
        if identifier == "recharge":
            command(sim, "wait", duration_s=2)
            assert sim.challenge_status()["completed_objectives"] == 0
            drive_to(sim, [.2, 0])
            assert command(sim, "wait", duration_s=1.2).status == "ok"
            assert sim.challenge_progress.survey_complete and sim.battery_sensor().low
            assert sim.challenge_status()["status"] != "completed"
            drive_to(sim, [-2.7, 0])
            for _ in range(3):
                assert command(sim, "wait", duration_s=2).status == "ok"
            assert sim.battery_sensor().charge_pct >= 90
        else:
            for point in ([-6.8, 0], [-2.7, 0], [-.3, 0]):
                drive_to(sim, point)
            assert sim.challenge_status()["status"] == "in_progress"
            if identifier == "flat_kitchen":
                for point in ([3.4, 0], [-.3, 0], [-.3, 3.8], [3.4, 3.8]):
                    drive_to(sim, point)
                assert command(sim, "wait", duration_s=.6).status == "ok"
            else:
                for point in ([-.3, -3.8], [2.7, -3.8], [2.7, -2.6], [5.45, -2.6]):
                    drive_to(sim, point)
                assert command(sim, "set_head", yaw_rad=0, pitch_rad=.35, duration_s=1).status == "ok"
                assert command(sim, "wait", duration_s=1.2).status == "ok"
        assert not sim.proximity_sensors().collisions
        assert sim.challenge_status()["status"] == "completed", sim.challenge_status()
        assert sim.path_length > 5
    finally:
        try:
            recording.finish(sys.exc_info()[0])
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