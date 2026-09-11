from uuid import uuid4

import numpy as np
import pybullet as bullet
import pytest

from backend.contracts import Command, ManualPlacement
from backend.simulation import BulletSimulation, MotionError


@pytest.fixture
def sim():
    simulation = BulletSimulation(width=160, height=120)
    yield simulation
    simulation.close()


def command(sim, tool, **arguments):
    return sim.execute(Command(run_id=sim.run_id, episode_epoch=sim.epoch, action_id=str(uuid4()),
                               observation_seq=sim.seq, tool=tool, arguments=arguments))


def test_distance_and_collision_sensors_follow_robot_frame_and_exclude_floor():
    import json
    scene = [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "obstacle", "size": [.1, 2, .5], "position": [1, 0, .25], "color": [.2, .3, .3, 1]}]
    sim = BulletSimulation(scene=scene, width=160, height=120)
    try:
        sensors = sim.observe().proximity
        assert not sensors.collisions and not sim.observe().bumpers
        assert len(sensors.distances) == 8
        assert sensors.distances[0].distance_m == pytest.approx(.77, abs=.01)
        assert sensors.distances[4].status == "clear" and sensors.distances[4].distance_m is None
        position = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0]
        bullet.resetBasePositionAndOrientation(sim.robot, position, bullet.getQuaternionFromEuler([0, 0, np.pi / 2]), physicsClientId=sim.client)
        turned = sim.proximity_sensors()
        assert turned.distances[6].distance_m == pytest.approx(.70, abs=.01)
        assert turned.distances[0].status == "clear"
        assert not any(value in json.dumps(turned.model_dump()) for value in ("obstacle", "body_id", "position", "geometry"))
        bullet.resetBasePositionAndOrientation(sim.robot, position, [0, 0, 0, 1], physicsClientId=sim.client)
        contacts = []
        sim.on_tick = lambda: contacts.extend(sim.proximity_sensors().collisions)
        for _ in range(3):
            command(sim, "drive_base", linear_mps=.3, angular_radps=0, duration_s=2)
        assert any(contact.direction == "front" and contact.force_n > .05 for contact in contacts)
        assert sim.observe().bumpers == [contact.direction for contact in sim.proximity_sensors().collisions]
    finally:
        sim.close()


def test_blocked_drive_reports_no_progress_and_allows_inspection_and_retreat():
    from backend.challenges import get_challenge
    from backend.feedback import model_tool_result
    simulation = BulletSimulation(challenge=get_challenge("apartment"), width=160, height=120)
    try:
        for _ in range(6):
            result = command(simulation, "drive_base", linear_mps=.1, angular_radps=0, duration_s=1)
        assert result.status == "error" and result.error == "NO_PROGRESS"
        assert "set_head" in result.message and result.observation.bumpers
        contacts = result.sensor_deltas["contacts_during_action"]
        assert any(contact["direction"] == "front" and contact["force_n"] > .05 for contact in contacts)
        assert "short, slow reverse" in model_tool_result(result.model_dump())["collision_feedback"]["guidance"]
        assert np.linalg.norm(result.sensor_deltas["odometry_m_rad"][:2]) < .005
        assert result.actual_duration_s == 1
        assert command(simulation, "drive_base", linear_mps=0, angular_radps=0, duration_s=.5).status == "ok"
        assert command(simulation, "set_head", yaw_rad=.5, pitch_rad=.2, duration_s=.5).status == "ok"
        retreat = command(simulation, "drive_base", linear_mps=-.1, angular_radps=0, duration_s=1)
        assert retreat.status == "ok" and not retreat.observation.bumpers
        assert retreat.sensor_deltas["contacts_during_action"]
        assert model_tool_result(retreat.model_dump())["collision_feedback"]["source"] == "during_action"
        clear = command(simulation, "wait", duration_s=.5)
        assert "contacts_during_action" not in clear.sensor_deltas
        assert "collision_feedback" not in model_tool_result(clear.model_dump())
        assert simulation.action_contacts is None
    finally:
        simulation.close()


def test_scene_textures_render_in_authoritative_camera_without_changing_physics(sim):
    from io import BytesIO
    from PIL import Image
    from backend.materials import ensure_textures
    paths = ensure_textures()
    assert len(paths) == 6
    for path in paths.values():
        with Image.open(path) as texture:
            assert texture.size == (512, 512)
            assert np.std(np.asarray(texture)) > 3
    command(sim, "set_head", yaw_rad=0, pitch_rad=.7, duration_s=.5)
    original = sim.snapshot()
    textured = np.asarray(Image.open(BytesIO(sim.capture()))).astype(float)
    assert any(item["texture"] for item in sim.geometry())
    for item in sim.objects:
        bullet.changeVisualShape(item["id"], -1, textureUniqueId=-1, physicsClientId=sim.client)
    plain = np.asarray(Image.open(BytesIO(sim.capture()))).astype(float)
    assert np.mean(np.abs(textured - plain)) > 1
    assert sim.snapshot() == original


def test_drive_head_and_private_observation(sim):
    initial = sim.observe()
    image = sim.frame(initial.frame_ref)
    result = command(sim, "drive_base", linear_mps=.2, angular_radps=0, duration_s=1)
    assert result.status == "ok"
    assert sim.evaluator_state().robot_position[0] > .08
    result = command(sim, "set_head", yaw_rad=.5, pitch_rad=.6, duration_s=1)
    assert result.status == "ok"
    assert sim.frame(result.observation.frame_ref) != image
    assert abs(result.observation.head_rad[0] - .5) < .03
    assert not any(key in result.observation.model_dump() for key in ("bodies", "objects", "depth", "goal", "robot_position"))


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_floor_reach(sim, side, sign):
    result = command(sim, "move_end_effector", arm=side, position_m=[.36, sign * .25, -.12],
                     orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
    assert result.status == "ok", (result.error, result.message)
    state = bullet.getLinkState(sim.robot, sim.palms[side], physicsClientId=sim.client)
    assert np.linalg.norm(np.array(state[4]) - [.36, sign * .25, .035]) < .025


def test_far_grasp_and_unreachable_target(sim):
    result = command(sim, "set_gripper", arm="left", opening_m=0, max_force_n=20)
    assert result.status == "ok"
    assert not sim.held
    before = sim.ticks
    result = command(sim, "move_end_effector", arm="left", position_m=[5, .25, 1], orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
    assert result.error == "UNREACHABLE_TARGET"
    assert sim.ticks == before


def test_stale_duplicate_and_stop(sim):
    sim.observe()
    envelope = Command(run_id=sim.run_id, episode_epoch=0, action_id="drive", observation_seq=sim.seq,
                       tool="drive_base", arguments={"linear_mps": .2, "angular_radps": 0, "duration_s": .5})
    first = sim.execute(envelope)
    ticks = sim.ticks
    assert sim.execute(envelope) is first
    assert sim.ticks == ticks
    envelope.action_id = "stale"
    assert sim.execute(envelope).error == "STALE_OBSERVATION"
    command(sim, "stop")
    assert command(sim, "drive_base", linear_mps=.2, angular_radps=0, duration_s=1).error == "CANCELLED"


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_floor_grasp_lift_release(side, sign):
    scene = [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "cube", "size": [.06, .06, .06], "position": [.36, sign * .25, .031], "color": [.9, .1, .2, 1], "mass": .08}]
    sim = BulletSimulation(width=160, height=120, scene=scene)
    try:
        result = command(sim, "move_end_effector", arm=side, position_m=[.36, sign * .25, -.12], orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
        assert result.status == "ok", (result.error, result.message)
        result = command(sim, "set_gripper", arm=side, opening_m=0, max_force_n=20)
        assert side in sim.held, result.observation.grippers
        result = command(sim, "move_end_effector", arm=side, position_m=[.36, sign * .25, .05], orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
        assert result.status == "ok", (result.error, result.message)
        assert sim.evaluator_state().bodies[1]["position"][2] > .17
        assert side in sim.held
        before_placement = sim.snapshot()
        sequence = sim.seq
        with pytest.raises(MotionError, match="Release held objects"):
            sim.reposition(ManualPlacement(run_id=sim.run_id, episode_epoch=sim.epoch, observation_seq=sequence, xy_m=[-.5, 0]))
        assert sim.snapshot() == before_placement and sim.seq == sequence and side in sim.held
        result = command(sim, "set_gripper", arm=side, opening_m=.11, max_force_n=20)
        assert result.status == "ok"
        assert not sim.held
        assert sim.evaluator_state().bodies[1]["position"][2] < .05
    finally:
        sim.close()


def test_obstacle_blocks_base_and_rotation_works():
    scene = [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "obstacle", "size": [.1, 2, .5], "position": [.5, 0, .25], "color": [.2, .3, .3, 1]}]
    sim = BulletSimulation(width=160, height=120, scene=scene)
    try:
        for _ in range(3):
            command(sim, "drive_base", linear_mps=.3, angular_radps=0, duration_s=2)
        assert sim.evaluator_state().robot_position[0] < .32
        command(sim, "drive_base", linear_mps=-.2, angular_radps=0, duration_s=2)
        result = command(sim, "drive_base", linear_mps=0, angular_radps=.6, duration_s=2)
        assert result.observation.odometry_m_rad[2] > .5
        orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[1]
        assert bullet.getEulerFromQuaternion(orientation)[2] > .4
    finally:
        sim.close()


def test_colliding_arm_path_does_not_move(sim):
    before = sim.ticks
    result = command(sim, "move_end_effector", arm="left", position_m=[.2, .05, -.06], orientation_xyzw=[0, 0, 0, 1], frame="base", duration_s=2)
    assert result.error in ("COLLISION_BLOCKED", "UNREACHABLE_TARGET")
    assert sim.ticks == before