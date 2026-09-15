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


def test_render_snapshot_is_calibrated_serializable_and_read_only(sim):
    import json
    from backend.camera import scene_snapshot
    from backend.contracts import AgentObservation
    before = sim.snapshot()
    sequence = sim.seq
    packet = scene_snapshot(sim, 320, 240)
    assert json.loads(json.dumps(packet)) == packet
    assert sim.snapshot() == before and sim.seq == sequence
    assert packet["snapshot"] == before
    assert packet["run_id"] == sim.run_id and packet["episode_epoch"] == sim.epoch
    assert packet["observation_seq"] == sequence
    assert packet["camera"]["width"] == 320 and packet["camera"]["height"] == 240
    eye, rotation = sim.camera_pose()
    assert np.allclose(packet["camera"]["eye"], eye)
    assert np.allclose(np.subtract(packet["camera"]["target"], eye), rotation[:, 0])
    assert np.allclose(packet["camera"]["up"], rotation[:, 2])
    assert all(asset["name"] for asset in packet["geometry"])
    assert not {"geometry", "snapshot", "camera", "robot_body_id"} & AgentObservation.model_fields.keys()
    for dimensions in [(0, 120), (320, 2049), (True, 120), (320.5, 240)]:
        with pytest.raises(ValueError):
            scene_snapshot(sim, *dimensions)


@pytest.mark.parametrize("wall_x", [1.5, 6.])
def test_render_depth_gate_rejects_shifted_or_missing_surfaces(wall_x):
    from scripts.benchmark_rendering import depth_agreement
    reference = np.full((120, 160), wall_x)
    assert depth_agreement(reference, reference.copy())["passed"]
    assert not depth_agreement(reference, reference + .04)["passed"]
    missing = reference.copy()
    missing[:, :20] = np.nan
    assert not depth_agreement(reference, missing)["passed"]
    assert not depth_agreement(reference, np.full_like(reference, np.nan))["passed"]


def test_persistent_enhanced_renderer_keeps_snapshot_identity_and_depth(sim):
    from io import BytesIO
    from PIL import Image
    from backend.camera import EnhancedRenderer, scene_snapshot
    before = sim.snapshot()
    renderer = EnhancedRenderer()
    try:
        packet = scene_snapshot(sim, 160, 120)
        image, depth = renderer.capture(packet)
        repeat, repeated_depth = renderer.capture(packet)
        assert image == repeat and np.allclose(depth, repeated_depth, equal_nan=True)
        assert Image.open(BytesIO(image)).size == (160, 120)
        assert np.isfinite(depth).sum() > 1000
        assert renderer.device and "NVIDIA" in renderer.device
        assert sim.snapshot() == before
    finally:
        renderer.close()
    assert renderer.process.poll() is not None


def test_enhanced_live_sensor_pair_never_uses_tiny_renderer(sim, monkeypatch):
    from io import BytesIO
    from PIL import Image
    baseline = sim.capture()
    sim.rendering = "enhanced"
    def forbidden(*args, **kwargs):
        raise AssertionError("TinyRenderer must not be used in enhanced mode")
    monkeypatch.setattr(bullet, "getCameraImage", forbidden)
    before = sim.snapshot()
    observation = sim.observe()
    image = sim.frame(observation.frame_ref)
    spatial, paired = sim.capture_spatial(7)
    assert sim.snapshot() == before and sim.seq == observation.seq
    assert image != baseline and Image.open(BytesIO(image)).size == (160, 120)
    assert paired == image
    assert spatial.sequence == 7 and spatial.simulated_time_s == observation.simulated_time_s
    assert all(value is None or spatial.calibration.near_m <= value <= spatial.calibration.usable_range_m for value in spatial.depth_m)
    assert not {"geometry", "rendering", "snapshot", "renderer"} & observation.model_dump().keys()
    sim.camera_renderer.close()
    with pytest.raises(RuntimeError, match="closed"):
        sim.capture()
    assert sim.cancel.is_set() and sim.snapshot() == before


@pytest.mark.parametrize("wall_x", [1.5, 6.])
def test_spatial_capture_pairs_rgb_and_axial_depth_without_advancing_physics(wall_x):
    from io import BytesIO
    from PIL import Image
    from backend.spatial import point_cloud
    scene = [{"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
             {"name": "wall", "size": [.1, 4, 2], "position": [wall_x, 0, 1], "color": [.2, .3, .3, 1]}]
    sim = BulletSimulation(scene=scene)
    try:
        before = sim.snapshot()
        sensor, image = sim.capture_spatial(1)
        assert sim.snapshot() == before and sensor.sequence == 1
        assert Image.open(BytesIO(image)).size == (160, 120)
        depth = np.array(sensor.depth_m, dtype=float).reshape(120, 160)
        assert depth[60, 80] == pytest.approx(wall_x - .105, abs=.025)
        eye, points = point_cloud(sensor)
        wall = points[(points[:, 2] > .3) & (points[:, 2] < 1.)]
        assert np.median(wall[:, 0]) == pytest.approx(wall_x - .05, abs=.025)
        assert not {"bodies", "objects", "robot_position", "segmentation", "geometry"} & sensor.model_dump().keys()
        assert command(sim, "set_head", yaw_rad=0, pitch_rad=.9, duration_s=1).status == "ok"
        downward, _ = sim.capture_spatial(2)
        assert any(value is None for value in downward.depth_m)
    finally:
        sim.close()


@pytest.mark.parametrize("head", [(0., 0.), (.6, .35), (-.6, .65)])
def test_spatial_projection_tracks_head_pose_and_ignores_world_origin(head):
    from backend.spatial import point_cloud
    scene = [{"name": "floor", "size": [8, 8, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
             {"name": "wall", "size": [.1, 6, 2], "position": [1.5, 0, 1], "color": [.4, .4, .4, 1]}]
    sim = BulletSimulation(scene=scene)
    try:
        assert command(sim, "set_head", yaw_rad=head[0], pitch_rad=head[1], duration_s=1).status == "ok"
        sensor, _ = sim.capture_spatial(1)
        _, points = point_cloud(sensor, stride=2)
        wall = points[(points[:, 2] > .25) & (points[:, 2] < 1.2)]
        assert len(wall) > 20
        assert np.percentile(np.abs(wall[:, 0] - 1.45), 90) < .045
        shifted = sensor.model_copy(update={"odometry_m_rad": [sensor.odometry_m_rad[0] + 5, sensor.odometry_m_rad[1] - 2, sensor.odometry_m_rad[2]]})
        _, moved = point_cloud(shifted, stride=2)
        assert np.allclose(moved - points, [5, -2, 0])
    finally:
        sim.close()


def test_spatial_map_keeps_occluded_floor_unknown_and_rejects_stale_frames():
    from backend.spatial import ObservedMap
    scene = [{"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
             {"name": "screen", "size": [.1, 2, 1.5], "position": [1, 0, .75], "color": [.4, .4, .4, 1]}]
    sim = BulletSimulation(scene=scene)
    try:
        command(sim, "set_head", yaw_rad=0, pitch_rad=.65, duration_s=1)
        observation, _ = sim.capture_spatial(1)
        observed = ObservedMap(sim.run_id, sim.epoch)
        observed.update(observation, now=observation.captured_at)
        state = observed.public(now=observation.captured_at)
        assert state["observed_floor_cells"] > 0 and state["obstacle_cells"] > 0
        assert state["unknown_cells"] > observed.size ** 2 * .8
        def cell(position):
            column, row = np.floor(np.array(position) / observed.resolution_m).astype(int) - observed.origin
            return observed.cells[row, column]
        assert cell([1.4, 0]) == -1
        assert cell([-.5, 0]) == -1
        assert observed.public(now=observation.captured_at + 2)["stale"]
        original = observed.cells.copy()
        for invalid in (observation, observation.model_copy(update={"sequence": 2, "captured_at": observation.captured_at - 2}),
                        observation.model_copy(update={"sequence": 2, "episode_epoch": 99})):
            with pytest.raises(ValueError):
                observed.update(invalid, now=observation.captured_at)
            assert np.array_equal(observed.cells, original)
        shifted = observation.model_copy(update={"sequence": 2, "odometry_m_rad": [20., 0., 0.],
            "depth_m": [None] * len(observation.depth_m)})
        observed.update(shifted, now=observation.captured_at)
        assert np.all(observed.cells == -1)
    finally:
        sim.close()


def test_spatial_map_clears_departed_obstacles_only_after_two_visible_depth_frames():
    from backend.spatial import ObservedMap
    scene = [{"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
             {"name": "background", "size": [.1, 3., 1.5], "position": [2.7, 0, .75], "color": [.5, .5, .5, 1]},
             {"name": "moving_box", "size": [.2, .4, .5], "position": [1.2, 0, .25], "color": [.3, .4, .5, 1]}]
    sim = BulletSimulation(scene=scene, width=160, height=120)
    try:
        assert command(sim, "set_head", yaw_rad=0, pitch_rad=.25, duration_s=1).status == "ok"
        observed = ObservedMap(sim.run_id, sim.epoch)
        for sequence in range(1, 4):
            sensor, _ = sim.capture_spatial(sequence)
            observed.update(sensor, now=sensor.captured_at)
        columns = np.indices(observed.cells.shape)[1]
        forward = (columns + observed.origin[0] + .5) * observed.resolution_m
        occupied = (observed.cells == 100) & (forward > 1.) & (forward < 1.4)
        assert occupied.any()
        box = next(item for item in sim.objects if item["name"] == "moving_box")
        bullet.resetBasePositionAndOrientation(box["id"], [1.2, 1.8, .25], [0, 0, 0, 1], physicsClientId=sim.client)
        for sequence in (4, 5):
            sensor, _ = sim.capture_spatial(sequence)
            hidden = sensor.model_copy(update={"depth_m": [None] * len(sensor.depth_m)})
            observed.update(hidden, now=hidden.captured_at)
            assert np.all(observed.cells[occupied] == 100)
        sensor, _ = sim.capture_spatial(6)
        observed.update(sensor, now=sensor.captured_at)
        assert np.all(observed.cells[occupied] == 100)
        sensor, _ = sim.capture_spatial(7)
        observed.update(sensor, now=sensor.captured_at)
        assert np.count_nonzero(observed.cells[occupied] == 100) < np.count_nonzero(occupied) * .3
        assert observed.cells[observed.cell_index([1.1, 0.])] == 0
    finally:
        sim.close()


def test_floor_regions_use_paired_depth_keep_targets_and_reject_partial_completion():
    from backend.spatial import FloorRegionTracker
    scene = [{"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
             {"name": "zone", "size": [.9, 1.05, .002], "position": [1.6, .2, .002], "color": [.15, .65, .31, 1], "marker": True},
             {"name": "green_object", "size": [.25, .3, .5], "position": [1.5, -.9, .25], "color": [.15, .65, .31, 1]}]
    sim = BulletSimulation(scene=scene, width=320, height=240)
    try:
        assert command(sim, "set_head", yaw_rad=0, pitch_rad=.35, duration_s=1).status == "ok"
        tracker = FloorRegionTracker(sim.run_id, sim.epoch)
        sensor, image = sim.capture_spatial(1)
        regions = tracker.update(sensor, image, now=sensor.captured_at)
        assert len(regions) == 1
        region = regions[0]
        assert region["color"] == "green" and region["complete_view"]
        assert np.linalg.norm(np.array(region["center_m"]) - [1.6, .2]) < .08
        footprint = np.array([[1.35, -.05], [1.85, -.05], [1.85, .45], [1.35, .45]])
        assert tracker.contains(region, footprint)
        assert not tracker.contains(region, footprint + [.4, 0.])
        assert not tracker.contains({**region, "complete_view": False}, footprint)
        with pytest.raises(ValueError, match="out of order"):
            tracker.update(sensor, image, now=sensor.captured_at)
        from io import BytesIO
        from PIL import Image, ImageDraw
        altered = Image.open(BytesIO(image)).convert("RGB")
        polygon = np.asarray(region["image_polygon"])
        midpoint = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
        ImageDraw.Draw(altered).rectangle((int(midpoint[0] * altered.width), int(polygon[:, 1].min() * altered.height),
            int(midpoint[0] * altered.width + 8), int(midpoint[1] * altered.height)), fill=(150, 150, 150))
        encoded = BytesIO()
        altered.save(encoded, format="PNG")
        partial = FloorRegionTracker(sim.run_id, sim.epoch).update(sensor, encoded.getvalue(), now=sensor.captured_at)
        assert partial and not any(item["complete_view"] for item in partial)
        assert command(sim, "set_head", yaw_rad=1.2, pitch_rad=.35, duration_s=1).status == "ok"
        sensor, image = sim.capture_spatial(2)
        retained = tracker.update(sensor, image, now=sensor.captured_at)
        assert any(item["id"] == region["id"] and item["center_m"] == region["center_m"] for item in retained)
        assert not tracker.public(now=sensor.captured_at + 121)
        with pytest.raises(ValueError):
            tracker.update(sensor.model_copy(update={"episode_epoch": 99}), image, now=sensor.captured_at)
    finally:
        sim.close()


def test_travel_posture_reduces_footprint_through_real_joint_motion():
    from backend.robot import TRAVEL
    from backend.challenges import get_challenge
    for challenge_id in ("park", "kitchen_bathroom", "recharge"):
        sim = BulletSimulation(challenge=get_challenge(challenge_id), width=160, height=120)
        try:
            before = sim.robot_footprint()
            for side in ("left", "right"):
                result = command(sim, "set_arm_joints", arm=side, joint_positions_rad=TRAVEL, duration_s=2)
                assert result.status == "ok", (challenge_id, result.message)
            after = sim.robot_footprint()
            assert after["upper_xy_m"][0] < before["upper_xy_m"][0] - .12
            assert after["radius_m"] < before["radius_m"]
            assert not sim.proximity_sensors().collisions
        finally:
            sim.close()


def test_robot_footprint_does_not_grow_when_only_base_heading_changes():
    from backend.robot import TRAVEL
    sim = BulletSimulation(width=160, height=120)
    try:
        for side in ("left", "right"):
            assert command(sim, "set_arm_joints", arm=side, joint_positions_rad=TRAVEL, duration_s=2).status == "ok"
        position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        measured = []
        for heading in (0., .08, .4, .8, 1.5, 3.1):
            bullet.resetBasePositionAndOrientation(sim.robot, position, bullet.getQuaternionFromEuler([0., 0., heading]),
                physicsClientId=sim.client)
            measured.append(sim.robot_footprint())
        for footprint in measured[1:]:
            assert footprint["radius_m"] == pytest.approx(measured[0]["radius_m"], abs=1e-6)
            np.testing.assert_allclose(footprint["lower_xy_m"], measured[0]["lower_xy_m"], atol=1e-6)
            np.testing.assert_allclose(footprint["upper_xy_m"], measured[0]["upper_xy_m"], atol=1e-6)
    finally:
        sim.close()


def test_spatial_depth_invalid_and_self_pixels_are_not_free_space():
    from backend.contracts import SpatialObservation
    from backend.spatial import calibration, metric_depth, point_cloud
    intrinsics = calibration(16, 16)
    depth = metric_depth(np.array([[0., 1., float("nan"), -.1, .5]]), np.array([[True, False, False, False, False]]), intrinsics)
    assert np.isnan(depth[0, :4]).all() and np.isfinite(depth[0, 4])
    observation = SpatialObservation(run_id="test", episode_epoch=0, sequence=1, captured_at=10,
        simulated_time_s=0, calibration=intrinsics, head_rad=[0, 0], odometry_m_rad=[0, 0, 0], depth_m=[None] * 256)
    assert point_cloud(observation)[1].shape == (0, 3)
    for values in ([1.], [float("nan")] * 256, [intrinsics.usable_range_m + 1] * 256):
        with pytest.raises(ValueError):
            SpatialObservation(**(observation.model_dump() | {"depth_m": values}))


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