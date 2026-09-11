import asyncio
from io import BytesIO
import threading
import time
from uuid import uuid4

import pytest
import pybullet as bullet
from PIL import Image

from backend.contracts import Command, ManualPlacement
from backend.simulation import MotionError
from backend.worker import CameraActivity, SimulationWorker


def test_camera_activity_uses_elapsed_time_and_ignores_tiny_pixel_noise():
    def png(color):
        output = BytesIO()
        Image.new("RGB", (64, 48), color).save(output, format="PNG")
        return output.getvalue()

    activity = CameraActivity()
    assert activity.sample(png((100, 100, 100)), 10) == {"revision": 1, "unchanged_s": 0}
    assert activity.sample(png((102, 102, 102)), 14.9)["unchanged_s"] == pytest.approx(4.9)
    assert activity.sample(png((100, 100, 100)), 15) == {"revision": 1, "unchanged_s": 5}
    assert activity.sample(png((130, 100, 100)), 16) == {"revision": 2, "unchanged_s": 0}


@pytest.mark.asyncio
async def test_camera_monitor_detects_real_head_changes_without_observing_or_stepping():
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        before = await worker.sample_camera_activity()
        await worker.sample_camera_activity()
        assert worker.latest["camera"] == initial["camera"]
        await worker.call(lambda sim: bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .8, physicsClientId=sim.client))
        changed = await worker.sample_camera_activity()
        assert changed["revision"] == before["revision"] + 1
        assert worker.latest["camera"]["seq"] > initial["camera"]["seq"]
        assert worker.sim.seq == initial["observation"]["seq"] and worker.sim.ticks == 0
        assert len(worker.sim.frames) == 1
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_live_camera_and_poses_update_during_motion_without_extra_model_observations():
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        initial_seq = worker.latest["observation"]["seq"]
        initial_frame = worker.sim.frame(worker.latest["observation"]["frame_ref"])
        assert worker.camera_frames[worker.latest["camera"]["frame_ref"]] == initial_frame
        envelope = Command(run_id=worker.sim.run_id, episode_epoch=0, action_id="live-head",
                           observation_seq=initial_seq, tool="set_head",
                           arguments={"yaw_rad": .8, "pitch_rad": .7, "duration_s": 2})
        started = time.monotonic()
        task = asyncio.create_task(worker.execute(envelope, assisted=False))
        images, poses, observation_sequences = {}, {}, set()
        async with asyncio.timeout(5):
            while not task.done():
                state = worker.latest
                if state["busy"]:
                    camera = state["camera"]
                    if image := worker.camera_frames.get(camera["frame_ref"]):
                        images[camera["seq"]] = image
                    poses[state["snapshot"]["simulated_time_s"]] = state["snapshot"]["poses"]
                    observation_sequences.add(state["observation"]["seq"])
                await asyncio.sleep(.02)
        result = await task
        assert result.status == "ok"
        assert len(images) >= 5 and len(set(images.values())) >= 3
        assert len(poses) >= 8 and next(iter(poses.values())) != list(poses.values())[-1]
        assert observation_sequences == {initial_seq}
        assert result.observation.seq == initial_seq + 1
        assert len(worker.sim.frames) == 2
        assert len(worker.camera_frames) <= worker.camera_capacity
        assert worker.camera_frames[worker.latest["camera"]["frame_ref"]] == worker.sim.frame(result.observation.frame_ref)
        assert worker.latest["camera"]["simulated_time_s"] == 2
        assert time.monotonic() - started < 4
    finally:
        await worker.close()
    assert not worker.camera_frames


@pytest.mark.asyncio
async def test_manual_placement_updates_physics_and_camera_without_driving():
    worker = SimulationWorker(pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        original_pose = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client))
        original_frame = worker.camera_frames[initial["camera"]["frame_ref"]]
        placement = ManualPlacement(run_id=initial["run_id"], episode_epoch=initial["episode_epoch"],
                                    observation_seq=initial["observation"]["seq"], xy_m=[-.6, .3])
        updated = await worker.reposition(placement)
        position, orientation = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client))
        assert position[:2] == pytest.approx([-.6, .3])
        assert position[2] == original_pose[0][2] and orientation == original_pose[1]
        assert updated["observation"]["seq"] == initial["observation"]["seq"] + 1
        assert updated["observation"]["odometry_m_rad"] == initial["observation"]["odometry_m_rad"]
        assert updated["snapshot"]["simulated_time_s"] == 0
        assert updated["manual_placements"] == 1 and not updated["busy"]
        assert worker.camera_frames[updated["camera"]["frame_ref"]] != original_frame
        assert "xy_m" not in updated["observation"] and "manual_placements" not in updated["observation"]
        with pytest.raises(MotionError, match="state changed"):
            await worker.reposition(placement)
        for destination, code in [([2.5, 0], "COLLISION_BLOCKED"), ([3, 0], "OUT_OF_BOUNDS")]:
            rejected = placement.model_copy(update={"xy_m": destination, "observation_seq": updated["observation"]["seq"]})
            with pytest.raises(MotionError) as failure:
                await worker.reposition(rejected)
            assert failure.value.code == code
            assert worker.latest == updated
            assert await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)) == (position, orientation)
        worker.stop()
        with pytest.raises(MotionError, match="Resume manual"):
            await worker.reposition(placement.model_copy(update={"observation_seq": updated["observation"]["seq"]}))
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_stop_cancels_a_queued_manual_placement():
    worker = SimulationWorker(pace=False)
    entered, release = threading.Event(), threading.Event()
    try:
        await asyncio.wrap_future(worker.ready)
        initial = worker.latest
        placement = ManualPlacement(run_id=initial["run_id"], episode_epoch=initial["episode_epoch"],
                                    observation_seq=initial["observation"]["seq"], xy_m=[-.6, .3])

        def block_worker(sim):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Placement test worker was not released")

        blocker = asyncio.create_task(worker.call(block_worker))
        assert await asyncio.to_thread(entered.wait, 2)
        queued = asyncio.create_task(worker.reposition(placement))
        await asyncio.sleep(0)
        worker.stop()
        release.set()
        await blocker
        with pytest.raises(MotionError, match="Resume manual"):
            await queued
        assert worker.latest["snapshot"] == initial["snapshot"]
        assert worker.latest["manual_placements"] == 0
    finally:
        release.set()
        await worker.close()


@pytest.mark.asyncio
async def test_stop_interrupts_worker_and_reset_isolates_clients():
    worker = SimulationWorker(pace=True)
    await asyncio.wrap_future(worker.ready)
    envelope = Command(run_id=worker.sim.run_id, episode_epoch=0, action_id=str(uuid4()), observation_seq=worker.sim.seq,
                       tool="drive_base", arguments={"linear_mps": .2, "angular_radps": 0, "duration_s": 2})
    task = asyncio.create_task(worker.execute(envelope))
    await asyncio.sleep(.12)
    worker.stop()
    result = await asyncio.wait_for(task, 2)
    assert result.error == "CANCELLED"
    assert result.actual_duration_s < 2
    old_id = worker.sim.run_id
    await worker.close()
    replacement = SimulationWorker(epoch=1, pace=False)
    try:
        await asyncio.wrap_future(replacement.ready)
        assert replacement.sim.run_id != old_id
        result = await replacement.execute(envelope)
        assert result.error == "CANCELLED"
        assert replacement.sim.ticks == 0
    finally:
        await replacement.close()