import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pybullet as bullet

from backend.policy import ACTION_NAMES, PolicyChunk, PolicyMetadata, SkillRuntime, StartSkill
from backend.smolvla_server import state_vector
from backend.simulation import MotionError
from backend.worker import SimulationWorker
from scripts.milo_dataset import ALIGNMENT, load_episode


TASK = "Pick up the red cube and place it on the green square with the left arm."


def scene_for(cube_xy):
    return [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "cube", "size": [.06, .06, .06], "position": [*cube_xy, .031],
         "color": [.9, .1, .2, 1], "mass": .08},
        {"name": "destination", "size": [.13, .13, .002], "position": [.36, .34, .001],
         "color": [.1, .7, .3, 1], "marker": True},
    ]


def smooth_targets(initial, target, duration_s):
    count = round(duration_s * 20)
    for index in range(1, count + 1):
        fraction = index / count
        blend = 10 * fraction ** 3 - 15 * fraction ** 4 + 6 * fraction ** 5
        yield (initial + (target - initial) * blend).tolist()


def demonstration_actions(sim, cube_xy):
    current = np.array([*state_vector(sim.observe(render=False))])
    actions = []
    prior_position = None
    for position, opening, duration in [
        ([*cube_xy, .08], .11, 6),
        ([*cube_xy, -.12], .11, 4),
        (None, 0., 4),
        ([*cube_xy, .08], 0., 4),
        ([.36, .34, .08], 0., 4),
        ([.36, .34, -.12], 0., 4),
        (None, .11, 4),
        ([.36, .34, .08], .11, 4),
        (None, .11, 2),
    ]:
        if position is not None and prior_position is not None and position[2] != prior_position[2]:
            for point in smooth_targets(np.array(prior_position), np.array(position), duration):
                joints = sim.inverse_kinematics("left", point, [0, 0, 0, 1], seed=current[:6])
                current = np.array([*joints, opening])
                actions.append(current.tolist())
        else:
            joints = sim.inverse_kinematics("left", position, [0, 0, 0, 1], seed=current[:6]) if position else current[:6]
            target = np.array([*joints, opening])
            actions.extend(smooth_targets(current, target, duration))
            current = target
        if position is not None:
            prior_position = position
    return actions


class EpisodeRecorder:
    def __init__(self, directory, metadata):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "images").mkdir()
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.stream = (directory / "frames.jsonl").open("w", encoding="utf-8")
        self.count = 0

    def append(self, observation, image, action):
        state = state_vector(observation)
        filename = f"images/{self.count:06d}.png"
        (self.directory / filename).write_bytes(image)
        self.stream.write(json.dumps({"frame_index": self.count, "timestamp": self.count / 20,
            "simulated_time_s": observation.simulated_time_s, "observation_wall_timestamp": observation.wall_timestamp,
            "observation.state": state, "observation.images.head": filename,
            "image_sha256": hashlib.sha256(image).hexdigest(), "action": action, "task": TASK}, allow_nan=False) + "\n")
        self.stream.flush()
        self.count += 1

    def close(self, result):
        self.stream.close()
        (self.directory / "result.json").write_text(json.dumps({**result, "frames": self.count}, indent=2), encoding="utf-8")


def execute_episode(sim, cube_xy, directory=None, actions=None, expected_frames=None):
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    for name, value in (("head_yaw", 0.), ("head_pitch", .7)):
        bullet.resetJointState(sim.robot, sim.joints[name], value, physicsClientId=sim.client)
        sim.targets[sim.joints[name]] = (value, 5)
    clock = lambda: sim.ticks / 240
    metadata = PolicyMetadata(backend="smolvla", checkpoint="scripted-demonstrator-NOT-trained-weights",
        embodiment="milo-left-arm-v1", action_names=ACTION_NAMES, camera_key="observation.images.head",
        fps=20, trained_for_milo=True, skills=["pick_place"])
    runtime = SkillRuntime(metadata, clock=clock)
    runtime.apply(sim, "start_skill", StartSkill(expected_revision=0, skill="pick_place", instruction=TASK, timeout_s=60))
    actions = demonstration_actions(sim, cube_xy) if actions is None else actions
    recorder = EpisodeRecorder(directory, {"format": "milo-demonstrations-v1", "fps": 20,
        "action_names": ACTION_NAMES, "state_names": ACTION_NAMES, "task": TASK,
        "source": "scripted IK using privileged target positions, not a learned policy",
        "clock": "offline fixed simulation clock; no inference latency or wall-clock deadline claim",
        "action_alignment": ALIGNMENT,
        "robot_asset_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "assets/milo.urdf").read_bytes()).hexdigest(),
        "head_rad": [0., .7], "grasp_assistance": False, "fixture": {"cube_xy": cube_xy}, "camera_size": [320, 240]}) if directory else None
    cube = next(item["id"] for item in sim.objects if item["name"] == "cube")
    peak_bottom = -1.
    contact_lift = False
    result = {"success": False, "executed_frames": 0, "error": None, "replay_state_max_error": 0. if expected_frames else None}
    started = time.perf_counter()
    try:
        for index, action in enumerate(actions):
            if expected_frames:
                state = np.array(state_vector(sim.observe(render=False)))
                error = float(np.max(np.abs(state - expected_frames[index]["observation.state"])))
                result["replay_state_max_error"] = max(result["replay_state_max_error"], error)
            if index % 10 == 0:
                observation = sim.observe(render=False)
                ticket = runtime.ticket(sim, observation)
                runtime.accept(sim, PolicyChunk(ticket=ticket, actions=actions[index:index + 20]))
            if recorder:
                observation = sim.observe()
                recorder.append(observation, sim.frame(observation.frame_ref), action)
            for _ in range(12):
                runtime.tick(sim)
                if runtime.status == "failed":
                    raise ValueError(runtime.reason)
                if bullet.getNumConstraints(physicsClientId=sim.client) or sim.held:
                    raise ValueError("Assisted attachment detected")
            bottom = bullet.getAABB(cube, physicsClientId=sim.client)[0][2]
            peak_bottom = max(peak_bottom, bottom)
            contacts = [bullet.getContactPoints(sim.robot, cube, linkIndexA=finger, physicsClientId=sim.client)
                        for finger in sim.fingers["left"]]
            contact_lift = contact_lift or (bottom > .08 and all(any(contact[9] > .02 for contact in group) for group in contacts))
            result["executed_frames"] += 1
        lower, upper = bullet.getAABB(cube, physicsClientId=sim.client)
        position = bullet.getBasePositionAndOrientation(cube, physicsClientId=sim.client)[0]
        speed = np.linalg.norm(bullet.getBaseVelocity(cube, physicsClientId=sim.client)[0])
        placed = all(lower[axis] >= center - .065 and upper[axis] <= center + .065
                     for axis, center in enumerate((.36, .34)))
        floor = next(item["id"] for item in sim.objects if item["name"] == "floor")
        grounded = any(contact[9] > .01 for contact in bullet.getContactPoints(cube, floor, physicsClientId=sim.client))
        result.update(success=bool(contact_lift and placed and grounded and speed < .025),
                      final_cube_position=list(position), final_speed_mps=float(speed), placed=bool(placed), grounded=grounded)
    except (ValueError, RuntimeError, MotionError) as error:
        result["error"] = str(error)
        result["failed_frame"] = index
        result["failed_action"] = action
    finally:
        result.update(peak_cube_bottom_m=float(peak_bottom), opposing_contact_lift=bool(contact_lift),
                      elapsed_wall_s=time.perf_counter() - started, simulated_s=sim.ticks / 240,
                      clock="simulation", grasp_assistance=False)
        if recorder:
            observation = sim.observe()
            (directory / "terminal.png").write_bytes(sim.frame(observation.frame_ref))
            (directory / "terminal.json").write_text(json.dumps({"observation.state": state_vector(observation),
                "simulated_time_s": observation.simulated_time_s}), encoding="utf-8")
            recorder.close(result)
        runtime.cancel(sim, "Isolated demonstration ended")
    return result


async def main(options):
    options.output.mkdir(parents=True, exist_ok=True)
    results = []
    for episode, cube_xy in enumerate(([.36, .25], [.35, .25], [.37, .25], [.36, .24], [.36, .26])[:options.episodes]):
        expected_frames = None
        directory = options.output / f"episode-{episode:03d}"
        if options.replay:
            metadata, expected_frames, _ = load_episode(directory)
            cube_xy = metadata["fixture"]["cube_xy"]
            actual_hash = hashlib.sha256((Path(__file__).resolve().parents[1] / "assets/milo.urdf").read_bytes()).hexdigest()
            if actual_hash != metadata["robot_asset_sha256"]:
                raise ValueError("Robot asset changed since recording")
        worker = SimulationWorker(scene=scene_for(cube_xy), pace=False)
        try:
            result = await worker.call(lambda sim: execute_episode(sim, cube_xy,
                None if options.probe or options.replay else directory,
                actions=[frame["action"] for frame in expected_frames] if expected_frames else None,
                expected_frames=expected_frames))
            if options.replay:
                result.update(source_frames_sha256=hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest(),
                              robot_asset_sha256=actual_hash)
                result["success"] = result["success"] and result["replay_state_max_error"] <= .001
                (directory / "replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append({"episode": episode, "cube_xy": cube_xy, **result})
            print(json.dumps(results[-1]), flush=True)
        finally:
            await worker.close()
    filename = "replay-summary.json" if options.replay else "probe-summary.json" if options.probe else "summary.json"
    (options.output / filename).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record isolated physical-only Milo demonstrations through the validated policy executor")
    parser.add_argument("--output", type=Path, default=Path(".runtime/milo-demonstrations"))
    parser.add_argument("--episodes", type=int, choices=range(1, 6), default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true", help="Check mechanics without rendering or saving training images")
    mode.add_argument("--replay", action="store_true", help="Replay saved numeric actions in a fresh isolated worker; never regenerate them with IK")
    asyncio.run(main(parser.parse_args()))