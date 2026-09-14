import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from backend.challenges import get_challenge
from backend.navigation_supervisor import waypoint_instruction
from scripts.navigation_policy import ACTION_NAMES, EMBODIMENT, STATE_NAMES, apply, prepare, state_vector
from scripts.recovery_policy import execute_sample


def case_for(index):
    if index not in range(20):
        raise ValueError("Unknown task-skill case")
    variation = index // 4
    kind = index % 4
    offset = [-.08, 0., .08, -.04, .04][variation]
    identifier = ["park", "kitchen_bathroom", "recharge", "park"][kind]
    challenge = get_challenge(identifier).model_copy(deep=True)
    challenge.initial_head_pitch = .45
    if kind == 1:
        challenge.initial_xy = [1.1 + offset, -1.5]
        route = [([0., -1.5], 0), ([0., 1.4 + offset], 0), ([1.25, 1.4 + offset], 2)]
    elif kind == 2:
        clearance = -.95 + offset
        route = [([0., clearance], 0), ([1.45, clearance], 0), ([1.45, 0.], 2)]
    else:
        challenge.initial_xy = [-.1, offset]
        center = challenge.objectives[0].center[:2]
        route = [([-.4, offset], 0), ([-.4, .3 + offset], 0), (center, 2)] if kind == 3 else [(center, 2)]
    return challenge, route


def local_route(challenge, route):
    initial = np.array(challenge.initial_xy, dtype=float)
    previous = initial.copy()
    points = []
    for destination, holds in route:
        destination = np.array(destination, dtype=float)
        count = max(1, math.ceil(float(np.linalg.norm(destination - previous)) / .65))
        for part in range(1, count + 1):
            points.append(((previous + (destination - previous) * part / count - initial).tolist(),
                           holds if part == count else 0))
        previous = destination
    return points


def teacher_action(observation, target):
    delta_x, delta_y = target[0] - observation.odometry_m_rad[0], target[1] - observation.odometry_m_rad[1]
    distance = math.hypot(delta_x, delta_y)
    if distance < .045:
        return [0., 0.]
    bearing = math.atan2(delta_y, delta_x) - observation.odometry_m_rad[2]
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    reverse = abs(bearing) > 2.8 and distance < .7
    if reverse:
        bearing = math.atan2(math.sin(bearing - math.pi), math.cos(bearing - math.pi))
    speed = min(.15, distance * 1.6) * (-1 if reverse else 1)
    return [speed if abs(bearing) < .12 else 0., float(np.clip(2 * bearing, -.5, .5))]


def demonstration_success(sim):
    status = sim.challenge_status()
    return status["completed_objectives"] >= 1 if status["id"] == "recharge" else status["status"] == "completed"


def record_episode(sim, index, directory):
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    challenge, route = case_for(index)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "images").mkdir()
    frames = []
    result = {"success": False, "scripted_demonstration": True, "case": index, "error": None}
    try:
        for target, holds in local_route(challenge, route):
            apply(runtime, sim, "begin_local_subgoal", {"goal": "Scripted local waypoint demonstration"})
            stopped = 0
            for _ in range(45):
                observation = sim.observe()
                image = sim.frame(observation.frame_ref)
                action = teacher_action(observation, target)
                frame_index = len(frames)
                filename = f"images/{frame_index:04d}.png"
                (directory / filename).write_bytes(image)
                frames.append({"frame_index": frame_index, "timestamp": frame_index,
                    "observation.state": state_vector(observation), "observation.images.head": filename,
                    "image_sha256": hashlib.sha256(image).hexdigest(), "action": action,
                    "task": waypoint_instruction(observation, target)})
                execute_sample(sim, runtime, action)
                stopped = stopped + 1 if action == [0., 0.] else 0
                if stopped >= max(2, holds):
                    break
            else:
                raise ValueError("Teacher did not reach its local waypoint")
        result.update(success=demonstration_success(sim), frames=len(frames),
            full_task_success=sim.challenge_status()["status"] == "completed",
            training_goal="survey approach only" if index % 4 == 2 else "park fully inside destination",
            physics_status=sim.challenge_status(), odometry=sim.odometry.tolist())
        if not result["success"]:
            raise ValueError("Scripted task demonstration did not satisfy unchanged physics scorer")
        return result
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        metadata = {"curriculum": "task-waypoints-v1", "case": index, "embodiment": EMBODIMENT,
            "fps": 1, "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "split": "train" if index < 12 else "validation" if index < 16 else "evaluation_only",
            "frames": len(frames), "asset_sha256": hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest()}
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (directory / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (directory / "terminal.png").write_bytes(sim.capture())
        runtime.cancel(sim, "Task-skill recording ended")


def load_episode(directory):
    from PIL import Image
    metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    frames = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (metadata.get("curriculum") != "task-waypoints-v1" or metadata.get("case") not in range(16) or
            metadata.get("embodiment") != EMBODIMENT or metadata.get("fps") != 1 or
            metadata.get("state_names") != STATE_NAMES or metadata.get("action_names") != ACTION_NAMES or
            metadata.get("asset_sha256") != hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest() or
            not result.get("success") or result.get("error") or not frames or len(frames) != metadata.get("frames")):
        raise ValueError("Invalid task-waypoint recording")
    for index, frame in enumerate(frames):
        if (set(frame) != {"frame_index", "timestamp", "observation.state", "observation.images.head", "image_sha256", "action", "task"}
                or frame["frame_index"] != index or frame["timestamp"] != index or
                np.shape(frame["observation.state"]) != (20,) or not np.isfinite(frame["observation.state"]).all() or
                np.shape(frame["action"]) != (2,) or not np.isfinite(frame["action"]).all() or
                abs(frame["action"][0]) > .15 or abs(frame["action"][1]) > .5 or
                not frame["task"].startswith("Reach the local waypoint:")):
            raise ValueError("Invalid task-waypoint sample or private fields")
        path = (directory / frame["observation.images.head"]).resolve(strict=True)
        if not path.is_relative_to((directory / "images").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise ValueError("Task-waypoint image path/hash mismatch")
        with Image.open(path) as image:
            if image.size != (320, 240) or image.mode != "RGB":
                raise ValueError("Task-waypoint image contract mismatch")
    return {**metadata, "episode": metadata["case"]}, frames


def replay_episode(sim, frames):
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    maximum = 0.
    try:
        for frame in frames:
            actual = state_vector(sim.observe(render=False))
            maximum = max(maximum, float(np.max(np.abs(np.array(actual) - frame["observation.state"]))))
            apply(runtime, sim, "begin_local_subgoal", {"goal": "Saved-action replay, not learned control"})
            execute_sample(sim, runtime, frame["action"])
        return {"verified": maximum < 1e-5 and demonstration_success(sim),
                "max_state_error": maximum, "frames": len(frames)}
    finally:
        runtime.cancel(sim, "Task-skill replay ended")


async def record(options):
    from backend.worker import SimulationWorker
    options.output.mkdir(parents=True, exist_ok=False)
    results = []
    for index in options.cases:
        directory = options.output / f"episode-{index:03d}"
        challenge, _ = case_for(index)
        worker = SimulationWorker(challenge=challenge, pace=False)
        try:
            result = await worker.call(lambda sim: record_episode(sim, index, directory))
        finally:
            await worker.close()
        _, frames = load_episode(directory)
        worker = SimulationWorker(challenge=challenge, pace=False)
        try:
            replay = await worker.call(lambda sim: replay_episode(sim, frames))
            replay["frames_sha256"] = hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest()
            (directory / "replay.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
            if not replay["verified"]:
                raise ValueError("Task-skill replay diverged")
        finally:
            await worker.close()
        results.append({**result, "replay": replay})
        print(json.dumps(results[-1]), flush=True)
    (options.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


async def evaluate(options):
    from backend.contracts import Command
    from backend.local_navigation import LocalNavigationClient
    from backend.worker import SimulationWorker
    options.output.mkdir(parents=True, exist_ok=False)
    policy = LocalNavigationClient(options.checkpoint)
    results = []
    try:
        await policy.start()
        for index in options.cases:
            challenge, route = case_for(index)
            directory = options.output / f"case-{index:03d}"
            directory.mkdir()
            worker = SimulationWorker(challenge=challenge, pace=True)
            report = {"case": index, "challenge": challenge.id, "checkpoint": str(options.checkpoint),
                "teacher_waypoints": True, "autonomous_task_success": False, "real_smolvla": True,
                "qualification": "Teacher-provided local goals; evaluates execution, not autonomous route discovery",
                "success": False, "error": None, "requests": [], "waypoints_completed": 0}
            began = time.monotonic()
            try:
                await policy.reset()
                await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
                observation, image = await worker.feedback()
                policy.instruction = waypoint_instruction(observation, local_route(challenge, route)[0][0])
                await policy.predict(observation, image)
                await worker.begin_navigation(worker.stop_revision)
                await worker.call(lambda sim: prepare(sim, worker.navigation))
                for target, holds in local_route(challenge, route):
                    await worker.call(lambda sim: apply(worker.navigation, sim, "begin_local_subgoal",
                        {"goal": "Controlled teacher waypoint benchmark"}))
                    stop_votes = 0
                    for _ in range(40):
                        observation, image = await worker.feedback()
                        policy.instruction = waypoint_instruction(observation, target)
                        started = time.monotonic()
                        reply = await policy.predict(observation, image)
                        if not 0 <= time.time() - observation.wall_timestamp <= 2:
                            raise ValueError("STALE_INFERENCE: local prediction exceeded two-second freshness")
                        action = reply["action"]
                        stopped = abs(action[0]) < .025 and abs(action[1]) < .05
                        action = [0., 0.] if stopped else action
                        entry = {"instruction": policy.instruction, "observation": observation.model_dump(),
                            **reply, "executed_action": action, "inference_s": time.monotonic() - started}
                        report["requests"].append(entry)
                        (directory / f"frame-{len(report['requests']):04d}.png").write_bytes(image)
                        result = await worker.execute_navigation(Command(run_id=observation.run_id,
                            episode_epoch=observation.episode_epoch, observation_seq=observation.seq, action_id=str(uuid4()),
                            tool="replace_motion_buffer", arguments={"expected_revision": observation.navigation.revision,
                                "segments": [{"kind": "drive", "linear_mps": action[0], "angular_radps": action[1], "duration_s": 1.}]}))
                        if result.status != "ok":
                            raise ValueError(f"{result.error}: {result.message}")
                        async with asyncio.timeout(4):
                            while worker.latest["navigation"]["remaining_s"] > 0:
                                await asyncio.sleep(.05)
                        if worker.latest["navigation"]["status"] == "failed":
                            raise ValueError(worker.latest["navigation"]["reason"])
                        current, _ = await worker.feedback()
                        entry["remaining_m"] = math.dist(current.odometry_m_rad[:2], target)
                        stop_votes = stop_votes + 1 if stopped else 0
                        if stop_votes >= max(2, holds):
                            if entry["remaining_m"] > .07:
                                raise ValueError("PREMATURE_STOP: model stopped outside its local waypoint")
                            report["waypoints_completed"] += 1
                            break
                    else:
                        raise ValueError("WAYPOINT_LIMIT: local policy did not finish its waypoint")
                report["success"] = await worker.call(demonstration_success)
            except Exception as error:
                report["error"] = str(error)
            finally:
                worker.stop()
                observation, image = await worker.feedback()
                report.update(wall_s=time.monotonic() - began, final_odometry=observation.odometry_m_rad,
                    physics_status=worker.latest["challenge"], manual_placements=worker.latest["manual_placements"],
                    full_task_physics_success=worker.latest["challenge"]["status"] == "completed")
                (directory / "terminal.png").write_bytes(image)
                (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                results.append({key: value for key, value in report.items() if key != "requests"} | {"requests": len(report["requests"])})
                print(json.dumps(results[-1]), flush=True)
                await worker.close()
    finally:
        await policy.close()
        (options.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay-verified local waypoint demonstrations in existing task scenes")
    parser.add_argument("--stage", choices=["record", "export", "evaluate"], default="record")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--retain-primitives", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, nargs="+", choices=range(20), default=list(range(16)))
    options = parser.parse_args()
    if options.stage == "export":
        if options.source is None:
            parser.error("Export requires --source")
        from scripts.recovery_policy import export
        def mixed_episode(directory):
            metadata, frames = load_episode(directory)
            if options.retain_primitives:
                from scripts.recovery_policy import load_episode as load_primitives
                primitive_directory = options.retain_primitives / directory.name
                _, primitives = load_primitives(primitive_directory)
                replay = json.loads((primitive_directory / "replay.json").read_text(encoding="utf-8"))
                if not replay.get("verified") or replay["frames_sha256"] != hashlib.sha256((primitive_directory / "frames.json").read_bytes()).hexdigest():
                    raise ValueError("Retention examples require exact replay verification")
                for frame in primitives:
                    frames.append({**frame, "observation.images.head": str((primitive_directory / frame["observation.images.head"]).resolve())})
            return metadata, frames
        export(options, loader=mixed_episode, curriculum="task-waypoints-v1-with-primitive-retention")
    elif options.stage == "evaluate":
        if options.checkpoint is None:
            parser.error("Evaluation requires --checkpoint")
        asyncio.run(evaluate(options))
    else:
        if any(index >= 16 for index in options.cases):
            parser.error("Evaluation-only cases cannot be recorded for training")
        asyncio.run(record(options))