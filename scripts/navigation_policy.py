import argparse
import asyncio
import base64
from contextlib import redirect_stdout
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np

from backend.challenges import Challenge, Objective


EMBODIMENT = "milo-navigation-v1"
ACTION_NAMES = ["linear_mps", "angular_radps"]
DIRECTIONS = ["front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right"]
STATE_NAMES = ["head_yaw_rad", "head_pitch_rad", "wheel_linear_mps", "wheel_angular_radps",
               *[f"distance_{name}_m" for name in DIRECTIONS], *[f"range_status_{name}" for name in DIRECTIONS]]
TASK = "Drive into the green floor bay and stop. Keep clear of the posts and walls."
FPS = 1
CHUNK_SIZE = 1
CASES = [
    {"id": index, "target": [distance, lateral], "doorway": doorway, "split": split}
    for index, (distance, lateral, doorway, split) in enumerate([
        (.95, -.55, False, "train"), (.95, .55, False, "train"), (1.2, -.35, False, "train"),
        (1.2, .35, False, "train"), (1.5, 0., False, "train"), (.95, 0., False, "train"),
        (1.4, -.3, True, "train"), (1.4, .3, True, "train"), (1.6, 0., True, "train"),
        (1.3, -.15, True, "train"), (1.3, .15, True, "train"), (1.5, 0., True, "train"),
        (1.1, -.45, False, "validation"), (1.1, .45, False, "validation"),
        (1.45, -.22, True, "validation"), (1.45, .22, True, "validation"),
    ])]

EVALUATION_CASES = [
    {"id": index, "target": [distance, lateral], "doorway": doorway, "split": "evaluation_only"}
    for index, (distance, lateral, doorway) in enumerate([
        (1.05, -.40, False), (1.05, .40, False), (1.55, -.18, True), (1.55, .18, True),
    ], start=len(CASES))]


def state_vector(observation):
    joints = {joint.name: joint for joint in observation.joints}
    left, right = joints["left_wheel"].velocity, joints["right_wheel"].velocity
    readings = {reading.direction: reading for reading in observation.proximity.distances}
    values = [*observation.head_rad, (left + right) * .09 / 2, (right - left) * .09 / .38]
    values.extend(readings[name].distance_m if readings[name].status == "hit" else
                  2. if readings[name].status == "clear" else 0. for name in DIRECTIONS)
    values.extend({"hit": 1., "clear": 0., "occluded": -1.}[readings[name].status] for name in DIRECTIONS)
    if len(values) != len(STATE_NAMES) or not np.isfinite(values).all():
        raise ValueError("Navigation requires finite wheel, head and eight-range observations")
    return values


def challenge_for(case):
    return Challenge(id="park", title="Local navigation pilot", category="Navigation", skill="Green-bay docking",
        goal=TASK, initial_head_pitch=.45, objects=[
            {"name": "back_wall", "size": [.1, 5.8, 1.2], "position": [2.9, 0, .6], "color": [.68, .73, .73, 1]},
            {"name": "north_wall", "size": [5.8, .1, 1.2], "position": [0, 2.9, .6], "color": [.78, .80, .77, 1]},
            {"name": "south_wall", "size": [5.8, .1, 1.2], "position": [0, -2.9, .6], "color": [.78, .80, .77, 1]},
            *([{"name": f"door_post_{side}", "size": [.14, .15, 1.1], "position": [.65, sign * 1.05, .55],
                "color": [.89, .60, .15, 1]} for side, sign in (("left", 1), ("right", -1))] if case["doorway"] else []),
        ], objectives=[Objective(label="Park fully in the green bay", body="robot", center=[*case["target"], 0],
                                 size=[.85, .9], color=[.08, .72, .25, 1], require_lift=False)])


def apply(runtime, sim, name, arguments):
    from backend.navigation import CompleteNavigationSkill, NavigationPlan, ReplaceMotionBuffer
    observation = sim.observe(render=False)
    runtime.observe(sim, observation)
    models = {"set_navigation_plan": NavigationPlan, "replace_motion_buffer": ReplaceMotionBuffer,
              "complete_navigation_skill": CompleteNavigationSkill}
    runtime.apply(sim, name, models[name](expected_revision=runtime.revision, **arguments), observation.seq)


def prepare(sim, runtime):
    apply(runtime, sim, "set_navigation_plan", {"steps": [
        {"skill": skill, "goal": "Local navigation pilot: inspect, locate and approach the green bay"}
        for skill in ("inspect_room", "locate_doorway", "approach", "cross")]})
    for _ in range(2):
        apply(runtime, sim, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": yaw, "pitch_rad": .45, "duration_s": .6} for yaw in (-.5, .5, 0)]})
        while runtime.buffer:
            runtime.tick(sim)
        if runtime.status == "failed":
            raise ValueError(runtime.reason)
        apply(runtime, sim, "complete_navigation_skill", {"evidence": "Scripted preparation scan; no learned semantic claim"})


def command_for(sim, case):
    import pybullet as bullet
    if sim.challenge_status()["status"] == "completed":
        return [0., 0.]
    position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
    delta = np.array(case["target"]) - position[:2]
    heading = bullet.getEulerFromQuaternion(orientation)[2]
    angle = math.atan2(math.sin(math.atan2(delta[1], delta[0]) - heading), math.cos(math.atan2(delta[1], delta[0]) - heading))
    return [0. if abs(angle) > .25 else .15, float(np.clip(angle * 1.5, -.5, .5))]


def record_episode(sim, case, directory):
    import pybullet as bullet
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "images").mkdir()
    frames = []
    start_ticks = sim.ticks
    result = {"success": False, "case": case, "error": None, "scripted_demonstration": True,
              "clock": "offline simulation time; not wall-clock control performance"}
    try:
        stopped_frames = 0
        for index in range(50):
            observation = sim.observe()
            image = sim.frame(observation.frame_ref)
            action = command_for(sim, case)
            filename = f"images/{index:04d}.png"
            (directory / filename).write_bytes(image)
            frames.append({"frame_index": index, "timestamp": (sim.ticks - start_ticks) / 240,
                "observation.state": state_vector(observation), "observation.images.head": filename,
                "image_sha256": hashlib.sha256(image).hexdigest(), "action": action, "task": TASK})
            apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive", "linear_mps": action[0],
                "angular_radps": action[1], "duration_s": 1.}]})
            before = sim.ticks
            while runtime.buffer:
                runtime.tick(sim)
            if runtime.status == "failed" or sim.ticks - before != 240:
                raise ValueError(runtime.reason)
            stopped_frames = stopped_frames + 1 if action == [0., 0.] else 0
            if stopped_frames == 2:
                result["success"] = sim.challenge_status()["status"] == "completed"
                break
        result.update(frames=len(frames), travel_m=sim.path_length,
                      final_position=bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0],
                      final_progress=sim.challenge_status()["status"])
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (directory / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        (directory / "episode.json").write_text(json.dumps({"embodiment": EMBODIMENT, "fps": FPS, "state_names": STATE_NAMES,
            "action_names": ACTION_NAMES, "case": case, "alignment": "head/state before one-second drive segment",
            "asset_sha256": hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
        (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (directory / "terminal.png").write_bytes(sim.capture())
        runtime.cancel(sim, "Isolated recording finished")
    if not result["success"]:
        raise ValueError("Demonstration did not park successfully")
    return result


async def record(options):
    from backend.worker import SimulationWorker
    options.output.mkdir(parents=True, exist_ok=False)
    results = []
    for case in CASES[:options.episodes]:
        worker = SimulationWorker(challenge=challenge_for(case), pace=False)
        try:
            result = await worker.call(lambda sim: record_episode(sim, case, options.output / f"episode-{case['id']:03d}"))
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            await worker.close()
    (options.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def load_episode(directory):
    from PIL import Image
    from backend.contracts import Drive
    metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    frames = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    if (metadata.get("embodiment") != EMBODIMENT or metadata.get("fps") != FPS or
            metadata.get("action_names") != ACTION_NAMES or metadata.get("state_names") != STATE_NAMES or
            metadata.get("alignment") != "head/state before one-second drive segment" or
            metadata.get("case") not in CASES or not result.get("success") or result.get("error") is not None or
            not frames or len(frames) != result.get("frames") or
            metadata.get("asset_sha256") != hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest()):
        raise ValueError("Invalid successful navigation recording contract")
    fields = {"frame_index", "timestamp", "observation.state", "observation.images.head", "image_sha256", "action", "task"}
    for index, frame in enumerate(frames):
        if (set(frame) != fields or frame["frame_index"] != index or frame["timestamp"] != index or frame["task"] != TASK or
                np.shape(frame["observation.state"]) != (len(STATE_NAMES),) or not np.isfinite(frame["observation.state"]).all() or
                np.shape(frame["action"]) != (2,)):
            raise ValueError("Invalid navigation sample alignment or private fields")
        action = Drive(linear_mps=frame["action"][0], angular_radps=frame["action"][1], duration_s=1)
        if abs(action.linear_mps) > .15 or abs(action.angular_radps) > .5:
            raise ValueError("Navigation sample exceeds buffered movement limits")
        path = (directory / frame["observation.images.head"]).resolve(strict=True)
        if not path.is_relative_to((directory / "images").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise ValueError("Navigation camera path or hash mismatch")
        with Image.open(path) as image:
            if image.size != (320, 240) or image.mode != "RGB":
                raise ValueError("Navigation camera shape mismatch")
            image.verify()
    return metadata, frames


def replay_episode(sim, metadata, frames):
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    errors = []
    try:
        for frame in frames:
            actual = state_vector(sim.observe(render=False))
            errors.append(float(np.max(np.abs(np.array(actual) - frame["observation.state"]))))
            linear, angular = frame["action"]
            apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive", "linear_mps": linear,
                "angular_radps": angular, "duration_s": 1.}]})
            before = sim.ticks
            while runtime.buffer:
                runtime.tick(sim)
            if runtime.status == "failed" or sim.ticks - before != 240:
                raise ValueError(runtime.reason)
        return {"success": sim.challenge_status()["status"] == "completed" and max(errors) < 1e-5,
                "max_state_error": max(errors), "frames": len(frames)}
    finally:
        runtime.cancel(sim, "Saved action replay finished")


async def replay(options):
    from backend.worker import SimulationWorker
    for directory in sorted(options.source.glob("episode-*")):
        metadata, frames = load_episode(directory)
        worker = SimulationWorker(challenge=challenge_for(metadata["case"]), pace=False)
        try:
            result = await worker.call(lambda sim: replay_episode(sim, metadata, frames))
            result["frames_sha256"] = hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest()
            (directory / "replay.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"episode": directory.name, **result}), flush=True)
            if not result["success"]:
                raise ValueError("Saved navigation actions failed deterministic replay")
        finally:
            await worker.close()


def export(options):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    directories = sorted(options.source.glob("episode-*"))
    if len(directories) != len(CASES) or options.output.exists():
        raise ValueError("Export requires all 16 replayed cases and a new output directory")
    records = []
    for directory in directories:
        metadata, frames = load_episode(directory)
        result = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
        if not result.get("success") or result.get("frames_sha256") != hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest():
            raise ValueError("Export requires a matching successful replay")
        records.append((directory, metadata, frames))
    if [metadata["case"]["id"] for _, metadata, _ in records] != list(range(len(CASES))):
        raise ValueError("Export requires each fixed case exactly once in episode order")
    dataset = LeRobotDataset.create(repo_id="local/milo-navigation", root=options.output, robot_type=EMBODIMENT,
        fps=FPS, use_videos=False, features={
            "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
            "observation.images.head": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}})
    try:
        for directory, metadata, frames in records:
            for frame in frames:
                with Image.open(directory / frame["observation.images.head"]) as image:
                    pixels = np.array(image, dtype=np.uint8)
                dataset.add_frame({"observation.state": np.array(frame["observation.state"], dtype=np.float32),
                    "action": np.array(frame["action"], dtype=np.float32), "observation.images.head": pixels, "task": TASK})
            dataset.save_episode()
    finally:
        dataset.finalize()
    loaded = LeRobotDataset("local/milo-navigation", root=options.output, video_backend="pyav", token=False)
    offset = 0
    for directory, metadata, frames in records:
        for frame in frames:
            sample = loaded[offset]
            for key in ("action", "observation.state"):
                if not np.array_equal(np.asarray(sample[key]), np.array(frame[key], dtype=np.float32)):
                    raise ValueError("Exported navigation values mismatch")
            with Image.open(directory / frame["observation.images.head"]) as image:
                pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
            if not np.array_equal(np.asarray(sample["observation.images.head"]), pixels):
                raise ValueError("Exported navigation pixels mismatch")
            offset += 1
    summary = {"verified": True, "frames": offset, "embodiment": EMBODIMENT, "fps": FPS,
               "train_episodes": [case["id"] for case in CASES if case["split"] == "train"],
               "validation_episodes": [case["id"] for case in CASES if case["split"] == "validation"],
               "source_hashes": {directory.name: hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest()
                                 for directory, _, _ in records}, "uploaded": False}
    (options.output / "navigation-export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def load_checkpoint(checkpoint):
    checkpoint = checkpoint.resolve(strict=True)
    report = json.loads((checkpoint.parent / "report.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if (report.get("status") != "trained_and_reloaded" or report.get("embodiment") != EMBODIMENT or
            report.get("weights_sha256") != digest or report.get("state_names") != STATE_NAMES or
            report.get("action_names") != ACTION_NAMES or report.get("reloaded_output", {}).get("shape") != [1, 1, 2]):
        raise ValueError("A verified navigation-specific training checkpoint is required")
    return checkpoint, report


def bounded_velocity(prediction):
    action = np.asarray(prediction, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Expected two finite navigation velocity outputs")
    bounded = np.clip(action, [-.15, -.5], [.15, .5])
    return bounded.tolist(), [name for name, requested, value in zip(ACTION_NAMES, action, bounded) if requested != value]


def inference(options):
    with redirect_stdout(sys.stderr):
        from PIL import Image
        import torch
        from backend.smolvla_server import PolicyRequest
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        checkpoint, _ = load_checkpoint(options.checkpoint)
        torch.set_num_threads(8)
        torch.manual_seed(options.seed)
        policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True).float().eval()
        config = policy.config
        if (config.device != "cuda" or config.chunk_size != 1 or config.n_obs_steps != 1 or
                tuple(config.input_features["observation.state"].shape) != (20,) or
                tuple(config.output_features["action"].shape) != (2,) or
                set(config.image_features) != {"observation.images.head"}):
            raise ValueError("The saved model is not a compatible navigation policy")
        preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    print(json.dumps({"ready": True, "device": torch.cuda.get_device_name(0)}), flush=True)
    for line in sys.stdin:
        with redirect_stdout(sys.stderr):
            request = PolicyRequest.model_validate_json(line)
            with Image.open(BytesIO(base64.b64decode(request.image, validate=True))) as image:
                if image.size != (320, 240):
                    raise ValueError("Expected the synchronized 320x240 head image")
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255
            frame = {"observation.state": torch.tensor(state_vector(request.observation), dtype=torch.float32),
                "observation.images.head": torch.from_numpy(pixels).permute(2, 0, 1), "task": request.instruction}
            policy.reset()
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(preprocess(frame))
                action = postprocess(prediction[:, 0])[0].cpu().tolist()
            if np.shape(action) != (2,) or not np.isfinite(action).all():
                raise ValueError("Nonfinite navigation policy output")
        print(json.dumps({"ticket": request.ticket.model_dump(), "action": action}), flush=True)


def evaluation_state(sim):
    import pybullet as bullet
    position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
    return {"position": list(position), "heading": bullet.getEulerFromQuaternion(orientation)[2],
            "travel_m": sim.path_length, "simulated_s": sim.ticks / 240,
            "parked": sim.challenge_status()["status"] == "completed",
            "contacts": sim.proximity_sensors().model_dump()["collisions"]}


async def evaluate(options):
    from backend.local_progress import track_navigation_progress
    async with track_navigation_progress(options.checkpoint, options.case, options.requests) as progress:
        report = await evaluate_with_progress(options, progress)
        progress.update(phase="completed", success=report["success"], outcome=report["status"])
        return report


async def evaluate_with_progress(options, progress):
    from backend.contracts import Command
    from backend.policy import PolicyTicket
    from backend.smolvla_server import PolicyRequest
    from backend.worker import SimulationWorker
    from scripts.evaluate_milo import visual_evidence

    checkpoint, trained = load_checkpoint(options.checkpoint)
    options.output.mkdir(parents=True, exist_ok=False)
    case = (CASES + EVALUATION_CASES)[options.case]
    report = {"case": case, "checkpoint": str(checkpoint), "weights_sha256": trained["weights_sha256"],
              "training_steps": trained.get("total_steps", trained.get("steps")), "balance_stops": trained.get("balance_stops", False),
              "seed": options.seed, "status": "running", "requests": [], "success": False,
              "controller": "Learned velocity targets, saturation logged; unchanged NavigationRuntime buffers/safety",
              "stop_rule": "Two consecutive predictions with abs(linear)<0.025 and abs(angular)<0.05; each executes zero velocity for 1 s",
              "preparation": "Two scripted head scans advance to approach; policy controls only subsequent driving",
              "inference": "Single fresh paired head camera and measured state; no target coordinates, private map, or evaluator data"}
    worker = None
    process = None
    began = time.perf_counter()
    with (options.output / "inference.log").open("w", encoding="utf-8") as log:
        try:
            progress.update(phase="loading")
            process = await asyncio.create_subprocess_exec(str(options.python.resolve(strict=True)), "-u", "-m", "scripts.navigation_policy",
                "--stage", "inference", "--checkpoint", str(checkpoint), "--seed", str(options.seed),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=2_500_000)
            ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 120))
            if not ready.get("ready"):
                raise RuntimeError("Navigation inference worker failed to start")
            progress.update(phase="warming")
            worker = SimulationWorker(challenge=challenge_for(case), pace=True)
            await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))

            async def predict(observation, image):
                revision = observation.navigation.revision if observation.navigation else 0
                ticket = PolicyTicket(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
                                      revision=revision, observation_seq=observation.seq)
                payload = PolicyRequest(ticket=ticket, observation=observation,
                    image=base64.b64encode(image).decode("ascii"), instruction=TASK)
                process.stdin.write((payload.model_dump_json() + "\n").encode())
                await process.stdin.drain()
                reply = json.loads(await asyncio.wait_for(process.stdout.readline(), 10))
                if reply.get("ticket") != ticket.model_dump():
                    raise ValueError("Inference replied to the wrong observation")
                return reply

            observation, image = await worker.feedback()
            await predict(observation, image)
            await worker.begin_navigation(worker.stop_revision)
            await worker.call(lambda sim: prepare(sim, worker.navigation))
            report["initial"] = await worker.call(evaluation_state)
            stop_votes = 0
            progress.update(phase="running")
            for index in range(options.requests):
                observation, image = await worker.feedback()
                (options.output / f"frame-{index:03d}.png").write_bytes(image)
                captured = time.perf_counter()
                reply = await predict(observation, image)
                elapsed = time.perf_counter() - captured
                action, saturated = bounded_velocity(reply["action"])
                entry = {"index": index, "observation": observation.model_dump(), "raw_action": reply["action"],
                         "bounded_action": action, "saturated_axes": saturated, "inference_s": elapsed,
                         "observation_age_s": time.time() - observation.wall_timestamp}
                report["requests"].append(entry)
                if entry["observation_age_s"] > 2 or entry["observation_age_s"] < 0:
                    report["status"] = "stale_inference"
                    break
                model_stopped = abs(action[0]) < .025 and abs(action[1]) < .05
                stop_votes = stop_votes + 1 if model_stopped else 0
                entry["model_stop_vote"] = model_stopped
                execution = [0., 0.] if model_stopped else action
                entry["executed_action"] = execution
                result = await worker.execute_navigation(Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
                    observation_seq=observation.seq, action_id=str(uuid4()), tool="replace_motion_buffer",
                    arguments={"expected_revision": observation.navigation.revision, "segments": [
                        {"kind": "drive", "linear_mps": execution[0], "angular_radps": execution[1], "duration_s": 1.}]}))
                if result.status != "ok":
                    entry.update(error=result.error, message=result.message)
                    report["status"] = "command_rejected"
                    break
                async with asyncio.timeout(4):
                    while worker.latest["navigation"]["remaining_s"] > 0:
                        await asyncio.to_thread(worker.sim.cancel.wait, .05)
                measurement = await worker.call(evaluation_state)
                entry["measured"] = measurement
                entry["navigation"] = dict(worker.latest["navigation"])
                progress.update(requests_completed=index + 1)
                print(json.dumps({"index": index, "action": execution, "travel_m": measurement["travel_m"],
                    "parked": measurement["parked"], "navigation": worker.latest["navigation"]["reason"]}), flush=True)
                if worker.latest["navigation"]["status"] == "failed":
                    report["status"] = "safety_stop"
                    break
                if stop_votes >= 2:
                    report["success"] = measurement["parked"] and not measurement["contacts"]
                    report["status"] = "parked" if report["success"] else "stopped_outside_bay"
                    break
            else:
                report["status"] = "request_limit"
            report["final"] = await worker.call(evaluation_state)
            report["base_displacement_m"] = float(np.linalg.norm(np.array(report["final"]["position"][:2]) - report["initial"]["position"][:2]))
            report["saturated_requests"] = sum(bool(entry["saturated_axes"]) for entry in report["requests"])
        except Exception as error:
            report.update(status="error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            progress.update(phase="saving")
            if worker:
                worker.stop()
                try:
                    (options.output / "terminal.png").write_bytes(await worker.call(lambda sim: sim.capture()))
                finally:
                    await worker.close()
            if process and process.returncode is None:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            report["wall_s"] = time.perf_counter() - began
            report["visual_evidence"] = visual_evidence(options.output)
            (options.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "requests"}), flush=True)
    return report


async def compare(options):
    options.output.mkdir(parents=True, exist_ok=False)
    summary = {"purpose": "Paired small-layout stopping evaluation; not a general navigation benchmark",
               "controller_changed": False, "request_limit": options.requests, "cases": EVALUATION_CASES, "trials": []}
    try:
        for case in EVALUATION_CASES:
            for label, checkpoint in (("baseline", options.baseline), ("candidate", options.checkpoint)):
                trial_options = argparse.Namespace(**{**vars(options), "case": case["id"], "checkpoint": checkpoint,
                    "seed": options.seed + case["id"], "output": options.output / f"{label}-case{case['id']}"})
                report = await evaluate(trial_options)
                arrived = [entry["index"] for entry in report["requests"] if entry.get("measured", {}).get("parked")]
                summary["trials"].append({"model": label, "case": case["id"], "seed": trial_options.seed,
                    "weights_sha256": report["weights_sha256"], "success": report["success"], "status": report["status"],
                    "requests": len(report["requests"]), "first_parked_request": min(arrived) if arrived else None,
                    "stopped_outside_bay": report["status"] == "stopped_outside_bay",
                    "saturated_requests": report["saturated_requests"], "final": report["final"],
                    "report": str(trial_options.output / "report.json")})
    finally:
        summary["totals"] = {label: {
            "trials": sum(trial["model"] == label for trial in summary["trials"]),
            "successes": sum(trial["model"] == label and trial["success"] for trial in summary["trials"]),
            "premature_stops": sum(trial["model"] == label and trial["stopped_outside_bay"] for trial in summary["trials"]),
            "requests": sum(trial["requests"] for trial in summary["trials"] if trial["model"] == label),
            "saturated_requests": sum(trial["saturated_requests"] for trial in summary["trials"] if trial["model"] == label),
        } for label in ("baseline", "candidate")}
        summary["complete"] = len(summary["trials"]) == 2 * len(EVALUATION_CASES)
        summary["qualification"] = "Candidate differs in both additional training and sampling; this comparison does not isolate the effect of stop balancing."
        (options.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["totals"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native Windows navigation recording, export and real-model evaluation; no cloud calls")
    parser.add_argument("--stage", choices=["record", "replay", "export", "inference", "evaluate", "compare"], default="record")
    parser.add_argument("--output", type=Path, default=Path(".runtime/navigation-demonstrations"))
    parser.add_argument("--source", type=Path, default=Path(".runtime/navigation-demonstrations"))
    parser.add_argument("--episodes", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--checkpoint", type=Path, default=Path(".runtime/navigation-training-1000/checkpoint"))
    parser.add_argument("--baseline", type=Path, default=Path(".runtime/navigation-training-1000/checkpoint"))
    parser.add_argument("--python", type=Path, default=Path(".runtime/smolvla-env/Scripts/python.exe"))
    parser.add_argument("--case", type=int, choices=range(len(CASES) + len(EVALUATION_CASES)), default=12)
    parser.add_argument("--requests", type=int, choices=range(1, 41), default=30)
    parser.add_argument("--seed", type=int, default=713)
    options = parser.parse_args()
    if options.stage == "export":
        export(options)
    elif options.stage == "inference":
        inference(options)
    elif options.stage == "evaluate":
        asyncio.run(evaluate(options))
    elif options.stage == "compare":
        asyncio.run(compare(options))
    else:
        asyncio.run(record(options) if options.stage == "record" else replay(options))