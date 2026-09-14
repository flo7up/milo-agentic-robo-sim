import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.challenges import get_challenge
from backend.navigation_supervisor import POLICY_TASKS
from scripts.navigation_policy import ACTION_NAMES, EMBODIMENT, STATE_NAMES, apply, prepare, state_vector


POSES = [
    ("park", [-.35, -.2]), ("recharge", [-1.6, -.8]),
    ("park", [-.3, .2]), ("recharge", [-1.6, .8]),
    ("park", [-.15, -.15]), ("recharge", [-1.5, -1.3]),
    ("park", [-.2, .1]), ("recharge", [1.25, -1.3]),
    ("park", [-.45, 0]), ("recharge", [-1.5, 1.3]),
    ("park", [-.1, 0]), ("recharge", [-1.7, 0]),
    ("park", [-.25, -.1]), ("recharge", [-1.55, -.6]),
    ("park", [-.25, .1]), ("recharge", [-1.55, .6]),
]
TEACHER_ACTIONS = {"forward": [.15, 0.], "backward": [-.15, 0.], "left": [0., .5], "right": [0., -.5], "hold": [0., 0.]}
SEQUENCE = ["forward"] * 2 + ["backward"] * 2 + ["left"] * 3 + ["right"] * 3 + ["hold"]


def challenge_for(index):
    identifier, position = POSES[index]
    return get_challenge(identifier).model_copy(deep=True, update={"initial_xy": position, "initial_head_pitch": .45})


def execute_sample(sim, runtime, action):
    apply(runtime, sim, "replace_motion_buffer", {"segments": [
        {"kind": "drive", "linear_mps": action[0], "angular_radps": action[1], "duration_s": 1.}]})
    before = sim.ticks
    while runtime.buffer:
        runtime.tick(sim)
    if runtime.status == "failed" or sim.ticks - before != 240:
        raise ValueError(runtime.reason)


def record_episode(sim, index, directory, varied_camera=False):
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    if varied_camera:
        yaw, pitch = [(-.6, .8), (0., .45), (.6, .8), (0., 1.)][index % 4]
        apply(runtime, sim, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": yaw, "pitch_rad": pitch, "duration_s": 1.}]})
        while runtime.buffer:
            runtime.tick(sim)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "images").mkdir()
    frames = []
    try:
        for frame_index, motion in enumerate(SEQUENCE * 4):
            observation = sim.observe()
            image = sim.frame(observation.frame_ref)
            filename = f"images/{frame_index:04d}.png"
            (directory / filename).write_bytes(image)
            frames.append({"frame_index": frame_index, "timestamp": frame_index,
                "observation.state": state_vector(observation), "observation.images.head": filename,
                "image_sha256": hashlib.sha256(image).hexdigest(), "action": TEACHER_ACTIONS[motion], "task": POLICY_TASKS[motion]})
            execute_sample(sim, runtime, TEACHER_ACTIONS[motion])
        (directory / "frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
        metadata = {"curriculum": "recovery-v1", "varied_camera": varied_camera, "episode": index, "challenge": POSES[index][0],
            "initial_xy": POSES[index][1], "embodiment": EMBODIMENT, "state_names": STATE_NAMES, "action_names": ACTION_NAMES,
            "fps": 1, "frames": len(frames), "scripted_demonstration": True,
            "asset_sha256": hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest(),
            "split": "train" if index < 12 else "validation", "task_success_claimed": False}
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        runtime.cancel(sim, "Recovery demonstration ended")


def load_episode(directory):
    from PIL import Image
    metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    frames = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    index = metadata["episode"]
    if (metadata.get("curriculum") != "recovery-v1" or index not in range(16) or
            (metadata["challenge"], metadata["initial_xy"]) != POSES[index] or metadata["fps"] != 1 or
            metadata["state_names"] != STATE_NAMES or metadata["action_names"] != ACTION_NAMES or
            metadata["asset_sha256"] != hashlib.sha256(Path("assets/milo.urdf").read_bytes()).hexdigest() or
            len(frames) != 44 or metadata["frames"] != len(frames)):
        raise ValueError("Invalid recovery recording metadata")
    for index, (frame, motion) in enumerate(zip(frames, SEQUENCE * 4)):
        if (set(frame) != {"frame_index", "timestamp", "observation.state", "observation.images.head", "image_sha256", "action", "task"}
                or frame["frame_index"] != index or frame["timestamp"] != index or frame["task"] != POLICY_TASKS[motion]
                or frame["action"] != TEACHER_ACTIONS[motion] or np.shape(frame["observation.state"]) != (20,)
                or not np.isfinite(frame["observation.state"]).all()):
            raise ValueError("Invalid recovery sample or private fields")
        path = (directory / frame["observation.images.head"]).resolve(strict=True)
        if not path.is_relative_to((directory / "images").resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise ValueError("Recovery image path/hash mismatch")
        with Image.open(path) as image:
            if image.size != (320, 240) or image.mode != "RGB":
                raise ValueError("Recovery image contract mismatch")
    return metadata, frames


def replay_episode(sim, frames, metadata=None):
    from backend.navigation import NavigationRuntime
    sim.on_tick = None
    sim.width, sim.height = 320, 240
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    prepare(sim, runtime)
    if metadata and metadata.get("varied_camera"):
        yaw, pitch = [(-.6, .8), (0., .45), (.6, .8), (0., 1.)][metadata["episode"] % 4]
        apply(runtime, sim, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": yaw, "pitch_rad": pitch, "duration_s": 1.}]})
        while runtime.buffer:
            runtime.tick(sim)
    maximum = 0.
    try:
        for frame in frames:
            actual = state_vector(sim.observe(render=False))
            maximum = max(maximum, float(np.max(np.abs(np.array(actual) - frame["observation.state"]))))
            execute_sample(sim, runtime, frame["action"])
        if maximum >= 1e-5:
            raise ValueError("Recovery replay diverged from recorded sensor states")
        return {"verified": True, "max_state_error": maximum, "frames": len(frames)}
    finally:
        runtime.cancel(sim, "Recovery replay ended")


async def record(options):
    from backend.worker import SimulationWorker
    options.output.mkdir(parents=True, exist_ok=False)
    for index in range(options.episodes):
        directory = options.output / f"episode-{index:03d}"
        worker = SimulationWorker(challenge=challenge_for(index), pace=False)
        try:
            result = await worker.call(lambda sim: record_episode(sim, index, directory, options.varied_camera))
        finally:
            await worker.close()
        metadata, frames = load_episode(directory)
        worker = SimulationWorker(challenge=challenge_for(index), pace=False)
        try:
            replay = await worker.call(lambda sim: replay_episode(sim, frames, metadata))
            replay["frames_sha256"] = hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest()
            (directory / "replay.json").write_text(json.dumps(replay, indent=2), encoding="utf-8")
        finally:
            await worker.close()
        print(json.dumps({"episode": index, "challenge": metadata["challenge"], "frames": result["frames"], **replay}), flush=True)


def export(options, loader=load_episode, curriculum="recovery-v1"):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    records = []
    for index in range(16):
        directory = options.source / f"episode-{index:03d}"
        metadata, frames = loader(directory)
        replay = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
        if (metadata["episode"] != index or not replay.get("verified") or
                replay["frames_sha256"] != hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest()):
            raise ValueError("Export requires all 16 matching verified replays")
        records.append((directory, frames))
    dataset = LeRobotDataset.create(repo_id="local/milo-recovery", root=options.output, robot_type=EMBODIMENT,
        fps=1, use_videos=False, features={
            "observation.state": {"dtype": "float32", "shape": (20,), "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": (2,), "names": ACTION_NAMES},
            "observation.images.head": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}})
    try:
        for directory, frames in records:
            for frame in frames:
                with Image.open(directory / frame["observation.images.head"]) as image:
                    pixels = np.array(image, dtype=np.uint8)
                dataset.add_frame({"observation.state": np.array(frame["observation.state"], dtype=np.float32),
                    "action": np.array(frame["action"], dtype=np.float32), "observation.images.head": pixels, "task": frame["task"]})
            dataset.save_episode()
    finally:
        dataset.finalize()
    loaded = LeRobotDataset("local/milo-recovery", root=options.output, token=False, video_backend="pyav")
    offset = 0
    for directory, frames in records:
        for frame in frames:
            sample = loaded[offset]
            if any(not np.array_equal(np.asarray(sample[key]), np.array(frame[key], dtype=np.float32)) for key in ("action", "observation.state")) or sample["task"] != frame["task"]:
                raise ValueError("Recovery export values/task changed")
            with Image.open(directory / frame["observation.images.head"]) as image:
                pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
            if not np.array_equal(np.asarray(sample["observation.images.head"]), pixels):
                raise ValueError("Recovery export pixels changed")
            offset += 1
    summary = {"verified": True, "embodiment": EMBODIMENT, "curriculum": curriculum, "frames": offset,
        "train_episodes": list(range(12)), "validation_episodes": list(range(12, 16)),
        "source_hashes": {directory.name: hashlib.sha256((directory / "frames.json").read_bytes()).hexdigest() for directory, _ in records},
        "task_success_claimed": False, "uploaded": False}
    (options.output / "navigation-export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Native Windows instruction/recovery demonstrations and replay-verified export")
    parser.add_argument("--stage", choices=["record", "export"], required=True)
    parser.add_argument("--source", type=Path, default=Path(".runtime/recovery-demonstrations"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--varied-camera", action="store_true")
    options = parser.parse_args()
    if options.stage == "record":
        asyncio.run(record(options))
    else:
        export(options)