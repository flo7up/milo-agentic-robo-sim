import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from backend.policy import ACTION_NAMES, PolicyChunk, PolicyTicket


ALIGNMENT = "state/image before motion; action is the next 50ms endpoint, interpolated over 12 physics ticks"
FRAME_FIELDS = {"frame_index", "timestamp", "simulated_time_s", "observation_wall_timestamp", "observation.state",
                "observation.images.head", "image_sha256", "action", "task"}


def load_episode(directory, require_replay=False):
    directory = Path(directory).resolve(strict=True)
    metadata = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (metadata.get("format") != "milo-demonstrations-v1" or metadata.get("fps") != 20 or
            metadata.get("action_names") != ACTION_NAMES or metadata.get("state_names") != ACTION_NAMES or
            metadata.get("action_alignment") != ALIGNMENT or metadata.get("grasp_assistance") is not False or
            metadata.get("camera_size") != [320, 240]):
        raise ValueError("Incompatible demonstration contract")
    if result.get("success") is not True or result.get("error") is not None or result.get("grasp_assistance") is not False:
        raise ValueError("Only successful physical-only episodes can be used for training/replay")
    frames_path = directory / "frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    if not frames or len(frames) != result.get("frames") or len(frames) != result.get("executed_frames"):
        raise ValueError("Incomplete demonstration recording")
    for index, frame in enumerate(frames):
        if set(frame) != FRAME_FIELDS or frame["frame_index"] != index or frame["task"] != metadata["task"]:
            raise ValueError("Unexpected frame schema, task or frame order")
        if any(not math.isfinite(frame[key]) for key in ("timestamp", "simulated_time_s", "observation_wall_timestamp")):
            raise ValueError("Nonfinite frame timestamp")
        if abs(frame["timestamp"] - index / 20) > 1e-6 or abs(frame["simulated_time_s"] - index / 20) > 1e-6:
            raise ValueError("Frames are not aligned to the 20 Hz action timebase")
        state = frame["observation.state"]
        if len(state) != 7 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in state):
            raise ValueError("Invalid measured state vector")
        action = frame["action"]
        if len(action) != 7 or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in action):
            raise ValueError("Invalid commanded action vector")
        image_path = (directory / frame["observation.images.head"]).resolve(strict=True)
        if not image_path.is_relative_to(directory / "images"):
            raise ValueError("Camera path escapes the episode image directory")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != frame["image_sha256"]:
            raise ValueError("Image hash mismatch")
        with Image.open(image_path) as image:
            if image.size != (320, 240) or image.mode != "RGB" or image.format != "PNG":
                raise ValueError("Camera image does not match the recorded RGB contract")
            image.verify()
    ticket = PolicyTicket(run_id="dataset-validation", episode_epoch=0, revision=0, observation_seq=0)
    for start in range(0, len(frames), 50):
        PolicyChunk(ticket=ticket, actions=[frame["action"] for frame in frames[start:start + 50]])
    if require_replay:
        replay = json.loads((directory / "replay.json").read_text(encoding="utf-8"))
        if (replay.get("success") is not True or replay.get("error") is not None or
                replay.get("source_frames_sha256") != hashlib.sha256(frames_path.read_bytes()).hexdigest() or
                replay.get("robot_asset_sha256") != metadata.get("robot_asset_sha256") or
                replay.get("replay_state_max_error", float("inf")) > .001):
            raise ValueError("Episode has no matching successful deterministic replay")
    return metadata, frames, result


def export_dataset(source, output, repo_id, dataset_type=None):
    import numpy as np
    episodes = sorted(source.glob("episode-*"))
    if not episodes:
        raise ValueError("No recorded episodes found")
    if output.exists():
        raise ValueError("Choose a new output directory; existing datasets are never overwritten")
    validated = [(directory, *load_episode(directory, require_replay=True)) for directory in episodes]
    if dataset_type is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_type = LeRobotDataset
    dataset = dataset_type.create(repo_id=repo_id, root=output, robot_type="milo-left-arm-v1", fps=20,
        use_videos=False, features={
            "observation.state": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
            "observation.images.head": {"dtype": "image", "shape": (240, 320, 3), "names": ["height", "width", "channels"]}})
    try:
        for directory, metadata, frames, result in validated:
            for frame in frames:
                with Image.open(directory / frame["observation.images.head"]) as image:
                    pixels = np.array(image, dtype=np.uint8)
                dataset.add_frame({"observation.state": np.array(frame["observation.state"], dtype=np.float32),
                    "action": np.array(frame["action"], dtype=np.float32),
                    "observation.images.head": pixels, "task": frame["task"]})
            dataset.save_episode()
    finally:
        dataset.finalize()
    return {"episodes": len(validated), "frames": sum(len(frames) for _, _, frames, _ in validated),
            "uploaded": False, "dataset": str(output)}


def verify_export(source, output, repo_id, dataset=None):
    import numpy as np
    episodes = sorted(source.glob("episode-*"))
    if not episodes:
        raise ValueError("No recorded episodes found")
    validated = [(directory, *load_episode(directory, require_replay=True)) for directory in episodes]
    reference_stats = None
    dataset_type = None
    if dataset is None:
        from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset_type = LeRobotDataset
        dataset = LeRobotDataset(repo_id, root=output, video_backend="pyav", token=False)
        reference_stats = aggregate_stats([
            {key: get_feature_stats(np.array([frame[key] for frame in frames], dtype=np.float32), axis=0, keepdims=False)
             for key in ("observation.state", "action")} for _, _, frames, _ in validated])
    expected_count = sum(len(frames) for _, _, frames, _ in validated)
    if dataset.fps != 20 or dataset.num_frames != expected_count or dataset.num_episodes != len(validated):
        raise ValueError("Exported dataset count or timebase mismatch")
    vectors = ("observation.state", "action")
    image_key = "observation.images.head"
    allowed = {*vectors, image_key, "timestamp", "frame_index", "episode_index", "index", "task_index"}
    if set(dataset.features) != allowed:
        raise ValueError("Unexpected exported features; private metadata must not enter model inputs")
    for key in vectors:
        feature = dataset.features[key]
        if feature["names"] != ACTION_NAMES or tuple(feature["shape"]) != (7,) or feature["dtype"] != "float32":
            raise ValueError("Exported action/state contract mismatch")
    feature = dataset.features[image_key]
    if feature["dtype"] != "image" or tuple(feature["shape"]) != (240, 320, 3):
        raise ValueError("Exported camera contract mismatch")
    samples = {key: [] for key in vectors}
    max_rounding = {key: 0. for key in vectors}
    max_image_error = 0.
    boundaries, offset = [], 0
    for episode_index, (directory, metadata, frames, _) in enumerate(validated):
        episode = dataset.meta.episodes[episode_index]
        if (int(episode["episode_index"]) != episode_index or int(episode["length"]) != len(frames) or
                int(episode["dataset_from_index"]) != offset or int(episode["dataset_to_index"]) != offset + len(frames)):
            raise ValueError("Exported episode boundaries mismatch")
        for frame_index, frame in enumerate(frames):
            sample = dataset[offset + frame_index]
            if set(sample) != allowed | {"task"}:
                raise ValueError("Unexpected loaded sample fields")
            if (int(sample["index"]) != offset + frame_index or int(sample["frame_index"]) != frame_index or
                    int(sample["episode_index"]) != episode_index or sample["task"] != metadata["task"] or
                    abs(float(sample["timestamp"]) - frame["timestamp"]) > 1e-4):
                raise ValueError("Exported frame/task alignment mismatch")
            for key in vectors:
                original = np.array(frame[key], dtype=np.float64)
                expected = original.astype(np.float32)
                actual = np.asarray(sample[key])
                if actual.dtype != np.float32 or actual.shape != (7,) or not np.array_equal(actual, expected):
                    raise ValueError(f"Exported {key} differs from the source action/state")
                samples[key].append(expected)
                max_rounding[key] = max(max_rounding[key], float(np.max(np.abs(actual - original))))
            with Image.open(directory / frame[image_key]) as image:
                pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
            actual = np.asarray(sample[image_key])
            if actual.dtype != np.float32 or actual.shape != (3, 240, 320) or not np.allclose(actual, pixels, atol=1e-7, rtol=0):
                raise ValueError("Exported camera pixels differ from their paired source frame")
            max_image_error = max(max_image_error, float(np.max(np.abs(actual - pixels))))
        boundaries.append({"episode": episode_index, "from_index": offset, "to_index": offset + len(frames),
                           "frames": len(frames), "source_frames_sha256": hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest()})
        offset += len(frames)
        print(json.dumps({"verified_episode": episode_index, "verified_frames": offset}), flush=True)
    statistics = {}
    for key in vectors:
        values = np.array(samples[key], dtype=np.float64)
        expected_stats = {"min": values.min(axis=0), "max": values.max(axis=0),
                          "mean": values.mean(axis=0), "std": values.std(axis=0)}
        saved = dataset.meta.stats[key]
        if np.asarray(saved["count"]).shape != (1,) or int(saved["count"][0]) != expected_count:
            raise ValueError("Normalization statistics sample count mismatch")
        errors = {}
        for name, expected in expected_stats.items():
            actual = np.asarray(saved[name])
            tolerance = 1e-3 if name == "std" else 1e-4
            if actual.shape != (7,) or not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-6, rtol=tolerance):
                raise ValueError(f"Normalization statistics mismatch: {key}.{name}")
            if reference_stats is not None and not np.allclose(actual, reference_stats[key][name], atol=1e-8, rtol=1e-7):
                raise ValueError(f"Statistics differ from LeRobot's source recomputation: {key}.{name}")
            errors[name] = float(np.max(np.abs(actual - expected)))
        statistics[key] = {"max_absolute_errors": errors, "mean": np.asarray(saved["mean"]).tolist(),
                           "std": np.asarray(saved["std"]).tolist(),
                           "low_variance_axes": [name for name, deviation in zip(ACTION_NAMES, saved["std"]) if deviation < 1e-4]}
    image_stats = dataset.meta.stats[image_key]
    for name in ("min", "max", "mean", "std"):
        values = np.asarray(image_stats[name])
        if values.shape != (3, 1, 1) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Invalid image statistics")
    window_checks = 0
    if dataset_type is not None:
        chunked = dataset_type(repo_id, root=output, video_backend="pyav", token=False,
                               delta_timestamps={"action": [step / 20 for step in range(50)]})
        offset = 0
        for _, _, frames, _ in validated:
            for frame_index in sorted({0, max(0, len(frames) - 2), len(frames) - 1}):
                sample = chunked[offset + frame_index]
                indices = [min(frame_index + step, len(frames) - 1) for step in range(50)]
                expected = np.array([frames[index]["action"] for index in indices], dtype=np.float32)
                padding = np.array([frame_index + step >= len(frames) for step in range(50)])
                if not np.array_equal(np.asarray(sample["action"]), expected) or not np.array_equal(np.asarray(sample["action_is_pad"]), padding):
                    raise ValueError("Action window or end-of-episode padding mismatch")
                window_checks += 1
            offset += len(frames)
    return {"verified": True, "dataset": str(output), "episodes": len(validated), "frames": expected_count,
            "action_window_checks": window_checks, "action_window_steps": 50,
            "fps": dataset.fps, "image_max_absolute_error": max_image_error,
            "float32_max_rounding_error": max_rounding, "boundaries": boundaries,
            "normalization_statistics": statistics,
            "lerobot_statistics_recomputed": reference_stats is not None,
            "statistics_tolerance": "Float64 comparison: absolute 1e-6, relative 0.1% for std and 0.01% otherwise; LeRobot recomputation: absolute 1e-8, relative 1e-7",
            "image_statistics": "finite per-channel [0,1] statistics; LeRobot samples images for statistics",
            "uploaded": False, "training_started": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and locally export successful replayed Milo episodes to LeRobot")
    parser.add_argument("--source", type=Path, default=Path(".runtime/milo-demonstrations"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/datasets/milo-pilot"))
    parser.add_argument("--repo-id", default="local/milo-pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--verify-export", action="store_true", help="Reopen local LeRobot data and compare every source frame and vector statistic")
    parser.add_argument("--report", type=Path, help="Write verification evidence outside the training dataset")
    options = parser.parse_args()
    if options.validate_only:
        episodes = sorted(options.source.glob("episode-*"))
        if not episodes:
            raise ValueError("No recorded episodes found")
        for directory in episodes:
            _, frames, _ = load_episode(directory, require_replay=True)
            print(json.dumps({"episode": directory.name, "frames": len(frames), "validated": True}))
    elif options.verify_export:
        report = verify_export(options.source, options.output, options.repo_id)
        if options.report:
            options.report.parent.mkdir(parents=True, exist_ok=True)
            options.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(export_dataset(options.source, options.output, options.repo_id)))