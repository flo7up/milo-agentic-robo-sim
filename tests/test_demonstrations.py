import hashlib
import json

import numpy as np
import pytest

from backend.policy import ACTION_NAMES
from backend.robot import NEUTRAL
from backend.simulation import BulletSimulation
from scripts.milo_dataset import ALIGNMENT, export_dataset, load_episode, verify_export
from scripts.record_milo import EpisodeRecorder, execute_episode, scene_for, smooth_targets


def small_episode(directory):
    sim = BulletSimulation(width=320, height=240)
    try:
        metadata = {"format": "milo-demonstrations-v1", "fps": 20, "action_names": ACTION_NAMES,
            "state_names": ACTION_NAMES, "action_alignment": ALIGNMENT, "grasp_assistance": False,
            "camera_size": [320, 240], "task": "Pick up the red cube and place it on the green square with the left arm.",
            "robot_asset_sha256": "fixture"}
        recorder = EpisodeRecorder(directory, metadata)
        observation = sim.observe()
        recorder.append(observation, sim.frame(observation.frame_ref), [*NEUTRAL, .1])
        recorder.close({"success": True, "error": None, "grasp_assistance": False, "executed_frames": 1})
        (directory / "replay.json").write_text(json.dumps({"success": True, "error": None,
            "source_frames_sha256": hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest(),
            "robot_asset_sha256": "fixture", "replay_state_max_error": 0}), encoding="utf-8")
    finally:
        sim.close()


def test_recorder_separates_state_action_and_validates_hashes_timing_and_success(tmp_path):
    directory = tmp_path / "episode-000"
    small_episode(directory)
    _, frames, _ = load_episode(directory, require_replay=True)
    assert frames[0]["observation.state"][6] > .10 and frames[0]["action"][6] == .1
    assert not {"objects", "fixture", "cube_xy", "robot_position"} & set(frames[0])
    result_path = directory / "result.json"
    original = result_path.read_text()
    result_path.write_text(original.replace('"success": true', '"success": false'))
    with pytest.raises(ValueError, match="successful"):
        load_episode(directory)
    result_path.write_text(original)
    image = directory / frames[0]["observation.images.head"]
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_episode(directory)


@pytest.mark.parametrize("field,value", [("timestamp", .1), ("action", [0] * 6),
                                       ("observation.state", [float("nan")] * 7), ("objects", [{"id": 1}])])
def test_dataset_validation_rejects_malformed_frames(tmp_path, field, value):
    directory = tmp_path / "episode-000"
    small_episode(directory)
    path = directory / "frames.jsonl"
    frame = json.loads(path.read_text())
    frame[field] = value
    path.write_text(json.dumps(frame) + "\n")
    with pytest.raises(ValueError):
        load_episode(directory)


def test_lerobot_export_uses_official_writer_contract_without_private_inputs(tmp_path):
    small_episode(tmp_path / "episode-000")
    calls = []

    class Dataset:
        @classmethod
        def create(cls, **kwargs):
            calls.append(kwargs)
            assert kwargs["fps"] == 20 and not kwargs["use_videos"]
            assert kwargs["features"]["action"]["names"] == ACTION_NAMES
            return cls()

        def add_frame(self, frame):
            assert set(frame) == {"observation.state", "observation.images.head", "action", "task"}
            assert frame["observation.images.head"].shape == (240, 320, 3)
            assert frame["action"].shape == (7,)
            calls.append("frame")

        def save_episode(self):
            calls.append("episode")

        def finalize(self):
            calls.append("finalized")

    result = export_dataset(tmp_path, tmp_path / "export", "local/test", Dataset)
    assert result["frames"] == 1 and not result["uploaded"]
    assert calls[1:] == ["frame", "episode", "finalized"]


def test_export_refuses_missing_or_tampered_replay_and_existing_output(tmp_path):
    directory = tmp_path / "episode-000"
    small_episode(directory)
    (directory / "replay.json").unlink()
    with pytest.raises(FileNotFoundError):
        export_dataset(tmp_path, tmp_path / "export", "local/test")
    (directory / "replay.json").write_text(json.dumps({"success": True, "error": None,
        "source_frames_sha256": "wrong", "robot_asset_sha256": "fixture", "replay_state_max_error": 0}))
    with pytest.raises(ValueError, match="matching successful"):
        export_dataset(tmp_path, tmp_path / "export", "local/test")
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="never overwritten"):
        export_dataset(tmp_path, output, "local/test")


def test_dataset_rejects_image_paths_outside_episode(tmp_path):
    directory = tmp_path / "episode-000"
    small_episode(directory)
    path = directory / "frames.jsonl"
    frame = json.loads(path.read_text())
    (tmp_path / "outside.png").write_bytes((directory / frame["observation.images.head"]).read_bytes())
    frame["observation.images.head"] = "../outside.png"
    path.write_text(json.dumps(frame) + "\n")
    with pytest.raises(ValueError, match="escapes"):
        load_episode(directory)


@pytest.mark.parametrize("corruption", [None, "action", "image", "timestamp", "boundary", "statistics", "private_field"])
def test_export_verifier_checks_reopened_data_and_statistics(tmp_path, corruption):
    from types import SimpleNamespace
    from PIL import Image
    directory = tmp_path / "episode-000"
    small_episode(directory)
    _, frames, _ = load_episode(directory, require_replay=True)
    source = frames[0]
    values = {key: np.array(source[key], dtype=np.float32) for key in ("action", "observation.state")}
    with Image.open(directory / source["observation.images.head"]) as image:
        pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255
    features = {key: {"names": ACTION_NAMES, "shape": (7,), "dtype": "float32"} for key in values}
    features["observation.images.head"] = {"shape": (240, 320, 3), "dtype": "image"}
    features.update({key: {} for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
    stats = {key: {"min": value.copy(), "max": value.copy(), "mean": value.copy(),
                   "std": np.zeros(7), "count": np.array([1])} for key, value in values.items()}
    stats["observation.images.head"] = {key: np.zeros((3, 1, 1)) for key in ("min", "max", "mean", "std")}
    episode = {"episode_index": 0, "length": 1, "dataset_from_index": 0, "dataset_to_index": 1}
    sample = {**values, "observation.images.head": pixels, "task": source["task"],
              **{key: np.array(0) for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")}}

    class Dataset:
        fps, num_frames, num_episodes = 20, 1, 1
        meta = SimpleNamespace(episodes=[episode], stats=stats)

        def __getitem__(self, index):
            assert index == 0
            return sample

    dataset = Dataset()
    dataset.features = features
    if corruption == "action":
        sample["action"][0] += .001
    elif corruption == "image":
        pixels[0, 0, 0] = .12345
    elif corruption == "timestamp":
        sample["timestamp"] = np.array(.05)
    elif corruption == "boundary":
        episode["dataset_to_index"] = 2
    elif corruption == "statistics":
        stats["action"]["mean"][0] += .1
    elif corruption == "private_field":
        dataset.features["fixture"] = {}
    if corruption:
        with pytest.raises(ValueError):
            verify_export(tmp_path, tmp_path / "export", "local/test", dataset)
    else:
        report = verify_export(tmp_path, tmp_path / "export", "local/test", dataset)
        assert report["verified"] and report["frames"] == report["episodes"] == 1
        assert report["image_max_absolute_error"] == 0
        assert not report["lerobot_statistics_recomputed"] and not report["training_started"]


def test_smooth_gripper_target_respects_speed_and_acceleration():
    initial = np.array([*NEUTRAL, .11])
    target = np.array([*NEUTRAL, 0.])
    values = np.array([initial.tolist(), *smooth_targets(initial, target, 4)])
    assert np.max(np.abs(np.diff(values[:, 6]) / .05)) < .06
    assert np.max(np.abs(np.diff(values[:, 6], n=2) / .05 ** 2)) < .3


def test_navigation_stop_sampling_balances_classes_without_changing_labels():
    from scripts.train_milo import navigation_sampling_weights
    labels = np.array([[.15, 0], [.1, .2], [0, -.4], [0, 0]], dtype=np.float32)
    original = labels.copy()
    weights = navigation_sampling_weights(labels)
    assert np.array_equal(labels, original)
    assert weights.sum() == pytest.approx(1)
    assert weights[-1] == pytest.approx(.5)
    assert weights[:3].sum() == pytest.approx(.5)
    for invalid in ([], [[0, 0]], [[.15, 0]], [[0, 0, 0]], [[float("nan"), 0], [0, 0]]):
        with pytest.raises(ValueError):
            navigation_sampling_weights(invalid)


def test_training_sample_excludes_privileged_fields_and_requires_alignment():
    from scripts.train_milo import MODEL_FIELDS, training_sample
    sample = {key: object() for key in MODEL_FIELDS}
    result = training_sample({**sample, "fixture": {"cube_xy": [.36, .25]}, "episode_index": 0})
    assert result == sample
    del sample["action_is_pad"]
    with pytest.raises(ValueError, match="paired"):
        training_sample(sample)


@pytest.mark.parametrize("corruption", [None, "action_order", "extra_camera", "fps"])
def test_training_requires_the_milo_camera_action_contract(corruption):
    from types import SimpleNamespace
    from scripts.train_milo import validate_dataset
    features = {key: {"shape": (7,), "names": ACTION_NAMES.copy(), "dtype": "float32"}
                for key in ("observation.state", "action")}
    features["observation.images.head"] = {"dtype": "image", "shape": (240, 320, 3)}
    dataset = SimpleNamespace(fps=20, num_episodes=5, features=features)
    if corruption == "action_order":
        features["action"]["names"].reverse()
    elif corruption == "extra_camera":
        features["observation.images.spectator"] = {"dtype": "image"}
    elif corruption == "fps":
        dataset.fps = 30
    if corruption:
        with pytest.raises(ValueError):
            validate_dataset(dataset)
    else:
        validate_dataset(dataset)


def test_candidate_evaluation_requires_a_matching_trained_checkpoint(tmp_path):
    from scripts.evaluate_milo import validate_pilot
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"test-weights")
    report = {"status": "trained_and_reloaded", "steps": 100,
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              "reloaded_output": {"shape": [1, 50, 7]}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert validate_pilot(checkpoint)[0] == checkpoint
    weights.write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="matching completed"):
        validate_pilot(checkpoint)


def test_candidate_adapter_keeps_limits_and_records_target_saturation():
    from backend.policy import PolicyTicket
    from scripts.evaluate_milo import adapted_chunk
    ticket = PolicyTicket(run_id="adapter-test", episode_epoch=0, revision=0, observation_seq=1)
    position = np.array([*NEUTRAL, .05])
    target = [1., -.1, 1.4, .2, -1., .1, .2]
    raw = {"ticket": ticket.model_dump(), "actions": [target] * 50}
    chunk, details = adapted_chunk(raw, position, np.zeros(7), ticket, .41)
    assert details["saturated_axes"] == [ACTION_NAMES[-1]]
    assert raw["actions"][19][6] == .2 and details["bounded_target"][6] == .11
    assert details["discarded_prefix_samples"] == 9 and len(chunk.actions) == 17
    trajectory = np.array([position.tolist(), *chunk.actions, chunk.actions[-1]])
    speed = np.diff(trajectory, axis=0) / .05
    acceleration = np.diff(np.vstack([np.zeros(7), speed]), axis=0) / .05
    assert np.all(np.abs(speed) <= np.array([.8] * 6 + [.06]))
    assert np.all(np.abs(acceleration) <= np.array([4.] * 6 + [.3]))
    assert np.max(np.abs(trajectory[-1] - position)) > .01
    for age in (-.1, 1.01):
        with pytest.raises(ValueError, match="Stale"):
            adapted_chunk(raw, position, np.zeros(7), ticket, age)
    with pytest.raises(ValueError, match="rest"):
        adapted_chunk(raw, position, np.ones(7), ticket, .4)
    with pytest.raises(ValueError, match="Stale"):
        adapted_chunk(raw, position, np.zeros(7), ticket.model_copy(update={"revision": 1}), .4)
    raw["actions"] = [[float("nan")] * 7] * 50
    with pytest.raises(ValueError, match="finite"):
        adapted_chunk(raw, position, np.zeros(7), ticket, .4)


def test_candidate_visual_evidence_measures_pixels_and_labels_timelapse(tmp_path):
    from PIL import Image
    from scripts.evaluate_milo import visual_evidence
    Image.new("RGB", (320, 240), "black").save(tmp_path / "frame-000.png")
    assert visual_evidence(tmp_path) is None
    Image.new("RGB", (320, 240), "white").save(tmp_path / "terminal.png")
    evidence = visual_evidence(tmp_path)
    assert evidence["changed_pixel_fraction"] == 1
    assert "timelapse" in evidence["animation_timing"]
    with Image.open(tmp_path / "motion.gif") as animation:
        assert animation.n_frames == 2
    with Image.open(tmp_path / "comparison.png") as comparison:
        assert comparison.size == (640, 270)


def test_candidate_validation_allows_bounded_continued_training(tmp_path):
    from scripts.evaluate_milo import validate_pilot
    from scripts.train_milo import MAX_TRAINING_STEPS
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"continued")
    report = {"status": "trained_and_reloaded", "steps": 1000, "total_steps": 1100,
              "weights_sha256": hashlib.sha256(b"continued").hexdigest(), "reloaded_output": {"shape": [1, 50, 7]}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert validate_pilot(checkpoint)[1]["total_steps"] == 1100
    report["steps"] = MAX_TRAINING_STEPS + 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matching completed"):
        validate_pilot(checkpoint)


def test_physical_only_demonstration_lifts_places_releases_without_constraint():
    sim = BulletSimulation(scene=scene_for([.36, .25]), width=160, height=120)
    try:
        result = execute_episode(sim, [.36, .25])
        assert result["success"], result
        assert result["opposing_contact_lift"] and result["peak_cube_bottom_m"] > .08
        assert result["grounded"] and result["placed"] and not result["grasp_assistance"]
    finally:
        sim.close()