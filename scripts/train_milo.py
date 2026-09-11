import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from backend.policy import ACTION_NAMES
from scripts.probe_smolvla import MODEL_REVISION, checked_actions


MODEL_FIELDS = {"observation.state", "observation.images.head", "action", "action_is_pad", "task"}
MAX_TRAINING_STEPS = 3000


def navigation_sampling_weights(actions):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Expected finite two-value navigation training labels")
    stopped = np.all(actions == 0, axis=1)
    if not stopped.any() or stopped.all():
        raise ValueError("Stop balancing requires both stopped and moving training examples")
    return np.where(stopped, .5 / stopped.sum(), .5 / (~stopped).sum())


def training_sample(sample):
    if not MODEL_FIELDS <= sample.keys():
        raise ValueError("Missing paired Milo training features")
    return {key: sample[key] for key in MODEL_FIELDS}


def validate_dataset(dataset, navigation=False):
    from scripts.navigation_policy import ACTION_NAMES as NAV_ACTIONS, STATE_NAMES
    if dataset.fps != (1 if navigation else 20) or dataset.num_episodes != (16 if navigation else 5):
        raise ValueError("Dataset episode count or timebase does not match the selected pilot")
    for key in ("observation.state", "action"):
        feature = dataset.features[key]
        names = (STATE_NAMES if key == "observation.state" else NAV_ACTIONS) if navigation else ACTION_NAMES
        if tuple(feature["shape"]) != (len(names),) or feature["names"] != names:
            raise ValueError("Dataset does not match the selected Milo state/action contract")
    images = {key for key, feature in dataset.features.items() if feature["dtype"] in {"image", "video"}}
    if images != {"observation.images.head"}:
        raise ValueError("Only Milo's paired head camera is supported")


def run(options):
    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from torch.utils.data import default_collate

    if not torch.cuda.is_available():
        raise RuntimeError("The Windows CUDA runtime must be available")
    from scripts.navigation_policy import ACTION_NAMES as NAV_ACTIONS, STATE_NAMES
    navigation = options.embodiment == "navigation"
    if options.balance_stops and not navigation:
        raise ValueError("Stop balancing is only supported for navigation training")
    embodiment = "milo-navigation-v1" if navigation else "milo-left-arm-v1"
    state_names = STATE_NAMES if navigation else ACTION_NAMES
    action_names = NAV_ACTIONS if navigation else ACTION_NAMES
    chunk_size, fps = (1, 1) if navigation else (50, 20)
    training_episodes = list(range(12)) if navigation else list(range(4))
    validation_episodes = list(range(12, 16)) if navigation else [4]
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(713)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report = {"status": "running", "steps": options.steps, "batch_size": options.batch_size,
              "base_revision": MODEL_REVISION, "training_episodes": training_episodes, "validation_episodes": validation_episodes,
              "embodiment": embodiment, "state_names": state_names, "action_names": action_names, "fps": fps,
              "balance_stops": options.balance_stops,
              "robot_motion_executed": False, "live_execution_enabled": False,
              "qualification": "Training pipeline pilot; closely spaced held-out episode is not generalization proof",
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "updates": []}
    try:
        parent_path = options.base.parent / "report.json"
        parent_steps = 0
        if parent_path.exists():
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            parent_hash = hashlib.sha256((options.base / "model.safetensors").read_bytes()).hexdigest()
            if parent.get("status") != "trained_and_reloaded" or parent.get("weights_sha256") != parent_hash:
                raise ValueError("Continuation requires a matching completed training report")
            if parent.get("embodiment", "milo-left-arm-v1") != embodiment:
                raise ValueError("Do not continue a checkpoint from a different embodiment")
            parent_steps = parent.get("total_steps", parent["steps"])
            report["parent_weights_sha256"] = parent_hash
        report.update(initialization_checkpoint=str(options.base.resolve()), total_steps=parent_steps + options.steps,
                      optimizer_state="new AdamW state; weights continued, not a full optimizer resume")
        if navigation:
            export = json.loads((options.dataset / "navigation-export.json").read_text(encoding="utf-8"))
            if (not export.get("verified") or export.get("embodiment") != embodiment or
                    export.get("train_episodes") != training_episodes or export.get("validation_episodes") != validation_episodes):
                raise ValueError("Navigation training requires a verified disjoint dataset export")
        dataset = LeRobotDataset("local/milo-navigation" if navigation else "local/milo-pilot", root=options.dataset,
                     token=False, video_backend="pyav", delta_timestamps={"action": [index / fps for index in range(chunk_size)]})
        validate_dataset(dataset, navigation)
        boundaries = [dataset.meta.episodes[index] for index in range(dataset.num_episodes)]
        train_indices = [index for episode in [boundaries[entry] for entry in training_episodes]
                         for index in range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"]))]
        validation_indices = [index for episode in [boundaries[entry] for entry in validation_episodes]
                      for index in range(int(episode["dataset_from_index"]), int(episode["dataset_to_index"]), 3 if navigation else 90)]
        stats = {key: get_feature_stats(np.array(dataset.hf_dataset.select(train_indices)[key], dtype=np.float32),
                                       axis=0, keepdims=False) for key in ("observation.state", "action")}
        sampling_weights = None
        if options.balance_stops:
            labels = np.array(dataset.hf_dataset.select(train_indices)["action"], dtype=np.float32)
            sampling_weights = torch.tensor(navigation_sampling_weights(labels), dtype=torch.float64)
            report["sampling"] = {"method": "training-only weighted sampling with replacement", "expected_stop_fraction": .5,
                                  "stop_examples": int(np.all(labels == 0, axis=1).sum()),
                                  "total_examples": len(labels), "stop_samples_drawn": 0, "samples_drawn": 0}
        config = PreTrainedConfig.from_pretrained(options.base, local_files_only=True)
        config.device = "cuda"
        config.load_vlm_weights = False
        config.vlm_model_name = str(options.backbone.resolve(strict=True))
        config.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(len(state_names),)),
                                 "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 240, 320))}
        config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(len(action_names),))}
        config.chunk_size = config.n_action_steps = chunk_size
        if navigation:
            config.normalization_mapping["STATE"] = NormalizationMode.IDENTITY
        config.empty_cameras = 0
        config.adapt_to_pi_aloha = False
        config.use_delta_joint_actions_aloha = False
        config.freeze_vision_encoder = True
        config.train_expert_only = True
        config.train_state_proj = True
        policy = SmolVLAPolicy.from_pretrained(options.base, config=config, local_files_only=True, strict=True).float()
        preprocess, postprocess = make_pre_post_processors(config, dataset_stats=stats)
        parameters = [value for value in policy.parameters() if value.requires_grad]
        report.update(parameters=sum(value.numel() for value in policy.parameters()),
                      trainable_parameters=sum(value.numel() for value in parameters),
                      train_frames=len(train_indices), validation_frames=len(validation_indices),
                      normalization_source="training episodes only; identity images; " +
                          ("identity state, mean/std action" if navigation else "mean/std state and action"))

        def batch(indices):
            return preprocess(default_collate([training_sample(dataset[index]) for index in indices]))

        def validation_loss():
            policy.eval()
            losses = []
            with torch.inference_mode():
                for index in validation_indices:
                    torch.manual_seed(1700 + index)
                    loss, _ = policy(batch([index]))
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite held-out loss")
                    losses.append(loss.item())
            return float(np.mean(losses))

        report["validation_loss_before"] = validation_loss()
        print(json.dumps({key: value for key, value in report.items() if key != "updates"}), flush=True)
        torch.manual_seed(713)
        sampler = torch.Generator().manual_seed(713)
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
        policy.train()
        for step in range(1, options.steps + 1):
            began = time.perf_counter()
            selected = (torch.multinomial(sampling_weights, options.batch_size, replacement=True, generator=sampler)
                        if sampling_weights is not None else torch.randint(len(train_indices), (options.batch_size,), generator=sampler)).tolist()
            indices = [train_indices[index] for index in selected]
            if sampling_weights is not None:
                report["sampling"]["stop_samples_drawn"] += int(np.all(labels[selected] == 0, axis=1).sum())
                report["sampling"]["samples_drawn"] += len(selected)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch(indices))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
            if not any(value.grad is not None for value in parameters):
                raise ValueError("The action expert received no gradients")
            optimizer.step()
            torch.cuda.synchronize()
            update = {"step": step, "loss": loss.item(), "gradient_norm": gradient.item(),
                      "wall_s": time.perf_counter() - began}
            report["updates"].append(update)
            if step == 1 or step % 100 == 0 or step == options.steps:
                print(json.dumps(update), flush=True)
        report["validation_loss_after"] = validation_loss()
        checkpoint = options.output / "checkpoint"
        policy.save_pretrained(checkpoint)
        preprocess.save_pretrained(checkpoint, config_filename="policy_preprocessor.json")
        postprocess.save_pretrained(checkpoint, config_filename="policy_postprocessor.json")
        report["weights_sha256"] = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
        del optimizer, parameters, policy
        gc.collect()
        torch.cuda.empty_cache()
        policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True).float().eval()
        preprocess, postprocess = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
        with torch.inference_mode():
            policy.reset()
            prediction = policy.predict_action_chunk(batch([validation_indices[0]]))
            decoded = torch.stack([postprocess(prediction[:, index]) for index in range(prediction.shape[1])], dim=1)
        report["reloaded_output"] = checked_actions(decoded.cpu().numpy(), chunk_size, len(action_names))
        report["status"] = "trained_and_reloaded"
        report["checkpoint"] = str(checkpoint)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(wall_s=time.perf_counter() - started, peak_allocated_vram_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                      peak_reserved_vram_mib=torch.cuda.max_memory_reserved() / 1024 ** 2)
        (options.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bounded native-Windows Milo SmolVLA training pilot; no robot connection or upload")
    parser.add_argument("--dataset", type=Path, default=Path(".runtime/datasets/milo-pilot"))
    parser.add_argument("--embodiment", choices=["manipulation", "navigation"], default="manipulation")
    parser.add_argument("--balance-stops", action="store_true", help="Sample navigation stop/move training labels with equal class probability")
    parser.add_argument("--base", type=Path, default=Path(".runtime/smolvla-base-cache/policy-c83c3163"))
    parser.add_argument("--backbone", type=Path, default=Path(".runtime/smolvla-base-cache/vlm-7b375e1b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=range(1, MAX_TRAINING_STEPS + 1), default=100)
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=1)
    run(parser.parse_args())