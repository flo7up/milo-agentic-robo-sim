import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def checked_actions(values, chunk_size, action_size):
    values = np.asarray(values)
    if values.shape != (1, chunk_size, action_size) or not np.isfinite(values).all():
        raise ValueError("Policy output has an unexpected shape or nonfinite values")
    return {"shape": list(values.shape), "finite": True, "min": float(values.min()),
            "max": float(values.max()), "first_action": values[0, 0].tolist(), "actions": values[0].tolist()}


def run(options):
    import psutil
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    options.output.mkdir(parents=True, exist_ok=False)
    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    report = {"model": MODEL_ID, "revision": MODEL_REVISION, "device": options.device, "dtype": "float32",
              "smoke_test_only": True, "robot_motion_executed": False, "fine_tuned": False,
              "input_note": "One saved Milo head-camera image repeated into the base checkpoint camera slots; synthetic zero state, not Milo joints or calibrated multi-view observations",
              "output_note": "Base checkpoint embodiment units; not Milo joint targets and never executed",
              "versions": {name: metadata.version(name) for name in ("lerobot", "torch", "transformers", "huggingface-hub")},
              "requests": [], "passed": False}
    torch.set_num_threads(options.threads)
    torch.manual_seed(713)
    report["cpu_threads"] = torch.get_num_threads()
    if options.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        report.update(cuda_runtime=torch.version.cuda, cuda_device=torch.cuda.get_device_name(0),
                      cuda_capability=list(torch.cuda.get_device_capability(0)))
    started = time.perf_counter()
    try:
        print("Downloading pinned public SmolVLA checkpoint; no robot connection", flush=True)
        checkpoint = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False,
            local_dir=options.cache / f"policy-{MODEL_REVISION[:8]}", max_workers=2,
            allow_patterns=["config.json", "model.safetensors", "policy_*.json", "policy_*.safetensors"]))
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        backbone_id = config.vlm_model_name
        backbone_revision = HfApi().model_info(backbone_id, token=False).sha
        backbone = snapshot_download(backbone_id, revision=backbone_revision, token=False,
            local_dir=options.cache / f"vlm-{backbone_revision[:8]}", max_workers=2,
            allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"])
        report.update(backbone=backbone_id, backbone_revision=backbone_revision,
                      download_s=time.perf_counter() - started)
        config.device = options.device
        config.load_vlm_weights = False
        config.vlm_model_name = str(Path(backbone).resolve())
        print("Loading full policy weights with strict key validation", flush=True)
        loaded_at = time.perf_counter()
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, local_files_only=True, strict=True).float().eval()
        preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": options.device},
                                    "tokenizer_processor": {"tokenizer_name": config.vlm_model_name}})
        report.update(load_s=time.perf_counter() - loaded_at, parameters=sum(value.numel() for value in policy.parameters()),
                      state_shape=list(config.input_features["observation.state"].shape),
                      action_shape=list(config.output_features["action"].shape),
                      cameras=list(config.image_features), chunk_size=config.chunk_size,
                      denoising_steps=config.num_steps, processor_image_size=list(config.resize_imgs_with_padding),
                      tokenizer_limit=config.tokenizer_max_length, use_cache=config.use_cache,
                      strict_weights_loaded=True)
        token_ids = policy.model.vlm_with_expert.processor.tokenizer(options.instruction + "\n", truncation=False)["input_ids"]
        if len(token_ids) > config.tokenizer_max_length:
            raise ValueError("Smoke instruction exceeds the checkpoint tokenizer limit")
        report["instruction"] = options.instruction
        report["instruction_tokens"] = len(token_ids)
        report["image_sha256"] = hashlib.sha256(options.image.read_bytes()).hexdigest()
        report["image_source"] = str(options.image)
        for index in range(options.requests + 1):
            policy.reset()
            began = time.perf_counter()
            with Image.open(options.image) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255
            frame = {"observation.state": torch.zeros(config.input_features["observation.state"].shape, dtype=torch.float32),
                     "task": options.instruction}
            for key, feature in config.image_features.items():
                image = torch.from_numpy(pixels).permute(2, 0, 1)
                frame[key] = torch.nn.functional.interpolate(image.unsqueeze(0), size=tuple(feature.shape[-2:]),
                    mode="bilinear", align_corners=False).squeeze(0)
            batch = preprocess(frame)
            processed_at = time.perf_counter()
            with torch.inference_mode():
                predicted = policy.predict_action_chunk(batch)
                inferred_at = time.perf_counter()
                decoded = torch.stack([postprocess(predicted[:, step, :]) for step in range(predicted.shape[1])], dim=1)
            result = checked_actions(decoded.detach().cpu().numpy(), config.chunk_size, config.action_feature.shape[0])
            normalized = checked_actions(predicted.detach().cpu().numpy(), config.chunk_size, config.action_feature.shape[0])
            result.update(index=index, warmup=index == 0, latency_s=time.perf_counter() - began,
                          preprocessing_s=processed_at - began, model_s=inferred_at - processed_at,
                          normalized_min=normalized["min"], normalized_max=normalized["max"])
            report["requests"].append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "actions"}), flush=True)
        report["passed"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        memory = psutil.Process().memory_info()
        report.update(total_wall_s=time.perf_counter() - started,
                      process_peak_ram_mib=getattr(memory, "peak_wset", memory.rss) / 1024 ** 2,
                      memory_note="Process RAM working-set peak on Windows; current RSS on platforms without peak_wset")
        if options.device == "cuda":
            report["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        (options.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-actuating SmolVLA base-model smoke test; no simulator or robot client")
    parser.add_argument("--image", type=Path, default=Path(".runtime/milo-demonstrations-pilot/episode-000/images/000000.png"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/smolvla-base-smoke"))
    parser.add_argument("--cache", type=Path, default=Path(".runtime/smolvla-base-cache"))
    parser.add_argument("--requests", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--threads", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--instruction", default="Pick up the red cube.")
    run(parser.parse_args())