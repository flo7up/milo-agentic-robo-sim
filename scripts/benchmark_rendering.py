import argparse
import asyncio
from io import BytesIO
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw
import pybullet as bullet
from scipy.ndimage import binary_dilation, binary_erosion, maximum_filter, minimum_filter

from backend.camera import scene_snapshot
from backend.challenges import PRESETS
from backend.contracts import Command
from backend.simulation import BulletSimulation

ROOT = Path(__file__).resolve().parents[1]


def reference_frame(sim, packet):
    camera = packet["camera"]
    view = bullet.computeViewMatrix(camera["eye"], camera["target"], camera["up"])
    projection = bullet.computeProjectionMatrixFOV(camera["vertical_fov_deg"], camera["width"] / camera["height"], camera["near_m"], camera["far_m"])
    rendered = bullet.getCameraImage(camera["width"], camera["height"], view, projection,
        renderer=bullet.ER_TINY_RENDERER, flags=bullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=sim.client)
    shape = (camera["height"], camera["width"])
    normalized = np.asarray(rendered[3]).reshape(shape)
    near, far = camera["near_m"], camera["far_m"]
    depth = near * far / (far - normalized.astype(np.float64) * (far - near))
    segments = np.asarray(rendered[4], dtype=np.int64).reshape(shape)
    self_mask = binary_dilation((segments >= 0) & ((segments & ((1 << 24) - 1)) == sim.robot))
    depth[self_mask | (normalized >= 1)] = np.nan
    image = Image.fromarray(np.asarray(rendered[2], dtype=np.uint8).reshape(*shape, 4)[:, :, :3])
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue(), depth


def export_cases(output, samples=5, widths=(320, 640), lighting_seed=0):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for scene in ("calibration", "kitchen_bathroom"):
        sim = BulletSimulation(challenge=PRESETS[scene]) if scene in PRESETS else BulletSimulation(scene=[
            {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
            {"name": "wall", "size": [.1, 4, 2], "position": [1.5, 0, 1], "color": [.4, .6, .5, 1]},
            {"name": "occluder", "size": [.12, .24, .65], "position": [.8, .35, .325], "color": [.7, .2, .15, 1]}])
        try:
            heads = [(0, 0), (.6, .4)] if scene == "calibration" else [(0, .12), (.7, .3), (-.6, .65)]
            for head_index, (yaw, pitch) in enumerate(heads):
                result = sim.execute(Command(run_id=sim.run_id, episode_epoch=sim.epoch, observation_seq=sim.seq,
                    action_id=str(uuid4()), tool="set_head", arguments={"yaw_rad": yaw, "pitch_rad": pitch, "duration_s": 1}))
                if result.status != "ok":
                    raise RuntimeError(f"Head setup failed: {result.message}")
                for width in widths:
                    name = f"{scene}-{head_index}-{width}"
                    case = output / name
                    case.mkdir()
                    before = sim.snapshot()
                    packet = scene_snapshot(sim, width, width * 3 // 4)
                    (case / "snapshot.json").write_text(json.dumps(packet), encoding="utf-8")
                    reference_frame(sim, packet)
                    timings = []
                    for _ in range(samples):
                        started = time.perf_counter()
                        rgb, depth = reference_frame(sim, packet)
                        timings.append((time.perf_counter() - started) * 1000)
                    if sim.snapshot() != before:
                        raise RuntimeError("Rendering mutated physics")
                    (case / "tiny.png").write_bytes(rgb)
                    np.save(case / "tiny-depth.npy", depth)
                    cases.append({"name": name, "scene": scene, "width": width, "height": width * 3 // 4,
                        "cpu_capture_ms": timings, "snapshot": f"{name}/snapshot.json"})
        finally:
            sim.close()
    manifest = {"schema": "milo-render-benchmark-v1", "cases": cases, "samples": samples, "lighting_seed": lighting_seed,
        "source": "scripted-physics-snapshots", "physics_mutated_by_capture": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def depth_agreement(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Depth dimensions differ")
    valid_reference, valid_candidate = np.isfinite(reference), np.isfinite(candidate)
    union = valid_reference | valid_candidate
    overlap = valid_reference & valid_candidate
    smooth = np.where(valid_reference, reference, 0)
    interior = binary_erosion(overlap, iterations=2) & (maximum_filter(smooth, size=5) - minimum_filter(smooth, size=5) < .10)
    errors = np.abs(reference[interior] - candidate[interior])
    iou = float(overlap.sum() / max(1, union.sum()))
    percentile = float(np.percentile(errors, 95)) if errors.size else None
    return {"valid_mask_iou": iou, "interior_pixels": int(errors.size), "interior_p95_error_m": percentile,
        "passed": bool(iou > .96 and errors.size > reference.size * .05 and percentile is not None and percentile < .03)}


def summarize(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    gpu = json.loads((output / "gpu-results.json").read_text(encoding="utf-8"))
    records = []
    contact = Image.new("RGB", (960, 272 * len(manifest["cases"]) + 32), "white")
    draw = ImageDraw.Draw(contact)
    for column, label in enumerate(("TinyRenderer / CPU", "Three.js / Standard", "Three.js / Enhanced")):
        draw.text((column * 320 + 12, 8), label, fill="black")
    for case_index, case in enumerate(manifest["cases"]):
        directory = output / case["name"]
        reference = np.load(directory / "tiny-depth.npy")
        top = 32 + case_index * 272
        draw.text((12, top + 8), case["name"], fill="black")
        contact.paste(Image.open(directory / "tiny.png").resize((320, 240)), (0, top + 32))
        for mode_index, quality in enumerate(("standard", "enhanced")):
            result = next(record for record in gpu["cases"] if record["name"] == case["name"] and record["quality"] == quality)
            depth = np.fromfile(directory / f"{quality}-depth.bin", dtype="<f4").reshape(reference.shape)
            image = Image.open(directory / f"{quality}.png").convert("RGB")
            colors = int(np.unique(np.asarray(image).reshape(-1, 3)[::16], axis=0).shape[0])
            alignment = depth_agreement(reference, depth)
            record = {"name": case["name"], "quality": quality, "width": case["width"], "height": case["height"],
                "depth": alignment, "sampled_rgb_colors": colors,
                "tiny_p50_ms": float(np.median(case["cpu_capture_ms"])), "tiny_p95_ms": float(np.percentile(case["cpu_capture_ms"], 95)),
                "gpu_p50_ms": float(np.median(result["capture_ms"])), "gpu_p95_ms": float(np.percentile(result["capture_ms"], 95)),
                "roundtrip_p50_ms": float(np.median(result["roundtrip_ms"])),
                "passed": bool(alignment["passed"] and colors > 32 and image.size == (case["width"], case["height"]))}
            records.append(record)
            contact.paste(image.resize((320, 240)), ((mode_index + 1) * 320, top + 32))
    contact.save(output / "comparison.png")
    report = {"source": manifest["source"], "renderer": gpu["renderer"], "hardware": gpu["hardware"],
        "gpu_memory_sample": gpu.get("gpu_memory_sample"), "samples_per_case": manifest["samples"], "lighting_seed": manifest["lighting_seed"],
        "passed": bool(gpu["hardware"] and all(record["passed"] for record in records)), "cases": records,
        "limitations": ["Offline frozen snapshots; no live controller or task-quality benchmark.",
            "Warm synchronous RGB PNG plus axial depth readback; excludes scene export and process startup.",
            "GPU roundtrip includes browser IPC; GPU memory is a whole-device sample, not a peak.",
            "Depth agreement excludes discontinuity edges; rounded visual edges retain conservative box collisions.",
            "New appearance is not qualified for existing policies or color-based perception."]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_gpu(output):
    subprocess.run(["node", str(ROOT / "scripts" / "render_camera.mjs"), "--input", str(output)], check=True, timeout=300, cwd=ROOT)


async def policy_comparison(output):
    from backend.local_navigation import LocalNavigationClient
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0])
    if free_mib < 4000:
        await asyncio.to_thread(run_gpu, output)
        return {"status": "skipped_insufficient_free_vram", "free_mib_before": free_mib, "required_free_mib": 4000,
            "reason": "Existing GPU workloads were left untouched; no additional policy process started."}
    sim = BulletSimulation(width=320, height=240, challenge=PRESETS["kitchen_bathroom"])
    try:
        observation = sim.observe()
        image = sim.frame(observation.frame_ref)
    finally:
        sim.close()
    client = LocalNavigationClient()
    try:
        started = time.perf_counter()
        await client.start()
        startup = time.perf_counter() - started
        await client.predict(observation, image)
        baseline = []
        for _ in range(5):
            started = time.perf_counter()
            await client.predict(observation, image)
            baseline.append((time.perf_counter() - started) * 1000)
        rendering = asyncio.create_task(asyncio.to_thread(run_gpu, output))
        concurrent = []
        try:
            while not rendering.done() and len(concurrent) < 120:
                started = time.perf_counter()
                await client.predict(observation, image)
                concurrent.append((time.perf_counter() - started) * 1000)
        finally:
            await rendering
        return {"status": "measured_shadow_inference", "checkpoint": str(client.checkpoint.relative_to(ROOT)),
            "startup_s": startup, "free_mib_before": free_mib, "baseline_ms": baseline, "concurrent_ms": concurrent,
            "outputs_executed": 0, "input": "One frozen TinyRenderer head image plus its AgentObservation; repeated for load only.",
            "limitations": "Serial predictions overlap the rendering batch including scene loads; no policy accuracy or training evaluation."}
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5, choices=range(1, 101))
    parser.add_argument("--widths", type=int, nargs="+", default=[320, 640], choices=[160, 320, 640])
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--lighting-seed", type=int, default=0)
    parser.add_argument("--with-local-policy", action="store_true")
    arguments = parser.parse_args()
    if not 0 <= arguments.lighting_seed <= 2147483647:
        parser.error("--lighting-seed must be between 0 and 2147483647; 0 uses nominal lighting")
    if arguments.export_only and arguments.with_local_policy:
        parser.error("--with-local-policy cannot be combined with --export-only")
    output = arguments.output.resolve()
    export_cases(output, arguments.samples, arguments.widths, arguments.lighting_seed)
    if arguments.export_only:
        print(json.dumps({"exported": str(output)}))
        return
    policy = asyncio.run(policy_comparison(output)) if arguments.with_local_policy else None
    if not arguments.with_local_policy:
        run_gpu(output)
    report = summarize(output)
    report["local_policy"] = policy
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Renderer qualification failed; inspect report.json")


if __name__ == "__main__":
    main()