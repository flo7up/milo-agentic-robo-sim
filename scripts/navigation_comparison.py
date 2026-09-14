import argparse
import asyncio
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / ".runtime/navigation-study"


def fingerprint(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def host_memory():
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    process = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb):
        raise ctypes.WinError()
    return {"rss_mib": memory.rss / 1024 ** 2, "peak_rss_mib": memory.peak_rss / 1024 ** 2}


def policy_process(options):
    sys.path[:0] = [str(STUDY / "packages"), str(STUDY / "upstream/nomad/train"),
                    str(STUDY / "upstream/diffusion_policy")]
    import contextlib
    import numpy as np
    from PIL import Image
    import psutil
    import torch
    from torchvision import transforms
    import yaml
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    config = yaml.safe_load((STUDY / "upstream/nomad/train/config/nomad.yaml").read_text())
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    began = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        encoder = replace_bn_with_gn(NoMaD_ViNT(obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"], mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"], mha_ff_dim_factor=config["mha_ff_dim_factor"]))
        network = ConditionalUnet1D(input_dim=2, global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"], cond_predict_scale=config["cond_predict_scale"])
        model = NoMaD(vision_encoder=encoder, noise_pred_net=network,
            dist_pred_net=DenseNetwork(embedding_dim=config["encoding_size"]))
        model.load_state_dict(torch.load(options.checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(options.device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225])])
    scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon")
    process = psutil.Process()
    if options.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"ready": True, "backend": "nomad", "parameters": sum(value.numel() for value in model.parameters()),
        "checkpoint_sha256": fingerprint(options.checkpoint), "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if options.device == "cuda" else "cpu",
        "load_s": time.perf_counter() - began, "history_frames": 4, "image_size": [96, 96],
        "goals": ["image", "explore"], "waypoints": 8, "waypoint_scale_m": .05}), flush=True)

    def image_tensor(encoded):
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            if image.width > 640 or image.height > 480:
                raise ValueError("Policy image too large")
            return transform(image.convert("RGB").resize((96, 96))).unsqueeze(0).to(options.device)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("reset"):
                torch.manual_seed(request.get("seed", options.seed))
                print(json.dumps({"reset": True}), flush=True)
                continue
            if set(request) - {"images", "goal_image", "kind"} or request["kind"] not in {"image", "explore"}:
                raise ValueError("NoMaD only receives RGB history and optional goal image")
            if len(request["images"]) != 4 or (request["kind"] == "image" and not request.get("goal_image")):
                raise ValueError("Four RGB observations and a real image goal are required")
            began = time.perf_counter()
            cpu_before = sum(process.cpu_times()[:2])
            observations = torch.cat([image_tensor(image) for image in request["images"]], dim=1)
            goal_image = image_tensor(request["goal_image"]) if request["kind"] == "image" else torch.randn((1, 3, 96, 96), device=options.device)
            with torch.inference_mode():
                condition = model("vision_encoder", obs_img=observations, goal_img=goal_image,
                    input_goal_mask=torch.tensor([int(request["kind"] == "explore")], device=options.device)).repeat(8, 1)
                action = torch.randn((8, 8, 2), device=options.device)
                scheduler.set_timesteps(10)
                for timestep in scheduler.timesteps:
                    noise = model("noise_pred_net", sample=action, timestep=timestep, global_cond=condition)
                    action = scheduler.step(model_output=noise, timestep=timestep, sample=action).prev_sample
                minimum = torch.tensor([-2.5, -4.], device=options.device)
                maximum = torch.tensor([5., 4.], device=options.device)
                paths = torch.cumsum((action + 1) / 2 * (maximum - minimum) + minimum, dim=1) * .05
                result = paths.cpu().tolist()
            memory = process.memory_info()
            print(json.dumps({"paths_m": result, "selected_index": 0, "latency_s": time.perf_counter() - began,
                "cpu_s": sum(process.cpu_times()[:2]) - cpu_before, "rss_mib": memory.rss / 1024 ** 2,
                "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 1024 ** 2,
                "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if options.device == "cuda" else 0,
                "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if options.device == "cuda" else 0}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": type(error).__name__, "message": str(error)}), flush=True)


def navdp_process(options):
    import contextlib
    sys.path[:0] = [str(STUDY / "packages"), str(STUDY / "upstream/navdp/baselines/navdp")]
    import torch
    import numpy as np
    import psutil
    from PIL import Image
    from policy_network import NavDP_Policy
    from policy_agent import NavDP_Agent
    torch.set_num_threads(4)
    torch.manual_seed(options.seed)
    began = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        model = NavDP_Policy(image_size=224, memory_size=8, predict_size=24, temporal_depth=16,
            heads=8, token_dim=384, device=options.device)
        weights = torch.load(options.checkpoint, map_location="cpu", weights_only=True)
        for name in ("pixel_aux_head", "image_aux_head"):
            weight, bias = weights[name + ".weight"], weights[name + ".bias"]
            model.add_module(name, torch.nn.Linear(weight.shape[1], bias.shape[0]))
        model.load_state_dict(weights, strict=True)
        model.to(options.device).eval()
        processor = NavDP_Agent.__new__(NavDP_Agent)
        processor.image_size = 224
    process = psutil.Process()
    if options.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"ready": True, "backend": "navdp", "parameters": sum(value.numel() for value in model.parameters()),
        "checkpoint_sha256": fingerprint(options.checkpoint), "checkpoint_source": "X-NavDP public navdp_pretrain artifact; auxiliary heads retained",
        "strict_load": True, "torch": torch.__version__, "device": torch.cuda.get_device_name() if options.device == "cuda" else "cpu",
        "load_s": time.perf_counter() - began, "history_frames": 8, "image_size": [224, 224],
        "goals": ["point", "explore"], "waypoints": 24, "dtype": "float32"}), flush=True)

    def pixels(encoded):
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            if image.width > 640 or image.height > 480:
                raise ValueError("Policy image too large")
            return np.array(image.convert("RGB"))[:, :, ::-1].copy()

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("reset"):
                torch.manual_seed(request.get("seed", options.seed))
                print(json.dumps({"reset": True}), flush=True)
                continue
            if set(request) - {"kind", "images", "depth_m", "goal_m"} or request["kind"] not in {"point", "explore"} or len(request["images"]) != 8:
                raise ValueError("NavDP probe supports eight RGB frames, paired metric depth and optional point goal")
            began = time.perf_counter()
            cpu_before = sum(process.cpu_times()[:2])
            rgb = [pixels(image) for image in request["images"]]
            depth = np.array(request["depth_m"], dtype=np.float32).reshape(*rgb[-1].shape[:2], 1)
            depth[~np.isfinite(depth)] = 0.
            images = processor.process_image(np.stack(rgb))[None]
            depths = processor.process_depth(depth[None])
            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                if request["kind"] == "point":
                    goal = np.asarray(request["goal_m"], dtype=float)
                    if goal.shape != (2,) or not np.isfinite(goal).all() or goal[0] < 0 or np.abs(goal).max() > 10:
                        raise ValueError("Point goal must be in the observed forward half-plane; turn and recapture first")
                    output = model.predict_pointgoal_action(np.array([[*goal, 0.]]), images, depths)
                else:
                    output = model.predict_nogoal_action(images, depths)
            paths, values, selected, _ = output
            def array(value):
                return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            paths, values, selected = array(paths), array(values), array(selected)
            memory = process.memory_info()
            print(json.dumps({"path_poses_m_rad": selected[0, 0].tolist(), "critic_values": values.tolist(),
                "paths_m": paths[0, :, :, :2].tolist(), "latency_s": time.perf_counter() - began,
                "cpu_s": sum(process.cpu_times()[:2]) - cpu_before, "rss_mib": memory.rss / 1024 ** 2,
                "peak_rss_mib": getattr(memory, "peak_wset", memory.rss) / 1024 ** 2,
                "peak_allocated_vram_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if options.device == "cuda" else 0,
                "peak_reserved_vram_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if options.device == "cuda" else 0,
                "low_critic_fallback": "not applied; unsafe trajectories remain subject to worker rejection"}, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": type(error).__name__, "message": str(error)}), flush=True)


def summarize(options):
    options.output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in ("conventional-v2", "two-room-conventional", "conventional-rescan-v1", "nomad-closed-loop-v2",
                 "navdp-closed-loop-v2", "nomad-lifecycle", "navdp-lifecycle"):
        root = STUDY / name
        for path in sorted(root.glob("*/report.json")):
            report = json.loads(path.read_text())
            recording = report.get("recording", {})
            events = report.get("events", [])
            row = {key: report.get(key) for key in ("case", "repeat", "backend", "status", "success", "false_success", "control_wall_s",
                "wall_s", "minimum_clearance_m", "final_clearance_m", "interventions", "observed_area_m2", "repeated_endpoints", "host_resources")}
            samples_path = path.parent / "recording/trajectory.jsonl"
            samples = [json.loads(line) for line in samples_path.read_text().splitlines()] if samples_path.exists() else []
            if report["case"] in {"return", "two_rooms_return"} and samples:
                return_error = math.dist(samples[-1]["position_m"][:2], samples[0]["position_m"][:2])
                row["physical_return_error_m"] = return_error
                row["physically_crossed_doorway"] = max(sample["position_m"][0] for sample in samples) > .92
                row["success"] = bool(row["success"] and return_error < .15 and
                    (report["case"] != "two_rooms_return" or row["physically_crossed_doorway"]))
            if report["case"] == "explore":
                row["success"] = None
                row["success_note"] = "Exploration is measured as coverage/travel; no binary target was specified"
            if report["backend"] == "navdp" and report["case"] == "straight" and report["status"] == "local_arrival_only":
                row["false_success"] = False
                row["scoring_note"] = "Corrected v2 harness classification: local trajectory arrival is not a model mission-completion claim"
            if report.get("control_wall_s", 0) and report.get("control_wall_s", 0) > 45 and report["status"] == "Following observed path":
                row["status"] = "time_limit"
            row.update(artifact=path.relative_to(ROOT).as_posix(), run_set=name, path_length_m=recording.get("actual_distance_m"),
                contact_episodes=recording.get("contact_episodes"), dropped_records=recording.get("dropped_records"),
                predictions=sum("prediction" in event for event in events), routes=sum("execution" in event for event in events),
                route_update_gaps_s=[event["execution"]["maximum_update_gap_s"] for event in events if "execution" in event],
                command_ages_s=[event["observation_age_s"] for event in events if "observation_age_s" in event],
                inference_s=[event["prediction"]["latency_s"] for event in events if "prediction" in event],
                roundtrip_s=[event["prediction"]["transport_roundtrip_s"] for event in events if "prediction" in event],
                allocated_vram_mib=max([event["prediction"]["peak_allocated_vram_mib"] for event in events if "prediction" in event], default=None),
                policy_peak_rss_mib=max([event["prediction"]["peak_rss_mib"] for event in events if "prediction" in event], default=None))
            results.append(row)
    def distribution(values):
        values = sorted(value for value in values if value is not None)
        return {"count": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)] if values else None}
    aggregate = []
    for backend in ("conventional", "navdp", "nomad"):
        selected = [row for row in results if row["backend"] == backend]
        aggregate.append({"backend": backend, "trials": len(selected), "contacts": sum(row["contact_episodes"] or 0 for row in selected),
            "false_success": sum(bool(row["false_success"]) for row in selected),
            "inference_s": distribution([value for row in selected for value in row["inference_s"]]),
            "roundtrip_s": distribution([value for row in selected for value in row["roundtrip_s"]]),
            "observation_age_s": distribution([value for row in selected for value in row["command_ages_s"]]),
            "maximum_allocated_vram_mib": max([row["allocated_vram_mib"] for row in selected if row["allocated_vram_mib"] is not None], default=None),
            "maximum_policy_rss_mib": max([row["policy_peak_rss_mib"] for row in selected if row["policy_peak_rss_mib"] is not None], default=None)})
    summary = {"results": results, "aggregate": aggregate, "qualifications": [
        "Different goal capabilities: NoMaD image/explore only; RGB-D safety common to all, not RGB-only robot safety",
        "Scripted executive, no new Luna semantic evaluation; repeated fixed starts, not independent layout generalization",
        "Trials span reported source revisions; compare matching sets, not a pooled success rate",
        "Per-process tensor VRAM differs from whole-device VRAM; renderer subprocess memory and host-wide peak are not measured"]}
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


async def probe(options):
    options.output.mkdir(parents=True, exist_ok=False)
    log = (options.output / "policy.log").open("w", encoding="utf-8")
    process = await asyncio.create_subprocess_exec(str(ROOT / ".runtime/smolvla-env/Scripts/python.exe"), "-u", "-m",
        "scripts.navigation_comparison", "--stage", "policy", "--checkpoint", str(options.checkpoint),
        "--device", options.device, "--backend", options.backend, "--seed", str(options.seed), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=log, limit=8_000_000, cwd=ROOT)
    report = {"stage": "no-motion-probe", "robot_motion": False, "results": []}
    try:
        ready = await asyncio.wait_for(process.stdout.readline(), 180)
        if not ready:
            raise RuntimeError("Policy startup failed; inspect policy.log")
        report["model"] = json.loads(ready)
        print(json.dumps(report["model"]), flush=True)
        encoded = base64.b64encode(options.image.read_bytes()).decode("ascii")
        for index in range(4):
            request = {"kind": "explore", "images": [encoded] * (8 if options.backend == "navdp" else 4)}
            if options.backend == "navdp":
                request["depth_m"] = json.loads(options.depth.read_text())["depth_m"]
            process.stdin.write((json.dumps(request) + "\n").encode())
            await process.stdin.drain()
            response = json.loads(await asyncio.wait_for(process.stdout.readline(), 30))
            if "error" in response:
                raise RuntimeError(response)
            report["results"].append({"warmup": index == 0, **response})
            print(json.dumps({key: value for key, value in response.items() if key != "paths_m"}), flush=True)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 10)
        except TimeoutError:
            process.kill()
            await process.wait()
        log.close()
        (options.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


async def study(options):
    import numpy as np
    import pybullet as bullet
    from uuid import uuid4
    from backend.challenges import get_challenge
    from backend.contracts import Command
    from backend.continuous_navigation import ContinuousScan, ContinuousTarget
    from backend.navigation_backends import NavigationGoal, NavigationImages, NOMAD_CAPABILITIES, NAVDP_CAPABILITIES, relative_proposal, navdp_camera_proposal, navdp_camera_goal
    from backend.recording import RunRecorder
    from backend.simulation import MotionError
    from backend.worker import SimulationWorker

    options.output.mkdir(parents=True, exist_ok=False)
    if not 0 < options.time_limit <= 180:
        raise ValueError("Study deadline must be positive and at most 180 seconds")
    versions = {}
    for name in ("navdp", "nomad", "microduck", "diffusion_policy"):
        versions[name] = subprocess.check_output(["git", "-C", str(STUDY / "upstream" / name), "rev-parse", "HEAD"], text=True).strip()
    files = [ROOT / name for name in ("backend/agent.py", "backend/worker.py", "backend/navigation.py", "backend/continuous_navigation.py",
        "backend/navigation_backends.py", "backend/spatial.py", "backend/simulation.py", "scripts/navigation_comparison.py")]
    manifest = {"backend": options.backend, "rendering": options.rendering, "seed": options.seed, "repeats": options.repeats,
        "time_limit_s": options.time_limit, "max_decisions": options.decisions, "upstream": versions,
        "study_speed_cap_mps": options.speed,
        "bounded_rescans": options.rescans,
        "source_sha256": {str(path.relative_to(ROOT)): fingerprint(path) for path in files},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"], text=True).strip(),
        "controller_inputs": "Paired RGB/depth and encoder observations only; no fixture/evaluator coordinates",
        "executive": "Scripted observed-candidate selection, not Luna semantic navigation",
        "safety": "Same depth-map and simulator-privileged full-body shield for all backends; NoMaD policy input remains RGB only",
        "nomad_execution": "Released sample 0, first three predicted waypoints, original 0.05 m scale; retimed by common controller; stop-and-observe, not asynchronous deployment"}
    manifest["navdp_execution"] = "Public NavDP pretrain checkpoint; first eight selected camera-local poses transformed through measured head calibration; predicted yaw retained in raw log but XY geometrically retimed, matching upstream tracker semantics; no low-critic lateral fallback"
    (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    process = None
    log = None
    results = []
    from backend import continuous_navigation
    original_speed = continuous_navigation.CONTINUOUS_SPEED_MPS
    continuous_navigation.CONTINUOUS_SPEED_MPS = options.speed
    if options.backend in {"nomad", "navdp"}:
        log = (options.output / "policy.log").open("w", encoding="utf-8")
        process = await asyncio.create_subprocess_exec(str(ROOT / ".runtime/smolvla-env/Scripts/python.exe"), "-u", "-m",
            "scripts.navigation_comparison", "--stage", "policy", "--checkpoint", str(options.checkpoint), "--device", options.device,
            "--seed", str(options.seed), "--backend", options.backend, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=8_000_000, cwd=ROOT)
        line = await asyncio.wait_for(process.stdout.readline(), 180)
        if not line:
            await process.wait()
            log.close()
            raise RuntimeError("NoMaD startup failed; inspect policy.log")
        manifest["model"] = json.loads(line)
        (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    async def infer(payload):
        began = time.perf_counter()
        process.stdin.write((json.dumps(payload) + "\n").encode())
        await process.stdin.drain()
        response = json.loads(await asyncio.wait_for(process.stdout.readline(), 15))
        if "error" in response:
            raise RuntimeError(response)
        response["transport_roundtrip_s"] = time.perf_counter() - began
        return response

    try:
        for case in options.cases:
            for repeat in range(options.repeats):
                if options.backend == "nomad" and case not in {"explore", "return", "cancel", "goal_change", "delayed"}:
                    results.append({"case": case, "repeat": repeat, "status": "unsupported_goal", "success": None})
                    continue
                directory = options.output / f"{case}-{repeat:02d}"
                directory.mkdir()
                fixture = "kitchen_bathroom" if case == "rooms_return" else "park" if case == "doorway" else "local_park"
                challenge = get_challenge(fixture)
                if case == "two_rooms_return":
                    challenge = challenge.model_copy(update={"objects": challenge.objects + [
                        {"name": f"study_partition_{side}", "size": [.12, 2.1, .8], "position": [.8, sign * 1.85, .4],
                         "color": [.6, .6, .65, 1.]} for side, sign in (("north", 1), ("south", -1))]})
                if case in {"obstacle", "dead_end"}:
                    objects = challenge.objects + [{"name": "study_obstacle", "size": [.15, .3 if case == "obstacle" else 3., .5],
                        "position": [.85, 0., .25], "color": [.6, .3, .2, 1.]}]
                    challenge = challenge.model_copy(update={"objects": objects})
                worker = SimulationWorker(challenge=challenge, rendering=options.rendering, pace=True)
                recorder = None
                state = {"phase": "preparing"}
                report = {"case": case, "repeat": repeat, "backend": options.backend, "status": "running", "success": False,
                    "false_success": False, "events": [], "minimum_clearance_m": None, "interventions": 0,
                    "qualification": "Bounded scripted mission; independent physical scoring, no hidden goals used for control"}
                seen = []
                began = time.perf_counter()
                motion_began = None
                cpu_started = time.process_time()
                try:
                    await asyncio.wrap_future(worker.ready)
                    initial_physics = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client))
                    recorder = RunRecorder(directory / "recording", context=lambda: state)
                    await worker.call(lambda sim: setattr(worker, "recorder", recorder))
                    def install_metrics(sim):
                        prior = sim.on_tick
                        last_tick = [-1000]
                        def sample():
                            prior()
                            if sim.ticks - last_tick[0] < 60:
                                return
                            last_tick[0] = sim.ticks
                            values = [point[8] for item in sim.objects if item["name"] != "floor" and not item.get("marker")
                                      for point in bullet.getClosestPoints(sim.robot, item["id"], 2., physicsClientId=sim.client)]
                            if values:
                                clearance = min(values)
                                report["minimum_clearance_m"] = min(clearance, report["minimum_clearance_m"]) if report["minimum_clearance_m"] is not None else clearance
                        sim.on_tick = sample
                    await worker.call(install_metrics)

                    async def command(tool, arguments):
                        result = await worker.execute(Command(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
                            observation_seq=worker.latest["observation"]["seq"], action_id=str(uuid4()), tool=tool, arguments=arguments))
                        if result.status != "ok":
                            raise MotionError(result.error or "COMMAND_FAILED", result.message)

                    await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch, compact_arms=True))
                    await command("set_head", {"yaw_rad": 0., "pitch_rad": .3, "duration_s": 1.})
                    goal = NavigationGoal(kind="explore", evidence="exploration")
                    images = NavigationImages(8 if options.backend == "navdp" else 4)
                    ticket, sensor, image = await worker.capture_navigation_frame(goal)
                    start_pose = list(ticket.odometry_m_rad)
                    report["initial_observed_area_m2"] = float(np.count_nonzero(worker.spatial_map.cells == 0) * .05 ** 2)
                    start_view = images.remember(ticket, image)
                    if process:
                        await infer({"reset": True, "seed": options.seed + repeat})
                        warmup = {"kind": "explore", "images": [base64.b64encode(image).decode("ascii")] * images.frames.maxlen}
                        if options.backend == "navdp":
                            warmup["depth_m"] = sensor.depth_m
                        await infer(warmup)
                    report["preparation_s"] = time.perf_counter() - began
                    motion_began = time.perf_counter()
                    visited_cells = set()
                    returning = False
                    rescans = 0
                    for decision in range(options.decisions):
                        if time.perf_counter() - motion_began > options.time_limit:
                            report["status"] = "time_limit"
                            break
                        decision_began = time.perf_counter()
                        if (options.backend == "nomad" and (case != "return" or returning)) or options.backend == "navdp":
                            state["phase"] = "thinking"
                            if returning and options.backend == "nomad":
                                goal = NavigationGoal(kind="image", evidence="remembered_view", view_id=start_view)
                            if options.backend == "navdp" and case != "explore":
                                sensor, image, candidates, _ = await worker.continuous_candidates(seen)
                                if not candidates:
                                    report["status"] = "no_observed_route"
                                    break
                                desired = np.array(start_pose[:2] if returning else [.8, .55] if case == "turning" else [1.2, 0.])
                                selected = min(candidates, key=lambda item: np.linalg.norm(np.array(item["target_m"]) - desired))
                                grounded_target = list(selected["target_m"])
                                goal = NavigationGoal(kind="point", evidence="observed_map", point_m=grounded_target)
                            ticket, sensor, image = await worker.capture_navigation_frame(goal)
                            images.frames.clear()
                            for history_index in range(images.frames.maxlen):
                                if history_index:
                                    await command("wait", {"duration_s": .25})
                                    ticket, sensor, image = await worker.capture_navigation_frame(goal)
                                images.append(ticket, image)
                            if len(images.frames) != images.frames.maxlen:
                                raise MotionError("INSUFFICIENT_FRESH_HISTORY", "Sensor latency interrupted the required temporal history")
                            payload = {"kind": goal.kind, "images": [base64.b64encode(image).decode("ascii") for _, image in images.frames]}
                            if returning and options.backend == "nomad":
                                payload["goal_image"] = base64.b64encode(images.goal_image(start_view, ticket, time.monotonic())).decode("ascii")
                            if options.backend == "navdp":
                                payload["depth_m"] = sensor.depth_m
                                if goal.kind == "point":
                                    payload["goal_m"] = navdp_camera_goal(ticket, goal)
                            response = await infer(payload)
                            report["events"].append({"decision": decision, "source": options.backend, "prediction": response,
                                "observation_age_s": time.monotonic() - ticket.captured_at, "goal": goal.model_dump()})
                            if options.backend == "navdp":
                                proposal = navdp_camera_proposal(ticket, response["path_poses_m_rad"][:8])
                            else:
                                proposal = relative_proposal(ticket, "nomad", response["paths_m"][0][:3])
                            if case == "delayed":
                                proposal.ticket = proposal.ticket.model_copy(update={"captured_at": proposal.ticket.captured_at - 2.})
                            age = time.monotonic() - ticket.captured_at
                            event = {"decision": decision, "source": options.backend, "prediction": response,
                                "observation_age_s": age, "proposal": proposal.model_dump(), "goal": goal.model_dump()}
                            report["events"][-1].update(event)
                            await worker.start_navigation_proposal(proposal, goal, NAVDP_CAPABILITIES if options.backend == "navdp" else NOMAD_CAPABILITIES)
                        else:
                            state["phase"] = "planning"
                            sensor, image, candidates, _ = await worker.continuous_candidates(seen)
                            if not candidates:
                                if rescans < options.rescans and not worker.sim.cancel.is_set():
                                    rescans += 1
                                    report["interventions"] += 1
                                    report["events"].append({"decision": decision, "source": "mission_executive", "action": "rescan",
                                        "reason": "No reachable observed candidate", "attempt": rescans})
                                    await command("drive_base", {"linear_mps": 0., "angular_radps": .4, "duration_s": 1.5})
                                    await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch))
                                    await command("set_head", {"yaw_rad": 0., "pitch_rad": .3, "duration_s": 1.})
                                    continue
                                report["status"] = "no_observed_route"
                                break
                            position = np.array(sensor.odometry_m_rad[:2])
                            if returning:
                                desired = np.array(start_pose[:2])
                            elif case == "turning":
                                desired = np.array([.8, .55])
                            elif case in {"explore", "rooms_return"}:
                                desired = np.array(max(candidates, key=lambda item: min(np.linalg.norm(np.array(item["target_m"]) - point) for point in seen or [position]))["target_m"])
                            else:
                                desired = np.array([1.2, 0.])
                            selected = min(candidates, key=lambda item: np.linalg.norm(np.array(item["target_m"]) - desired))
                            grounded_target = list(selected["target_m"])
                            report["events"].append({"decision": decision, "source": "conventional", "selected": selected,
                                "observation_age_s": time.monotonic() - sensor.captured_at})
                            request = ContinuousTarget(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
                                spatial_sequence=sensor.sequence, pixel=selected.get("pixel") or [.5, .5])
                            if case == "delayed":
                                sensor = sensor.model_copy(update={"captured_at": sensor.captured_at - 46.})
                            await worker.start_continuous(request, selected_sensor=sensor, selected_target=grounded_target)
                        state["phase"] = "executing"
                        route_started = time.perf_counter()
                        while worker.continuous.active:
                            if case in {"cancel", "goal_change"} and time.perf_counter() - route_started > .5:
                                worker.stop()
                                break
                            if time.perf_counter() - motion_began > options.time_limit:
                                worker.stop()
                                report["status"] = "time_limit"
                                break
                            await asyncio.sleep(.1)
                            recorder.flush()
                        route = worker.continuous.state()
                        report["events"][-1]["execution"] = route
                        report["events"][-1]["capture_to_route_end_s"] = time.perf_counter() - decision_began
                        pose = await worker.call(lambda sim: sim.odometry.tolist())
                        physical_pose = await worker.call(lambda sim: bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0])
                        seen.append(pose[:2])
                        visited_cells.add((int(pose[0] / .25), int(pose[1] / .25)))
                        if case in {"cancel", "goal_change"}:
                            ticks = worker.sim.ticks
                            await worker.call(lambda sim: None)
                            report["success"] = worker.sim.ticks == ticks and worker.sim.cancel.is_set() and not worker.navigation.buffer
                            report["status"] = "cancelled_safely" if report["success"] else "cancel_failed"
                            if case == "goal_change" and report["success"]:
                                await worker.resume_manual(expected_stop_revision=worker.stop_revision)
                                old_goal = NavigationGoal(kind="explore", evidence="exploration")
                                old_frame, _, _ = await worker.capture_navigation_frame(old_goal)
                                obsolete = relative_proposal(old_frame, "nomad", [[.1, 0.], [.2, 0.]])
                                new_goal = NavigationGoal(kind="explore", evidence="exploration")
                                await worker.capture_navigation_frame(new_goal)
                                try:
                                    await worker.start_navigation_proposal(obsolete, old_goal, NOMAD_CAPABILITIES)
                                except MotionError as error:
                                    report["goal_change_rejection"] = error.code
                                    report["success"] = error.code == "STALE_PLAN"
                                else:
                                    report["success"] = False
                                report["status"] = "old_goal_rejected" if report["success"] else "goal_change_failed"
                            break
                        if route["status"] != "arrived":
                            if report["status"] != "time_limit":
                                report["status"] = route["reason"]
                            break
                        if case in {"straight", "turning", "doorway", "obstacle"}:
                            initial_yaw = bullet.getEulerFromQuaternion(initial_physics[1])[2]
                            rotation = np.array([[math.cos(initial_yaw), -math.sin(initial_yaw)], [math.sin(initial_yaw), math.cos(initial_yaw)]])
                            world_target = np.array(initial_physics[0][:2]) + rotation @ np.array(grounded_target)
                            error = float(np.linalg.norm(np.array(physical_pose[:2]) - world_target))
                            objective_reached = error < .08 and (case not in {"doorway", "obstacle"} or pose[0] > 1.05)
                            report.update(success=objective_reached, false_success=options.backend == "conventional" and error >= .08, status="arrived" if objective_reached else "local_arrival_only", goal_error_m=error,
                                grounded_goal_m=grounded_target, independent_physics_goal_m=world_target.tolist())
                            if objective_reached or case in {"straight", "turning"}:
                                break
                        if case in {"return", "rooms_return", "two_rooms_return"}:
                            if returning and math.dist(pose[:2], start_pose[:2]) < .15:
                                report.update(success=True, status="returned")
                                break
                            returning = decision >= (1 if case == "rooms_return" else 0)
                        if case == "dead_end":
                            report["status"] = "stopped_before_obstacle"
                            break
                    else:
                        report["status"] = "decision_limit"
                    report["control_wall_s"] = time.perf_counter() - motion_began
                    report["distinct_endpoint_cells_025m"] = len(visited_cells)
                    report["repeated_endpoints"] = len(seen) - len(visited_cells)
                except Exception as error:
                    report["status"] = getattr(error, "code", type(error).__name__)
                    report["error"] = str(error)
                    if case == "delayed" and report["status"] in {"SPATIAL_STALE", "STALE_OBSERVATION"}:
                        report["success"] = True
                finally:
                    if motion_began is not None:
                        report.setdefault("control_wall_s", time.perf_counter() - motion_began)
                    worker.stop()
                    if worker.sim:
                        def finish(sim):
                            if worker.navigation:
                                worker.navigation.cancel(sim, "Evaluation finished")
                            if recorder:
                                recorder.capture(worker)
                                worker.recorder = None
                            readings = []
                            for item in sim.objects:
                                if item["name"] != "floor" and not item.get("marker"):
                                    readings.extend(point[8] for point in bullet.getClosestPoints(sim.robot, item["id"], 2., physicsClientId=sim.client))
                            return {"final_clearance_m": min(readings) if readings else None, "final_odometry": sim.odometry.tolist(),
                                "physical_challenge": sim.challenge_status(), "observed_area_m2": float(np.count_nonzero(worker.spatial_map.cells == 0) * .05 ** 2) if worker.spatial_map else 0.}
                        report.update(await worker.call(finish))
                    if recorder:
                        report["recording"] = recorder.finish({"backend": options.backend, "real_model": options.backend != "conventional",
                            "mission": "scripted local navigation study", "seed": options.seed + repeat})
                    await worker.close()
                    report["wall_s"] = time.perf_counter() - began
                    report["host_resources"] = {**host_memory(), "cpu_s": time.process_time() - cpu_started,
                        "scope": "Physics/executive process; excludes policy and renderer child processes. Peak RSS is process-lifetime."}
                    report["host_resources"]["average_cpu_core_pct"] = 100 * report["host_resources"]["cpu_s"] / max(.001, report["wall_s"])
                    report["observed_area_gain_m2"] = report.get("observed_area_m2", 0.) - report.get("initial_observed_area_m2", 0.)
                    report["whole_device_gpu_sample"] = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
                    if report.get("recording"):
                        report["success"] = report["success"] and report["recording"]["contact_episodes"] == 0 and report["recording"]["complete_recording"]
                    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    results.append({key: value for key, value in report.items() if key != "events"})
                    (options.output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
                    print(json.dumps({key: report.get(key) for key in ("case", "repeat", "backend", "status", "success", "control_wall_s", "final_odometry")}), flush=True)
    finally:
        continuous_navigation.CONTINUOUS_SPEED_MPS = original_speed
        if process:
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated navigation policy comparison; no changes to live sessions")
    parser.add_argument("--stage", choices=["policy", "probe", "study", "summarize"], default="probe")
    parser.add_argument("--checkpoint", type=Path, default=STUDY / "nomad.pth")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--output", type=Path, default=STUDY / "nomad-probe")
    parser.add_argument("--backend", choices=["conventional", "nomad", "navdp"], default="conventional")
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--rendering", choices=["tiny", "enhanced"], default="tiny")
    parser.add_argument("--cases", nargs="+", choices=["straight", "turning", "doorway", "obstacle", "explore", "return", "rooms_return", "two_rooms_return", "dead_end", "cancel", "goal_change", "delayed"], default=["straight", "turning", "doorway", "obstacle", "explore", "return", "rooms_return", "dead_end", "cancel", "goal_change", "delayed"])
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--decisions", type=int, choices=range(1, 17), default=6)
    parser.add_argument("--time-limit", type=float, default=60.)
    parser.add_argument("--speed", type=float, choices=[.15, .2, .35, .5], default=.2)
    parser.add_argument("--rescans", type=int, choices=[0, 1, 2], default=0)
    options = parser.parse_args()
    if options.stage == "summarize":
        summarize(options)
    elif options.stage == "policy":
        navdp_process(options) if options.backend == "navdp" else policy_process(options)
    elif options.stage == "study":
        asyncio.run(study(options))
    else:
        asyncio.run(probe(options))