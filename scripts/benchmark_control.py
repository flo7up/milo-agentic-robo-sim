import argparse
import asyncio
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from scripts.probe_hybrid import ollama_result
from backend.contracts import Command
from backend.worker import SimulationWorker


class SkillDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    skill: Literal["inspect", "approach", "grasp", "hold"]
    target_visible: bool
    target_x: float = Field(ge=0, le=1, allow_inf_nan=False)


PROMPT = (
    "Select the next skill for a wheeled robot looking for the red cube. "
    "Use only this head-camera image. If the red cube is absent choose inspect, "
    "target_visible false and target_x 0.5. Otherwise report its horizontal center "
    "as target_x in [0,1], left=0, right=1. Choose approach if visible, hold if "
    "the path is clearly blocked, grasp only if already within gripper reach. "
    "Return only the required JSON object. No explanation."
)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {"count": len(ordered), **{
        name: ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]
        for name, fraction in (("p50", .5), ("p95", .95), ("max", 1))}}


def summarize(records):
    valid = [record for record in records if record["valid"]]
    return {
        "requests": len(records), "valid_actions": len(valid),
        "valid_fraction": len(valid) / len(records) if records else None,
        "all_attempt_latency_s": distribution([record["latency_s"] for record in records]),
        "valid_image_to_action_s": distribution([record["latency_s"] for record in valid]),
        "unique_images": len({record["image_sha256"] for record in records}),
        "stale_actions": sum(record["stale"] for record in records),
        "usable_within_deadline": sum(record["valid"] and not record["stale"] for record in records),
    }


async def image_to_skill(client, model, image, width=320, provider="ollama", deadline_s=2):
    started = time.perf_counter()
    with Image.open(BytesIO(image)) as source:
        resized = source.convert("RGB").resize((width, round(source.height * width / source.width)))
        buffer = BytesIO()
        resized.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    record = {"image_sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
              "preprocess_s": time.perf_counter() - started, "valid": False}
    try:
        if provider == "ollama":
            response = await client.post("/api/chat", json={
                "model": model, "stream": False, "think": False, "keep_alive": "10m",
                "format": SkillDecision.model_json_schema(),
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 96},
                "messages": [{"role": "system", "content": PROMPT},
                             {"role": "user", "content": "Next skill?", "images": [encoded]}]})
            response.raise_for_status()
            result = ollama_result(response.json())
            completed = result["done"] and result["finish_reason"] == "stop"
            answer = result.pop("answer")
        else:
            response = await client.post("/v1/chat/completions", json={
                "model": model, "stream": False, "temperature": 0, "max_tokens": 96,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "skill", "strict": True, "schema": SkillDecision.model_json_schema()}},
                "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": [
                    {"type": "text", "text": "Next skill?"}, {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + encoded}}]}]})
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            answer = choice["message"]["content"]
            completed = choice["finish_reason"] == "stop"
            result = {"response_model": data.get("model"), "usage": data.get("usage"),
                      "finish_reason": choice["finish_reason"]}
        record.update(result)
        record["decision"] = SkillDecision.model_validate_json(answer).model_dump()
        record["valid"] = completed
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
        record["error"] = type(error).__name__
    record["latency_s"] = time.perf_counter() - started
    record["stale"] = record["latency_s"] > deadline_s
    return record


async def live_image_to_skill(worker, client, options):
    started = time.perf_counter()
    observation, image = await worker.feedback()
    acquired = time.perf_counter()
    record = await image_to_skill(client, options.model, image, options.width, options.provider, options.deadline)
    record["request_after_capture_s"] = record["latency_s"]
    record["acquisition_s"] = acquired - started
    record["latency_s"] = time.perf_counter() - started
    record["stale"] = record["latency_s"] > options.deadline
    record["observation_seq"] = observation.seq
    record["observation_wall_timestamp"] = observation.wall_timestamp
    record["simulated_time_s"] = observation.simulated_time_s
    return record


def cadence(timestamps, period):
    gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    return {"gap_s": distribution(gaps),
            "absolute_jitter_s": distribution([abs(gap - period) for gap in gaps]),
            "gaps_over_period": sum(gap > period for gap in gaps),
            "missed_slots": sum(max(0, math.ceil(gap / period) - 1) for gap in gaps),
            "effective_hz": (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else None}


class TimedWorker(SimulationWorker):
    def __init__(self, **kwargs):
        self.tick_times = []
        self.publish_times = []
        self.capture_times = []
        self.capture_costs = []
        self.motion_samples = []
        super().__init__(**kwargs)

    async def instrument(self):
        def install(sim):
            original_ticks, original_capture = sim._ticks, sim.capture

            def measured_ticks(count, callback=None, allow_depleted=False):
                def measured_control(tick):
                    self.tick_times.append(time.perf_counter())
                    if callback:
                        callback(tick)
                return original_ticks(count, measured_control, allow_depleted)

            def measured_capture():
                started = time.perf_counter()
                result = original_capture()
                self.capture_times.append(time.perf_counter())
                self.capture_costs.append(time.perf_counter() - started)
                return result

            sim._ticks, sim.capture = measured_ticks, measured_capture
        await self.call(install)

    def _publish(self):
        self.publish_times.append(time.perf_counter())
        super()._publish()
        import pybullet as bullet
        self.motion_samples.append(bullet.getJointState(
            self.sim.robot, self.sim.joints["head_yaw"], physicsClientId=self.sim.client)[1])

    def measurements(self):
        tick_times, publish_times = list(self.tick_times), list(self.publish_times)
        capture_times, capture_costs = list(self.capture_times), list(self.capture_costs)
        motion_samples = list(self.motion_samples)
        return {"physics_step_240hz": cadence(tick_times, 1 / 240),
                "physics_steps_against_100hz_gap_budget": cadence(tick_times, .01),
                "worker_publish_20hz": cadence(publish_times, .05),
                "head_camera_10hz": cadence(capture_times, .1),
                "capture_cost_s": distribution(capture_costs),
                "head_moving_sample_fraction": sum(abs(speed) > .02 for speed in motion_samples) /
                    len(motion_samples) if motion_samples else None}


class PolicyTimedWorker(TimedWorker):
    def __init__(self, **kwargs):
        self.policy_costs = {}
        self.wait_requested = []
        self.wait_actual = []
        self.policy_tick_times = []
        super().__init__(**kwargs)

    async def instrument_policy(self):
        await self.instrument()

        def install(sim):
            def measure(owner, name, label):
                original = getattr(owner, name)

                def measured(*args, **kwargs):
                    started = time.perf_counter()
                    if label == "policy_tick":
                        self.policy_tick_times.append(started)
                    try:
                        return original(*args, **kwargs)
                    finally:
                        self.policy_costs.setdefault(label, []).append(time.perf_counter() - started)
                setattr(owner, name, measured)

            measure(self.skill, "tick", "policy_tick")
            measure(sim, "_check_trajectory", "collision_validation")
            measure(self, "_render_update", "render_update")
            measure(self, "_publish_skill", "skill_publish")
            measure(sim, "_ticks", "physics_with_callbacks")
            original_wait = sim.cancel.wait

            def measured_wait(timeout=None):
                started = time.perf_counter()
                result = original_wait(timeout)
                if threading.current_thread() is self.thread and timeout is not None and timeout > 0:
                    self.wait_requested.append(timeout)
                    self.wait_actual.append(time.perf_counter() - started)
                return result

            sim.cancel.wait = measured_wait
        await self.call(install)

    def policy_measurements(self):
        return {"cost_s": {name: {**distribution(values), "total": sum(values)}
                           for name, values in self.policy_costs.items()},
                "wait_requested_s": distribution(self.wait_requested),
                "wait_actual_s": {**distribution(self.wait_actual), "total": sum(self.wait_actual)},
                "physics_cadence": cadence(self.tick_times, 1 / 240),
                "note": "Inclusive timings overlap; thread wait samples exclude evaluator waits and idle queue waits"}


async def benchmark_policy_timing(output, repeats=5):
    import numpy as np
    from backend.policy import ACTION_NAMES, PolicyChunk, PolicyMetadata, StartSkill
    from scripts.record_milo import smooth_targets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    worker = PolicyTimedWorker(pace=True)
    report = {"kind": "scripted_real_physics_timing", "model_calls": 0, "trials": []}
    finished = threading.Event()
    try:
        await worker.begin_skill_mode(PolicyMetadata(backend="smolvla", checkpoint="scripted-timing-NOT-trained-policy",
            embodiment="milo-left-arm-v1", action_names=ACTION_NAMES, camera_key="observation.images.head",
            fps=20, trained_for_milo=True, skills=["pick_place"]), worker.stop_revision)

        def begin(sim):
            worker.skill.apply(sim, "start_skill", StartSkill(expected_revision=0, skill="pick_place",
                instruction="Scripted timing validation only", timeout_s=60))
            original_brake = worker.skill.brake

            def brake(sim, status, reason):
                original_brake(sim, status, reason)
                finished.set()
            worker.skill.brake = brake
        await worker.call(begin)
        await worker.instrument_policy()
        for index in range(repeats):
            finished.clear()

            def submit_chunk(sim):
                initial = worker.skill.position.copy()
                target = initial.copy()
                target[0] += .03 if index % 2 == 0 else -.03
                ticket = worker.skill.ticket(sim, sim.observe(render=False))
                actions = list(smooth_targets(initial, target, .4))
                worker.skill.accept(sim, PolicyChunk(ticket=ticket, actions=actions))
                return {"start_ticks": sim.ticks, "accepted_steps": len(worker.skill.buffer) * 12,
                        "accepted_at": time.monotonic(), "lease_s": worker.skill.expires_at - time.monotonic(),
                        "initial": initial.tolist(), "target": target.tolist()}

            trial = await worker.call(submit_chunk)
            if not await asyncio.to_thread(finished.wait, 3):
                raise TimeoutError("Policy trajectory did not finish or expire")

            def result(sim):
                import pybullet as bullet
                positions = [bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0] for joint in sim.arms["left"]]
                return {"actual_steps": sim.ticks - trial["start_ticks"], "reason": worker.skill.reason,
                        "status": worker.skill.status, "wall_s": time.monotonic() - trial["accepted_at"],
                        "max_joint_target_error_rad": float(np.max(np.abs(np.array(positions) - trial["target"][:6])))}
            trial.update(await worker.call(result))
            trial["completed"] = trial["actual_steps"] == trial["accepted_steps"] and trial["status"] == "awaiting_policy"
            report["trials"].append(trial)
            print(json.dumps(trial), flush=True)
        report["timing"] = await worker.call(lambda sim: worker.policy_measurements())
        report["completed"] = sum(trial["completed"] for trial in report["trials"])
    finally:
        await worker.close()
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class GpuSampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    used, total, utilization = [float(value.strip()) for value in result.stdout.strip().split(",")]
                    self.samples.append({"time": time.time(), "used_mib": used, "total_mib": total,
                                         "gpu_utilization_pct": utilization})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(.25)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {"scope": "whole GPU, including desktop and other processes; sampled, not exact allocation peak",
                "samples": len(self.samples), "peak_used_mib": max(
                    (sample["used_mib"] for sample in self.samples), default=None),
                "readings": self.samples}


async def submit(worker, tool, arguments, navigation=True):
    observation, _ = await worker.feedback()
    if navigation:
        arguments = {"expected_revision": observation.navigation.revision, **arguments}
    command = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
        observation_seq=observation.seq, action_id=str(uuid4()), tool=tool, arguments=arguments)
    result = await (worker.execute_navigation(command) if navigation else worker.execute(command))
    if result.status != "ok":
        raise RuntimeError(f"{tool}: {result.error}")
    return result


async def scanning_motion(worker, done):
    await worker.begin_navigation(worker.stop_revision)
    started = 0
    while not done.is_set():
        if time.monotonic() - started > 45:
            await submit(worker, "set_navigation_plan", {"steps": [
                {"skill": "inspect_room", "goal": "Scripted load benchmark; not autonomous planning"}]})
            started = time.monotonic()
        await submit(worker, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": .7 * math.sin(time.monotonic() * 1.5),
             "pitch_rad": .4, "duration_s": 1}] * 2})
        try:
            await asyncio.wait_for(done.wait(), .5)
        except asyncio.TimeoutError:
            pass


async def capture_corpus(directory, count):
    from backend.challenges import get_challenge
    import pybullet as bullet
    directory.mkdir(parents=True, exist_ok=True)
    metadata = []
    for scene_index, challenge_id in enumerate(("bench", "park", "apartment", "kitchen_bathroom")):
        worker = SimulationWorker(challenge=get_challenge(challenge_id), pace=False)
        try:
            for index in range(scene_index, count, 4):
                yaw = -1.1 + 2.2 * ((index // 4) / max(1, math.ceil(count / 4) - 1))
                pitch = .15 + .65 * ((index * 7 % 13) / 12)

                def capture(sim):
                    bullet.resetJointState(sim.robot, sim.joints["head_yaw"], yaw, physicsClientId=sim.client)
                    bullet.resetJointState(sim.robot, sim.joints["head_pitch"], pitch, physicsClientId=sim.client)
                    return sim.capture()

                started = time.perf_counter()
                image = await worker.call(capture)
                filename = f"{index:04d}.png"
                (directory / filename).write_bytes(image)
                metadata.append({"file": filename, "scene": challenge_id,
                                 "capture_s": time.perf_counter() - started})
        finally:
            await worker.close()
    (directory / "manifest.json").write_text(json.dumps({
        "setup": "Privileged fixed scene selection and head joint placement, not autonomous exploration; RGB only sent to model",
        "frames": sorted(metadata, key=lambda entry: entry["file"])}, indent=2), encoding="utf-8")


async def run_benchmark(options):
    paths = sorted(options.frames.glob("*.png"))
    if len(paths) < options.requests + options.warmup:
        raise ValueError("Capture at least requests + warmup distinct frames first")
    if options.requests < 100 and not options.smoke:
        raise ValueError("Use at least 100 requests, or explicitly label a --smoke run")
    if options.live_camera and options.stage != "concurrent":
        raise ValueError("Live camera requires concurrent mode")
    parsed = urlparse(options.endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password:
        raise ValueError("Benchmark accepts unauthenticated loopback HTTP only")
    options.output.mkdir(parents=True, exist_ok=True)
    name = f"{options.stage}-{options.width}"
    report = {"model": options.model, "provider": options.provider, "width": options.width,
              "stage": options.stage, "smoke_only": options.smoke, "batch_size": 1,
              "context_limit": 4096 if options.provider == "ollama" else "server configured",
              "output_token_limit": 96, "deadline_s": options.deadline,
              "model_outputs_executed": False,
              "input": "live RGB head camera, no state/depth" if options.live_camera else "replayed RGB head camera, no state/depth",
              "cache": "Five separate warmups by default; distinct PNG hashes; identical system prefix, runtime caching defaults unchanged",
              "camera_width": options.camera_width,
              "rendering": f"PyBullet CPU TinyRenderer, {options.camera_width}x{options.camera_width * 3 // 4}",
              "graphics_load": options.graphics_load,
              "timing": "PNG decode/resize/encode through complete response and schema validation; server vision/reasoning included but not separately exposed; live mode also includes worker queue and observation capture",
              "warmup": [], "records": []}
    worker, motion = None, None
    done = asyncio.Event()
    sampler = GpuSampler()
    sampler.thread.start()
    try:
        async with httpx.AsyncClient(base_url=options.endpoint, timeout=options.timeout, trust_env=False) as client:
            if options.provider == "ollama":
                response = await client.get("/api/version")
                response.raise_for_status()
                report["runtime"] = response.json()
                response = await client.post("/api/show", json={"model": options.model})
                response.raise_for_status()
                report["model_details"] = response.json().get("details")
            for path in paths[:options.warmup]:
                report["warmup"].append(await image_to_skill(client, options.model, path.read_bytes(),
                    options.width, options.provider, options.deadline))
            if options.stage == "concurrent":
                worker = TimedWorker(pace=True)
                await worker.instrument()
                await worker.call(lambda sim: configure_camera(sim, options.camera_width))
                motion = asyncio.create_task(scanning_motion(worker, done))
            with (options.output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for index, path in enumerate(paths[options.warmup:options.warmup + options.requests]):
                    if motion and motion.done():
                        await motion
                    started = time.perf_counter()
                    before = worker.latest["snapshot"]["simulated_time_s"] if worker else 0
                    record = (await live_image_to_skill(worker, client, options) if options.live_camera else
                              await image_to_skill(client, options.model, path.read_bytes(), options.width,
                                                   options.provider, options.deadline))
                    record.update(index=index, frame="live" if options.live_camera else path.name, request_wall_time=time.time(),
                                  request_with_read_s=time.perf_counter() - started)
                    if worker:
                        record["physics_advanced_s"] = worker.latest["snapshot"]["simulated_time_s"] - before
                    report["records"].append(record)
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if (index + 1) % 10 == 0:
                        print(json.dumps({"stage": name, "completed": index + 1,
                                          "last_latency_s": record["latency_s"], "valid": record["valid"]}), flush=True)
            if options.provider == "ollama":
                response = await client.get("/api/ps")
                response.raise_for_status()
                report["loaded_models"] = response.json()
    finally:
        done.set()
        try:
            if motion:
                await motion
        finally:
            if worker:
                await worker.close()
                report["control"] = worker.measurements()
            report["gpu"] = sampler.finish()
            report["summary"] = summarize(report["records"])
            (options.output / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"file": str(options.output / f"{name}.json"), "summary": report["summary"],
                      "control": report.get("control"), "peak_vram_mib": report["gpu"]["peak_used_mib"]}, indent=2), flush=True)


def configure_camera(sim, width):
    sim.width, sim.height = width, width * 3 // 4


async def control_baseline(options):
    worker = TimedWorker(pace=True)
    done = asyncio.Event()
    await worker.instrument()
    await worker.call(lambda sim: configure_camera(sim, options.camera_width))
    task = asyncio.create_task(scanning_motion(worker, done))
    try:
        await asyncio.sleep(options.seconds)
    finally:
        done.set()
        try:
            await task
        finally:
            await worker.close()
    report = {"live_inference": False, "scripted": True, "camera_width": options.camera_width,
              "seconds": options.seconds, "control": worker.measurements()}
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / f"control-{options.camera_width}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


async def wait_for_brake(worker, timeout=8):
    async with asyncio.timeout(timeout):
        while worker.latest["navigation"]["remaining_s"]:
            await asyncio.sleep(.01)


def red_center(image):
    import numpy as np
    pixels = np.asarray(Image.open(BytesIO(image)).convert("RGB"), dtype=float)
    red = (pixels[:, :, 0] > 100) & (pixels[:, :, 0] > pixels[:, :, 1] * 1.7) & (pixels[:, :, 0] > pixels[:, :, 2] * 1.7)
    columns = np.where(red)[1]
    return float(columns.mean() / pixels.shape[1]) if len(columns) >= 8 else None


async def shadow_planner(worker, client, options, done, records):
    while not done.is_set():
        def latest(sim):
            reference = worker.latest["camera"]["frame_ref"]
            return worker.camera_frames[reference], worker.capture_times[-1] if worker.capture_times else time.perf_counter()
        image, captured = await worker.call(latest)
        age = time.perf_counter() - captured
        record = await image_to_skill(client, options.model, image, options.width, options.provider, options.deadline)
        record["frame_age_at_dispatch_s"] = age
        record["age_at_validation_s"] = age + record["latency_s"]
        record["stale"] = record["age_at_validation_s"] > options.deadline
        records.append(record)


async def perturbation(worker, kind):
    import pybullet as bullet
    if kind == "failed_grasp":
        started = time.perf_counter()
        result = await submit(worker, "set_gripper", {"arm": "left", "opening_m": 0, "max_force_n": 20}, navigation=False)
        gripper = result.observation.grippers["left"]
        detected = not all(gripper.contact) and gripper.load_n == 0
        latency = time.perf_counter() - started
        await submit(worker, "set_gripper", {"arm": "left", "opening_m": .11, "max_force_n": 20}, navigation=False)
        return {"task_success": False, "failure_detected": detected, "close_to_feedback_s": latency,
                "recovery": "Scripted reopen only; autonomous regrasp not implemented", "sensor": gripper.model_dump()}

    await worker.begin_navigation(worker.stop_revision)
    await submit(worker, "set_navigation_plan", {"steps": [
        {"skill": "inspect_room", "goal": "Scripted inspection"},
        {"skill": "locate_doorway", "goal": "Scripted scan"},
        {"skill": "approach", "goal": "Scripted obstacle response test"}]})
    if kind == "delayed_response":
        await submit(worker, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": .7, "pitch_rad": .3, "duration_s": 1}]})
        observation, _ = await worker.feedback()
        pending = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
            observation_seq=observation.seq, action_id=str(uuid4()), tool="replace_motion_buffer", arguments={
                "expected_revision": observation.navigation.revision, "segments": [
                    {"kind": "head", "yaw_rad": -.7, "pitch_rad": .3, "duration_s": 1}]})
        started = time.perf_counter()
        await wait_for_brake(worker)
        brake_s = time.perf_counter() - started
        await asyncio.sleep(max(0, 3 - brake_s))
        result = await worker.execute_navigation(pending)
        return {"injected_response_delay_s": 3, "braked_after_s": brake_s,
                "stale_rejected": result.status == "error", "error": result.error,
                "task_success": False, "note": "Scripted delayed buffer while live shadow inference runs; finite horizon correctly pauses"}

    if kind == "moving_target":
        errors, updates, rejections = [], [], []
        first_response = None
        started = time.perf_counter()
        for index in range(12):
            def move_target(sim):
                body = next(item["id"] for item in sim.objects if item["name"] == "target")
                bullet.resetBasePositionAndOrientation(body, [1, .4 * math.sin(index * .5), .6],
                    [0, 0, 0, 1], physicsClientId=sim.client)
            await worker.call(move_target)
            observed_at = time.perf_counter()
            observation, image = await worker.feedback()
            center = red_center(image)
            if center is None:
                errors.append(None)
                continue
            errors.append(abs(center - .5))
            yaw = max(-1.5, min(1.5, observation.head_rad[0] - (center - .5) * 1.13))
            command = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
                observation_seq=observation.seq, action_id=str(uuid4()), tool="replace_motion_buffer", arguments={
                    "expected_revision": observation.navigation.revision, "segments": [
                        {"kind": "head", "yaw_rad": yaw, "pitch_rad": 0, "duration_s": .4}]})
            result = await worker.execute_navigation(command)
            if result.status != "ok":
                rejections.append(result.error)
                continue
            if first_response is None:
                first_response = time.perf_counter() - started
            updates.append(time.perf_counter() - observed_at)
            await asyncio.sleep(.1)
        return {"task_success": all(error is not None and error < .15 for error in errors[-3:]),
                "tracking_pixel_error_fraction": errors, "first_update_s": first_response,
            "rejected_updates": rejections,
                "capture_to_buffer_accept_s": distribution(updates),
                "note": "RGB color centroid plus head encoder only; target teleported by test fixture, no pose sent to controller; acceptance is not measured motor response"}

    for _ in range(2):
        await submit(worker, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": -.6, "pitch_rad": .2, "duration_s": 1},
            {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": 1}]})
        await wait_for_brake(worker)
        await submit(worker, "complete_navigation_skill", {"evidence": "Scripted scan completed; no semantic doorway claim"})
    await submit(worker, "replace_motion_buffer", {"segments": [
        {"kind": "drive", "linear_mps": .15, "angular_radps": 0, "duration_s": 1}] * 2})
    await asyncio.sleep(.2)

    def insert_obstacle(sim):
        front = max(bullet.getAABB(sim.robot, index, physicsClientId=sim.client)[1][0]
                    for index in [-1, *sim.joints.values()])
        body = next(item["id"] for item in sim.objects if item["name"] == "obstacle")
        bullet.resetBasePositionAndOrientation(body, [front + .075, 0, .4], [0, 0, 0, 1], physicsClientId=sim.client)
        return time.perf_counter()

    inserted = await worker.call(insert_obstacle)
    await wait_for_brake(worker)
    latency = time.perf_counter() - inserted
    observation, _ = await worker.feedback()
    state = worker.latest["navigation"]
    return {"task_success": False, "safety_stop_success": state["status"] == "failed" and
                "CLEARANCE_STOP" in state["reason"] and not observation.bumpers,
            "insert_to_brake_s": latency, "reason": state["reason"], "contacts": observation.bumpers,
            "note": "Privileged fixture placement AND existing whole-body ground-truth collision lookahead; not camera-only avoidance"}


async def run_scenarios(options):
    report = {"scripted_control": True, "live_model_shadow_only": True, "camera_width": options.camera_width, "scenarios": {}}
    async with httpx.AsyncClient(base_url=options.endpoint, timeout=options.timeout, trust_env=False) as client:
        for kind in ("moving_target", "unexpected_obstacle", "delayed_response", "failed_grasp"):
            scene = [{"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
                     {"name": "target", "size": [.15, .15, .15], "position": [1, 0, .6], "color": [.9, .05, .05, 1]},
                     {"name": "obstacle", "size": [.1, 2, .8], "position": [2, 2, .4], "color": [.2, .3, .3, 1]}]
            worker = TimedWorker(scene=scene, pace=True)
            await worker.instrument()
            await worker.call(lambda sim: configure_camera(sim, options.camera_width))
            done, records = asyncio.Event(), []
            inference = asyncio.create_task(shadow_planner(worker, client, options, done, records))
            try:
                result = await perturbation(worker, kind)
            finally:
                done.set()
                try:
                    await inference
                finally:
                    await worker.close()
            report["scenarios"][kind] = {**result, "inference": summarize(records), "requests": records,
                                         "control": worker.measurements()}
            print(json.dumps({"scenario": kind, **result}), flush=True)
    options.output.mkdir(parents=True, exist_ok=True)
    (options.output / f"scenarios-{options.camera_width}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description="Isolated RGB-to-skill latency and concurrent real-physics benchmark")
    result.add_argument("--stage", choices=["capture", "alone", "concurrent", "control", "scenarios", "policy-timing"], default="alone")
    result.add_argument("--policy-repeats", type=int, choices=range(1, 21), default=5)
    result.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    result.add_argument("--endpoint", default="http://127.0.0.1:11434")
    result.add_argument("--model", default="gemma4:e2b-it-qat")
    result.add_argument("--frames", type=Path, default=Path(".runtime/control-benchmark/frames"))
    result.add_argument("--output", type=Path, default=Path(".runtime/control-benchmark"))
    result.add_argument("--width", type=int, choices=[320, 640], default=320)
    result.add_argument("--camera-width", type=int, choices=[320, 640], default=640)
    result.add_argument("--graphics-load", default="External WebGL load not controlled", help="Operator description, independently verify actual draw activity")
    result.add_argument("--live-camera", action="store_true", help="Acquire a fresh worker observation per request, including capture in latency")
    result.add_argument("--seconds", type=float, default=20)
    result.add_argument("--requests", type=int, default=100)
    result.add_argument("--warmup", type=int, default=5)
    result.add_argument("--timeout", type=float, default=45)
    result.add_argument("--deadline", type=float, default=2)
    result.add_argument("--smoke", action="store_true")
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    stages = {"capture": lambda: capture_corpus(options.frames, options.requests + options.warmup),
              "control": lambda: control_baseline(options), "scenarios": lambda: run_scenarios(options),
              "policy-timing": lambda: benchmark_policy_timing(options.output, options.policy_repeats)}
    asyncio.run(stages.get(options.stage, lambda: run_benchmark(options))())