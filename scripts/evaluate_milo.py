import argparse
import asyncio
import base64
from contextlib import redirect_stdout
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from backend.policy import ACTION_NAMES, PolicyChunk, PolicyMetadata
from backend.smolvla_server import PolicyRequest, state_vector
from scripts.train_milo import MAX_TRAINING_STEPS


def validate_pilot(checkpoint):
    checkpoint = checkpoint.resolve(strict=True)
    report = json.loads((checkpoint.parent / "report.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    if (report.get("status") != "trained_and_reloaded" or not 1 <= report.get("steps", 0) <= MAX_TRAINING_STEPS or
            report.get("weights_sha256") != actual or report.get("reloaded_output", {}).get("shape") != [1, 50, 7]):
        raise ValueError("A matching completed Milo training-pilot report is required")
    return checkpoint, report


def adapted_chunk(raw, position, velocity, expected_ticket, age_s):
    import math
    import numpy as np
    from backend.policy import PolicyTicket
    from backend.robot import ARM_LIMITS
    from scripts.record_milo import smooth_targets

    if PolicyTicket.model_validate(raw["ticket"]) != expected_ticket or not 0 <= age_s <= 1:
        raise ValueError("Stale candidate target; capture a new observation")
    values = np.asarray(raw["actions"], dtype=float)
    if values.shape != (50, 7) or not np.isfinite(values).all():
        raise ValueError("Candidate must produce 50 finite seven-axis targets")
    position, velocity = np.asarray(position), np.asarray(velocity)
    if position.shape != (7,) or not np.isfinite(position).all() or np.any(np.abs(velocity) > 1e-6):
        raise ValueError("Experimental adapter requires the worker to be at rest")
    limits = np.array([*ARM_LIMITS, (0, .11)])
    if np.any(position < limits[:, 0]) or np.any(position > limits[:, 1]):
        raise ValueError("Worker starting position is outside the physical joint limits")
    target_index = 19
    requested = values[target_index]
    bounded = np.clip(requested, limits[:, 0], limits[:, 1])
    delta = bounded - position
    duration = .4
    speed = np.array([.8] * 6 + [.06])
    acceleration = np.array([4.] * 6 + [.3])
    allowance = .8 * np.minimum(speed * duration / 1.875, acceleration * duration ** 2 / (10 / np.sqrt(3)))
    fraction = min(1., float(np.min(allowance / np.maximum(np.abs(delta), 1e-12))))
    endpoint = position + delta * fraction
    skipped = math.ceil(age_s * 20)
    actions = [position.tolist()] * skipped + list(smooth_targets(position, endpoint, duration))
    return PolicyChunk(ticket=expected_ticket, actions=actions), {
        "kind": "bounded_target_adapter", "target_index": target_index, "requested_target": requested.tolist(),
        "bounded_target": bounded.tolist(), "saturated_axes": [name for name, original, clipped in
            zip(ACTION_NAMES, requested, bounded) if original != clipped], "progress_fraction": fraction,
        "endpoint": endpoint.tolist(), "duration_s": duration, "discarded_prefix_samples": skipped,
        "original_observation_age_s": age_s,
        "note": "Model target retimed from rest; this is not unmodified policy chunk execution"}


def visual_evidence(directory):
    import numpy as np
    from PIL import Image, ImageDraw
    paths = [*sorted(directory.glob("frame-*.png")), directory / "terminal.png"]
    if len(paths) < 2 or not all(path.exists() for path in paths):
        return None
    frames = []
    for path in paths:
        with Image.open(path) as source:
            frames.append(source.convert("RGB").copy())
    first, last = np.array(frames[0], dtype=np.int16), np.array(frames[-1], dtype=np.int16)
    if first.shape != last.shape:
        raise ValueError("Camera dimensions changed during candidate evaluation")
    changed = np.max(np.abs(last - first), axis=2) > 12
    width, height = frames[0].size
    comparison = Image.new("RGB", (width * 2, height + 30), "white")
    comparison.paste(frames[0], (0, 30))
    comparison.paste(frames[-1], (width, 30))
    draw = ImageDraw.Draw(comparison)
    draw.text((8, 8), "Before policy motion", fill="black")
    draw.text((width + 8, 8), "After policy motion", fill="black")
    comparison.save(directory / "comparison.png")
    frames[0].save(directory / "motion.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)
    return {"camera_frames": len(frames), "changed_pixel_fraction": float(np.mean(changed)),
            "camera_size": [width, height], "comparison": "comparison.png", "animation": "motion.gif",
            "animation_timing": "250 ms per request snapshot; timelapse, not wall-clock playback"}


def inference_worker(options):
    with redirect_stdout(sys.stderr):
        import numpy as np
        from PIL import Image
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from scripts.probe_smolvla import checked_actions

        checkpoint, _ = validate_pilot(options.checkpoint)
        torch.set_num_threads(8)
        torch.manual_seed(options.seed)
        policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True, strict=True).float().eval()
        config = policy.config
        if (config.device != "cuda" or tuple(config.input_features["observation.state"].shape) != (7,) or
                tuple(config.output_features["action"].shape) != (7,) or
                set(config.image_features) != {"observation.images.head"}):
            raise ValueError("Candidate features do not match Milo's CUDA policy contract")
        preprocess, postprocess = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    print(json.dumps({"ready": True, "device": torch.cuda.get_device_name(0)}), flush=True)
    for line in sys.stdin:
        with redirect_stdout(sys.stderr):
            request = PolicyRequest.model_validate_json(line)
            with Image.open(BytesIO(base64.b64decode(request.image, validate=True))) as image:
                if image.size != (320, 240):
                    raise ValueError("Expected the paired 320x240 head camera")
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255
            frame = {"observation.state": torch.tensor(state_vector(request.observation), dtype=torch.float32),
                     "observation.images.head": torch.from_numpy(pixels).permute(2, 0, 1), "task": request.instruction}
            policy.reset()
            with torch.inference_mode():
                prediction = policy.predict_action_chunk(preprocess(frame))
                decoded = torch.stack([postprocess(prediction[:, index]) for index in range(prediction.shape[1])], dim=1)
            actions = checked_actions(decoded.cpu().numpy(), 50, 7)["actions"]
        print(json.dumps({"ticket": request.ticket.model_dump(), "actions": actions}, allow_nan=False), flush=True)


def outcome(sim):
    import numpy as np
    import pybullet as bullet
    cube = next(item["id"] for item in sim.objects if item["name"] == "cube")
    lower, upper = bullet.getAABB(cube, physicsClientId=sim.client)
    floor = next(item["id"] for item in sim.objects if item["name"] == "floor")
    contacts = [[contact for contact in bullet.getContactPoints(sim.robot, cube, linkIndexA=finger,
                 physicsClientId=sim.client) if contact[9] > .02] for finger in sim.fingers["left"]]
    velocity, angular = bullet.getBaseVelocity(cube, physicsClientId=sim.client)
    return {"state": state_vector(sim.observe(render=False)), "simulated_s": sim.ticks / 240,
            "cube_bottom_m": lower[2], "placed": all(lower[axis] >= center - .065 and upper[axis] <= center + .065
                                                       for axis, center in enumerate((.36, .34))),
            "grounded": any(contact[9] > .01 for contact in bullet.getContactPoints(cube, floor, physicsClientId=sim.client)),
            "cube_speed_mps": float(np.linalg.norm(velocity)), "cube_angular_radps": float(np.linalg.norm(angular)),
            "released": not any(contacts),
            "assisted_constraints": bullet.getNumConstraints(physicsClientId=sim.client),
            "opposing_contact": any(np.dot(inner[7], outer[7]) < -.7 for inner in contacts[0] for outer in contacts[1])}


async def evaluate(options):
    from pydantic import ValidationError
    from backend.contracts import Command
    from backend.policy import PolicyTicket
    from backend.simulation import MotionError
    from backend.worker import SimulationWorker
    from scripts.record_milo import TASK, scene_for

    checkpoint, training = validate_pilot(options.checkpoint)
    options.output.mkdir(parents=True, exist_ok=False)
    training_steps = training.get("total_steps", training["steps"])
    report = {"status": "running", "training_steps": training_steps, "weights_sha256": training["weights_sha256"],
              "checkpoint": str(checkpoint), "cube_xy": options.cube_xy, "seed": options.seed,
              "held_out_episode": 4 if options.cube_xy == [.36, .26] else None,
              "motion_adapter": options.motion_adapter,
              "qualification": "Experimental candidate; no live-app enablement or worker guard relaxation. Adapter-assisted results are separate from raw-policy validation.",
              "clock": "wall-clock, existing worker freshness and motion limits", "requests": [], "success": False,
              "lift_observed": False, "constraints_observed": False}
    worker = None
    process = None
    started = time.perf_counter()
    with (options.output / "inference.log").open("w", encoding="utf-8") as log:
        try:
            process = await asyncio.create_subprocess_exec(str(options.python.resolve(strict=True)), "-u", "-m",
                "scripts.evaluate_milo", "--inference-worker", "--checkpoint", str(checkpoint), "--seed", str(options.seed),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=2_500_000)
            ready = json.loads(await asyncio.wait_for(process.stdout.readline(), 120))
            if not ready.get("ready"):
                raise RuntimeError("Candidate inference worker did not become ready")
            if options.timing:
                from scripts.benchmark_control import PolicyTimedWorker
                worker = PolicyTimedWorker(scene=scene_for(options.cube_xy), pace=True)
            else:
                worker = SimulationWorker(scene=scene_for(options.cube_xy), pace=True)
            await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
            result = await worker.execute(Command(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                observation_seq=worker.latest["observation"]["seq"], action_id=str(uuid4()), tool="set_head",
                arguments={"yaw_rad": 0., "pitch_rad": .7, "duration_s": 1.}), assisted=False)
            if result.status != "ok":
                raise RuntimeError("Could not establish the recorded camera pose")

            async def predict(ticket, observation, image):
                payload = PolicyRequest(ticket=ticket, observation=observation,
                                        image=base64.b64encode(image).decode("ascii"), instruction=TASK)
                process.stdin.write((payload.model_dump_json() + "\n").encode())
                await process.stdin.drain()
                return json.loads(await asyncio.wait_for(process.stdout.readline(), 10))

            observation, image = await worker.feedback()
            ticket = PolicyTicket(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
                                  revision=0, observation_seq=observation.seq)
            await predict(ticket, observation, image)
            metadata = PolicyMetadata(backend="smolvla", checkpoint=f"experimental-{training_steps}-step-candidate-NOT-reviewed",
                embodiment="milo-left-arm-v1", action_names=ACTION_NAMES, camera_key="observation.images.head",
                fps=20, trained_for_milo=True, skills=["pick_place"])
            await worker.begin_skill_mode(metadata, worker.stop_revision)
            result = await worker.execute_skill(Command(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                observation_seq=worker.latest["observation"]["seq"], action_id=str(uuid4()), tool="start_skill",
                arguments={"expected_revision": 0, "skill": "pick_place", "instruction": TASK, "timeout_s": 60}))
            if result.status != "ok":
                raise RuntimeError(f"Could not start isolated candidate evaluation: {result.message}")
            if options.timing:
                await worker.instrument_policy()
            report["initial"] = await worker.call(outcome)
            report["max_arm_displacement_rad"] = 0.
            report["max_gripper_displacement_m"] = 0.
            for index in range(options.requests):
                ticket, observation, image = await worker.policy_feedback()
                (options.output / f"frame-{index:03d}.png").write_bytes(image)
                began = time.perf_counter()
                raw = await predict(ticket, observation, image)
                record = {"request": index, "observation": observation.model_dump(), "prediction": raw,
                          "latency_s": time.perf_counter() - began, "accepted": False}
                report["requests"].append(record)
                try:
                    if options.motion_adapter:
                        def accept_target(sim):
                            if not worker.skill or not worker.skill.pending:
                                raise MotionError("STALE_POLICY", "Candidate request was cancelled")
                            expected, captured_at = worker.skill.pending
                            chunk, details = adapted_chunk(raw, worker.skill.position, worker.skill.velocity,
                                expected, worker.skill.clock() - captured_at)
                            record["adapter"] = details
                            record["executed_chunk"] = chunk.model_dump()
                            worker.skill.accept(sim, chunk)
                            record["execution"] = {"start_ticks": sim.ticks, "accepted_at": time.perf_counter(),
                                "planned_ticks": len(worker.skill.buffer) * 12,
                                "lease_s": worker.skill.expires_at - worker.skill.clock()}
                            worker._publish_skill()
                        await worker.call(accept_target)
                    else:
                        await worker.accept_policy(PolicyChunk.model_validate(raw))
                except (ValueError, ValidationError, MotionError) as error:
                    record.update(rejection=getattr(error, "code", "INVALID_POLICY_CHUNK"), message=str(error))
                    report["status"] = "rejected_by_safety"
                    await worker.fail_policy(f"Candidate evaluation stopped: {record['rejection']}")
                    break
                record["accepted"] = True
                deadline = time.monotonic() + 2
                while worker.latest["skill"]["remaining_s"] > 0 and time.monotonic() < deadline:
                    measurement = await worker.call(outcome)
                    report["lift_observed"] |= measurement["cube_bottom_m"] > .08 and measurement["opposing_contact"]
                    report["constraints_observed"] |= measurement["assisted_constraints"] != 0
                    report["max_arm_displacement_rad"] = max(report["max_arm_displacement_rad"],
                        max(abs(value - initial) for value, initial in zip(measurement["state"][:6], report["initial"]["state"][:6])))
                    report["max_gripper_displacement_m"] = max(report["max_gripper_displacement_m"],
                        abs(measurement["state"][6] - report["initial"]["state"][6]))
                    await asyncio.to_thread(worker.sim.cancel.wait, .05)
                if "execution" in record:
                    def completed_chunk(sim):
                        execution = record["execution"]
                        actual = sim.ticks - execution["start_ticks"]
                        return {"actual_ticks": actual, "wall_s": time.perf_counter() - execution["accepted_at"],
                            "complete": actual == execution["planned_ticks"], "reason": worker.skill.reason}
                    record["execution"].update(await worker.call(completed_chunk))
                if worker.latest["skill"]["status"] == "failed":
                    report.update(status="stopped_by_safety", reason=worker.latest["skill"]["reason"])
                    break
            else:
                report["status"] = "request_limit"
            report["final"] = await worker.call(outcome)
            report["max_arm_displacement_rad"] = max(report["max_arm_displacement_rad"],
                max(abs(value - initial) for value, initial in zip(report["final"]["state"][:6], report["initial"]["state"][:6])))
            report["max_gripper_displacement_m"] = max(report["max_gripper_displacement_m"],
                abs(report["final"]["state"][6] - report["initial"]["state"][6]))
            report["policy_simulated_s"] = report["final"]["simulated_s"] - report["initial"]["simulated_s"]
            report["movement_observed"] = report["max_arm_displacement_rad"] > .01 and report["policy_simulated_s"] > 0
            executions = [record["execution"] for record in report["requests"] if "execution" in record]
            report["chunks"] = {"accepted": sum(record["accepted"] for record in report["requests"]),
                "completed": sum(execution.get("complete", False) for execution in executions),
                "expired": sum("expired" in execution.get("reason", "").lower() for execution in executions),
                "planned_ticks": sum(execution["planned_ticks"] for execution in executions),
                "actual_ticks": sum(execution.get("actual_ticks", 0) for execution in executions)}
            report["skill"] = dict(worker.latest["skill"])
            report["success"] = bool(report["lift_observed"] and report["final"]["placed"] and report["final"]["grounded"] and
                                     report["final"]["cube_speed_mps"] < .025 and report["final"]["cube_angular_radps"] < .15 and
                                     report["final"]["released"] and not report["constraints_observed"])
        except Exception as error:
            report.update(status="evaluation_error", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            if worker:
                worker.stop()
                if options.timing:
                    report["worker_timing"] = await worker.call(lambda sim: worker.policy_measurements())
                image = await worker.call(lambda sim: sim.capture())
                (options.output / "terminal.png").write_bytes(image)
                await worker.close()
            if process and process.returncode is None:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.wait(), 15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            report["wall_s"] = time.perf_counter() - started
            report["visual_evidence"] = visual_evidence(options.output)
            (options.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"requests", "worker_timing"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained experimental Milo policy in isolated Windows physics")
    parser.add_argument("--checkpoint", type=Path, default=Path(".runtime/milo-training-pilot-100/checkpoint"))
    parser.add_argument("--python", type=Path, default=Path(".runtime/smolvla-env/Scripts/python.exe"))
    parser.add_argument("--output", type=Path, default=Path(".runtime/milo-policy-evaluation"))
    parser.add_argument("--requests", type=int, choices=range(1, 41), default=40)
    parser.add_argument("--seed", type=int, default=713)
    parser.add_argument("--cube-xy", type=float, nargs=2, default=[.36, .26])
    parser.add_argument("--motion-adapter", action="store_true", help="Experimental bounded target trajectories; raw-policy validation remains the default")
    parser.add_argument("--timing", action="store_true", help="Record worker computation, wait and physics cadence diagnostics")
    parser.add_argument("--inference-worker", action="store_true")
    options = parser.parse_args()
    if options.inference_worker:
        inference_worker(options)
    else:
        asyncio.run(evaluate(options))