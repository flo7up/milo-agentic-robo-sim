import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from uuid import uuid4

import numpy as np

from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
from backend.challenges import get_challenge
from backend.contracts import Command
from backend.local_navigation import CHECKPOINT, MODEL_PYTHON, LocalNavigationClient, ResidentNavigationModel
from backend.navigation_supervisor import POLICY_TASKS
from backend.worker import SimulationWorker
from scripts.navigation_policy import TASK, EVALUATION_CASES, challenge_for, bounded_velocity, evaluation_state, prepare
from scripts.recovery_policy import TEACHER_ACTIONS


DOCKING_CASES = [
    {"id": index, "target": target, "initial_xy": initial, "heading_rad": heading,
     "doorway": doorway, "split": "docking-evaluation-only-v1"}
    for index, (target, initial, heading, doorway) in enumerate([
        ([1.05, -.4], [0., 0.], 0., False), ([1.05, .4], [0., 0.], 0., False),
        ([1.05, -.4], [-.08, .06], -.2, False), ([1.05, .4], [-.08, -.06], .2, False),
        ([1.55, -.18], [.08, .04], -.12, True), ([1.55, .18], [.08, -.04], .12, True),
    ])]


class SensorDockingPolicy:
    def __init__(self):
        self.target = None
        self.instruction = TASK

    async def predict(self, observation, image):
        from backend.navigation_supervisor import marker_target
        if self.target is None:
            self.target = marker_target(observation, image, "green")
        delta = np.array(self.target) - observation.odometry_m_rad[:2]
        distance = float(np.linalg.norm(delta))
        angle = math.atan2(delta[1], delta[0]) - observation.odometry_m_rad[2]
        angle = math.atan2(math.sin(angle), math.cos(angle))
        linear = min(.15, .7 * distance) * max(0., math.cos(angle)) if abs(angle) < .45 else 0.
        action = [linear, float(np.clip(1.5 * angle, -.5, .5))] if distance >= .055 else [0., 0.]
        return {"action": action, "raw_action": action, "saturated_axes": [], "observed_target_m": self.target}


class WaypointDockingPolicy:
    def __init__(self, client):
        self.client = client
        self.target = None
        self.episode = None
        self.instruction = TASK

    async def predict(self, observation, image):
        from backend.navigation_supervisor import marker_target, waypoint_instruction
        episode = (observation.run_id, observation.episode_epoch)
        if self.episode is not None and self.episode != episode:
            raise ValueError("Waypoint target belongs to another episode")
        self.episode = episode
        if self.target is None:
            self.target = marker_target(observation, image, "green")
        self.instruction = waypoint_instruction(observation, self.target)
        self.client.instruction = self.instruction
        result = await self.client.predict(observation, image)
        return {**result, "observed_target_m": self.target, "instruction": self.instruction}


def docking_score(final, target, stopped, wheel_velocity, requests):
    resting = abs(wheel_velocity[0]) < .01 and abs(wheel_velocity[1]) < .02
    interrupted = any(entry.get("status") != "ok" or entry.get("execution_status") == "failed" for entry in requests)
    return {"autonomous_stop": stopped, "at_rest": resting, "center_error_m": math.dist(final["position"][:2], target),
        "success": bool(final["parked"] and stopped and resting and not interrupted),
        "shield_or_execution_failure": interrupted}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def distribution(values):
    values = sorted(values)
    return {"n": len(values), "median": statistics.median(values) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "p95": values[math.ceil(.95 * len(values)) - 1] if values else None}


def summarize_artifacts(root):
    rows = []
    for path in sorted(root.glob("*/*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        actions_path = path.with_name("actions.json")
        if not actions_path.exists():
            continue
        events = json.loads(actions_path.read_text(encoding="utf-8"))
        trace_path = path.with_name("trace.json")
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
        manifest_path = path.parent.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        navigation_feedback = {}
        for event in trace:
            navigation = event.get("payload", {}).get("observation", {}).get("navigation")
            if navigation and navigation.get("reason"):
                navigation_feedback[navigation["revision"]] = navigation["reason"]
        proposals = [event for event in events if event["event"] == "proposed"]
        commands = [event for event in events if event["event"] in {"accepted", "rejected"}]
        drives = [event for event in commands if any(segment.get("kind") == "drive" for segment in event["command"]["arguments"].get("segments", []))]
        samples = [event for event in events if event["event"] == "executed_tick" and event["requested"]["kind"] == "drive"]
        policy_steps = [event for event in trace if event["title"] == "SmolVLA supervised motion"]
        rows.append({"path": path.as_posix(), **{key: report.get(key) for key in ("task", "source", "seed", "variant", "success", "wall_s", "interventions", "false_success", "error")},
            "checkpoint": manifest.get("checkpoint") if report.get("source") == "smolvla" else None,
            "travel_m": report.get("final", {}).get("travel_m"), "termination": report.get("outcome"),
            "observed_navigation_reasons_by_revision": dict(Counter(navigation_feedback.values())),
            "last_request": {key: report["requests"][-1].get(key) for key in ("status", "error", "requested", "execution_status", "execution_reason")}
                if report.get("requests") else None,
            "luna_latency_s": distribution([event["payload"]["latency_s"] for event in trace if event["kind"] == "response"]),
            "policy_latency_s": distribution([event["latency_s"] for event in proposals] or [entry["latency_s"] for entry in report.get("requests", [])]),
            "drive_requests": len(drives), "drive_accepted": sum(event["event"] == "accepted" for event in drives),
            "rejection_reasons": dict(Counter(event["result"]["error"] for event in commands if event["event"] == "rejected")),
            "sampled_drive_sim_s": len(samples) * .05,
            "sampled_nonzero_drive_sim_s": sum(abs(event["velocity"][0]) > .001 or abs(event["velocity"][1]) > .001 for event in samples) * .05,
            "contact_samples": sum(bool(event["collisions"]) for event in samples),
            "clipped_requests": sum(bool(event["saturated_axes"]) for event in proposals) if proposals else sum(bool(entry["clipped_axes"]) for entry in report.get("requests", [])),
            "deadband_requests": sum(event["payload"]["executed_action"] == [0., 0.] and event["payload"]["raw_action"] != [0., 0.] for event in policy_steps)
                if policy_steps else sum(entry["deadband_override"] for entry in report.get("requests", [])),
            "tokens": [report.get("input_tokens"), report.get("output_tokens")],
            "counting_note": "Warmup proposals included in inference counts; accepted means queued, nonzero execution is sampled every12physics ticks, not inferred from proposals. Navigation reasons count observed revisions, not independently counted stop episodes."})
    write(root / "results.json", rows)
    print(json.dumps(rows, indent=2))


def inventory(output):
    output.mkdir(parents=True, exist_ok=False)
    result = {"datasets": [], "checkpoints": []}
    for directory in Path(".runtime/datasets").iterdir():
        path = directory / "meta/info.json"
        if path.is_file():
            info = json.loads(path.read_text(encoding="utf-8"))
            result["datasets"].append({"name": directory.name,
                **{key: info.get(key) for key in ("fps", "total_episodes", "total_frames", "total_tasks", "features", "splits")},
                "file_hashes": {file.relative_to(directory).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
                    for file in directory.rglob("*") if file.is_file() and file.suffix in {".json", ".parquet"}}})
    for directory in sorted(Path(".runtime").glob("*training*")) + sorted(Path(".runtime").glob("navigation-recovery-*")) + sorted(Path(".runtime").glob("navigation-task-waypoints-*")):
        report_path = directory / "report.json"
        weights = directory / "checkpoint/model.safetensors"
        if not report_path.exists() or not weights.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
        result["checkpoints"].append({"name": directory.name,
            **{key: report.get(key) for key in ("status", "embodiment", "steps", "total_steps", "initialization_checkpoint", "train_frames", "validation_frames", "training_episodes", "validation_episodes", "parent_weights_sha256")},
            "weights_sha256": actual_hash, "hash_matches_report": actual_hash == report.get("weights_sha256"),
            "qualified_arm_manifest_present": (weights.parent / "milo-policy.json").exists()})
    from scripts.recovery_policy import load_episode
    recovery = [load_episode(path) for path in sorted(Path(".runtime/recovery-demonstrations-v1").glob("episode-*"))]
    train_hashes = {frame["image_sha256"] for metadata, frames in recovery if metadata["split"] == "train" for frame in frames}
    validation_hashes = {frame["image_sha256"] for metadata, frames in recovery if metadata["split"] == "validation" for frame in frames}
    result["recovery"] = {"episodes_verified": len(recovery), "frames_verified": sum(len(frames) for _, frames in recovery),
        "exact_image_hash_overlap": len(train_hashes & validation_hashes),
        "actions_per_instruction": {text: sorted({tuple(frame["action"]) for _, frames in recovery for frame in frames if frame["task"] == text}) for text in POLICY_TASKS.values()}}
    write(output / "inventory.json", result)
    print(json.dumps({"dataset_counts": [{key: row[key] for key in ("name", "total_episodes", "total_frames", "total_tasks")} for row in result["datasets"]],
        "recovery": result["recovery"], "checkpoint_hashes_match": all(row["hash_matches_report"] for row in result["checkpoints"])}))


def primitive_pass(name, before, after):
    heading = before[2]
    delta = np.array(after[:2]) - before[:2]
    forward = float(delta @ np.array([math.cos(heading), math.sin(heading)]))
    lateral = float(delta @ np.array([-math.sin(heading), math.cos(heading)]))
    turn = after[2] - before[2]
    if name in {"forward", "backward"}:
        return forward * (1 if name == "forward" else -1) > .12 and abs(lateral) < .1 and abs(turn) < .25
    if name in {"left", "right"}:
        return turn * (1 if name == "left" else -1) > .35 and np.linalg.norm(delta) < .06
    return np.linalg.norm(delta) < .02 and abs(turn) < .08


class SeededClient(LocalNavigationClient):
    def __init__(self, checkpoint, seed):
        super().__init__(checkpoint)
        self.seed = seed

    async def start(self):
        directory = Path(".runtime/local-navigation")
        directory.mkdir(exist_ok=True, parents=True)
        self.log = (directory / f"audit-{uuid4()}.log").open("w", encoding="utf-8")
        self.process = await asyncio.create_subprocess_exec(str(MODEL_PYTHON), "-u", "-m", "scripts.navigation_policy",
            "--stage", "inference", "--checkpoint", str(self.checkpoint), "--seed", str(self.seed),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log, limit=2_500_000)
        if (await self.reply(300)).get("ready") is not True:
            raise ValueError("Model did not become ready")


class PrimitivePolicy:
    def __init__(self, zero=False):
        self.instruction = ""
        self.zero = zero

    @staticmethod
    def check_available():
        pass

    async def start(self):
        pass

    async def predict(self, observation, image):
        name = next((name for name, text in POLICY_TASKS.items() if text == self.instruction), None)
        if name is None:
            raise ValueError("Lookup cannot interpret a new instruction")
        action = [0., 0.] if self.zero else TEACHER_ACTIONS[name]
        return {"action": list(action), "raw_action": list(action), "saturated_axes": []}

    async def close(self):
        pass


class AuditedWorker(SimulationWorker):
    def __init__(self, *args, **kwargs):
        self.audit = []
        self.pending_prediction = None
        self.audit_source = "unknown"
        super().__init__(*args, **kwargs)

    async def execute_navigation(self, command):
        before = await self.call(lambda sim: {"odometry": sim.odometry.tolist(), "ticks": sim.ticks})
        result = await super().execute_navigation(command)
        is_drive = any(segment.get("kind") == "drive" for segment in command.arguments.get("segments", []))
        self.audit.append({"event": "accepted" if result.status == "ok" else "rejected", "source": self.audit_source if is_drive else "geometric_head_or_executive",
            "command": command.model_dump(), "prediction": self.pending_prediction if is_drive else None,
            "result": result.model_dump(), "before": before})
        return result

    async def install_audit(self):
        def operation(sim):
            original = sim.on_tick
            def tick():
                import pybullet as bullet
                original()
                if self.navigation and self.navigation.buffer:
                    segment = self.navigation.buffer[0][0]
                    self.audit.append({"event": "executed_tick", "sample_period_sim_s": .05,
                        "source": self.audit_source if segment.kind == "drive" else "geometric_head_or_executive", "simulated_s": sim.ticks / 240,
                        "wall_s": time.monotonic(), "revision": self.navigation.revision,
                        "evaluator_completed": self.latest["challenge"]["status"] == "completed",
                        "odometry": sim.odometry.tolist(), "velocity": self.navigation.velocity.tolist(),
                        "requested": segment.model_dump(),
                        "measured_wheel_radps": [bullet.getJointState(sim.robot, index, physicsClientId=sim.client)[1] for index in sim.wheels],
                        "collisions": sim.proximity_sensors().model_dump()["collisions"]})
            sim.on_tick = tick
        await self.call(operation)


class LoggedPolicy:
    def __init__(self, policy, worker, source):
        self.policy, self.worker, self.source = policy, worker, source
        self.instruction = ""

    async def start(self):
        self.policy.instruction = self.instruction
        await self.policy.start()

    async def predict(self, observation, image):
        self.policy.instruction = self.instruction
        began = time.monotonic()
        result = await self.policy.predict(observation, image)
        entry = {"event": "proposed", "source": self.source, "instruction": self.instruction, "seq": observation.seq,
            "wall_s": time.monotonic(), "latency_s": time.monotonic() - began, "image_sha256": hashlib.sha256(image).hexdigest(),
            "observation": observation.model_dump(), **result}
        self.worker.audit.append(entry)
        self.worker.pending_prediction = entry
        return result

    async def close(self):
        await self.policy.close()


async def setup(challenge, heading=0.):
    worker = AuditedWorker(challenge=challenge, pace=True)
    await asyncio.wrap_future(worker.ready)
    await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
    if heading:
        def initialize_heading(sim):
            import pybullet as bullet
            position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
            roll, pitch, _ = bullet.getEulerFromQuaternion(orientation)
            bullet.resetBasePositionAndOrientation(sim.robot, position, bullet.getQuaternionFromEuler([roll, pitch, heading]), physicsClientId=sim.client)
        await worker.call(initialize_heading)
    await worker.begin_navigation(worker.stop_revision)
    await worker.call(lambda sim: prepare(sim, worker.navigation))
    await worker.install_audit()
    return worker


async def skill_trial(client, options, name, seed, variant):
    docking = options.stage == "docking"
    case = DOCKING_CASES[variant] if docking else None
    if name == "parking":
        challenge = challenge_for(case if docking else EVALUATION_CASES[variant])
        if docking:
            challenge.initial_xy = case["initial_xy"]
    else:
        challenge = get_challenge("park").model_copy(deep=True, update={"initial_xy": [-.25, -.1], "initial_head_pitch": .45})
        if variant == 1:
            challenge.initial_xy = [-.19, -.04]
            challenge.initial_head_pitch = .55
        if variant == 2:
            challenge.initial_xy = [-.19, -.04]
            challenge.objects.append({"name": "audit_obstacle", "size": [.16, .16, .35], "position": [.9, .4, .175], "color": [.5, .5, .5, 1.]})
    worker = await setup(challenge, case["heading_rad"] if docking else 0.)
    worker.audit_source = options.source
    directory = options.output / f"{name}-{seed}-{variant}"
    directory.mkdir()
    report = {"task": name, "seed": seed, "variant": variant, "source": options.source, "success": False, "error": None,
        "instruction": TASK if name == "parking" else POLICY_TASKS[name], "requests": [], "interventions": 0}
    if docking:
        report.update(case=case, protocol="docking-v1", termination="step_limit",
            conditioning=getattr(options, "conditioning", "task"),
            safety="Unchanged privileged simulator shield; not hardware-safe autonomy",
            controller_inputs="AgentObservation and paired RGB only; no evaluator or scene coordinates")
    began = time.monotonic()
    try:
        if client:
            await client.reset()
        policy = client if client else SensorDockingPolicy() if options.source == "sensor" else PrimitivePolicy(options.source == "zero")
        if docking and getattr(options, "conditioning", "task") == "waypoint":
            policy = WaypointDockingPolicy(client)
        policy.instruction = report["instruction"]
        before = await worker.call(evaluation_state)
        votes = 0
        for index in range(options.steps if name == "parking" else 2):
            if docking and time.monotonic() - began >= min(options.seconds, 60.):
                report["termination"] = "wall_deadline"
                break
            observation, image = await worker.feedback()
            (directory / f"camera-{index}.png").write_bytes(image)
            started = time.monotonic()
            proposal = await policy.predict(observation, image)
            bounded, modified = bounded_velocity(proposal.get("raw_action", proposal["action"]))
            stopped = abs(bounded[0]) < .025 and abs(bounded[1]) < .05
            action = [0., 0.] if stopped else bounded
            entry = {"index": index, "source": options.source, "proposed": proposal.get("raw_action", proposal["action"]),
                "instruction": proposal.get("instruction", policy.instruction),
                "bounded": bounded, "clipped_axes": modified, "deadband_override": stopped and action != bounded,
                "requested": action, "latency_s": time.monotonic() - started, "observation_age_s": time.time() - observation.wall_timestamp,
                "observation": observation.model_dump()}
            report["requests"].append(entry)
            if "observed_target_m" in proposal:
                entry["observed_target_m"] = proposal["observed_target_m"]
            if not 0 <= entry["observation_age_s"] <= 2:
                entry.update(status="rejected", reason="STALE_OBSERVATION")
                break
            command = Command(run_id=worker.sim.run_id, episode_epoch=0, observation_seq=observation.seq, action_id=str(uuid4()),
                tool="replace_motion_buffer", arguments={"expected_revision": observation.navigation.revision,
                    "segments": [{"kind": "drive", "linear_mps": action[0], "angular_radps": action[1], "duration_s": 1.}]})
            result = await worker.execute_navigation(command)
            entry.update(status=result.status, reason=result.message, error=result.error)
            async with asyncio.timeout(5):
                while worker.navigation.buffer:
                    await asyncio.sleep(.02)
            entry["measured"] = await worker.call(evaluation_state)
            entry["execution_status"] = worker.navigation.status
            entry["execution_reason"] = worker.navigation.reason
            if worker.sim.cancel.is_set() or result.status != "ok":
                break
            votes = votes + 1 if stopped else 0
            if name == "parking" and votes >= 2:
                report["termination"] = "controller_stop"
                break
        after = await worker.call(evaluation_state)
        report.update(initial=before, final=after, wall_s=time.monotonic() - began)
        report["success"] = after["parked"] if name == "parking" else primitive_pass(name,
            [*before["position"][:2], before["heading"]], [*after["position"][:2], after["heading"]])
        report["success"] = bool(report["success"] and not any(event.get("collisions") for event in worker.audit) and not worker.sim.cancel.is_set())
        if docking:
            from scripts.navigation_policy import state_vector
            terminal, _ = await worker.feedback()
            score = docking_score(after, case["target"], votes >= 2, state_vector(terminal)[2:4], report["requests"])
            score["success"] = bool(score["success"] and report["success"])
            report.update(score)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        worker.stop()
        await worker.hold_stopped()
        write(directory / "report.json", report)
        write(directory / "actions.json", worker.audit)
        await worker.close()
    print(json.dumps({key: report.get(key) for key in ("task", "seed", "variant", "success", "wall_s", "error")}), flush=True)
    return report


async def mission(options, seed, identifier, resident):
    from dotenv import load_dotenv
    load_dotenv("backend/.env", override=False)
    load_dotenv(".env", override=False)
    config = FoundryConfig.from_environment()
    worker = AuditedWorker(challenge=get_challenge(identifier), pace=True)
    await asyncio.wrap_future(worker.ready)
    worker.audit_source = options.source
    await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
    await worker.install_audit()
    class Factory:
        @staticmethod
        def check_available():
            if options.source == "smolvla" and not (options.checkpoint / "model.safetensors").exists():
                raise ValueError("Checkpoint unavailable")
        def __call__(self):
            return LoggedPolicy(resident() if options.source == "smolvla" else PrimitivePolicy(options.source == "zero"), worker, options.source)
    controller = AgentController(config, local_navigation_factory=Factory(), evaluation_budget=NavigationEvaluationBudget(timeout_s=options.seconds))
    controller.trace_capacity = 10000
    directory = options.output / f"mission-{identifier}-{seed}"
    directory.mkdir()
    try:
        if options.source == "smolvla":
            session = resident()
            await session.start()
            await session.close()
        began = time.monotonic()
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_navigation",
            goal=get_challenge(identifier).goal, reasoning="high", feedback_interval_s=.25, max_turns=80))
        while not controller.task.done():
            if controller.trace_revision != getattr(controller, "audit_printed", 0):
                controller.audit_printed = controller.trace_revision
                print(json.dumps({"task": identifier, "source": options.source, "phase": controller.state["phase"],
                    "turn": controller.state["turns"], "elapsed_s": round(time.monotonic() - began, 1)}), flush=True)
            await asyncio.sleep(.2)
        await controller.task
        final = await worker.call(evaluation_state)
        events = controller.trace()["events"]
        report = {"source": options.source, "task": identifier, "seed": seed, "success": final["parked"], "final": final,
            "wall_s": time.monotonic() - began, "phase": controller.state["phase"], "error": controller.state["error"],
            "outcome": controller.state["outcome"], "turns": controller.state["turns"], "input_tokens": controller.state["input_tokens"],
            "output_tokens": controller.state["output_tokens"], "interventions": worker.latest["manual_placements"],
            "false_success": (controller.state["outcome"] or {}).get("kind") == "completed" and not final["parked"],
            "deployment": next(profile.deployment for profile in config.models if profile.id == "luna"), "identity_verified": False,
            "matched": "Same Luna prompt/tools/RGB/proprioception, primitive execution, safety and time budget; only local policy source differs"}
        write(directory / "report.json", report)
        write(directory / "actions.json", worker.audit)
        write(directory / "trace.json", events)
        for event_id, image in controller.trace_images.items():
            (directory / f"camera-{event_id}.png").write_bytes(image)
        print(json.dumps(report), flush=True)
        return report
    finally:
        await controller.halt()
        await worker.close()


async def main(options):
    if options.stage == "summarize":
        summarize_artifacts(options.output)
        return
    if options.stage == "inventory":
        inventory(options.output)
        return
    if options.variants is None:
        options.variants = list(range(len(DOCKING_CASES))) if options.stage == "docking" else [0, 1, 2]
    if options.steps is None:
        options.steps = 40 if options.stage == "docking" else 20
    if options.source == "sensor" and options.stage != "docking":
        raise ValueError("The sensor baseline is restricted to the fixed docking benchmark")
    if getattr(options, "conditioning", "task") == "waypoint" and (options.stage != "docking" or options.source != "smolvla"):
        raise ValueError("Waypoint conditioning is an isolated SmolVLA docking experiment")
    options.output.mkdir(parents=True, exist_ok=False)
    files = [path for folder in ("backend", "scripts") for path in Path(folder).glob("*.py")]
    manifest = {"protocol": "architecture-audit-v1", "stage": options.stage, "source": options.source,
        "conditioning": getattr(options, "conditioning", "task"),
        "seeds": options.seeds, "steps": options.steps, "seconds": options.seconds, "variants": options.variants,
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "checkpoint": str(options.checkpoint), "weights_sha256": hashlib.sha256((options.checkpoint / "model.safetensors").read_bytes()).hexdigest()
            if options.source == "smolvla" else None,
        "primitive_postconditions": "2x1s actions: signed translation>.12m lateral<.1m yaw<.25rad; turn>.35rad translation<.06m; hold translation<.02m yaw<.08rad; no contacts/safety failure",
        "seed_note": "Explicit local Torch seeds, reset per trial; cloud sampling not controlled", "privileged": "Reports only. No ground-truth position or scorer is passed to either policy"}
    write(options.output / "manifest.json", manifest)
    if options.stage == "docking":
        manifest.update(protocol="docking-v1", cases=[DOCKING_CASES[index] for index in options.variants],
            pass_criteria="Physical full-bay containment, two controller stop votes, measured v<.01m/s w<.02rad/s, no contacts or execution failures",
            policy_control="Same RGB320x240, state, preparation scans, one-second buffers, clipping/deadband and 60s skill limit for all sources",
            sensor_controller="Existing green marker segmentation and nominal floor projection once, then wheel-odometry proportional tracking")
        write(options.output / "manifest.json", manifest)
    reports = []
    try:
        for seed in options.seeds:
            if options.stage == "missions":
                resident = ResidentNavigationModel(lambda: SeededClient(options.checkpoint, seed))
                try:
                    for identifier in options.challenges:
                        reports.append(await mission(options, seed, identifier, resident))
                finally:
                    await resident.close()
            else:
                client = SeededClient(options.checkpoint, seed) if options.source == "smolvla" else None
                try:
                    if client:
                        await client.start()
                    names = ["parking"] if options.stage in {"parking", "docking"} else list(POLICY_TASKS)
                    for variant in options.variants:
                        for name in names:
                            reports.append(await skill_trial(client, options, name, seed, variant))
                finally:
                    if client:
                        await client.close()
    finally:
        write(options.output / "summary.json", [{key: value for key, value in report.items() if key != "requests"} for report in reports])
        write(options.output / "aggregate.json", {"trials": len(reports), "successes": sum(report["success"] for report in reports),
            "wall_s": distribution([report["wall_s"] for report in reports if "wall_s" in report]),
            "policy_latency_s": distribution([entry["latency_s"] for report in reports for entry in report.get("requests", [])])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["primitives", "parking", "docking", "missions", "inventory", "summarize"], required=True)
    parser.add_argument("--source", choices=["smolvla", "lookup", "zero", "sensor"], default="smolvla")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--conditioning", choices=["task", "waypoint"], default="task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[716, 717, 718])
    parser.add_argument("--variants", type=int, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--challenges", nargs="+", default=["park", "kitchen_bathroom"])
    asyncio.run(main(parser.parse_args()))