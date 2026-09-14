import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from uuid import uuid4

import uvicorn

from backend.agent import AgentStart, NavigationEvaluationBudget
from backend.challenges import get_challenge, shared_apartment
from scripts.evaluate_supervised import ROOT, design_snapshot, execution_metrics, workload_snapshot


def trial_plan():
    cases = [("park", "standalone"), ("flat_kitchen", "shared_apartment_v1"),
        ("apartment", "shared_apartment_v1")]
    trials = []
    for index, (identifier, environment) in enumerate(cases):
        challenge = get_challenge(identifier) if environment == "standalone" else shared_apartment(identifier)
        for backend in (("builtin", "nav2") if index % 2 == 0 else ("nav2", "builtin")):
            trials.append({"case_id": f"{identifier}-{backend}", "challenge": identifier,
                "environment": environment, "navigation_backend": backend, "goal": challenge.goal,
                "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()})
    return trials


def local_trial_plan():
    challenge = get_challenge("park")
    digest = hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()
    return [{"case_id": f"straight-{distance:.1f}-{repeat}-{backend}", "challenge": "park",
        "environment": "standalone", "navigation_backend": backend, "target_m": [distance, 0.],
        "repeat": repeat, "challenge_sha256": digest}
        for repeat in range(2) for distance in (.7, 1.2)
        for backend in (("builtin", "nav2") if repeat == 0 else ("nav2", "builtin"))]


async def prepare_local_trial(worker):
    from backend.continuous_navigation import ContinuousScan

    await worker.scan_continuous(ContinuousScan(run_id=worker.latest["run_id"],
        episode_epoch=worker.epoch, compact_arms=True))
    scanned_at = time.monotonic()
    await until(lambda: worker.nav2_status()["ready"] and worker.ros_bridge_at >= scanned_at,
        90, "Fresh post-scan bridge readiness")


async def local_trial(lab, trial, directory, port):
    from websockets.asyncio.client import connect
    from backend.continuous_navigation import ContinuousTarget
    from backend.recording import RunRecorder
    from backend.ros_navigation import RosStart

    worker = lab.worker
    await prepare_local_trial(worker)
    target = trial["target_m"]

    def observe(sim):
        worker._sample_spatial(force=True)
        if worker.spatial_error:
            raise RuntimeError(worker.spatial_error)
        allowed = worker.spatial_map.traversable(sim.odometry[:2], sim.robot_footprint()["radius_m"])
        if not worker.spatial_map.contains_path(allowed, [target]):
            raise RuntimeError("Fixed probe target is not observed traversable floor")
        return next(reversed(worker.spatial_frames.values()))[0]

    sensor = await worker.call(observe)
    recorder = RunRecorder(directory / "recording", context=lambda: {"phase": "scripted_local_goal"})
    await worker.call(lambda sim: (setattr(worker, "recorder", recorder), recorder.capture(worker)))
    result = {**trial, "evidence": "scripted_local_goal", "sensor_only": True,
        "rendering": worker.rendering, "initial_odometry": sensor.odometry_m_rad,
        "target_heading_rad": 0., "arrival_tolerance_m": .04, "timeout_s": 60}
    began = time.monotonic()
    try:
        async with connect(f"ws://127.0.0.1:{port}/api/live", max_size=4 * 1024 * 1024) as connection:
            await asyncio.wait_for(connection.recv(), 5)

            async def drain():
                async for _message in connection:
                    pass

            reader = asyncio.create_task(drain())
            try:
                began = time.monotonic()
                if trial["navigation_backend"] == "nav2":
                    await worker.start_ros(RosStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                        target_m_rad=[*target, 0.]))
                else:
                    await worker.start_continuous(ContinuousTarget(run_id=worker.latest["run_id"],
                        episode_epoch=worker.epoch, spatial_sequence=sensor.sequence, pixel=[.5, .5]),
                        selected_sensor=sensor, selected_target=target)
                await until(lambda: not (worker.ros_navigation.active if trial["navigation_backend"] == "nav2"
                    else worker.continuous.active), 65, "Local goal ended")
                elapsed = time.monotonic() - began
                navigation = worker.ros_navigation.state() if trial["navigation_backend"] == "nav2" else worker.continuous.state()
                odometry = (await worker.ros_odometry())["odometry_m_rad"]
                result.update(elapsed_s=elapsed, navigation=navigation, final_odometry=odometry,
                    endpoint_error_m=math.dist(odometry[:2], target),
                    local_arrival=navigation["status"] == "arrived" and math.dist(odometry[:2], target) < .04)
            finally:
                worker.stop()
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", local_arrival=False,
            elapsed_s=time.monotonic() - began)
    finally:
        worker.stop()
        await worker.call(lambda sim: (recorder.capture(worker), setattr(worker, "recorder", None)))
        score = await asyncio.to_thread(recorder.finish, {"evidence": "scripted_test", "real_model": False,
            "label": trial["case_id"], "goal": target})
        result.update(recording_scorecard=score, contacts=score.get("contact_episodes"),
            distance_m=score.get("actual_distance_m"), recording_complete=score["complete_recording"])
        result["verified_local_arrival"] = bool(result.get("local_arrival") and score["complete_recording"]
            and not score["operator_assisted"] and score.get("contact_episodes") == 0)
        write_json(directory / "report.json", result)
    return result


def frozen_sources():
    snapshot = design_snapshot("nav2-task-pilot-v1")
    hashes = snapshot["code_sha256"]
    for path in sorted(ROOT.joinpath("ros").glob("*")):
        if path.is_file() and (path.suffix in {".py", ".yaml"} or path.name == "Dockerfile"):
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["start.ps1"] = hashlib.sha256(ROOT.joinpath("start.ps1").read_bytes()).hexdigest()
    snapshot["source_sha256"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return snapshot


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


async def until(predicate, timeout_s, message):
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(.2)
    return message


async def evaluate(options):
    from backend.app import app, lab
    from backend.session_recording import SESSION_RESULTS_ROOT

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", options.port))
    options.output.mkdir(parents=True, exist_ok=False)
    os.environ["MILO_ROS_ENABLED"] = "1"
    os.environ["MILO_RENDERER"] = "enhanced"
    design = frozen_sources()
    settings = {"execution_mode": "luna_continuous", "model_id": "luna", "reasoning": "high",
        "continuous_handoff": False, "skill_composer": False, "adaptive_navigation": True,
        "compact_arms": True, "images_per_request": 2, "context_tokens": 8192,
        "feedback_interval_s": .25, "max_turns": 80}
    manifest = {"design": design, "started_at": datetime.now(timezone.utc).isoformat(),
        "evidence": "preflight_no_inference" if options.preflight else "scripted_local_goal" if options.local_goals else "real_model",
        "trials": local_trial_plan() if options.local_goals else trial_plan(), "settings": settings,
        "budget": {"timeout_s": 60 if options.local_goals else 180, "max_turns": None}, "results": [],
        "workload_before": workload_snapshot(), "source_changed_during_run": False,
        "limitations": ["One attempt per backend/task; descriptive pilot, not reliability or causal evidence",
            "Nav2 receives an additional simulated laser; this compares integrated systems",
            "Semantic candidate generation and marked-floor approach remain built-in",
            "Both backends share bridge/camera polling overhead; cloud seed and workload are uncontrolled",
            "No tuning, retries, operator steering or budget changes during measurement"]}
    write_json(options.output / "experiment.json", manifest)
    log = (options.output / "nav2.log").open("w", encoding="utf-8")
    container = f"milo-nav2-pilot-{uuid4().hex[:8]}"
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=options.port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    created = False
    try:
        await until(lambda: server.started or serving.done(), 60, "Server ready")
        if serving.done():
            await serving
            raise RuntimeError("Evaluation server exited during startup")
        profile = next((profile for profile in lab.agent.config.models if profile.id == "luna"), None)
        if not options.local_goals and (profile is None or not lab.agent.config.configured(profile)):
            raise RuntimeError("Configured inference is unavailable")
        manifest["supervisor_deployment"] = profile.deployment if profile and not options.local_goals else None
        manifest["underlying_model_identity_verified"] = False
        command = ["docker", "run", "--detach", "--name", container, "--restart", "unless-stopped",
            "--env", f"ROS_DOMAIN_ID={options.port % 200}",
            "--mount", f"type=bind,source={ROOT / 'ros'},target=/opt/milo,readonly", "milo-nav2:jazzy",
            "bash", "-lc", f"source /opt/ros/jazzy/setup.bash && exec ros2 launch /opt/milo/milo.launch.py backend:=http://host.docker.internal:{options.port}"]
        launched = await asyncio.to_thread(subprocess.run, command, capture_output=True, text=True, check=True, timeout=30)
        created = True
        manifest["container_id"] = launched.stdout.strip()
        image = await asyncio.to_thread(subprocess.run, ["docker", "image", "inspect", "milo-nav2:jazzy",
            "--format", "{{.Id}}"], capture_output=True, text=True, check=True, timeout=15)
        manifest["image_id"] = image.stdout.strip()
        if options.preflight:
            await until(lambda: lab.worker.nav2_status()["ready"], 90, "Nav2 ready")
            if options.local_goals:
                await prepare_local_trial(lab.worker)
            manifest["preflight_passed"] = bool(lab.agent.record_sessions and not lab.agent.active and not lab.worker.latest["busy"])
            if not manifest["preflight_passed"]:
                raise RuntimeError("Evaluation lifecycle is not idle and recording-enabled")
            print("Preflight passed: recording enabled, Nav2 ready; no inference or wheel command", flush=True)
            return
        for trial in manifest["trials"]:
            if frozen_sources()["code_sha256"] != design["code_sha256"]:
                raise RuntimeError("Source drift before trial; remaining trials cancelled")
            await lab.reset(trial["challenge"], environment=trial["environment"])
            await until(lambda: lab.worker.nav2_status()["ready"], 90, "Nav2 ready")
            if options.local_goals:
                directory = options.output / trial["case_id"]
                directory.mkdir()
                print(json.dumps({"started": trial["case_id"], "evidence": "scripted_local_goal"}), flush=True)
                result = await local_trial(lab, trial, directory, options.port)
                manifest["results"].append(result)
                write_json(options.output / "experiment.json", manifest)
                print(json.dumps(result), flush=True)
                continue
            lab.agent.evaluation_budget = NavigationEvaluationBudget(timeout_s=180)
            lab.agent.evaluation_camera_history = True
            await lab.worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
            existing = set(SESSION_RESULTS_ROOT.glob("browser-session-*"))
            request = AgentStart(**settings, navigation_backend=trial["navigation_backend"],
                run_id=lab.worker.latest["run_id"], episode_epoch=lab.worker.epoch, goal=trial["goal"])
            began = time.monotonic()
            lab.agent.start(lab.worker, request)
            session_id = lab.agent.state["session_id"]
            print(json.dumps({"started": trial["case_id"], "session_id": session_id}), flush=True)
            await asyncio.wait_for(asyncio.shield(lab.agent.task), 210)
            await lab.agent.recording_finished.wait()
            directories = [path for path in SESSION_RESULTS_ROOT.glob("browser-session-*") if path not in existing
                and json.loads((path / "experiment.json").read_text(encoding="utf-8")).get("session_id") == session_id]
            if len(directories) != 1:
                raise RuntimeError("Trial recording could not be uniquely identified")
            directory = directories[0]
            report_path = directory / trial["challenge"] / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            score = report["recording_scorecard"]
            trace = lab.agent.trace()["events"]
            outcome = lab.agent.state.get("outcome") or {}
            completed = bool(report["physics_success"] and report["verification_eligible"]
                and score["complete_recording"] and not score["operator_assisted"]
                and score.get("completion_time_s") is not None)
            result = {**trial, "session_id": session_id, "report": report_path.relative_to(ROOT).as_posix(),
                "verified_success": completed, "elapsed_s": report["evaluation_elapsed_s"],
                "completion_s": score.get("completion_time_s"), "distance_m": score.get("actual_distance_m"),
                "contacts": score.get("contact_episodes"), "recording_complete": score["complete_recording"],
                "dropped_records": score.get("dropped_records"), "outcome": outcome,
                "error": lab.agent.state.get("error"), "input_tokens": report["input_tokens"],
                "output_tokens": report["output_tokens"], "supervisor_turns": report["supervisor_turns"],
                "challenge_status": report["challenge_status"], "rendering": report["rendering"],
                "wall_including_finalization_s": time.monotonic() - began, **execution_metrics(trace)}
            manifest["results"].append(result)
            write_json(options.output / "experiment.json", manifest)
            print(json.dumps(result), flush=True)
            if any(f"HTTP {status}" in (result["error"] or "") for status in (401, 403)):
                raise RuntimeError("Inference access denied; remaining model trials cancelled")
            if outcome.get("kind") == "interrupted":
                raise RuntimeError("Trial interrupted; remaining trials cancelled")
    except BaseException as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if lab.worker is not None:
            lab.worker.stop()
            await lab.agent.halt("Evaluation finished")
        if created:
            await asyncio.to_thread(subprocess.run, ["docker", "logs", container], stdout=log, stderr=subprocess.STDOUT, timeout=20)
            await asyncio.to_thread(subprocess.run, ["docker", "rm", "--force", container], capture_output=True, check=True, timeout=30)
        log.close()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["workload_after"] = workload_snapshot()
        manifest["source_changed_during_run"] = frozen_sources()["code_sha256"] != design["code_sha256"]
        if manifest["source_changed_during_run"] and not manifest.get("error"):
            manifest["error"] = "Source changed during final trial or cleanup; comparison is not frozen"
        write_json(options.output / "experiment.json", manifest)
        if manifest["source_changed_during_run"]:
            raise RuntimeError(manifest["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Six frozen-code real-model Built-in/Nav2 trials")
    parser.add_argument("--plan", action="store_true", help="Validate paired settings without motion or inference")
    parser.add_argument("--preflight", action="store_true", help="Check isolated server/container readiness without inference")
    parser.add_argument("--local-goals", action="store_true", help="Eight paired scripted local-goal probes; no inference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=8014)
    arguments = parser.parse_args()
    if arguments.plan:
        planned = local_trial_plan() if arguments.local_goals else trial_plan()
        if not arguments.local_goals:
            for trial in planned:
                AgentStart(run_id="plan", episode_epoch=0, goal=trial["goal"],
                    execution_mode="luna_continuous", navigation_backend=trial["navigation_backend"])
        assert len(planned) == (8 if arguments.local_goals else 6)
        assert all(planned[index]["challenge_sha256"] == planned[index + 1]["challenge_sha256"] for index in range(0, len(planned), 2))
        print(json.dumps(planned, indent=2))
    elif arguments.output is None:
        parser.error("--output is required unless --plan is selected")
    else:
        asyncio.run(evaluate(arguments))