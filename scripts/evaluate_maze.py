"""Bounded, recorded hybrid maze diagnostics; not a household benchmark series."""

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from backend.agent import AgentController, AgentStart, FoundryConfig, check_local_supervisor, prepare_local_supervisor
from backend.challenges import get_challenge
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission
from backend.regression import evaluation_snapshot, score_case
from backend.worker import SimulationWorker
from scripts.benchmark_household import changed_sources
from scripts.evaluate_supervised import ROOT, design_snapshot, workload_snapshot


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


async def preflight():
    # Let the application's normal loader consume credentials; never print them.
    load_dotenv(ROOT / "backend/.env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    config = FoundryConfig.from_environment()
    profiles = {profile.id: profile for profile in config.models}
    qwen, luna = profiles.get("qwen"), profiles.get("luna")
    local = await check_local_supervisor(config, "qwen")
    ready = bool(qwen and qwen.provider == "ollama" and local.get("ready")
                 and luna and luna.provider == "foundry" and config.configured(luna)
                 and "low" in luna.reasoning_efforts)
    return config, {"ready": ready, "qwen_ready": local.get("ready", False),
        "qwen_status": local.get("status"), "luna_configured": bool(luna and config.configured(luna)),
        "models": [profiles[name].model_dump() for name in ("qwen", "luna") if name in profiles],
        "luna_access": "Not verified until the first supervised request; no fallback"}


async def trial(config, directory, options, design):
    directory.mkdir()
    challenge = get_challenge("maze")
    report = {"evidence": "real_model_diagnostic", "status": "invalid", "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(), "challenge": "maze",
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest(),
        "source_sha256": design["source_sha256"], "saved_map_reused": False}
    worker = controller = None
    began = None
    try:
        worker = SimulationWorker(challenge=challenge, pace=True, rendering="enhanced")
        await asyncio.wait_for(asyncio.wrap_future(worker.ready), 90.)
        worker.home_mission = HomeMission(worker, MapStore(directory / "maps.sqlite3"))
        observation, image = await worker.feedback()
        (directory / "initial.png").write_bytes(image)
        report["vision_preparation"] = await prepare_local_supervisor(config, "qwen", image)
        report["initial_evaluation"] = await worker.call(evaluation_snapshot)
        settings = AgentStart(run_id=observation.run_id, episode_epoch=worker.epoch, goal=challenge.goal,
            model_id="qwen", reasoning="none", execution_mode="luna_continuous", unified_mission=True,
            task_supervisor_model_id="luna", task_supervisor_reasoning="low",
            max_task_supervisor_requests=4, max_task_supervisor_tokens=100000,
            mission_budget_s=options.seconds, max_turns=30, max_model_requests=30,
            max_model_tokens=200000, images_per_request=2, context_tokens=8192,
            feedback_interval_s=.25, map_context=True, compact_arms=True)
        report["settings"] = settings.model_dump(mode="json")
        write(directory / "plan.json", report)
        controller = AgentController(config)
        controller.record_sessions = True
        controller.recording_root = directory / "sessions"
        controller.recording_evidence = "real_model"
        began = time.monotonic()
        controller.start(worker, settings)
        while not controller.task.done():
            await asyncio.wait([controller.task], timeout=10.)
            print(json.dumps({"trial": directory.name, "elapsed_s": round(time.monotonic()-began, 1),
                "phase": controller.state["phase"], "turns": controller.state["turns"]}), flush=True)
            if time.monotonic()-began > options.seconds + 60:
                report["error"] = "Evaluation wall-time ceiling exceeded"
                await controller.halt()
                break
        await controller.task
        report["control_elapsed_s"] = time.monotonic()-began
        report["final_evaluation"] = await worker.call(evaluation_snapshot)
        report["challenge_status"] = copy.deepcopy(worker.latest["challenge"])
        report["controller"] = copy.deepcopy(controller.state)
        report["mission"] = controller.mission.state() if controller.mission else None
        report["recording_directory"] = str(controller.recording_directory) if controller.recording_directory else None
        score = None
        if controller.recording_directory:
            score_path = Path(controller.recording_directory) / "scorecard.json"
            if score_path.is_file():
                score = json.loads(score_path.read_text(encoding="utf-8"))
        report["scorecard"] = score
        report["status"] = score_case(report["challenge_status"], score, report["error"] or controller.recording_error)
        _, image = await worker.feedback()
        (directory / "terminal.png").write_bytes(image)
    except Exception as error:
        # Exception strings can include provider details: retain the type only.
        report["error"] = type(error).__name__
    finally:
        if controller:
            await controller.halt()
            report.setdefault("controller", copy.deepcopy(controller.state))
            write(directory / "trace.json", controller.trace())
        if worker:
            worker.stop()
            await worker.close()
        report["changed_sources"] = changed_sources(design["code_sha256"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        if report["changed_sources"]:
            report["status"] = "invalid"
        write(directory / "result.json", report)
    return report


async def main(options):
    config, checks = await preflight()
    print(json.dumps(checks, indent=2), flush=True)
    if options.check:
        return 0 if checks["ready"] else 1
    options.output.mkdir(parents=True, exist_ok=False)
    design = design_snapshot("maze-hybrid-diagnostic-v1")
    manifest = {"evidence": "real_model_diagnostic", "design": design, "preflight": checks,
        "planned_trials": options.repeats, "budget_per_trial_s": options.seconds,
        "workload_before": workload_snapshot(), "trials": [],
        "criteria": "Unmodified maze scorer completed, zero contacts, complete unassisted recording, unchanged source",
        "limitations": "Same-layout diagnostic repeats, not a fixed household baseline or a reliability estimate. Configured Luna deployment name does not verify underlying model identity."}
    write(options.output / "experiment.json", manifest)
    if not checks["ready"]:
        manifest["blocked"] = "Required Qwen + Luna configuration unavailable; no fallback or motion"
        write(options.output / "experiment.json", manifest)
        return 1
    for repeat in range(1, options.repeats + 1):
        if changed_sources(design["code_sha256"]):
            manifest["blocked"] = "Source changed before trial; remaining trials not run"
            break
        result = await trial(config, options.output / f"trial-{repeat:02d}", options, design)
        score, controller = result.get("scorecard") or {}, result.get("controller") or {}
        row = {"trial": repeat, "status": result["status"], "error": result["error"],
            "elapsed_s": result.get("control_elapsed_s"), "physics_complete": (result.get("final_evaluation") or {}).get("physics_complete"),
            "contact_episodes": score.get("contact_episodes"), "travel_m": score.get("actual_distance_m"),
            "outcome": controller.get("outcome"), "input_tokens": controller.get("input_tokens"),
            "output_tokens": controller.get("output_tokens"), "changed_sources": result["changed_sources"],
            "recording_directory": result.get("recording_directory")}
        manifest["trials"].append(row)
        write(options.output / "experiment.json", manifest)
        print(json.dumps(row, indent=2), flush=True)
        if result["error"] or result["changed_sources"] or controller.get("phase") == "error":
            manifest["blocked"] = "Stopped on infrastructure/controller error or source drift; remaining trials not run"
            break
    manifest["workload_after"] = workload_snapshot()
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write(options.output / "experiment.json", manifest)
    return int(bool(manifest.get("blocked")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check configured profiles and local availability without motion")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/performance" / f"maze-hybrid-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--seconds", type=float, choices=[120., 180., 300.], default=300.)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
