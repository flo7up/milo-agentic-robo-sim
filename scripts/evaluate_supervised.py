import argparse
import asyncio
from collections import Counter
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

from dotenv import load_dotenv

from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
from backend.challenges import PRESETS, get_challenge, shared_apartment
from backend.local_navigation import LocalNavigationClient, ResidentNavigationModel
from backend.navigation_supervisor import POLICY_TASKS
from backend.worker import SimulationWorker
from backend.recording import RunRecorder


ROOT = Path(__file__).resolve().parents[1]


def design_snapshot(label):
    paths = sorted({path for pattern in ("backend/*.py", "scripts/*.py", "scripts/*.mjs", "frontend/src/*.ts", "frontend/src/*.tsx",
                                        "frontend/src/*.css", "frontend/dist/assets/*.js", "frontend/dist/assets/*.css", "assets/textures/*.png")
                    for path in ROOT.glob(pattern)} | {ROOT / name for name in (
                        "backend/replay.html", "assets/milo.urdf", "frontend/index.html", "frontend/dist/index.html", "frontend/package.json", "frontend/package-lock.json",
                        "pyproject.toml", "environment.yml") if (ROOT / name).is_file()})
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    packages = {}
    for name in ("pybullet", "numpy", "openai", "Pillow", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    gpu = None
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        gpu = [dict(zip(("name", "memory_total_mib", "driver"), [value.strip() for value in row])) for row in csv.reader(probe.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"label": label, "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "code_sha256": hashes, "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "gpu": gpu}}


def workload_snapshot():
    gpu = None
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True)
        gpu = [dict(zip(("name", "memory_used_mib", "memory_free_mib", "utilization_pct"), [value.strip() for value in row]))
            for row in csv.reader(probe.stdout.splitlines())]
    except (OSError, subprocess.SubprocessError):
        pass
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
        "scope": "Whole-device instant snapshot; not per-process or continuous workload measurement"}


def performance_rows(manifest, results):
    by_case = {result["case_id"]: result for result in results if "case_id" in result}
    rows = []
    for case in manifest["cases"]:
        result = by_case.get(case["case_id"], {})
        score = result.get("recording_scorecard") or {}
        verified = bool(result.get("physics_success") and result.get("test_passed") is not False and not result.get("manual_placements", 0) and score.get("complete_recording")
                        and not score.get("operator_assisted") and score.get("completion_time_s") is not None)
        rows.append({"experiment_id": manifest["experiment_id"], "design": manifest["design"]["label"],
            "source_sha256": manifest["design"]["source_sha256"], "case_id": case["case_id"], "challenge": case["challenge"],
            "environment": case.get("environment", manifest.get("environment", "standalone")),
            "challenge_sha256": case["challenge_sha256"], "evidence": manifest["evidence"],
            "status": result.get("termination_reason", "missing_report"), "physics_success": result.get("physics_success"),
            "verified_success": verified, "recording_complete": score.get("complete_recording", False),
            "completion_s": score.get("completion_time_s") if verified else None,
            "elapsed_s": result.get("evaluation_elapsed_s"), "distance_m": score.get("actual_distance_m"),
            "contact_episodes": score.get("contact_episodes"), "false_completion_claim": result.get("unverified_completion_claim"),
            "input_tokens": result.get("input_tokens"), "output_tokens": result.get("output_tokens"),
            "inference_median_s": result.get("inference_median_s"), "inference_p95_s": result.get("inference_p95_s"),
            "rendering": result.get("rendering"), "error": result.get("error"),
            "camera_history": manifest.get("camera_history"), "images_sent": result.get("images_sent"),
            "history_sheets_sent": result.get("history_sheets_sent"), "history_originals_sent": result.get("history_originals_sent"),
            "sensing": result.get("sensing"), "circuit_sensing_recoveries": result.get("circuit_sensing_recoveries"),
            "report": f"{case['case_id']}/report.json"})
    return rows


def index_performance(manifest, rows, output, index_path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"experiment_id": manifest["experiment_id"], "started_at": manifest["started_at"],
             "design": manifest["design"]["label"], "source_sha256": manifest["design"]["source_sha256"],
             "evidence": manifest["evidence"], "camera_history": manifest["camera_history"],
             "output": str(output.resolve()), "planned": len(rows), "reported": sum(row["status"] != "missing_report" for row in rows),
             "verified_successes": sum(row["verified_success"] for row in rows)}
    with index_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


def compare_performance(baseline, candidate):
    manifests = [json.loads((directory / "experiment.json").read_text(encoding="utf-8")) for directory in (baseline, candidate)]
    rows = [json.loads((directory / "performance.json").read_text(encoding="utf-8")) for directory in (baseline, candidate)]
    if any(manifest.get("schema_version") != 2 for manifest in manifests):
        raise ValueError("Comparison requires version-2 tracking manifests; older runs need explicitly documented retrospective analysis")
    warnings = []
    if any(manifest.get("source_changed_during_run") for manifest in manifests):
        warnings.append("Source changed during an experiment; no frozen-design improvement claim")
    if any(manifest.get("stage") == "intentions" for manifest in manifests):
        raise ValueError("Intention probes are not physical challenge comparisons")
    for field in ("mode", "stage", "reasoning", "session_timeout_s", "max_turns", "exploration", "continuous_handoff", "skill_composer", "feedback_interval_s",
                  "images_per_request", "context_tokens", "camera_size", "supervisor_deployment", "evidence"):
        if manifests[0].get(field) != manifests[1].get(field):
            warnings.append(f"Mismatched {field}")
    if manifests[0]["design"]["runtime"] != manifests[1]["design"]["runtime"]:
        warnings.append("Mismatched runtime or hardware")
    if any(manifest["design"]["runtime"].get("gpu") is None for manifest in manifests):
        warnings.append("GPU identity not recorded for both experiments")
    expected = [Counter((case["case_id"], case["challenge_sha256"]) for case in manifest["cases"]) for manifest in manifests]
    if expected[0] != expected[1]:
        warnings.append("Mismatched cases, repeats or challenge definitions")
    for manifest, batch in zip(manifests, rows):
        observed = Counter((row["case_id"], row["challenge_sha256"]) for row in batch)
        planned = Counter((case["case_id"], case["challenge_sha256"]) for case in manifest["cases"])
        if observed != planned:
            raise ValueError("Performance rows do not match planned cases")
        if any(row["experiment_id"] != manifest["experiment_id"] or row["source_sha256"] != manifest["design"]["source_sha256"]
               or row["evidence"] != manifest["evidence"] for row in batch):
            raise ValueError("Performance provenance does not match its experiment")
    if manifests[0]["experiment_id"] == manifests[1]["experiment_id"]:
        raise ValueError("Choose two different experiments")
    if any(row["status"] == "missing_report" or not row["recording_complete"] for batch in rows for row in batch):
        warnings.append("Missing or incomplete trial records; no improvement claim")
    if {row["rendering"] for row in rows[0]} != {row["rendering"] for row in rows[1]} or any(row["rendering"] is None for batch in rows for row in batch):
        warnings.append("Mismatched or unknown actual renderer")
    changed = [name for name in sorted(manifests[0]["code_sha256"].keys() | manifests[1]["code_sha256"].keys())
               if manifests[0]["code_sha256"].get(name) != manifests[1]["code_sha256"].get(name)]
    if changed:
        warnings.append("Source files differ; review changed_files before attributing an effect")
    groups = []
    for challenge in sorted({row["challenge"] for batch in rows for row in batch}):
        summaries = []
        for batch in rows:
            cases = [row for row in batch if row["challenge"] == challenge]
            completed = [row["completion_s"] for row in cases if row["verified_success"]]
            tokens = [row["input_tokens"] + row["output_tokens"] for row in cases if row["input_tokens"] is not None and row["output_tokens"] is not None]
            contacts = [row["contact_episodes"] for row in cases if row["contact_episodes"] is not None]
            medians = {}
            for field in ("elapsed_s", "distance_m", "input_tokens", "output_tokens", "inference_median_s", "inference_p95_s",
                          "images_sent", "history_sheets_sent", "history_originals_sent"):
                values = [row[field] for row in cases if row.get(field) is not None]
                medians[field] = statistics.median(values) if values else None
            summaries.append({"attempts": len(cases), "successes": sum(row["verified_success"] for row in cases),
                "success_rate": sum(row["verified_success"] for row in cases) / len(cases) if cases else None,
                "successful_completion_median_s": statistics.median(completed) if completed else None,
                "tokens_median": statistics.median(tokens) if tokens else None, "token_reports": len(tokens),
                "contact_episodes": sum(contacts) if contacts else None, "contact_reports": len(contacts),
                "false_completion_claims": sum(bool(row["false_completion_claim"]) for row in cases),
                "missing_reports": sum(row["status"] == "missing_report" for row in cases), "all_attempt_medians": medians})
        groups.append({"challenge": challenge, "baseline": summaries[0], "candidate": summaries[1]})
    return {"baseline": {"path": str(baseline.resolve()), "design": manifests[0]["design"]["label"], "camera_history": manifests[0]["camera_history"], "evidence": manifests[0]["evidence"]},
        "candidate": {"path": str(candidate.resolve()), "design": manifests[1]["design"]["label"], "camera_history": manifests[1]["camera_history"], "evidence": manifests[1]["evidence"]},
        "warnings": warnings, "changed_files": changed, "per_challenge": groups,
        "interpretation": "Descriptive comparison only. Completion times include successes only, not speed gains when success rates differ. Cloud seed/model identity and concurrent machine load are uncontrolled. No automatic winner."}


def comparison_markdown(report):
    lines = ["# Performance Comparison", "", f"Baseline: {report['baseline']['design']} ({report['baseline']['camera_history']})",
             f"Candidate: {report['candidate']['design']} ({report['candidate']['camera_history']})",
             f"Evidence: {report['baseline']['evidence']} / {report['candidate']['evidence']}", "",
             "| Challenge | Baseline Pass | Candidate Pass | Completion Median A/B (s) | Tokens Median A/B | Contacts A/B |",
             "| --- | --- | --- | --- | --- | --- |"]
    for group in report["per_challenge"]:
        before, after = group["baseline"], group["candidate"]
        def display(value):
            return "unknown" if value is None else str(round(value, 2))
        lines.append(f"| {group['challenge']} | {before['successes']}/{before['attempts']} | {after['successes']}/{after['attempts']} | "
            f"{display(before['successful_completion_median_s'])} / {display(after['successful_completion_median_s'])} | "
            f"{display(before['tokens_median'])} / {display(after['tokens_median'])} | {display(before['contact_episodes'])} / {display(after['contact_episodes'])} |")
    lines.extend(["", report["interpretation"], "", *[f"- {warning}" for warning in report["warnings"]]])
    return "\n".join(lines) + "\n"


def benchmark_cases(challenges, suite=None, case_index=None, environment="standalone"):
    if environment not in {"standalone", "shared_apartment_v1"}:
        raise ValueError("Unknown environment")
    if suite and environment != "standalone":
        raise ValueError("Existing suites require the standalone environment")
    if not suite:
        if case_index is not None:
            raise ValueError("--case-index requires --suite")
        if environment == "shared_apartment_v1":
            return [(f"shared-apartment-v1-{identifier}", shared_apartment(identifier)) for identifier in challenges]
        return [(identifier, get_challenge(identifier)) for identifier in challenges]
    if challenges != ["kitchen_bathroom"]:
        raise ValueError("kitchen-v1 requires --challenges kitchen_bathroom")
    offsets = [(0., 0.), (-.06, 0.), (.06, 0.), (0., -.08), (0., .08),
        (-.04, -.06), (-.04, .06), (.04, -.06), (.04, .06), (.02, .03)]
    cases = []
    for index, (horizontal, vertical) in enumerate(offsets):
        if case_index is not None and index != case_index:
            continue
        challenge = get_challenge("kitchen_bathroom").model_copy(deep=True)
        challenge.initial_xy = [challenge.initial_xy[0] + horizontal, challenge.initial_xy[1] + vertical]
        cases.append((f"kitchen-{index:02d}", challenge))
    return cases


def summarize_benchmark(results, planned_count=None):
    attempts = max(len(results), planned_count if planned_count is not None else len(results))
    completed = [result["recording_scorecard"]["completion_time_s"] for result in results
        if result.get("physics_success") and not result.get("manual_placements", 0)
        and result.get("verification_eligible") is not False
        and result.get("recording_scorecard", {}).get("complete_recording")
        and not result["recording_scorecard"].get("operator_assisted")
        and result["recording_scorecard"].get("completion_time_s") is not None]
    return {"attempts": attempts, "reported_attempts": len(results), "missing_reports": attempts - len(results),
        "successes": len(completed), "success_rate": len(completed) / attempts if attempts else 0.,
        "successful_completion_median_s": statistics.median(completed) if completed else None,
        "successful_completion_p90_s": sorted(completed)[math.ceil(.9 * len(completed)) - 1] if completed else None,
        "failures": attempts - len(completed), "false_completion_claims": sum(result.get("unverified_completion_claim", False) for result in results),
        "note": "Completion times describe successes only; failures remain in the success-rate denominator. Development cases are not held-out generalization tests."}


async def run_reference(worker, challenge, controller, timeout_s):
    from backend.continuous_navigation import ContinuousScan
    from backend.continuous_supervisor import settle
    from scripts.task_navigation import case_for, local_route, teacher_action
    from scripts.navigation_policy import apply
    controller.active = True
    controller.state["phase"] = "acting"
    settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, goal=challenge.goal)
    try:
        async with asyncio.timeout(timeout_s):
            await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch, compact_arms=True))
            _, route = case_for(1)
            for target, _ in local_route(challenge, route):
                await worker.begin_navigation(worker.stop_revision)
                await worker.call(lambda sim: apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Scripted known-map reference"}))
                for _ in range(45):
                    observation, _ = await worker.feedback()
                    linear, angular = teacher_action(observation, target)
                    if linear == 0 and angular == 0:
                        break
                    await worker.call(lambda sim: apply(worker.navigation, sim, "replace_motion_buffer", {"segments": [
                        {"kind": "drive", "linear_mps": linear, "angular_radps": angular, "duration_s": 1.}]}))
                    while worker.latest["navigation"]["remaining_s"] > 0:
                        await asyncio.sleep(.02)
                    if worker.sim.cancel.is_set():
                        raise ValueError(worker.latest["navigation"]["reason"])
                else:
                    raise ValueError("Reference waypoint was not reached")
                await worker.call(lambda sim: worker.navigation.cancel(sim, "Reference waypoint reached"))
            await settle(worker, settings)
            controller.state.update(phase="completed", message="Scripted reference finished; not learned autonomy")
    except TimeoutError:
        controller.state.update(phase="completed", message="Reference time budget exhausted")
    except ValueError as error:
        controller.state.update(phase="error", error=str(error))
    finally:
        worker.stop()
        await worker.hold_stopped()
        controller.active = False


def execution_metrics(events):
    latencies = sorted(event["payload"]["latency_s"] for event in events
        if event["kind"] == "response" and event["payload"].get("latency_s") is not None)
    results = [event["payload"]["result"] for event in events
        if event["kind"] == "result" and "result" in event["payload"]]
    continuous = [event["payload"] for event in events if event["title"] == "Continuous execution feedback" and "path_m" in event["payload"]]
    timings = Counter()
    for event in events:
        if event["title"] == "Local action timing":
            timings[event["payload"]["action"]] += event["payload"]["wall_s"]
    return {
        "preparation_wall_s": sum(event["payload"].get("wall_s", 0.) for event in events if event["title"] == "Exploration preparation"),
        "inference_wall_s": sum(latencies), "local_action_wall_s": dict(timings),
        "recovery_interventions": sum(event["title"] == "No-progress recovery" and event["payload"].get("replacement") is not None for event in events),
        "recovery_exhausted": any(event["title"] == "No-progress recovery" and event["payload"].get("replacement") is None for event in events),
        "circuit_sensing_recoveries": sum(event["title"] == "Circuit sensing recovery" for event in events),
        "timing_note": "Action wall time includes execution and settling; inference and motion may overlap with handoffs. Do not sum these as disjoint phases.",
        "tool_requests": dict(Counter(event["payload"]["tool"] for event in events if event["kind"] == "tool")),
        "tool_errors": dict(Counter(result["error"] for result in results if result.get("error"))),
        "inference_responses": len(latencies),
        "inference_median_s": statistics.median(latencies) if latencies else None,
        "inference_p95_s": latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
        "discarded_navigation_requests": sum(event["title"] == "Navigation feedback refreshed" for event in events),
        "images_sent": sum(event["payload"].get("images_in_request", 0) for event in events if event["kind"] == "feedback"),
        "history_sheets_sent": sum(bool(event["payload"].get("camera_history", {}).get("frames")) for event in events if event["kind"] == "feedback"),
        "history_originals_sent": sum(bool(event["payload"].get("historical_original")) for event in events if event["kind"] == "feedback"),
        "continuous_routes": len(continuous),
        "continuous_arrivals": sum(result["status"] == "arrived" for result in continuous),
        "continuous_buffer_stops": sum(result["buffer_stops"] for result in continuous),
        "continuous_maximum_update_gap_s": max((result["maximum_update_gap_s"] for result in continuous), default=None),
    }


def evaluation_budget(options):
    time_limit = getattr(options, "time_limit_s", None)
    session_limit = getattr(options, "session_limit_s", None)
    turns = getattr(options, "turns", None)
    if time_limit is not None and session_limit is not None:
        raise ValueError("Choose either a time-only budget or a session ceiling for request-limited tests")
    if time_limit is not None and turns is not None:
        raise ValueError("Choose either --turns or --time-limit-s")
    if turns is None:
        return NavigationEvaluationBudget(timeout_s=time_limit if time_limit is not None else session_limit if session_limit is not None else 1800)
    return NavigationEvaluationBudget(max_turns=turns,
        timeout_s=time_limit if time_limit is not None else session_limit if session_limit is not None else
            1200 if getattr(options, "mode", "luna_navigation") == "luna_navigation" else 600)


async def prepare_home_map(worker, options):
    from backend.home_mission import HomeRequest
    map_id = getattr(options, "home_map_id", None)
    if not map_id:
        return None
    started = time.monotonic()
    identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
    await worker.home_command(HomeRequest(**identity, action="load_map", map_id=map_id))
    localized = await worker.home_command(HomeRequest(**identity, action="localize", map_id=map_id,
        place_id=getattr(options, "localization_place_id", None), pose_m_rad=getattr(options, "localization_pose", None)))
    worker.home_mission.allow_expansion = False
    document = worker.home_mission.home.document()
    return {"map_id": map_id, "revision": document["revision"], "environment_id": document["environment_id"],
        "sha256": hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest(),
        "localization": localized["localization"], "setup_wall_s": time.monotonic() - started,
        "seed_source": "operator_supplied" if getattr(options, "localization_place_id", None) or getattr(options, "localization_pose", None) else "scan_only",
        "geometry_updates": False}


async def evaluate(options):
    if getattr(options, "compare", None):
        report = compare_performance(*options.compare)
        options.output.mkdir(parents=True, exist_ok=False)
        (options.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (options.output / "comparison.md").write_text(comparison_markdown(report), encoding="utf-8")
        return report
    mode = getattr(options, "mode", "luna_navigation")
    if getattr(options, "home_map_id", None) and (mode != "luna_continuous" or options.stage != "challenges"):
        raise ValueError("Saved-home evaluation requires luna_continuous challenge mode")
    budget = evaluation_budget(options)
    environment = getattr(options, "environment", "standalone")
    cases = benchmark_cases(options.challenges, getattr(options, "suite", None), getattr(options, "case_index", None), environment)
    repeats = getattr(options, "repeats", 1)
    if repeats > 1:
        cases = [(f"{name}-repeat-{repeat:02d}", challenge.model_copy(deep=True))
            for repeat in range(repeats) for name, challenge in cases]
    if mode == "reference" and (options.challenges != ["kitchen_bathroom"] or options.stage != "challenges"):
        raise ValueError("The scripted reference supports the kitchen challenge only")
    needs_local = options.stage == "intentions" or mode == "luna_navigation"
    if needs_local and options.checkpoint is None:
        raise ValueError("Local policy evaluation requires --checkpoint")
    options.output.mkdir(parents=True, exist_ok=False)
    if mode != "reference":
        load_dotenv(Path("backend/.env"), override=False)
        load_dotenv(Path(".env"), override=False)
    config = FoundryConfig() if mode == "reference" else FoundryConfig.from_environment()
    luna = next((profile for profile in config.models if profile.id == "luna" and profile.provider == "foundry"), None)
    design = design_snapshot(getattr(options, "design", "unlabelled"))
    manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
        "design": design, "evidence": getattr(options, "evidence", "real_model") if mode != "reference" else "scripted_reference",
        "mode": mode, "stage": options.stage, "challenges": options.challenges, "environment": environment,
        "exploration": getattr(options, "exploration", "adaptive"), "repeats": repeats,
        "suite": getattr(options, "suite", None), "cases": [{"case_id": name, "challenge": challenge.id, "environment": challenge.environment, "initial_xy": challenge.initial_xy,
            "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()} for name, challenge in cases],
        "continuous_handoff": getattr(options, "continuous_handoff", False),
        "skill_composer": getattr(options, "skill_composer", False),
        "home_map": {"map_id": getattr(options, "home_map_id", None),
            "localization_place_id": getattr(options, "localization_place_id", None),
            "localization_pose": getattr(options, "localization_pose", None)},
        "camera_history": getattr(options, "camera_history", "enabled") if mode == "luna_continuous" else "not_applicable",
        "max_turns": budget.max_turns, "reasoning": options.reasoning, "feedback_interval_s": .25,
        "images_per_request": getattr(options, "images", 2), "context_tokens": getattr(options, "context_tokens", 8192),
        "camera_size": [320, 240], "session_timeout_s": budget.timeout_s,
        "budget_kind": "time_only" if budget.max_turns is None else "requests_with_time_ceiling",
        "budget_clock": "monotonic wall time from controller start; model preload and scene initialization excluded",
        "supervisor_deployment": luna.deployment if luna else None, "cloud_seed_controlled": False,
        "workload_before": workload_snapshot(),
        "code_sha256": design["code_sha256"]}
    if mode != "reference" and luna is not None:
        from backend.experiment_variants import variant_snapshot
        settings = AgentStart(run_id="variant-snapshot", episode_epoch=0, goal="Variant snapshot", execution_mode=mode,
            model_id="luna", reasoning=options.reasoning, continuous_handoff=manifest["continuous_handoff"],
            skill_composer=manifest["skill_composer"],
            adaptive_navigation=manifest["exploration"] == "adaptive", max_turns=min(budget.max_turns or 80, 200),
            feedback_interval_s=.25, images_per_request=manifest["images_per_request"], context_tokens=manifest["context_tokens"])
        effective = {"max_turns": budget.max_turns, "session_timeout_s": budget.timeout_s, "camera_history": manifest["camera_history"]}
        if needs_local:
            effective["local_checkpoint"] = str(options.checkpoint)
        variant = variant_snapshot(settings, luna, effective_configuration=effective)
        manifest.update({key: variant[key] for key in ("architecture", "model_variant", "variant_id")})
    (options.output / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    class Candidate(LocalNavigationClient):
        def __init__(self):
            super().__init__(options.checkpoint)

    resident = ResidentNavigationModel(Candidate)
    results = []
    try:
        for case_id, challenge in cases:
            challenge_id = challenge.id
            directory = options.output / case_id
            directory.mkdir()
            worker = SimulationWorker(challenge=challenge, pace=True)
            controller = AgentController(config, local_navigation_factory=resident, evaluation_budget=budget)
            controller.evaluation_camera_history = manifest["camera_history"] != "disabled"
            began = time.monotonic()
            phases = []
            sampled = -1
            recorder = None
            try:
                await asyncio.wrap_future(worker.ready)
                await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
                home_map = await prepare_home_map(worker, options)
                if home_map:
                    (directory / "home-map.json").write_text(json.dumps(home_map, indent=2), encoding="utf-8")
                if options.stage == "intentions":
                    session = resident()
                    await session.start()
                    observation, image = await worker.feedback()
                    decisions = []
                    for name, task in POLICY_TASKS.items():
                        session.instruction = task
                        for sample in range(3):
                            reply = await session.predict(observation, image)
                            decisions.append({"intention": name, "sample": sample, **reply})
                    await session.close()
                    result = {"challenge": challenge_id, "predictions": decisions, "motion_executed": False}
                    (directory / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    continue
                preload_started = time.monotonic()
                if needs_local:
                    session = resident()
                    try:
                        await session.start()
                    finally:
                        await session.close()
                preload_s = time.monotonic() - preload_started
                control_started = time.monotonic()
                recorder = RunRecorder(directory / "recording", context=lambda: controller.state)
                await worker.call(lambda sim: (setattr(worker, "recorder", recorder), recorder.capture(worker)))
                if mode == "reference":
                    controller.active = True
                    controller.task = asyncio.create_task(run_reference(worker, challenge, controller, budget.timeout_s))
                else:
                    controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                        goal=challenge.goal, execution_mode=mode, model_id="luna",
                        continuous_handoff=getattr(options, "continuous_handoff", False),
                        skill_composer=manifest["skill_composer"],
                        adaptive_navigation=manifest["exploration"] == "adaptive",
                        reasoning=options.reasoning, max_turns=min(budget.max_turns or 80, 200), feedback_interval_s=.25,
                        images_per_request=manifest["images_per_request"], context_tokens=manifest["context_tokens"]))
                while controller.active:
                    await worker.call(lambda sim: recorder.capture(worker))
                    await asyncio.to_thread(recorder.flush)
                    trace = controller.trace()
                    if trace["revision"] != sampled:
                        sampled = trace["revision"]
                        new = [entry for entry in trace["events"] if entry["id"] > (phases[-1]["id"] if phases else 0)]
                        phases.extend(new)
                        for event in new:
                            if event["title"] in {"Luna navigation decision", "Continuous goal selected", "Continuous execution feedback"}:
                                print(json.dumps({"challenge": challenge_id, "turn": event["turn"], **event["payload"]}), flush=True)
                            elif event["kind"] == "result":
                                print(json.dumps({"challenge": challenge_id, "mode": mode, "turn": event["turn"],
                                    "tool": event["payload"].get("tool"),
                                    "status": event["payload"].get("result", {}).get("status"),
                                    "error": event["payload"].get("result", {}).get("error")}), flush=True)
                            if event["id"] in controller.trace_images:
                                (directory / f"frame-{event['id']:04d}.png").write_bytes(controller.trace_images[event["id"]])
                            for index, image in enumerate(controller.trace_image_batches.get(event["id"], [])):
                                (directory / f"frame-{event['id']:04d}-{index}.png").write_bytes(image)
                        (directory / "trace.json").write_text(json.dumps(phases, indent=2), encoding="utf-8")
                    await asyncio.sleep(.2)
                await controller.task
                phases.extend([entry for entry in controller.trace()["events"] if entry["id"] > (phases[-1]["id"] if phases else 0)])
                (directory / "trace.json").write_text(json.dumps(phases, indent=2), encoding="utf-8")
                observation, image = await worker.feedback()
                (directory / "terminal.png").write_bytes(image)
                current = worker.latest
                local_state = controller.state.get("local_model") or {}
                outcome = controller.state.get("outcome") or {}
                sensing = await worker.call(lambda sim: {"timing": dict(worker.spatial_timing),
                    "maximum_processing_age_s": worker.spatial_processing_max_s, "scope": "Worker lifetime, including preparation"})
                termination = ("time_limit" if "Evaluation time budget exhausted" in outcome.get("message", "") else
                    "request_limit" if outcome.get("kind") == "limited" else
                    "error" if controller.state["error"] else outcome.get("kind", controller.state["phase"]))
                result = {"challenge": challenge_id, "environment": challenge.environment, "case_id": case_id, "execution_mode": mode,
                    "experiment_id": manifest["experiment_id"], "design": design["label"], "source_sha256": design["source_sha256"],
                    "evidence": manifest["evidence"], "rendering": worker.sim.rendering, "camera_history": manifest["camera_history"],
                    "home_map": home_map,
                    "checkpoint": str(options.checkpoint) if needs_local else None,
                    "physics_success": current["challenge"]["status"] == "completed", "challenge_status": current["challenge"],
                    "completion_source": outcome.get("source"),
                    "unverified_completion_claim": outcome.get("source") == "agent" and outcome.get("kind") == "completed"
                        and current["challenge"]["status"] != "completed",
                    "phase": controller.state["phase"], "error": controller.state["error"], "message": controller.state["message"],
                    "outcome": controller.state.get("outcome"),
                    "supervisor_turns": controller.state["turns"], "model_commands": local_state.get("requests_completed"),
                    "exploration": manifest["exploration"], "final_progress": controller.navigation_memory.progress(),
                    "wall_s": time.monotonic() - began, "simulated_s": current["snapshot"]["simulated_time_s"],
                    "evaluation_elapsed_s": time.monotonic() - control_started, "preload_s": preload_s,
                    "budget": budget.model_dump(), "termination_reason": termination,
                    "sensing": sensing,
                    "input_tokens": controller.state["input_tokens"], "output_tokens": controller.state["output_tokens"],
                    "manual_placements": current["manual_placements"], "real_luna": mode != "reference" and manifest["evidence"] == "real_model",
                    "real_smolvla": needs_local and manifest["evidence"] == "real_model",
                    "supervisor_deployment": luna.deployment if luna else None,
                    "underlying_supervisor_identity_verified": False,
                    "sensor_input_only": mode != "reference", "scripted_known_map_reference": mode == "reference", "final_odometry": observation.odometry_m_rad,
                    **execution_metrics(phases)}
                await worker.call(lambda sim: (recorder.capture(worker), setattr(worker, "recorder", None)))
                result["recording_scorecard"] = await asyncio.to_thread(recorder.finish, {"challenge": challenge_id,
                    "case_id": case_id, "initial_xy": challenge.initial_xy,
                    "label": f"{case_id} / {mode}", "real_model": result["real_luna"], "evidence": manifest["evidence"], "execution_mode": mode,
                    "budget": budget.model_dump(), "code_sha256": manifest["code_sha256"]})
                recorder = None
                (directory / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                results.append(result)
                (options.output / "performance.json").write_text(json.dumps(performance_rows(manifest, results), indent=2), encoding="utf-8")
                print(json.dumps(result), flush=True)
            finally:
                await controller.halt()
                if recorder is not None:
                    await worker.call(lambda sim: (recorder.capture(worker), setattr(worker, "recorder", None)))
                    await asyncio.to_thread(recorder.finish, {"challenge": challenge_id, "real_model": mode != "reference" and manifest["evidence"] == "real_model",
                        "evidence": manifest["evidence"], "interrupted": True})
                await worker.close()
    finally:
        await resident.close()
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["workload_after"] = workload_snapshot()
        manifest["source_changed_during_run"] = any(not (ROOT / name).is_file()
            or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in manifest["code_sha256"].items())
        (options.output / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (options.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (options.output / "benchmark.json").write_text(json.dumps(summarize_benchmark(results, len(cases)), indent=2), encoding="utf-8")
        if options.stage == "challenges":
            rows = performance_rows(manifest, results)
            (options.output / "performance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            index_performance(manifest, rows, options.output, getattr(options, "performance_index", ROOT / ".runtime/performance/index.jsonl"))


def argument_parser():
    parser = argparse.ArgumentParser(description="Isolated real Luna control-mode evaluation; no live browser reset")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--design", default="unlabelled", help="Human-readable design label; exact source hashes are also recorded")
    parser.add_argument("--camera-history", choices=["enabled", "disabled"], default="enabled", help="Continuous-mode input ablation only; not a recreation of historical code")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BASELINE", "CANDIDATE"), help="Compare tracked experiment directories offline; makes no model calls")
    parser.add_argument("--performance-index", type=Path, default=ROOT / ".runtime/performance/index.jsonl", help="Append one reference per completed/interrupted experiment; run evaluations serially")
    parser.add_argument("--mode", choices=["single_step", "navigation_plan", "luna_navigation", "luna_continuous", "reference"], default="luna_navigation")
    parser.add_argument("--suite", choices=["kitchen-v1"])
    parser.add_argument("--environment", choices=["standalone", "shared_apartment_v1"], default="standalone")
    parser.add_argument("--home-map-id", help="Reuse a saved sensor-built map; never starts a new mapping phase")
    localization = parser.add_mutually_exclusive_group()
    localization.add_argument("--localization-place-id", help="Optional operator-supplied initial place hint, checked by scan matching")
    localization.add_argument("--localization-pose", type=float, nargs=3, metavar=("X_M", "Y_M", "YAW_RAD"), help="Optional approximate map-frame pose; never copied from simulator state")
    parser.add_argument("--exploration", choices=["legacy", "adaptive"], default="adaptive")
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=1)
    parser.add_argument("--continuous-handoff", action="store_true", help="Opt-in bounded moving proposals for the continuous controller")
    parser.add_argument("--skill-composer", action="store_true", help="Opt-in bounded cruise/curve/approach/inspect composition")
    parser.add_argument("--case-index", type=int, choices=range(10))
    parser.add_argument("--stage", choices=["intentions", "challenges"], default="challenges")
    parser.add_argument("--challenges", nargs="+", choices=sorted(PRESETS), default=["park", "recharge"])
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--turns", type=int, choices=range(1, 1001), metavar="TURNS", help="Explicit request-limited comparison (up to 1,000 turns)")
    limits.add_argument("--time-limit-s", type=float, metavar="SECONDS", help="Wall-clock-only budget; default 1,800 seconds when --turns is omitted")
    parser.add_argument("--session-limit-s", type=float, metavar="SECONDS", help="Additional time ceiling for request-limited trials")
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--images", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--context-tokens", type=int, choices=range(32769), default=8192, metavar="TOKENS")
    return parser


if __name__ == "__main__":
    asyncio.run(evaluate(argument_parser().parse_args()))