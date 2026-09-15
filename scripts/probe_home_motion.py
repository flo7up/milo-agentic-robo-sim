import argparse
import asyncio
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import time
from types import SimpleNamespace

from backend.camera import EnhancedRenderer, EnhancedResources
from backend.continuous_navigation import ContinuousNavigation
from backend.home_mapping import HomeMap, MapStore
from backend.home_mission import DEFAULT_MAP_PATH, HomeMission, HomeRequest
from backend.navigation import NavigationRuntime
from backend.recording import RunRecorder
from backend.simulation import BulletSimulation
from backend.worker import SimulationWorker
from scripts.benchmark_household import benchmark_map, changed_sources, digest, execute_attempt, planned_cases, read_map, suite_definition
from scripts.evaluate_supervised import design_snapshot


async def fresh_exploration(output, resources, hashes):
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan

    output.mkdir(parents=True, exist_ok=False)
    original_map = digest(read_map(DEFAULT_MAP_PATH))
    worker = SimulationWorker(challenge=get_challenge("flat_kitchen"), rendering="enhanced", render_resources=resources)
    recorder = None
    started = None
    final_task = None
    state = {}
    initial_free = 0.
    status = "failed"
    error = None
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch, compact_arms=True))
        recorder = RunRecorder(output / "recording", context=lambda: {"phase": "scripted_fresh_exploration_diagnostic"})
        def attach(sim):
            worker.home_mission = HomeMission(worker, MapStore(output / "maps.sqlite3"))
            worker.recorder = recorder
            recorder.capture(worker)
        await worker.call(attach)
        started = time.monotonic()
        await worker.home_command(HomeRequest(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            action="start_exploration", time_budget=30.))
        initial_free = (await worker.home_state(compact=True))["coverage"]["free_m2"]
        async with asyncio.timeout(40.):
            while worker.home_mission.active:
                await asyncio.sleep(1.)
                await worker.home_state(compact=True)
                await asyncio.to_thread(recorder.flush)
        final_task = dict(worker.home_mission.task)
        status = final_task["status"]
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        worker.stop()
        if worker.ready.done() and worker.ready.exception() is None and worker.thread.is_alive():
            await worker.hold_stopped()
            state = await worker.home_state()
            (output / "final-state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            if recorder:
                await worker.call(lambda sim: recorder.capture(worker))
        score = await asyncio.to_thread(recorder.finish, {"evidence": "scripted_test", "real_model": False}) if recorder else {}
        await worker.close()
    metrics = {"controller_task": final_task, "error": error, "elapsed_s": time.monotonic() - started if started else None,
        "travel_m": score.get("actual_distance_m"), "contacts": score.get("contact_episodes"),
        "new_free_m2": (state.get("coverage") or {}).get("free_m2", initial_free) - initial_free,
        "recording_complete": score.get("complete_recording"), "source_unchanged": not changed_sources(hashes),
        "original_map_unchanged": digest(read_map(DEFAULT_MAP_PATH)) == original_map, "model_calls": 0}
    result = {"status": status, "metrics": metrics, "recording_scorecard": score,
        "evidence": "scripted_test", "note": "Fresh standalone-map diagnostic, not a replacement household baseline"}
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


async def probe(output, case_id):
    output.mkdir(parents=True, exist_ok=False)
    timings = defaultdict(list)
    events = []
    patched = []
    started = time.monotonic()

    def instrument(owner, name):
        original = getattr(owner, name)
        key = owner.__name__ + "." + name
        @wraps(original)
        def measured(instance, *args, **kwargs):
            began = time.perf_counter()
            worker = instance if isinstance(instance, SimulationWorker) else getattr(instance, "worker", None)
            navigation = getattr(worker, "navigation", None)
            before = None if navigation is None else {"status": navigation.status,
                "lease_s": navigation.expires_at - time.monotonic(), "buffered": bool(navigation.buffer)}
            try:
                return original(instance, *args, **kwargs)
            finally:
                elapsed = time.perf_counter() - began
                timings[key].append(elapsed)
                if (elapsed >= .015 or key in {"ContinuousNavigation.finish", "SimulationWorker._update_continuous", "SimulationWorker._sample_spatial", "SimulationWorker._receive_mapped_capture"}) and len(events) < 10000:
                    after = None if navigation is None else {"status": navigation.status, "reason": navigation.reason,
                        "lease_s": navigation.expires_at - time.monotonic(), "buffered": bool(navigation.buffer)}
                    events.append({"wall_s": time.monotonic() - started, "operation": key, "duration_s": elapsed,
                        "before": before, "after": after, "finish": str(args[2:]) if key == "ContinuousNavigation.finish" else None,
                        "sensor": None if worker is None else {"age_s": None if worker.spatial_map is None or worker.spatial_map.captured_at is None else time.monotonic() - worker.spatial_map.captured_at,
                            "pending": worker.spatial_pending is not None, "ready": worker.spatial_pending.done() if worker.spatial_pending else None,
                            "raw_depth_age_s": time.monotonic() - worker.mapped_depth_sensor.captured_at if worker.mapped_depth_sensor else None,
                            "capture_pending": worker.spatial_capture_pending is not None,
                            "capture_ready": worker.spatial_capture_pending.done() if worker.spatial_capture_pending else None,
                            "request_age_s": time.monotonic() - worker.spatial_sampled_at,
                            "sequence": worker.spatial_sequence,
                            "error": worker.spatial_error, "timing": dict(worker.spatial_timing)}})
        setattr(owner, name, measured)
        patched.append((owner, name, original))

    for owner, names in ((HomeMission, ("sample", "observe_depth", "path_valid", "state", "tick")),
            (HomeMap, ("observe", "allowed", "scan_quality", "route")),
            (SimulationWorker, ("_sample_spatial", "_receive_spatial", "_receive_mapped_capture", "_update_continuous", "_publish_navigation", "_publish", "_feedback")),
            (BulletSimulation, ("capture_spatial", "robot_footprint")),
            (EnhancedRenderer, ("capture",)), (NavigationRuntime, ("tick", "apply", "check_clearance")), (ContinuousNavigation, ("update", "finish")),
            (RunRecorder, ("capture", "capture_home"))):
        for name in names:
            instrument(owner, name)
    resources = None
    try:
        design = design_snapshot("mapped-motion-timing-probe-v1")
        resources = await asyncio.to_thread(EnhancedResources)
        if case_id == "fresh-exploration":
            case = {"id": case_id, "title": "Fresh standalone-map exploration", "budget_s": 30.}
            result = await fresh_exploration(output / case_id, resources, design["code_sha256"])
        elif case_id == "survey":
            from scripts.benchmark_rooms import attempt
            case = {"id": "survey", "title": "Instrumented sensor-only survey", "budget_s": 180.}
            result = await attempt(SimpleNamespace(map_store=DEFAULT_MAP_PATH, original_map=DEFAULT_MAP_PATH),
                output / case_id, case, resources, design["code_sha256"])
        else:
            case = next(case for case in planned_cases(suite_definition()) if case["case_id"] == case_id)
            document = benchmark_map(read_map(DEFAULT_MAP_PATH))
            result = await execute_attempt(case, output / case_id, document, resources, design["code_sha256"], DEFAULT_MAP_PATH)
        summary = {}
        for name, values in timings.items():
            ordered = sorted(values)
            summary[name] = {"calls": len(values), "total_s": sum(values), "max_s": max(values),
                "p95_s": ordered[min(len(values) - 1, int(.95 * len(values)))]}
        report = {"design": design, "case": case, "result": result,
            "timings": summary, "events": events,
            "evidence": "instrumented_scripted_physics_diagnostic_not_baseline",
            "note": "Inclusive wall times overlap. Timing wrappers add overhead. No budgets, motion settings, map inputs or criteria changed."}
        (output / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "metrics": result["metrics"],
            "timings": summary, "artifact": str(output / "timing.json")}, indent=2))
    finally:
        if resources:
            await asyncio.to_thread(resources.close)
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="navigate-r1", choices=["survey", "fresh-exploration", *[case["case_id"] for case in planned_cases(suite_definition())]])
    arguments = parser.parse_args()
    asyncio.run(probe(arguments.output, arguments.case))