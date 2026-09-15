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
from backend.home_mapping import HomeMap
from backend.home_mission import DEFAULT_MAP_PATH, HomeMission
from backend.navigation import NavigationRuntime
from backend.recording import RunRecorder
from backend.simulation import BulletSimulation
from backend.worker import SimulationWorker
from scripts.benchmark_household import benchmark_map, execute_attempt, planned_cases, read_map, suite_definition
from scripts.evaluate_supervised import design_snapshot


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
        if case_id == "survey":
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
    parser.add_argument("--case", default="navigate-r1", choices=["survey", *[case["case_id"] for case in planned_cases(suite_definition())]])
    arguments = parser.parse_args()
    asyncio.run(probe(arguments.output, arguments.case))