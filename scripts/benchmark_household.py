import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import time
from uuid import uuid4


SUITE_ID = "household-foundation-v1"
MAP_ID = "6d764d3e-8eeb-4810-b575-110fc115f0ce"
MAP_SHA256 = "b9e33c46d2062f5524994b48ead331226b7e47892b25d6e65401a4a6246b902d"
HOME_ID = "52bc9051-ae0a-4d35-910a-3cfb999cbb7b"
CHECKPOINT_ID = "benchmark-observed-checkpoint-v1"
START_OFFSETS_M = ([0., 0.], [.1, 0.], [-.1, 0.])
TASKS = (
    {"id": "localize", "title": "Localize after restart", "budget_s": 20.,
        "criteria": "Scan localization within 0.15 m and 0.15 rad of evaluator-only ground truth; no wheel travel over 0.03 m"},
    {"id": "navigate", "title": "Navigate to observed checkpoint", "budget_s": 60.,
        "criteria": "Controller reports arrival; independent endpoint error <=0.15 m with a recorded 0.5 simulated-second stable dwell"},
    {"id": "return_home", "title": "Checkpoint then return Home", "budget_s": 90.,
        "criteria": "Both checkpoint and Home independently reached and held; entire two-leg task within budget"},
    {"id": "blocked_route", "title": "Reject an occupied destination", "budget_s": 20.,
        "criteria": "Fresh sensors reject the fixed obstructed checkpoint as unreachable, <=0.05 m travel and no contact; not closed-door traversal"},
    {"id": "explore", "title": "Expand a partial map", "budget_s": 30.,
        "criteria": "At least 1 m2 newly observed free area and 0.5 m actual travel within budget, followed by stopped buffers; original map unchanged"},
    {"id": "cancel", "title": "Stop during motion and reject late commands", "budget_s": 30.,
        "criteria": "Stop triggered after >=0.10 m actual travel, acknowledgement <=0.5 wall seconds, frozen ticks for 0.75 s after acknowledgement, stale motion rejected after Resume"},
    {"id": "room_to_room", "title": "Navigate between named rooms", "budget_s": 90.,
        "criteria": "Two saved room annotations >=2 m apart, independently verified arrival/dwell at both; missing room annotations are blocked prerequisites"},
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def suite_definition():
    return {"suite_id": SUITE_ID, "version": 1, "environment": "shared_apartment_v1", "scenario": "flat_kitchen",
        "evidence": "scripted_test", "controller": "builtin_mapped_navigation", "renderer": "enhanced",
        "repetitions": 3, "start_offsets_m": list(START_OFFSETS_M), "map_id": MAP_ID,
        "map_document_sha256": MAP_SHA256, "home_place_id": HOME_ID, "checkpoint_id": CHECKPOINT_ID,
        "checkpoint_map_pose_m_rad": [.7, 0., 0.], "localization_hint": "saved Home pose, scan validated",
        "setup_budget_s": 60., "tasks": list(TASKS),
        "common_pass_requirements": {"recording_complete": True, "source_unchanged": True,
            "original_map_unchanged": True, "no_manual_relocation": True, "contact_episodes": 0},
        "qualification": "Development baseline; three fixed starting offsets are not a reliability estimate or held-out layout study",
        "model_inference": False}


def planned_cases(definition):
    return [{"case_id": f"{task['id']}-r{repeat + 1}", "task_id": task["id"], "title": task["title"],
        "repeat": repeat + 1, "start_offset_m": list(offset), "budget_s": task["budget_s"], "criteria": task["criteria"],
        "challenge": definition["scenario"], "environment": definition["environment"]}
        for repeat, offset in enumerate(definition["start_offsets_m"]) for task in definition["tasks"]]


def score_attempt(task_id, metrics):
    if metrics.get("blocked_prerequisite"):
        return {"status": "blocked", "passed": False, "reason": metrics["blocked_prerequisite"]}
    invalid = [name for name in ("recording_complete", "source_unchanged", "original_map_unchanged") if metrics.get(name) is not True]
    if invalid or metrics.get("infrastructure_error"):
        return {"status": "invalid", "passed": False, "reason": ", ".join(invalid) or metrics["infrastructure_error"]}
    safe = metrics.get("contact_episodes") == 0 and metrics.get("manual_placements") == 0
    timed = metrics.get("elapsed_s", math.inf) <= metrics.get("budget_s", 0.)
    arrivals = metrics.get("arrivals", [])
    valid_arrival = lambda arrival: (arrival.get("controller_status") == "completed"
        and arrival.get("position_error_m", math.inf) <= .15 and arrival.get("stable_dwell_sim_s", 0.) >= .5)
    if task_id == "localize":
        achieved = metrics.get("localized") is True and metrics.get("position_error_m", math.inf) <= .15 and metrics.get("heading_error_rad", math.inf) <= .15 and metrics.get("travel_m", math.inf) <= .03
    elif task_id in {"navigate", "return_home", "room_to_room"}:
        required = 1 if task_id == "navigate" else 2
        achieved = len(arrivals) == required and all(valid_arrival(arrival) for arrival in arrivals)
    elif task_id == "blocked_route":
        achieved = metrics.get("unreachable_rejected") is True and metrics.get("travel_m", math.inf) <= .05
    elif task_id == "explore":
        achieved = metrics.get("new_free_m2", 0.) >= 1. and metrics.get("travel_m", 0.) >= .5 and metrics.get("buffers_empty") is True
        timed = metrics.get("elapsed_s", math.inf) <= metrics.get("budget_s", 0.) + 1.
    elif task_id == "cancel":
        achieved = (metrics.get("stop_triggered") is True and metrics.get("travel_before_stop_m", 0.) >= .1
            and metrics.get("stop_ack_s", math.inf) <= .5 and metrics.get("frozen_after_stop") is True
            and metrics.get("late_command_rejected") is True and metrics.get("no_resume_motion") is True)
    else:
        raise ValueError("Unknown benchmark task")
    passed = bool(safe and timed and achieved)
    return {"status": "passed" if passed else "failed", "passed": passed,
        "reason": "All fixed checks passed" if passed else metrics.get("error") or "Fixed capability, time or safety criteria not met"}


def summarize(definition, results):
    planned = planned_cases(definition)
    by_case = {result["case_id"]: result for result in results}
    if len(by_case) != len(results) or any(identity not in {case["case_id"] for case in planned} for identity in by_case):
        raise ValueError("Duplicate or unknown benchmark attempt")
    tasks = []
    for task in definition["tasks"]:
        entries = [by_case.get(case["case_id"], {"status": "not_run", "passed": False}) for case in planned if case["task_id"] == task["id"]]
        counts = Counter(entry["status"] for entry in entries)
        successful_times = [entry["metrics"]["elapsed_s"] for entry in entries if entry.get("passed") and "metrics" in entry]
        tasks.append({"task_id": task["id"], "title": task["title"], "planned": len(entries),
            "passed": counts["passed"], "failed": counts["failed"], "blocked": counts["blocked"],
            "invalid": counts["invalid"], "not_run": counts["not_run"],
            "successful_median_s": statistics.median(successful_times) if successful_times else None})
    return {"suite_id": definition["suite_id"], "suite_sha256": digest(definition), "planned": len(planned),
        "reported": len(results), "tasks": tasks, "evidence": definition["evidence"],
        "aggregate_success_rate": None, "note": "Capability tasks are heterogeneous; report per-task denominators, not a single autonomy score"}


def argument_parser():
    parser = argparse.ArgumentParser(description="Versioned household navigation benchmark; real physics, no model calls")
    parser.add_argument("--stage", choices=["plan", "preflight", "run", "compare"], default="plan")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--map-store", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    return parser


def read_map(path):
    with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT document FROM maps WHERE map_id=?", (MAP_ID,)).fetchone()
    if row is None:
        raise ValueError("The suite's frozen source map is unavailable")
    data = json.loads(row[0])
    if digest(data) != MAP_SHA256:
        raise ValueError("Source map changed; use the preserved v1 map or define a new benchmark version")
    return data


def benchmark_map(document):
    from backend.home_mapping import HomeMap
    home = HomeMap.restore(document)
    if not any(place["place_id"] == HOME_ID for place in home.places):
        raise ValueError("Saved Home annotation is required")
    home.route([0., 0.], [.7, 0.], .4)
    derived = deepcopy(document)
    derived["places"].append({"place_id": CHECKPOINT_ID, "name": "Observed checkpoint 0.7 m", "kind": "destination",
        "pose_m_rad": [.7, 0., 0.], "map_id": MAP_ID, "frame": "map",
        "source": "fixed_benchmark_annotation_in_saved_observed_free_space"})
    return derived


def write_map_copy(path, document):
    from backend.home_mapping import MapStore
    store = MapStore(path)
    with store.connect() as connection:
        connection.execute("INSERT INTO maps VALUES (?, ?, ?, ?)",
            (document["map_id"], document["environment_id"], document["revision"], json.dumps(document, allow_nan=False)))
    return store


def fixture(case):
    from backend.challenges import shared_apartment
    challenge = shared_apartment("flat_kitchen")
    origin = list(challenge.initial_xy)
    challenge.initial_xy = [value + offset for value, offset in zip(origin, case["start_offset_m"])]
    if case["task_id"] == "blocked_route":
        challenge.objects.append({"name": "benchmark_moved_obstacle", "size": [.3, .3, .8],
            "position": [origin[0] + .7, origin[1], .4], "color": [.85, .25, .2, 1.]})
    return challenge, origin


class PhysicsMeasurements:
    def __init__(self, origin):
        self.origin = origin
        self.previous = None
        self.travel = 0.
        self.contacts = 0
        self.contact_active = False
        self.manual_placements = 0
        self.target = None
        self.dwell_start = None
        self.dwell_s = 0.
        self.maximum_dwell_s = 0.
        self.pose = None
        self.grounded = False
        self.linear_speed = 0.
        self.angular_speed = 0.

    def sample(self, worker):
        import pybullet as bullet
        sim = worker.sim
        position, rotation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        linear, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
        self.pose = [position[0] - self.origin[0], position[1] - self.origin[1], bullet.getEulerFromQuaternion(rotation)[2]]
        self.linear_speed = math.hypot(*linear[:2])
        self.angular_speed = abs(angular[2])
        self.grounded = any(contact[2] != sim.robot and contact[7][2] > .8 and contact[9] > .01
            for contact in bullet.getContactPoints(bodyA=sim.robot, physicsClientId=sim.client))
        if self.previous is not None:
            self.travel += math.dist(position[:2], self.previous)
        self.previous = list(position[:2])
        contact = bool(sim.proximity_sensors().collisions)
        self.contacts += int(contact and not self.contact_active)
        self.contact_active = contact
        self.manual_placements = max(self.manual_placements, worker.latest.get("manual_placements", 0))
        at_target = self.target is not None and math.dist(self.pose[:2], self.target) <= .15
        if at_target and self.grounded and self.linear_speed <= .025 and self.angular_speed <= .15 and not contact:
            self.dwell_start = sim.ticks / 240 if self.dwell_start is None else self.dwell_start
            self.dwell_s = sim.ticks / 240 - self.dwell_start
            self.maximum_dwell_s = max(self.maximum_dwell_s, self.dwell_s)
        else:
            self.dwell_start, self.dwell_s = None, 0.

    def select_target(self, position):
        self.target = list(position)
        self.dwell_start, self.dwell_s, self.maximum_dwell_s = None, 0., 0.


async def execute_attempt(case, directory, document, resources, source_hashes, map_path):
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mission import HomeMission, HomeRequest
    from backend.recording import RunRecorder, write_recording_json
    from backend.simulation import MotionError
    from backend.worker import SimulationWorker
    from scripts.evaluate_supervised import ROOT
    directory.mkdir(parents=True, exist_ok=False)
    challenge, origin = fixture(case)
    metrics = {"budget_s": case["budget_s"], "arrivals": [], "elapsed_s": 0., "setup_s": 0.}
    rooms = [place for place in document["places"] if place["kind"] == "room"]
    pair = next(((first, second) for first in rooms for second in rooms
        if math.dist(first["pose_m_rad"][:2], second["pose_m_rad"][:2]) >= 2.), None)
    if case["task_id"] == "room_to_room" and pair is None:
        metrics["blocked_prerequisite"] = "Frozen source map lacks two named room destinations at least 2 m apart"
        outcome = score_attempt(case["task_id"], metrics)
        result = {**case, "metrics": metrics, **outcome, "evidence": "scripted_test", "physics_success": False,
            "verification_eligible": False, "initial_goal": case["title"], "final_goal": case["title"],
            "termination_reason": "blocked", "request_error": metrics["blocked_prerequisite"],
            "benchmark": {"suite_id": SUITE_ID, "task_id": case["task_id"], **outcome}}
        write_recording_json(directory / "report.json", result)
        return result
    worker = SimulationWorker(challenge=challenge, pace=True, rendering="enhanced", render_resources=resources)
    monitor = PhysicsMeasurements(origin)
    recorder = None
    writer = None
    done = asyncio.Event()
    writer_errors = []
    score = {}
    began = time.monotonic()
    task_started = None
    run_identity = {}

    async def flush_recording():
        try:
            while not done.is_set():
                await worker.call(lambda sim: recorder.capture(worker))
                await asyncio.to_thread(recorder.flush)
                try:
                    await asyncio.wait_for(done.wait(), .5)
                except TimeoutError:
                    pass
        except Exception as error:
            writer_errors.append(f"{type(error).__name__}: {error}")
            worker.stop()

    async def command(action, **values):
        return await worker.home_command(HomeRequest(**run_identity, action=action, **values))

    async def navigate(place):
        await worker.call(lambda sim: monitor.select_target(place["pose_m_rad"][:2]))
        remaining = case["budget_s"] - (time.monotonic() - task_started)
        if remaining < 1.:
            raise TimeoutError("Task budget exhausted before next leg")
        await command("navigate_to", place_id=place["place_id"], time_budget=remaining)
        while worker.home_mission.active:
            if time.monotonic() - task_started >= case["budget_s"] + .5:
                raise TimeoutError("Task budget expired")
            if writer_errors:
                raise RuntimeError(writer_errors[0])
            await asyncio.sleep(.05)
        await worker.call(lambda sim: monitor.sample(worker))
        task = dict(worker.home_mission.task)
        arrival = {"place_id": place["place_id"], "controller_status": task["status"],
            "reason": task["reason"], "position_error_m": math.dist(monitor.pose[:2], place["pose_m_rad"][:2]),
            "stable_dwell_sim_s": round(monitor.dwell_s, 6), "grounded": monitor.grounded,
            "retries": task["retries"], "segments": task["segments"]}
        metrics["arrivals"].append(arrival)
        if task["status"] != "completed":
            raise ValueError(task["reason"])

    try:
        async with asyncio.timeout(60.):
            await asyncio.wrap_future(worker.ready)
            run_identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
            store = write_map_copy(directory / "maps.sqlite3", document)
            recorder = RunRecorder(directory / "recording", context=lambda: {"phase": "benchmark_scripted"})
            def attach(sim):
                worker.home_mission = HomeMission(worker, store)
                worker.recorder = recorder
                original_tick = sim.on_tick
                def tick():
                    original_tick()
                    monitor.sample(worker)
                sim.on_tick = tick
                monitor.sample(worker)
                recorder.capture(worker)
            await worker.call(attach)
            writer = asyncio.create_task(flush_recording())
            await command("load_map", map_id=MAP_ID)
            await worker.scan_continuous(ContinuousScan(**run_identity, compact_arms=True))
            if case["task_id"] != "localize":
                localized = await command("localize", place_id=HOME_ID)
                if localized["localization"]["status"] != "localized":
                    raise ValueError("Setup localization did not establish a pose")
            metrics["setup_s"] = time.monotonic() - began
        initial_travel = monitor.travel
        initial_free = int((worker.home_mission.home.cells == 0).sum()) * .01
        task_started = time.monotonic()
        metrics["measurement_start_wall_s"] = task_started - recorder.started
        async with asyncio.timeout(case["budget_s"] + 1.):
            if case["task_id"] == "localize":
                state = await command("localize", place_id=HOME_ID)
                await worker.call(lambda sim: monitor.sample(worker))
                pose = state["localization"]["pose_m_rad"]
                metrics.update(localized=state["localization"]["status"] == "localized",
                    position_error_m=math.dist(pose[:2], monitor.pose[:2]) if pose else None,
                    heading_error_rad=abs(math.atan2(math.sin(pose[2] - monitor.pose[2]), math.cos(pose[2] - monitor.pose[2]))) if pose else None,
                    scan_quality=state["localization"]["quality"], estimated_pose_m_rad=pose, evaluator_pose_m_rad=monitor.pose)
            elif case["task_id"] in {"navigate", "return_home", "room_to_room"}:
                checkpoint = next(place for place in document["places"] if place["place_id"] == CHECKPOINT_ID)
                home = next(place for place in document["places"] if place["place_id"] == HOME_ID)
                destinations = list(pair) if pair else [checkpoint, home] if case["task_id"] == "return_home" else [checkpoint]
                for place in destinations:
                    await navigate(place)
            elif case["task_id"] == "blocked_route":
                try:
                    await command("navigate_to", place_id=CHECKPOINT_ID, time_budget=case["budget_s"])
                except ValueError as error:
                    metrics.update(unreachable_rejected=str(error).startswith("UNREACHABLE:"), rejection=str(error))
                if worker.home_mission.active:
                    while worker.home_mission.active:
                        await asyncio.sleep(.05)
                    metrics["error"] = "An occupied destination was admitted rather than explicitly rejected"
            elif case["task_id"] == "explore":
                await command("explore", time_budget=case["budget_s"])
                while worker.home_mission.active:
                    await asyncio.sleep(.05)
                metrics.update(controller_task=dict(worker.home_mission.task),
                    new_free_m2=int((worker.home_mission.home.cells == 0).sum()) * .01 - initial_free,
                    buffers_empty=not worker.navigation or not worker.navigation.buffer)
            elif case["task_id"] == "cancel":
                await command("navigate_to", place_id=CHECKPOINT_ID, time_budget=case["budget_s"])
                while monitor.travel - initial_travel < .1:
                    if not worker.home_mission.active:
                        raise ValueError("Motion ended before the Stop injection point: " + worker.home_mission.task["reason"])
                    await asyncio.sleep(.01)
                stop_revision, task_revision = worker.stop_revision, worker.task_revision
                stopped_at = time.monotonic()
                metrics.update(stop_triggered=True, travel_before_stop_m=monitor.travel - initial_travel)
                worker.stop()
                await worker.hold_stopped()
                metrics["stop_ack_s"] = time.monotonic() - stopped_at
                ticks = await worker.call(lambda sim: sim.ticks)
                await asyncio.sleep(.75)
                metrics["frozen_after_stop"] = await worker.call(lambda sim: sim.ticks == ticks and not worker.navigation.buffer)
                await worker.resume_manual()
                try:
                    await worker.home_command(HomeRequest(**run_identity, action="navigate_to", place_id=CHECKPOINT_ID), stop_revision, task_revision)
                except MotionError as error:
                    metrics["late_command_rejected"] = "invalidated" in str(error)
                await asyncio.sleep(.25)
                metrics["no_resume_motion"] = await worker.call(lambda sim: sim.ticks == ticks and not worker.navigation.buffer and not worker.home_mission.active)
        metrics["elapsed_s"] = time.monotonic() - task_started
        metrics["travel_m"] = monitor.travel - initial_travel
    except (ValueError, MotionError, TimeoutError) as error:
        metrics["error"] = f"{type(error).__name__}: {error}"
        if task_started is None:
            metrics["setup_failure"] = True
            metrics["setup_s"] = time.monotonic() - began
        else:
            metrics["elapsed_s"] = time.monotonic() - task_started
            metrics["travel_m"] = monitor.travel - initial_travel
    except Exception as error:
        metrics["infrastructure_error"] = f"{type(error).__name__}: {error}"
    finally:
        worker.stop()
        done.set()
        if writer:
            await writer
        try:
            if worker.ready.done() and worker.ready.exception() is None and worker.thread.is_alive():
                await worker.hold_stopped()
                def detach(sim):
                    monitor.sample(worker)
                    if recorder:
                        recorder.capture(worker)
                    worker.recorder = None
                    return worker.camera_frames.get((worker.latest.get("camera") or {}).get("frame_ref"))
                image = await worker.call(detach)
                if image:
                    (directory / "terminal.png").write_bytes(image)
            if recorder:
                score = await asyncio.to_thread(recorder.finish, {"evidence": "scripted_test", "real_model": False,
                    "suite_id": SUITE_ID, "case_id": case["case_id"], "evaluator_only": True})
            if writer_errors:
                metrics["infrastructure_error"] = writer_errors[0]
        finally:
            if recorder and not recorder.stream.closed:
                recorder.stream.close()
            await worker.close()
    metrics.update(recording_complete=bool(score.get("complete_recording") and score.get("spatial_recording", {}).get("complete") and not writer_errors),
        contact_episodes=monitor.contacts, manual_placements=monitor.manual_placements,
        source_unchanged=not changed_sources(source_hashes), original_map_unchanged=digest(read_map(map_path)) == MAP_SHA256,
        maximum_sample_gap_s=score.get("maximum_sample_gap_s"), measurement="evaluator_only_world_pose_and_contacts")
    outcome = score_attempt(case["task_id"], metrics)
    result = {**case, **outcome, "metrics": metrics, "recording_scorecard": score,
        "run_id": run_identity.get("run_id"), "evidence": "scripted_test", "real_luna": False,
        "rendering": "enhanced", "verification_eligible": False, "physics_success": False,
        "initial_goal": case["title"], "final_goal": case["title"], "termination_reason": outcome["status"],
        "evaluation_elapsed_s": metrics["elapsed_s"], "test_passed": outcome["passed"], "input_tokens": 0, "output_tokens": 0,
        "home_map": {"map_id": MAP_ID, "source_sha256": MAP_SHA256, "derived_sha256": digest(document)},
        "benchmark": {"suite_id": SUITE_ID, "task_id": case["task_id"], **outcome},
        "error": metrics.get("error") or metrics.get("infrastructure_error")}
    write_recording_json(directory / "report.json", result)
    return result


def changed_sources(hashes):
    from scripts.evaluate_supervised import ROOT
    return [name for name, expected in hashes.items() if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]


def markdown_report(summary, results):
    lines = [f"# {summary['suite_id']} Baseline", "", "Evidence: scripted executive, real simulated sensors and physics. No Luna inference.", "",
        "| Task | Passed / planned | Failed | Blocked | Invalid | Not run | Successful median (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for task in summary["tasks"]:
        median = "Not measured" if task["successful_median_s"] is None else f"{task['successful_median_s']:.3f}"
        lines.append(f"| {task['title']} | {task['passed']}/{task['planned']} | {task['failed']} | {task['blocked']} | {task['invalid']} | {task['not_run']} | {median} |")
    lines.extend(["", "## Attempts", "", "| Attempt | Result | Detail | Evidence |", "| --- | --- | --- | --- |"])
    for result in results:
        detail = result["reason"].replace("|", "/").replace("\n", " ")
        replay = f"[Replay]({result['case_id']}/recording/replay.html)" if result.get("recording_scorecard") else "No execution"
        lines.append(f"| {result['case_id']} | {result['status']} | {detail} | [Report]({result['case_id']}/report.json) / {replay} |")
    lines.extend(["", "## Interpretation", "", "Three fixed starting offsets are development repetitions, not a statistical reliability estimate. Missing prerequisites remain in the planned denominator. Safe rejection does not establish successful navigation. Controller completion is checked against evaluator-only position and stopped dwell. The suite does not test real Luna instruction interpretation, complete home coverage, or physical hardware.", "",
        "The original partial map is frozen. The benchmark adds one explicit annotation at an already observed free coordinate to isolated map copies; it never imports simulator geometry into the map. Room-to-room tasks remain blocked until a new benchmark version includes a sensor-built map with room annotations."])
    return "\n".join(lines) + "\n"


def compare_runs(baseline, candidate):
    before = json.loads((baseline / "experiment.json").read_text(encoding="utf-8"))
    after = json.loads((candidate / "experiment.json").read_text(encoding="utf-8"))
    for key in ("suite_sha256", "derived_map_sha256", "fixture_sha256", "evidence"):
        if before.get(key) != after.get(key):
            raise ValueError(f"Unmatched benchmark input: {key}")
    if before["experiment_id"] == after["experiment_id"] or any(manifest.get("source_changed_during_run") or not manifest.get("finished_at") for manifest in (before, after)):
        raise ValueError("Comparison requires two distinct completed frozen-code runs")
    results = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in (baseline, candidate)]
    if any(len(rows) != len(planned_cases(before["suite"])) for rows in results):
        raise ValueError("Comparison requires every planned outcome, including blocked and invalid cases")
    summaries = [summarize(before["suite"], rows) for rows in results]
    return {"suite_id": SUITE_ID, "baseline": before["experiment_id"], "candidate": after["experiment_id"],
        "tasks": [{"task_id": first["task_id"], "baseline_passes": first["passed"], "candidate_passes": second["passed"],
            "planned": first["planned"], "baseline": first, "candidate": second} for first, second in zip(summaries[0]["tasks"], summaries[1]["tasks"])],
        "runtime_mismatch": before["design"]["runtime"] != after["design"]["runtime"],
        "warning": "Descriptive fixed-suite comparison only; inspect per-case outcomes and shared-machine load before improvement claims"}


async def run_suite(options):
    from backend.camera import EnhancedResources
    from backend.home_mission import DEFAULT_MAP_PATH
    from backend.recording import write_recording_json
    from scripts.evaluate_supervised import ROOT, design_snapshot, workload_snapshot
    definition = suite_definition()
    if options.stage == "plan":
        return {"suite": definition, "cases": planned_cases(definition), "suite_sha256": digest(definition)}
    if options.stage == "compare":
        if options.baseline is None or options.candidate is None:
            raise ValueError("Compare requires --baseline and --candidate directories")
        return compare_runs(options.baseline, options.candidate)
    if options.output is None:
        raise ValueError("A fresh --output directory is required")
    map_path = options.map_store or DEFAULT_MAP_PATH
    source_document = read_map(map_path)
    document = benchmark_map(source_document)
    options.output.mkdir(parents=True, exist_ok=False)
    design = design_snapshot(SUITE_ID + ("-preflight" if options.stage == "preflight" else "-baseline"))
    cases = planned_cases(definition)
    for case in cases:
        challenge, _ = fixture(case)
        case["challenge_sha256"] = digest(challenge.model_dump())
    manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "mode": "home_mapping", "stage": "spatial_workflow", "evidence": "scripted_test",
        "supervisor_deployment": "None", "suite": definition, "suite_sha256": digest(definition),
        "derived_map_sha256": digest(document), "fixture_sha256": digest([case["challenge_sha256"] for case in cases]),
        "cases": cases, "design": design, "preflight_only": options.stage == "preflight", "workload_before": workload_snapshot(),
        "note": "Fixed multi-task scripted executive benchmark; independent evaluator-only scoring, no model calls. No controller tuning during baseline."}
    write_recording_json(options.output / "experiment.json", manifest)
    write_recording_json(options.output / "frozen-map.json", document)
    write_recording_json(options.output / "source-map.json", source_document)
    write_recording_json(options.output / "suite.json", definition)
    results = []
    resources = None
    try:
        if options.stage == "preflight":
            resources = await asyncio.to_thread(EnhancedResources)
            case = cases[0]
            result = await execute_attempt(case, options.output / case["case_id"], document, resources, design["code_sha256"], map_path)
            results.append(result)
        else:
            resources = await asyncio.to_thread(EnhancedResources)
            for case in cases:
                changed = changed_sources(design["code_sha256"])
                if changed:
                    raise RuntimeError("Source changed during frozen benchmark: " + ", ".join(changed))
                result = await execute_attempt(case, options.output / case["case_id"], document, resources, design["code_sha256"], map_path)
                results.append(result)
                write_recording_json(options.output / case["case_id"] / "report.json", result)
                write_recording_json(options.output / "summary.json", results)
                write_recording_json(options.output / "benchmark.json", summarize(definition, results))
                print(json.dumps({"case_id": case["case_id"], "status": result["status"], "reason": result["reason"],
                    "elapsed_s": result["metrics"]["elapsed_s"]}), flush=True)
    finally:
        if resources:
            await asyncio.to_thread(resources.close)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["changed_sources"] = changed_sources(design["code_sha256"])
        manifest["source_changed_during_run"] = bool(manifest["changed_sources"])
        manifest["original_map_unchanged"] = digest(read_map(map_path)) == MAP_SHA256
        manifest["workload_after"] = workload_snapshot()
        summary = summarize(definition, results)
        write_recording_json(options.output / "experiment.json", manifest)
        write_recording_json(options.output / "summary.json", results)
        write_recording_json(options.output / "benchmark.json", summary)
        (options.output / "summary.md").write_text(markdown_report(summary, results), encoding="utf-8")
    if manifest["source_changed_during_run"] or not manifest["original_map_unchanged"]:
        raise RuntimeError("Run is not eligible as a frozen baseline; input drift retained in manifest")
    return summary


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_suite(argument_parser().parse_args())), indent=2))