import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from scripts.benchmark_household import HOME_ID, MAP_ID, MAP_SHA256, PhysicsMeasurements, changed_sources, digest, read_map


SUITE_ID = "household-room-v2"
TASKS = (
    {"id": "room_return", "title": "Kitchen, fresh visual report, then Home", "budget_s": 180.,
        "criteria": "Kitchen and Home independently reached within 0.15 m and held 0.5 simulated seconds; fresh post-arrival room report; semantic truth scored separately"},
    {"id": "blocked_doorway", "title": "Reject a blocked recorded doorway", "budget_s": 30.,
        "criteria": "Saved doorway with sensor-backed room evidence required; fresh obstacle prevents arrival, bounded stopped outcome and no contact"},
    {"id": "unknown_room", "title": "Unknown room cannot become an invented destination", "budget_s": 10.,
        "criteria": "Unknown place and frontier IDs rejected, no motion, no visual verification claim"},
)


def definition():
    return {"suite_id": SUITE_ID, "version": 2, "environment": "shared_apartment_v1", "scenario": "flat_kitchen",
        "start_offsets_m": [[0., 0.]], "tasks": list(TASKS), "evidence": "scripted_test",
        "renderer": "enhanced", "model_inference": False, "original_map_sha256": MAP_SHA256,
        "semantic_accuracy": "not_measured", "qualification": "Development mechanics baseline; no Luna or independent semantic accuracy claim"}


def copy_store(source, target):
    if target.exists() or source.resolve() == target.resolve():
        raise ValueError("Working map requires a new separate database")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as copied:
            original.backup(copied)


def planned_cases(tasks):
    return [{"case_id": task["id"], "task_id": task["id"], "challenge": "flat_kitchen",
        "environment": "shared_apartment_v1", **task} for task in tasks]


def prerequisite(task, home, records):
    if task == "unknown_room":
        return None
    kind = "doorway" if task == "blocked_doorway" else "room"
    target = next((place for place in home.places if place["kind"] == kind
        and (kind == "doorway" or place["name"].casefold() == "kitchen")
        and math.dist(place["pose_m_rad"][:2], next(item["pose_m_rad"][:2] for item in home.places if item["place_id"] == HOME_ID)) >= 2.
        and any(record["place_id"] == place["place_id"] and record.get("room_matches") is not False for record in records)), None)
    return target


def score_room(task, metrics):
    if metrics.get("blocked_prerequisite"):
        return {"status": "blocked", "passed": False, "reason": metrics["blocked_prerequisite"]}
    if not all(metrics.get(key) is True for key in ("recording_complete", "source_unchanged", "original_map_unchanged")) or metrics.get("infrastructure_error"):
        return {"status": "invalid", "passed": False, "reason": metrics.get("infrastructure_error") or "Recording or input provenance incomplete"}
    safe = metrics.get("contacts") == 0 and metrics.get("manual_placements") == 0 and metrics.get("buffers_empty") is True
    timed = metrics.get("elapsed_s", math.inf) <= metrics.get("budget_s", 0.) + 1.
    if task == "room_return":
        arrivals = metrics.get("arrivals", [])
        achieved = len(arrivals) == 2 and all(item["status"] == "completed" and item["error_m"] <= .15 and item["dwell_s"] >= .5 for item in arrivals)
        achieved = achieved and metrics.get("post_arrival_report") is True
    elif task == "blocked_doorway":
        achieved = metrics.get("blocked_stopped") is True and metrics.get("reported_arrival") is False
    elif task == "unknown_room":
        achieved = metrics.get("unknown_place_rejected") is True and metrics.get("unknown_frontier_rejected") is True and metrics.get("travel_m", math.inf) <= .03
    else:
        raise ValueError("Unknown room benchmark task")
    passed = bool(safe and timed and achieved)
    return {"status": "passed" if passed else "failed", "passed": passed,
        "reason": "Fixed mechanics checks passed; semantic accuracy unmeasured" if passed else metrics.get("error") or "Fixed mechanics criteria not met"}


async def attempt(options, directory, task, resources, hashes):
    from backend.challenges import shared_apartment
    from backend.continuous_navigation import ContinuousScan
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission, HomeRequest
    from backend.recording import RunRecorder, write_recording_json
    from backend.worker import SimulationWorker
    directory.mkdir(parents=True, exist_ok=False)
    database = directory / "maps.sqlite3"
    copy_store(options.map_store, database)
    store = MapStore(database)
    home = store.load(MAP_ID, "shared_apartment_v1")
    model_trial = task["id"] == "luna_room_return"
    evidence = "real_model" if model_trial else "scripted_test"
    target = prerequisite(task["id"], home, store.room_observations(MAP_ID)) if task["id"] not in {"survey", "luna_room_return"} else None
    metrics = {"budget_s": task["budget_s"], "arrivals": [], "semantic_accuracy": "not_measured"}
    if task["id"] not in {"survey", "unknown_room", "luna_room_return"} and target is None:
        metrics["blocked_prerequisite"] = "Working map lacks a camera-backed " + ("doorway" if task["id"] == "blocked_doorway" else "Kitchen") + " annotation at least 2 m from Home"
        outcome = score_room(task["id"], metrics)
        result = {"case_id": task["id"], "challenge": "flat_kitchen", "environment": "shared_apartment_v1",
            "title": task["title"], "criteria": task["criteria"], "budget_s": task["budget_s"],
            "evidence": "scripted_test", "physics_success": False, "verification_eligible": False,
            "metrics": metrics, **outcome, "benchmark": {"suite_id": SUITE_ID, "task_id": task["id"], **outcome}}
        write_recording_json(directory / "report.json", result)
        return result
    challenge = shared_apartment("flat_kitchen")
    origin = list(challenge.initial_xy)
    if task["id"] == "blocked_doorway":
        challenge.objects.append({"name": "room_v2_doorway_obstruction", "size": [.7, .7, 1.3],
            "position": [origin[0] + target["pose_m_rad"][0], origin[1] + target["pose_m_rad"][1], .65], "color": [.8, .2, .2, 1.]})
    worker = SimulationWorker(challenge=challenge, pace=True, rendering="enhanced", render_resources=resources)
    monitor = PhysicsMeasurements(origin)
    recorder, writer = None, None
    done = asyncio.Event()
    failures, observations = [], []
    started = None
    controller = None
    if model_trial:
        from dotenv import load_dotenv
        from backend.agent import AgentController, FoundryConfig, NavigationEvaluationBudget
        load_dotenv("backend/.env", override=False)
        load_dotenv(".env", override=False)
        config = FoundryConfig.from_environment()
        controller = AgentController(config, evaluation_budget=NavigationEvaluationBudget(max_turns=None, timeout_s=task["budget_s"]))
        profile = next(profile for profile in config.models if profile.id == "luna")
        metrics.update(deployment=profile.deployment, underlying_model_identity_verified=False, reasoning="high",
            images_per_request=2, context_tokens=8192, kitchen_physics_seen=False)

    async def flush():
        try:
            while not done.is_set():
                await worker.call(lambda sim: recorder.capture(worker))
                await asyncio.to_thread(recorder.flush)
                try:
                    await asyncio.wait_for(done.wait(), .5)
                except TimeoutError:
                    pass
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            worker.stop()

    async def command(action, **values):
        return await worker.home_command(HomeRequest(run_id=worker.sim.run_id, episode_epoch=worker.epoch, action=action, **values))

    async def travel(place, return_place_id=None):
        await worker.call(lambda sim: monitor.select_target(place["pose_m_rad"][:2]))
        await command("navigate_to", place_id=place["place_id"], return_place_id=return_place_id,
            time_budget=max(1., task["budget_s"] - (time.monotonic() - started)))
        while worker.home_mission.active:
            await asyncio.sleep(.05)
        await worker.call(lambda sim: monitor.sample(worker))
        current = worker.home_mission.task
        metrics["arrivals"].append({"place_id": place["place_id"], "status": current["status"], "reason": current["reason"],
            "error_m": math.dist(monitor.pose[:2], place["pose_m_rad"][:2]), "dwell_s": round(monitor.dwell_s, 6)})
        if current["status"] != "completed":
            raise ValueError(current["reason"])

    async def capture():
        def paired(sim):
            worker._sample_spatial(force=True)
            sensor, image = worker.spatial_frames[worker.spatial_map.sequence][:2]
            return sensor, image
        sensor, image = await worker.call(paired)
        name = f"view-{len(observations):03d}.png"
        (directory / name).write_bytes(image)
        observations.append({"image": name, "spatial_sequence": sensor.sequence, "odometry_m_rad": sensor.odometry_m_rad,
            "head_rad": sensor.head_rad, "simulated_s": sensor.simulated_time_s, "room_identity": "not_labelled"})
        return sensor, image

    try:
        async with asyncio.timeout(90.):
            await asyncio.wrap_future(worker.ready)
            recorder = RunRecorder(directory / "recording", context=lambda: controller.state if controller else {"phase": "room_v2_scripted"})
            def attach(sim):
                worker.home_mission = HomeMission(worker, store)
                worker.recorder = recorder
                previous = sim.on_tick
                def tick():
                    previous()
                    monitor.sample(worker)
                    if model_trial and (sim.challenge_status() or {}).get("status") == "completed":
                        metrics["kitchen_physics_seen"] = True
                sim.on_tick = tick
                monitor.sample(worker)
            await worker.call(attach)
            writer = asyncio.create_task(flush())
            await command("load_map", map_id=MAP_ID)
            await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch, compact_arms=True))
            await command("localize", place_id=HOME_ID)
            await capture()
        initial_free = (await worker.home_state(compact=True))["coverage"]["free_m2"]
        initial_travel = monitor.travel
        started = time.monotonic()
        async with asyncio.timeout(task["budget_s"] + 1.):
            if model_trial:
                from backend.agent import AgentStart
                controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                    goal=task["title"], execution_mode="luna_continuous", model_id="luna", reasoning="high",
                    max_turns=80, feedback_interval_s=.25, images_per_request=2, context_tokens=8192,
                    continuous_handoff=False, skill_composer=False, adaptive_navigation=True))
                while controller.active:
                    for event_id, image in list(controller.trace_images.items()):
                        path = directory / f"frame-{event_id:04d}.png"
                        if not path.exists():
                            path.write_bytes(image)
                    write_recording_json(directory / "trace.json", controller.trace())
                    await asyncio.sleep(.2)
                await controller.task
                write_recording_json(directory / "trace.json", controller.trace())
                state = await worker.home_state(compact=True)
                metrics.update(controller_outcome=controller.state.get("outcome"), controller_error=controller.state.get("error"),
                    room_workflow=state.get("room_workflow"), input_tokens=controller.state["input_tokens"], output_tokens=controller.state["output_tokens"],
                    turns=controller.state["turns"], room_observations=len(state.get("room_observations", [])))
                await capture()
            elif task["id"] == "survey":
                await command("explore", time_budget=task["budget_s"])
                while worker.home_mission.active:
                    await asyncio.sleep(.1)
                metrics["mission"] = dict(worker.home_mission.task)
                await capture()
                await command("save_map", name="Room v2 working survey")
                metrics["new_free_m2"] = (await worker.home_state(compact=True))["coverage"]["free_m2"] - initial_free
            elif task["id"] == "unknown_room":
                for action, values, field, code in (("navigate_to", {"place_id": "unobserved-room-v2"}, "unknown_place_rejected", "UNKNOWN_PLACE"),
                        ("explore_frontier", {"frontier_id": "unobserved-frontier-v2"}, "unknown_frontier_rejected", "UNKNOWN_FRONTIER")):
                    try:
                        await command(action, **values)
                    except ValueError as error:
                        metrics[field] = code in str(error)
            elif task["id"] == "room_return":
                await travel(target, HOME_ID)
                sensor, image = await capture()
                return_place = next(place for place in home.places if place["place_id"] == HOME_ID)
                await worker.call(lambda sim: monitor.select_target(return_place["pose_m_rad"][:2]))
                request = HomeRequest(run_id=worker.sim.run_id, episode_epoch=worker.epoch, action="observe_room", place_id=target["place_id"],
                    room_matches=True, confidence=0., spatial_sequence=sensor.sequence,
                    evidence_text="Scripted contract probe on this post-arrival image; this is not a semantic identity judgment.")
                state = await worker.home_command(request, selected_evidence=(sensor, image))
                metrics["post_arrival_report"] = bool(state["room_workflow"]["room_evidence_id"])
                while worker.home_mission.active:
                    await asyncio.sleep(.05)
                await worker.call(lambda sim: monitor.sample(worker))
                current = worker.home_mission.task
                metrics["arrivals"].append({"place_id": HOME_ID, "status": current["status"],
                    "error_m": math.dist(monitor.pose[:2], return_place["pose_m_rad"][:2]), "dwell_s": round(monitor.dwell_s, 6)})
            else:
                try:
                    await travel(target)
                except ValueError as error:
                    metrics["rejection"] = str(error)
                    metrics["blocked_stopped"] = "UNREACHABLE" in str(error) or "BLOCKED" in str(error)
                metrics["reported_arrival"] = bool(worker.home_mission.task and worker.home_mission.task.get("completion_verified"))
        metrics.update(elapsed_s=time.monotonic() - started, travel_m=monitor.travel - initial_travel)
    except Exception as error:
        metrics.update(error=f"{type(error).__name__}: {error}", elapsed_s=time.monotonic() - started if started else 0.)
    finally:
        if controller:
            await controller.halt()
        worker.stop()
        done.set()
        if writer:
            await writer
        score = {}
        try:
            if worker.thread.is_alive() and worker.ready.done() and worker.ready.exception() is None:
                await worker.hold_stopped()
                metrics.update(contacts=monitor.contacts, manual_placements=monitor.manual_placements, buffers_empty=worker.navigation is None or not worker.navigation.buffer)
                await worker.call(lambda sim: recorder.capture(worker) if recorder else None)
                state = await worker.home_state()
                write_recording_json(directory / "final-state.json", state)
            if recorder:
                score = await asyncio.to_thread(recorder.finish, {"evidence": evidence, "real_model": model_trial, "suite_id": SUITE_ID})
        finally:
            await worker.close()
    metrics.update(recording_complete=bool(score.get("complete_recording") and score.get("spatial_recording", {}).get("complete")),
        source_unchanged=not changed_sources(hashes), original_map_unchanged=digest(read_map(options.original_map)) == MAP_SHA256)
    if failures:
        metrics["infrastructure_error"] = "; ".join(failures)
    if model_trial:
        outcome = {"status": "unqualified", "passed": False, "reason": metrics.get("controller_error") or metrics.get("error") or "Real-model diagnostic; inspect separate physical and semantic evidence"}
    else:
        outcome = score_room(task["id"], metrics) if task["id"] != "survey" else {"status": "failed" if metrics.get("error") or metrics.get("mission", {}).get("status") == "failed" else "recorded", "passed": False, "reason": metrics.get("error") or metrics.get("mission", {}).get("reason") or "Bounded survey; no whole-home or semantic qualification"}
    result = {"case_id": task["id"], "challenge": "flat_kitchen", "environment": "shared_apartment_v1", "title": task["title"],
        "evidence": evidence, "real_luna": model_trial, "rendering": "enhanced", "verification_eligible": False, "physics_success": False,
        "initial_goal": task["title"], "final_goal": task["title"], "input_tokens": metrics.get("input_tokens", 0), "output_tokens": metrics.get("output_tokens", 0),
        "evaluation_elapsed_s": metrics.get("elapsed_s"), "termination_reason": outcome["status"],
        "criteria": task.get("criteria"), "budget_s": task["budget_s"],
        "recording_scorecard": score, "metrics": metrics, **outcome,
        "benchmark": {"suite_id": SUITE_ID, "task_id": task["id"], **outcome} if not model_trial and task["id"] != "survey" else None}
    write_recording_json(directory / "views.json", observations)
    write_recording_json(directory / "report.json", result)
    return result


async def run(options):
    from backend.camera import EnhancedResources
    from backend.recording import write_recording_json
    from scripts.evaluate_supervised import design_snapshot
    if options.stage == "plan":
        return definition()
    read_map(options.original_map)
    options.output.mkdir(parents=True, exist_ok=False)
    design = design_snapshot("room-memory-v1-" + options.stage)
    with sqlite3.connect(f"file:{options.map_store.resolve().as_posix()}?mode=ro", uri=True) as connection:
        input_map = json.loads(connection.execute("SELECT document FROM maps WHERE map_id=?", (MAP_ID,)).fetchone()[0])
    tasks = [{"id": "survey", "title": "Sensor-only working-map expansion", "budget_s": options.budget}] if options.stage == "survey" else list(TASKS)
    manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "evidence": "scripted_test", "mode": "home_mapping", "stage": "spatial_workflow", "design": design,
        "suite": definition() if options.stage == "run" else None, "suite_sha256": digest(definition()) if options.stage == "run" else None,
        "map_document_sha256": digest(input_map), "derived_map_sha256": digest(input_map),
        "map_id": MAP_ID, "map_revision": input_map["revision"],
        "cases": planned_cases(tasks)}
    write_recording_json(options.output / "input-map.json", input_map)
    results = []
    if options.stage == "model":
        tasks = [{"id": "luna_room_return", "title": "Go to the kitchen, verify it from a fresh camera image, then return to the saved Home. Use the observed home map; if Kitchen is unknown, explore within the time budget and record tentative room evidence. Report limitations honestly.", "budget_s": options.budget}]
        manifest.update(evidence="real_model", mode="luna_continuous", cases=planned_cases(tasks),
            note="One local Luna diagnostic on an isolated map copy; no independent semantic identity qualification")
    resources = None
    try:
        resources = await asyncio.to_thread(EnhancedResources)
        for task in tasks:
            if changed_sources(design["code_sha256"]):
                raise RuntimeError("Source changed before attempt")
            result = await attempt(options, options.output / task["id"], task, resources, design["code_sha256"])
            results.append(result)
            print(json.dumps({"case": task["id"], "status": result["status"], "metrics": result["metrics"]}), flush=True)
    finally:
        if resources:
            await asyncio.to_thread(resources.close)
        manifest.update(finished_at=datetime.now(timezone.utc).isoformat(), source_changed_during_run=bool(changed_sources(design["code_sha256"])),
            original_map_unchanged=digest(read_map(options.original_map)) == MAP_SHA256)
        write_recording_json(options.output / "experiment.json", manifest)
        write_recording_json(options.output / "summary.json", results)
    if manifest["source_changed_during_run"] or not manifest["original_map_unchanged"]:
        raise RuntimeError("Frozen input check failed")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate room-v2 mechanics benchmark and recorded sensor-only survey")
    parser.add_argument("--stage", choices=["plan", "survey", "run", "model"], default="plan")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-store", type=Path, default=Path(".runtime/maps/homes.sqlite3"))
    parser.add_argument("--original-map", type=Path, default=Path(".runtime/maps/homes.sqlite3"))
    parser.add_argument("--budget", type=float, default=180.)
    options = parser.parse_args()
    if not 1 <= options.budget <= 300:
        parser.error("Survey budget must be 1-300 seconds")
    print(json.dumps(asyncio.run(run(options)), indent=2))