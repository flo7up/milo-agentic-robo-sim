import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
from backend.challenges import get_challenge
from backend.continuous_navigation import ContinuousScan
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission, HomeRequest
from backend.memory_session import configure_memory, persist_memory, record_observation
from backend import session_recording
from backend.worker import SimulationWorker
from scripts.benchmark_household import PhysicsMeasurements, changed_sources
from scripts.evaluate_supervised import design_snapshot, workload_snapshot


TARGETS = {
    "kitchen": {
        "goal": "Use lookup_room for Kitchen, navigate to the returned remembered place, then inspect the current camera and report whether visible fixtures still identify a kitchen. Do not treat memory as current room proof.",
        "lookup": "lookup_room",
        "verification": "observe_room",
    },
    "tv": {
        "goal": "Use find_object_sightings for TV, navigate to the returned remembered observation viewpoint, then reacquire the TV in the current image, approach it safely, and verify it. Do not treat the old sighting as current identity proof.",
        "lookup": "find_object_sightings",
        "verification": "verify_object",
    },
}


def tv_challenge():
    return get_challenge("park").model_copy(update={"objectives": [], "objects": [
        {"name": "back_wall", "size": [.1, 6., 1.], "position": [2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "front_wall", "size": [.1, 6., 1.], "position": [-2.9, 0., .5], "color": [.7, .7, .7, 1.]},
        {"name": "left_wall", "size": [5.8, .1, 1.], "position": [0., 2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "right_wall", "size": [5.8, .1, 1.], "position": [0., -2.9, .5], "color": [.7, .7, .7, 1.]},
        {"name": "localization_landmark_tall", "size": [.35, .45, .9], "position": [-1.2, .9, .45], "color": [.3, .5, .7, 1.]},
        {"name": "localization_landmark_low", "size": [.55, .3, .4], "position": [1.4, -1.1, .2], "color": [.7, .5, .3, 1.]},
        {"name": "tv_screen", "size": [.12, .8, .6], "position": [1.75, 0., .65], "color": [.9, .03, .03, 1.]},
    ]})


def red_bounds(image):
    pixels = np.asarray(Image.open(BytesIO(image))).astype(float)
    rows, columns = np.where((pixels[:, :, 0] > 100)
        & (pixels[:, :, 0] > pixels[:, :, 1] * 1.8)
        & (pixels[:, :, 0] > pixels[:, :, 2] * 1.8))
    if len(rows) <= 8:
        raise ValueError("Evaluator setup did not retain a visible TV screen")
    height, width = pixels.shape[:2]
    return [float(columns.min() / width), float(rows.min() / height),
        float((columns.max() + 1) / width), float((rows.max() + 1) / height)]


async def wait_home(worker, timeout=50.):
    async with asyncio.timeout(timeout):
        while worker.home_mission.active:
            await asyncio.sleep(.02)
    if worker.home_mission.task["status"] != "completed":
        raise ValueError(worker.home_mission.task["reason"])


async def setup_memory(worker, target, store):
    worker.home_mission = HomeMission(worker, store)
    await configure_memory(worker, instance_id=f"real-memory-{target}-fixture")
    identity = {"run_id": worker.latest["run_id"], "episode_epoch": worker.epoch}
    await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
    await worker.home_command(HomeRequest(**identity, action="start_mapping"))
    await worker.home_command(HomeRequest(**identity, action="add_place", name="Home"))
    home = next(place for place in worker.home_mission.home.places if place["name"] == "Home")
    if target == "tv":
        await worker.home_command(HomeRequest(**identity, action="guided_to", pose_m_rad=[.55, 0., 0.], time_budget=45.))
        await wait_home(worker)
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        remembered = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="object", label="TV",
            description="Current-camera TV evidence seeded for the isolated recall evaluation", bounds=red_bounds(paired[1]),
            selected_evidence=paired, source="fixture")
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Real-model TV recall"))
        await persist_memory(worker)
        await worker.home_command(HomeRequest(**identity, action="navigate_to", place_id=home["place_id"], time_budget=45.))
        await wait_home(worker)
    else:
        paired = await worker.call(lambda sim: next(reversed(worker.spatial_frames.values()))[:2])
        remembered = await record_observation(worker, context_id=worker.memory.scope.context_id, kind="room", label="Kitchen",
            description="Current camera shows the furnished starting kitchen", selected_evidence=paired, source="fixture")
        await worker.home_command(HomeRequest(**identity, action="save_map", name="Real-model Kitchen recall"))
        await persist_memory(worker)
        await worker.home_command(HomeRequest(**identity, action="guided_to", pose_m_rad=[.35, 0., 0.], time_budget=45.))
        await wait_home(worker)
    await worker.home_command(HomeRequest(**identity, action="add_place", name="Reload anchor",
        pose_m_rad=list(worker.home_mission.pose)))
    anchor = next(place for place in worker.home_mission.home.places if place["name"] == "Reload anchor")
    await persist_memory(worker)
    profile_id = worker.memory.scope.profile_id
    await configure_memory(worker, fresh=True, name=f"Temporary {target} profile")
    await configure_memory(worker, profile_id=profile_id)
    await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
    await worker.home_command(HomeRequest(**identity, action="localize", place_id=anchor["place_id"]))
    if worker.home_mission.localization["status"] != "localized":
        raise ValueError("Restored profile did not localize from the explicit reload anchor")
    return identity, {"profile_id": profile_id, "remembered": remembered, "anchor": anchor,
        "evidence": "Current paired camera/depth with evaluator-provided semantic label; no hidden geometry enters model input"}


def action_sequence(trace):
    actions = []
    for event in trace["events"]:
        if event["title"] != "Luna mission decision":
            continue
        for call in event["payload"].get("calls", []):
            try:
                actions.append(json.loads(call["arguments"])["action"])
            except (KeyError, TypeError, json.JSONDecodeError):
                actions.append("invalid")
    return actions


def position_criterion(target, final_error, fresh_success):
    return bool(final_error is not None and final_error <= .15) if target == "kitchen" else fresh_success


def self_check():
    assert set(TARGETS) == {"kitchen", "tv"}
    assert TARGETS["kitchen"]["lookup"] != TARGETS["tv"]["lookup"]
    image = Image.new("RGB", (32, 24), (220, 220, 220))
    pixels = np.asarray(image).copy()
    pixels[6:18, 10:22] = [220, 10, 10]
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    bounds = red_bounds(stream.getvalue())
    assert bounds == [10 / 32, 6 / 24, 22 / 32, 18 / 24]
    assert position_criterion("kitchen", .15, False) and not position_criterion("kitchen", .151, True)
    assert position_criterion("tv", .4, True) and not position_criterion("tv", .01, False)
    return {"status": "passed", "model_calls": 0, "targets": TARGETS,
        "limits": {"mission_budget_s": 180, "max_requests": 8, "max_tokens": 100000}}


async def run(target, output):
    self_check()
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(Path("backend/.env"), override=False)
    load_dotenv(Path(".env"), override=False)
    config = FoundryConfig.from_environment()
    profile = next(item for item in config.models if item.id == "luna" and item.provider == "foundry")
    if not config.configured(profile):
        raise RuntimeError("Existing Luna profile is not configured")
    design = design_snapshot("memory-recall-real-v1")
    challenge = get_challenge("kitchen_bathroom") if target == "kitchen" else tv_challenge()
    plan = {"target": target, "goal": TARGETS[target]["goal"], "evidence": "real_model_diagnostic",
        "budget_s": 180, "max_requests": 8, "max_tokens": 100000, "reasoning": "low",
        "started_at": datetime.now(timezone.utc).isoformat(), "design": design,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload_before": workload_snapshot(),
        "limitations": "One isolated real-model trial. Initial semantic memory label is evaluator-provided from a genuine current camera frame."}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    session_recording.SESSION_RESULTS_ROOT = output / "sessions"
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    controller = AgentController(config, evaluation_budget=NavigationEvaluationBudget(max_turns=8, timeout_s=180.))
    controller.record_sessions = True
    controller.recording_evidence = "real_model"
    physical = PhysicsMeasurements(challenge.initial_xy)
    setup = None
    error = None
    trace = {"events": []}
    final_state = dict(controller.state)
    final_mission = None
    final_object = None
    final_pose = None
    began = time.monotonic()
    try:
        await asyncio.wrap_future(worker.ready)
        setup = await setup_memory(worker, target, MapStore(output / "memory.sqlite3"))
        identity, setup_result = setup
        (output / "setup.json").write_text(json.dumps(setup_result, indent=2), encoding="utf-8")

        def attach(sim):
            previous = sim.on_tick
            def sample():
                previous()
                physical.sample(worker)
            sim.on_tick = sample
            physical.sample(worker)
        await worker.call(attach)
        controller.start(worker, AgentStart(**identity, goal=TARGETS[target]["goal"], execution_mode="luna_continuous",
            unified_mission=True, model_id="luna", reasoning="low", max_turns=8, feedback_interval_s=.25,
            images_per_request=2, context_tokens=8192, map_context=True, mission_budget_s=180.,
            max_model_requests=8, max_model_tokens=100000))
        revision = -1
        while not controller.task.done():
            await asyncio.sleep(.5)
            if revision == controller.trace_revision:
                continue
            revision = controller.trace_revision
            trace = controller.trace()
            (output / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
            for event_id, image in controller.trace_images.items():
                path = output / f"frame-{event_id:04d}.png"
                if not path.exists():
                    path.write_bytes(image)
            print(json.dumps({"target": target, "phase": controller.state["phase"],
                "turns": controller.state["turns"], "actions": action_sequence(trace)}), flush=True)
        await controller.task
        await worker.memory.flush()
        trace = controller.trace()
        final_state = dict(controller.state)
        final_mission = controller.mission.state() if controller.mission else None
        final_object = await worker.object_state()
        final_pose = list(worker.home_mission.pose) if worker.home_mission.pose is not None else None
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        trace = controller.trace()
        final_state = dict(controller.state)
        final_mission = controller.mission.state() if controller.mission else None
    finally:
        (output / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        for event_id, image in controller.trace_images.items():
            path = output / f"frame-{event_id:04d}.png"
            if not path.exists():
                path.write_bytes(image)
        await controller.halt()
        worker.stop()
        if worker.ready.done() and worker.ready.exception() is None and worker.thread.is_alive():
            await worker.hold_stopped()
        await worker.close()
    actions = action_sequence(trace)
    remembered_pose = setup[1]["remembered"]["pose_m_rad"] if setup else None
    final_error = math.dist(final_pose[:2], remembered_pose[:2]) if final_pose and remembered_pose else None
    mission_success = bool(final_mission and final_mission["phase"] == "completed" and "target" in final_mission["receipts"])
    required = [TARGETS[target]["lookup"], "navigate_place", TARGETS[target]["verification"]]
    action_success = all(action in actions for action in required)
    fresh_success = bool(final_object and final_object.get("verified")) if target == "tv" else "observe_room" in actions
    position_success = position_criterion(target, final_error, fresh_success)
    result = {"target": target, "passed": bool(not error and not final_state.get("error") and mission_success and action_success
            and fresh_success and position_success and physical.contacts == 0 and physical.manual_placements == 0),
        "error": error or final_state.get("error"), "controller_outcome": final_state.get("outcome"),
        "mission": final_mission, "actions": actions, "required_actions": required,
        "fresh_verification": fresh_success, "position_criterion_passed": position_success,
        "final_memory_position_error_m": final_error,
        "travel_m": physical.travel, "contact_episodes": physical.contacts, "manual_placements": physical.manual_placements,
        "turns": final_state.get("turns"), "input_tokens": final_state.get("input_tokens"),
        "output_tokens": final_state.get("output_tokens"),
        "model_requests": (final_state.get("inference_budget") or {}).get("requests"), "elapsed_s": time.monotonic() - began,
        "source_unchanged": not changed_sources(design["code_sha256"]),
        "finished_at": datetime.now(timezone.utc).isoformat(), "workload_after": workload_snapshot(),
        "limitations": "Single real-model diagnostic with evaluator-seeded semantic memory; no model identity, reliability, or speed claim."}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated real-Luna spatial-memory recall diagnostic")
    parser.add_argument("--stage", choices=["check", "run"], default="check")
    parser.add_argument("--target", choices=sorted(TARGETS), default="kitchen")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.stage == "check":
        print(json.dumps(self_check(), indent=2))
    elif arguments.output is None:
        parser.error("--output is required for run")
    else:
        outcome = asyncio.run(run(arguments.target, arguments.output))
        raise SystemExit(0 if outcome["passed"] else 1)