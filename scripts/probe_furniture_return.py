import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from dotenv import load_dotenv

from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
from backend.challenges import shared_apartment
from backend.continuous_navigation import ContinuousScan
from backend.home_mapping import MapStore
from backend.home_mission import DEFAULT_MAP_PATH, HomeMission, HomeRequest
from backend import session_recording
from backend.worker import SimulationWorker
from scripts.benchmark_household import HOME_ID, MAP_ID, PhysicsMeasurements, changed_sources, digest, read_map
from scripts.evaluate_supervised import design_snapshot, workload_snapshot


GOAL = "Go to the wooden table in the living area. Stop safely beside it, without touching it, then return to the place where you started and stop. Use the saved Home place for the return if available. Do not circle the table. Report completion only after both the approach and return are done."
CRITERIA = {"table_edge_distance_m": [.4, 1.], "minimum_departure_m": .5,
    "stable_dwell_sim_s": .5, "return_distance_m": .15,
    "max_linear_speed_mps": .025, "max_angular_speed_radps": .15, "contacts_allowed": 0}


class RoundTripMonitor:
    def __init__(self):
        self.stage = "approach"
        self.stable_since = None
        self.approach = None
        self.returned = None
        self.last = None

    def observe(self, sample):
        self.last = sample
        stable = (sample["grounded"] and sample["linear_speed_mps"] <= CRITERIA["max_linear_speed_mps"]
            and sample["angular_speed_radps"] <= CRITERIA["max_angular_speed_radps"] and not sample["contacts"])
        if self.stage == "approach":
            eligible = (.4 <= sample["table_edge_distance_m"] <= 1. and sample["home_distance_m"] >= .5)
        else:
            eligible = sample["home_distance_m"] <= .15
        if not stable or not eligible:
            self.stable_since = None
            return
        if self.stable_since is None:
            self.stable_since = sample["simulated_s"]
        if sample["simulated_s"] - self.stable_since < .5:
            return
        if self.stage == "approach":
            self.approach = dict(sample)
            self.stage = "return"
            self.stable_since = None
        elif self.returned is None:
            self.returned = dict(sample)

    def result(self):
        final = self.last or {}
        return {"approach_verified": self.approach is not None, "return_verified": self.returned is not None,
            "physical_round_trip": bool(self.approach and self.returned and final.get("home_distance_m", math.inf) <= .15
                and final.get("linear_speed_mps", math.inf) <= .025 and final.get("angular_speed_radps", math.inf) <= .15
                and final.get("grounded") and final.get("contacts") == 0),
            "approach_sample": self.approach, "return_sample": self.returned, "final_sample": self.last}


def self_check():
    sample = {"grounded": True, "linear_speed_mps": 0., "angular_speed_radps": 0., "contacts": 0,
        "home_distance_m": 0., "table_edge_distance_m": 2., "simulated_s": 0.}
    monitor = RoundTripMonitor()
    for elapsed in (0., 1.):
        monitor.observe({**sample, "simulated_s": elapsed})
    assert not monitor.result()["physical_round_trip"] and monitor.approach is None
    for elapsed in (2., 2.25, 2.5):
        monitor.observe({**sample, "home_distance_m": 1., "table_edge_distance_m": .6, "simulated_s": elapsed})
    assert monitor.approach and monitor.returned is None
    for elapsed in (3., 3.5):
        monitor.observe({**sample, "simulated_s": elapsed})
    assert monitor.result()["physical_round_trip"]
    monitor.observe({**sample, "contacts": 1, "simulated_s": 4.})
    assert not monitor.result()["physical_round_trip"]
    for override in ({"grounded": False}, {"linear_speed_mps": .1}, {"table_edge_distance_m": 1.1}):
        monitor = RoundTripMonitor()
        for elapsed in (0., 1.):
            monitor.observe({**sample, "home_distance_m": 1., "table_edge_distance_m": .6, "simulated_s": elapsed, **override})
        assert monitor.approach is None
    return {"scorer_checks": "passed", "model_calls": 0, "goal": GOAL, "criteria": CRITERIA}


async def run(output, *, unified=False, map_context=True):
    self_check()
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(Path("backend/.env"), override=False)
    load_dotenv(Path(".env"), override=False)
    config = FoundryConfig.from_environment()
    profile = next(profile for profile in config.models if profile.id == "luna" and profile.provider == "foundry")
    if not config.configured(profile):
        raise RuntimeError("Existing Luna profile is not configured")
    original_hash = digest(read_map(DEFAULT_MAP_PATH))
    reader = sqlite3.connect(f"file:{DEFAULT_MAP_PATH.resolve().as_posix()}?mode=ro", uri=True)
    writer = sqlite3.connect(output / "maps.sqlite3")
    try:
        reader.backup(writer)
    finally:
        writer.close()
        reader.close()
    challenge = shared_apartment("flat_kitchen")
    table = next(item for item in challenge.objects if item["name"] == "circuit_table_top")
    design = design_snapshot("unified-map-mission-v1" if unified else "table-return-diagnostic-v1")
    plan = {"goal": GOAL, "criteria": CRITERIA, "budget_s": 180., "evidence": "real_model_diagnostic",
        "started_at": datetime.now(timezone.utc).isoformat(), "design": design,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "map_sha256": original_hash, "map_id": MAP_ID, "localization_hint": HOME_ID,
        "evaluator_only_table": table, "workload_before": workload_snapshot(),
        "unified_mission": unified, "map_context": map_context if unified else None,
        "note": "One custom-goal trial, not a replacement fixed baseline or model identity qualification"}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    session_recording.SESSION_RESULTS_ROOT = output
    worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
    controller = AgentController(config, evaluation_budget=NavigationEvaluationBudget(timeout_s=180.))
    controller.record_sessions = True
    controller.recording_evidence = "real_model"
    physical = PhysicsMeasurements(challenge.initial_xy)
    monitor = RoundTripMonitor()
    samples = []
    error = None
    began = time.monotonic()
    try:
        async with asyncio.timeout(90.):
            await asyncio.wrap_future(worker.ready)
            worker.home_mission = HomeMission(worker, MapStore(output / "maps.sqlite3"))
            identity = {"run_id": worker.sim.run_id, "episode_epoch": worker.epoch}
            await worker.home_command(HomeRequest(**identity, action="load_map", map_id=MAP_ID))
            await worker.scan_continuous(ContinuousScan(**identity, compact_arms=True))
            await worker.home_command(HomeRequest(**identity, action="localize", place_id=HOME_ID))
            worker.home_mission.allow_expansion = False
        def sample():
            physical.sample(worker)
            position = [physical.pose[index] + physical.origin[index] for index in (0, 1)]
            distance = math.hypot(*[max(abs(position[index] - table["position"][index]) - table["size"][index] / 2, 0.) for index in (0, 1)])
            value = {"wall_s": time.monotonic() - began, "simulated_s": worker.sim.ticks / 240.,
                "position_m": position, "table_edge_distance_m": distance, "home_distance_m": math.hypot(*physical.pose[:2]),
                "linear_speed_mps": physical.linear_speed, "angular_speed_radps": physical.angular_speed,
                "grounded": physical.grounded, "contacts": physical.contacts}
            monitor.observe(value)
            samples.append(value)
        def attach(sim):
            previous = sim.on_tick
            def tick():
                previous()
                sample()
            sim.on_tick = tick
            sample()
        await worker.call(attach)
        controller.start(worker, AgentStart(**identity, goal=GOAL, execution_mode="luna_continuous", model_id="luna",
            reasoning="high", max_turns=80, feedback_interval_s=.25, images_per_request=2, context_tokens=8192,
            continuous_handoff=False, skill_composer=False, adaptive_navigation=True,
            unified_mission=unified, map_context=map_context, mission_budget_s=180.))
        last_revision = -1
        while not controller.task.done():
            await asyncio.sleep(.5)
            if controller.trace_revision != last_revision:
                last_revision = controller.trace_revision
                (output / "trace.json").write_text(json.dumps(controller.trace(), indent=2), encoding="utf-8")
                for event_id, image in list(controller.trace_images.items()):
                    path = output / f"frame-{event_id:04d}.png"
                    if not path.exists():
                        path.write_bytes(image)
                print(json.dumps({"phase": controller.state["phase"], "turns": controller.state["turns"],
                    "approach": monitor.approach is not None, "returned": monitor.returned is not None}), flush=True)
        await controller.task
        await worker.call(lambda sim: sample())
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        final_controller_state = dict(controller.state)
        await controller.halt()
        worker.stop()
        if worker.thread.is_alive() and worker.ready.done() and worker.ready.exception() is None:
            await worker.hold_stopped()
        await worker.close()
    result = {**monitor.result(), "controller_outcome": final_controller_state.get("outcome"),
        "error": error or final_controller_state.get("error"), "turns": final_controller_state["turns"],
        "input_tokens": final_controller_state["input_tokens"], "output_tokens": final_controller_state["output_tokens"],
        "travel_m": physical.travel, "source_unchanged": not changed_sources(design["code_sha256"]),
        "original_map_unchanged": digest(read_map(DEFAULT_MAP_PATH)) == original_hash,
        "map_copy_unchanged": digest(read_map(output / "maps.sqlite3")) == original_hash,
        "finished_at": datetime.now(timezone.utc).isoformat(), "workload_after": workload_snapshot(),
        "limitations": "Single real-model diagnostic. Object identity is scored by physical geometry, not independent semantic recognition; trace exports contain primary images only."}
    (output / "measurements.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One recorded real-Luna furniture approach and return diagnostic")
    parser.add_argument("--stage", choices=["check", "run"], default="check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unified", action="store_true")
    parser.add_argument("--without-map", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage == "check":
        print(json.dumps(self_check(), indent=2))
    elif arguments.output is None:
        parser.error("--output is required for run")
    else:
        asyncio.run(run(arguments.output, unified=arguments.unified, map_context=not arguments.without_map))