import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

from backend.agent import AgentController, AgentStart, ConfiguredModel, FoundryConfig, ModelProfile, check_local_supervisor, prepare_local_supervisor
from backend.challenges import get_challenge
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission
from backend.worker import SimulationWorker
from scripts.benchmark_household import PhysicsMeasurements
from scripts.evaluate_supervised import ROOT, design_snapshot, workload_snapshot


MODEL = "qwen3-vl:4b-instruct-q4_K_M"
GOALS = {
    "find": "Find the green parking bay. Stop at your current position as soon as you identify it. Do not approach it.",
    "explore": "Explore the observed surroundings safely. Do not claim to identify any particular room or object.",
}
COMPARISON = [("explore", "qwen", 60), ("explore", "luna", 60),
    ("circle", "luna", 120), ("circle", "qwen", 120),
    ("object", "qwen", 120), ("object", "luna", 120)]


def trial_case(task):
    challenge = get_challenge({"circle": "furniture_circuit", "object": "apartment"}.get(task, "park"))
    return challenge, GOALS.get(task, challenge.goal)


def trial_settings(worker, task, provider, budget_s, requests, tokens):
    return AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, goal=trial_case(task)[1],
        model_id=provider, reasoning="none", execution_mode="luna_continuous", unified_mission=True,
        images_per_request=2, context_tokens=8192, map_context=True, mission_budget_s=budget_s,
        max_turns=requests, max_model_requests=requests, max_model_tokens=tokens, feedback_interval_s=.25)


def local_config():
    return FoundryConfig(endpoint="", models=[ModelProfile(id="qwen", label="Qwen3-VL 4B (local)",
        provider="ollama", deployment=MODEL, context_window=16384)])


async def warmup(config):
    profile = config.models[0]
    readiness = await check_local_supervisor(config, profile.id)
    if not readiness["ready"]:
        raise ValueError(readiness["message"])
    began = time.perf_counter()
    async with httpx.AsyncClient(base_url=config.ollama_endpoint, trust_env=False, follow_redirects=False, timeout=180.) as client:
        version = (await client.get("/api/version")).json()
        tags = (await client.get("/api/tags")).json()
        installed = next(item for item in tags["models"] if item["name"] == profile.deployment)
        response = await client.post("/api/chat", json={"model": profile.deployment, "messages": [],
            "stream": False, "keep_alive": "10m", "options": {"num_ctx": profile.context_window}})
        response.raise_for_status()
        body = response.json()
        if body.get("done") is not True or body.get("done_reason") != "load":
            raise ValueError("Expected a motion-free model load acknowledgement")
        resident = (await client.get("/api/ps")).json()
    return {"readiness": readiness, "ollama_version": version.get("version"), "model": profile.deployment,
        "digest": installed["digest"], "download_bytes": installed["size"], "elapsed_s": time.perf_counter() - began,
        "resident_models": resident.get("models", []), "workload_after": workload_snapshot(), "robot_motion_executed": False}


async def run(output, *, stage="run", task="find", provider="qwen", budget_s=60, requests=4, tokens=40000, design_label="qwen-value-pilot-v1"):
    output.mkdir(parents=True, exist_ok=False)
    if provider == "qwen":
        config = local_config()
    else:
        from dotenv import load_dotenv
        load_dotenv(ROOT / "backend/.env", override=False)
        load_dotenv(ROOT / ".env", override=False)
        config = FoundryConfig.from_environment()
    profile = next(item for item in config.models if item.id == provider)
    challenge, goal = trial_case(task)
    design = design_snapshot(design_label)
    report = {"evidence": "real_model_diagnostic", "stage": stage, "task": task, "provider": provider,
        "started_at": datetime.now(timezone.utc).isoformat(), "design": design,
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest(),
        "workload_before": workload_snapshot(), "error": None,
        "limitations": "Small real-model pilot, not a fixed household benchmark or reliability estimate. GPU snapshots are not peaks; Luna deployment identity is unverified."}
    (output / "plan.json").write_text(json.dumps({**report, "goal": goal,
        "budget_s": budget_s, "max_requests": requests, "max_tokens": tokens,
        "reasoning": "none", "model": profile.model_dump()}, indent=2), encoding="utf-8")
    worker = None
    controller = None
    try:
        if provider == "qwen":
            report["warmup"] = await warmup(config)
            (output / "warmup.json").write_text(json.dumps(report["warmup"], indent=2), encoding="utf-8")
            print(json.dumps({"warmup_s": report["warmup"]["elapsed_s"]}), flush=True)
        if stage == "run":
            calls = []

            class MeasuredModel(ConfiguredModel):
                async def respond(self, profile, reasoning, goal, inputs):
                    began = time.perf_counter()
                    entry = {"profile": profile.id, "started_at": datetime.now(timezone.utc).isoformat()}
                    calls.append(entry)
                    try:
                        response = await super().respond(profile, reasoning, goal, inputs)
                        entry["response"] = response.model_dump(mode="json", exclude_none=True)
                        return response
                    except Exception as error:
                        entry["error"] = type(error).__name__
                        raise
                    finally:
                        entry["elapsed_s"] = time.perf_counter() - began

            worker = SimulationWorker(challenge=challenge, rendering="enhanced", pace=True)
            await asyncio.wait_for(asyncio.wrap_future(worker.ready), 90.)
            worker.home_mission = HomeMission(worker, MapStore(output / "maps.sqlite3"))
            physical = PhysicsMeasurements(challenge.initial_xy)
            observation, image = await worker.feedback()
            report["initial_image_sha256"] = hashlib.sha256(image).hexdigest()
            (output / "initial.png").write_bytes(image)
            if provider == "qwen":
                report["vision_preparation"] = await prepare_local_supervisor(config, "qwen", image)
                report["vision_preparation"]["observation_seq"] = observation.seq

            def attach(sim):
                previous = sim.on_tick
                def tick():
                    previous()
                    physical.sample(worker)
                sim.on_tick = tick
                physical.sample(worker)
            await worker.call(attach)
            controller = AgentController(config, model_factory=MeasuredModel)
            controller.record_sessions = True
            controller.recording_root = output / "sessions"
            controller.recording_evidence = "real_model"
            settings = trial_settings(worker, task, provider, budget_s, requests, tokens)
            report["settings"] = settings.model_dump(mode="json")
            began = time.perf_counter()
            controller.start(worker, settings)
            await asyncio.wait_for(asyncio.shield(controller.task), budget_s + 40.)
            await worker.call(lambda sim: physical.sample(worker))
            challenge_status = copy.deepcopy(worker.latest["challenge"])
            report.update(control_elapsed_s=time.perf_counter()-began, controller=copy.deepcopy(controller.state),
                mission=controller.mission.state() if controller.mission else None,
                travel_m=physical.travel, contact_episodes=physical.contacts, final_pose=physical.pose,
                challenge_status=challenge_status,
                physics_success=challenge_status["status"] == "completed" if task in {"circle", "object"} else None,
                exploration_progress=physical.travel >= .5 and physical.contacts == 0 if task == "explore" else None,
                recording_directory=str(controller.recording_directory), inference=calls)
            _, terminal_image = await worker.feedback()
            (output / "terminal.png").write_bytes(terminal_image)
    except Exception as error:
        report["error"] = type(error).__name__
    finally:
        if controller:
            report.setdefault("controller", copy.deepcopy(controller.state))
            report.setdefault("inference", calls)
            (output / "trace.json").write_text(json.dumps(controller.trace(), indent=2), encoding="utf-8")
            for event_id, images in controller.trace_image_batches.items():
                for index, image in enumerate(images):
                    (output / f"input-{event_id}-{index}.png").write_bytes(image)
            await controller.halt()
        if worker:
            worker.stop()
            await worker.close()
        report["changed_sources"] = [name for name, expected in design["code_sha256"].items()
            if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["workload_after"] = workload_snapshot()
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("error", "control_elapsed_s", "mission", "travel_m", "contact_episodes", "changed_sources")}, indent=2), flush=True)
    return report


async def compare(output, *, design_label="qwen-value-pilot-v1"):
    output.mkdir(parents=True, exist_ok=False)
    design = design_snapshot(design_label)
    plan = {"design": design, "evidence": "real_model", "cases": [
        {"task": task, "provider": provider, "budget_s": budget, "max_requests": 12, "max_tokens": 80000,
            "goal": trial_case(task)[1]} for task, provider, budget in COMPARISON],
        "criteria": {"circle_object": "Existing independent challenge completion, no contacts, complete unassisted recording",
            "explore": "At least 0.5m measured travel without contacts; progress only, not full-task success"},
        "limitations": "One trial per task/provider, alternating order; unequal provider transports/context limits; no causal or reliability claim. No price estimate without known tariff.",
        "started_at": datetime.now(timezone.utc).isoformat()}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    rows = []
    for index, (task, provider, budget) in enumerate(COMPARISON, 1):
        directory = output / f"{index:02d}-{task}-{provider}"
        print(f"START {index}/6 {task} {provider} budget={budget}s", flush=True)
        result = await run(directory, task=task, provider=provider, budget_s=budget, requests=12, tokens=80000,
            design_label=design_label)
        state = result.get("controller", {})
        rows.append({"task": task, "provider": provider, "artifact": str(directory / "result.json"),
            "error": result.get("error"), "controller_error": state.get("error"), "outcome": state.get("outcome"),
            **{key: result.get(key) for key in ("physics_success", "exploration_progress", "control_elapsed_s", "travel_m", "contact_episodes", "initial_image_sha256", "challenge_sha256", "changed_sources")},
            "input_tokens": state.get("input_tokens"), "output_tokens": state.get("output_tokens"),
            "latencies_s": [call["elapsed_s"] for call in result.get("inference", []) if "response" in call],
            "failed_requests": sum("error" in call for call in result.get("inference", []))})
        (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if result["changed_sources"] or result["error"]:
            raise RuntimeError("Pilot stopped on infrastructure error or source drift; retained all completed evidence")
    plan["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return rows


def check_comparison():
    from types import SimpleNamespace
    worker = SimpleNamespace(sim=SimpleNamespace(run_id="11111111-1111-4111-8111-111111111111"), epoch=1)
    for task in ("explore", "circle", "object"):
        cases = [case for case in COMPARISON if case[0] == task]
        assert len(cases) == 2 and {case[1] for case in cases} == {"qwen", "luna"}
        settings = [trial_settings(worker, task, provider, budget, 12, 80000).model_dump(exclude={"model_id"})
            for _, provider, budget in cases]
        assert settings[0] == settings[1]
        assert trial_case(task)[0].id == {"explore": "park", "circle": "furniture_circuit", "object": "apartment"}[task]
    print("PASS: six matched cases; valid budgets/settings; existing challenge scorers; no inference or motion", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated recorded Qwen diagnostic or explicit Qwen/Luna value comparison; no automatic fallback")
    parser.add_argument("--stage", choices=["warmup", "run", "compare", "check"], default="warmup")
    parser.add_argument("--task", choices=[*GOALS, "circle", "object"], default="find")
    parser.add_argument("--provider", choices=["qwen", "luna"], default="qwen")
    parser.add_argument("--design", default="qwen-value-pilot-v1")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.stage == "check":
        check_comparison()
    elif options.stage == "compare":
        asyncio.run(compare(options.output, design_label=options.design))
    else:
        result = asyncio.run(run(options.output, stage=options.stage, task=options.task, provider=options.provider,
            design_label=options.design))
        if result["error"]:
            raise SystemExit(1)