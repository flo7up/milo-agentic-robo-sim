import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from backend.recording import RunRecorder, write_recording_json
from backend.experiment_variants import variant_snapshot


ROOT = Path(__file__).resolve().parents[1]
SESSION_RESULTS_ROOT = ROOT / ".runtime/performance"


def create_session(controller, worker, settings, profile):
    from backend.agent import ConfiguredModel
    evidence = controller.recording_evidence
    if evidence == "real_model" and controller.model_factory is not ConfiguredModel:
        evidence = "unknown"
    directory = SESSION_RESULTS_ROOT / f"browser-session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    challenge = worker.challenge
    case_id = challenge.id if challenge else "bench"
    case = {"case_id": case_id, "challenge": case_id, "environment": challenge.environment if challenge else "standalone",
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump() if challenge else {}, sort_keys=True).encode()).hexdigest()}
    budget_s = round(max(0., controller.session_deadline - time.monotonic())) if getattr(controller, "session_deadline", None) else None
    effective = {"max_turns": controller.state.get("max_turns", settings.max_turns), "session_timeout_s": budget_s,
        "camera_history": "enabled" if settings.execution_mode == "luna_continuous" else "not_applicable"}
    if settings.execution_mode in {"luna_navigation", "local_navigation"}:
        from backend.local_navigation import CHECKPOINT
        effective["local_checkpoint"] = CHECKPOINT.parent.name
    variant = variant_snapshot(settings, profile, ROOT, effective)
    hashes = variant["code_sha256"]
    recorder = RunRecorder(directory / case_id / "recording", context=lambda: controller.state)
    from backend.reference_paths import reference_path
    reference = reference_path(challenge) if challenge is not None and settings.goal == challenge.goal else None
    if reference is not None:
        (recorder.directory / "reference-path.json").write_text(json.dumps(reference), encoding="utf-8")
    manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None, "mode": settings.execution_mode, "stage": "challenges", "cases": [case],
        "evidence": evidence, "supervisor_deployment": profile.deployment,
        "reasoning": settings.reasoning, "camera_history": "enabled" if settings.execution_mode == "luna_continuous" else "not_applicable",
        "design": {"label": variant["architecture"]["version_key"], "code_sha256": hashes,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()},
        "architecture": variant["architecture"], "model_variant": variant["model_variant"], "variant_id": variant["variant_id"],
        "session_id": controller.state["session_id"], "goal": settings.goal,
        "ai_generated_routes": getattr(settings, "ai_generated_routes", False),
        "session_timeout_s": budget_s,
        "command_revision": getattr(controller, "command_revision", None),
        "reference_path_version": reference["version"] if reference else None,
        "note": "Interactive browser session, not a controlled benchmark. Recording begins at control start, not episode reset."}
    try:
        write_recording_json(directory / "experiment.json", manifest)
    except OSError:
        recorder.stream.close()
        raise
    return directory, recorder, manifest


def publish_live_report(directory, recorder, manifest, state, challenge):
    case = manifest["cases"][0]
    report = {"case_id": case["case_id"], "challenge": case["challenge"], "environment": case["environment"],
        "challenge_status": challenge, "evidence": manifest["evidence"], "run_status": "running", "updated_at": time.time(),
        "session_id": manifest["session_id"], "verification_eligible": False,
        "initial_goal": manifest.get("goal"), "final_goal": state.get("goal"),
        "phase": state.get("phase"), "termination_reason": "running", "physics_success": None,
        "input_tokens": state.get("input_tokens"), "output_tokens": state.get("output_tokens"),
        "supervisor_turns": state.get("turns"), "evaluation_elapsed_s": recorder.clock() - recorder.started,
        "recording_scorecard": {"complete_recording": False, "dropped_records": recorder.dropped, "samples": recorder.sample_sequence}}
    path = directory / case["case_id"] / "report.json"
    write_recording_json(path, report)


async def run_recorded_session(controller, worker, settings, profile, stop_revision):
    directory = recorder = manifest = None
    writer = None
    done = asyncio.Event()
    initial_status = (worker.latest.get("challenge") or {}).get("status")
    try:
        directory, recorder, manifest = create_session(controller, worker, settings, profile)

        def attach(sim):
            worker.recorder = recorder
            recorder.capture(worker)

        await worker.call(attach)
        await asyncio.to_thread(publish_live_report, directory, recorder, manifest, dict(controller.state), worker.latest.get("challenge"))

        async def flush_pending():
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), .5)
                except TimeoutError:
                    try:
                        await asyncio.to_thread(recorder.flush)
                        await asyncio.to_thread(publish_live_report, directory, recorder, manifest, dict(controller.state), worker.latest.get("challenge"))
                    except OSError:
                        controller.interrupt("Recording write failed")
                        raise

        writer = asyncio.create_task(flush_pending())
        await controller._run_controller(worker, settings, profile, stop_revision)
    finally:
        done.set()
        if recorder:
            finalizing = asyncio.create_task(finish_session(controller, worker, settings, directory, recorder, manifest, initial_status, writer))
            try:
                await asyncio.shield(finalizing)
            except asyncio.CancelledError:
                await finalizing
                raise


async def finish_session(controller, worker, settings, directory, recorder, manifest, initial_status, writer):
    writer_error = None
    if writer:
        try:
            await writer
        except OSError as error:
            writer_error = error

    def detach(sim):
        if worker.recorder is recorder:
            worker.recorder = None
            recorder.capture(worker)
        image = worker.camera_frames.get((worker.latest.get("camera") or {}).get("frame_ref"))
        return worker.latest.get("challenge"), image

    challenge, image = await worker.call(detach)
    state = dict(controller.state)
    trace = controller.trace()
    final_revision = getattr(controller, "command_revision", None)
    task_matches = bool(challenge and settings.goal == challenge.get("goal") and state.get("goal") == settings.goal
        and manifest.get("command_revision") == final_revision)

    def finish():
        try:
            score = recorder.finish({"real_model": manifest["evidence"] == "real_model",
                "evidence": manifest["evidence"], "challenge": manifest["cases"][0]["challenge"],
                "session_id": manifest["session_id"], "goal": settings.goal, "initial_physics_status": initial_status})
            report = {"case_id": manifest["cases"][0]["case_id"], "challenge": manifest["cases"][0]["challenge"],
                "challenge_status": challenge, "environment": manifest["cases"][0]["environment"],
                "evidence": manifest["evidence"], "real_luna": manifest["evidence"] == "real_model" and state["model_id"] == "luna",
                "physics_success": score["final_physics_success"],
                "verification_eligible": task_matches and initial_status != "completed" and writer_error is None,
                "initial_goal": settings.goal, "final_goal": state.get("goal"),
                "ai_generated_routes": manifest.get("ai_generated_routes", False), "final_command_revision": final_revision,
                "verification_note": "Original scenario task and command unchanged" if task_matches else "Custom or revised task lacks a matching independent scorer",
                "termination_reason": (state.get("outcome") or {}).get("kind", state["phase"]),
                "phase": state["phase"], "input_tokens": state["input_tokens"], "output_tokens": state["output_tokens"],
                "session_id": manifest["session_id"], "run_status": "finished", "updated_at": time.time(),
                "recording_error": str(writer_error) if writer_error else None,
                "supervisor_turns": state["turns"], "evaluation_elapsed_s": score["elapsed_wall_s"],
                "rendering": worker.rendering, "recording_scorecard": score}
            case_directory = directory / report["case_id"]
            write_recording_json(case_directory / "report.json", report)
            (case_directory / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
            if image:
                (case_directory / "terminal.png").write_bytes(image)
            write_recording_json(directory / "summary.json", [report])
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest["source_changed_during_run"] = any(not (ROOT / name).is_file()
                or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                for name, digest in manifest["design"]["code_sha256"].items())
            write_recording_json(directory / "experiment.json", manifest)
        finally:
            recorder.stream.close()

    await asyncio.to_thread(finish)
    if writer_error:
        raise writer_error