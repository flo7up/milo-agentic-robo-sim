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


def map_provenance(worker):
    mission = getattr(worker, "home_mission", None)
    home = mission.home if mission else None
    if home is None:
        return {"mode": "none", "map_id": None, "revision": None, "sha256": None}
    return {"mode": "saved" if home.revision else "draft", "map_id": home.identity, "name": home.name,
        "revision": home.revision, "environment_id": home.environment_id,
        "sha256": hashlib.sha256(json.dumps(home.document(), sort_keys=True).encode()).hexdigest(),
        "localization": mission.localization["status"], "geometry_updates": not home.saved}


class HomeSessionRecording:
    def __init__(self, worker, request, evidence="operator_session", directory=None):
        self.worker = worker
        self.request = request
        self.directory = Path(directory) if directory else SESSION_RESULTS_ROOT / f"home-session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        self.case_id = worker.challenge.id if worker.challenge else "bench"
        self.recorder = RunRecorder(self.directory / self.case_id / "recording")
        self.done = asyncio.Event()
        self.finalize_lock = asyncio.Lock()
        self.finished = False
        self.executing = True
        self.writer = None
        self.error = None
        self.request_error = None
        self.reason = None
        self.source = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "backend").glob("*.py"))}
        challenge = worker.challenge
        self.manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "mode": "home_mapping", "stage": "spatial_workflow", "evidence": evidence,
            "supervisor_deployment": "None", "cases": [{"case_id": self.case_id, "challenge": self.case_id,
                "environment": challenge.environment if challenge else "standalone",
                "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump() if challenge else {}, sort_keys=True).encode()).hexdigest()}],
            "design": {"label": "spatial-replay-v1", "code_sha256": self.source,
                "source_sha256": hashlib.sha256(json.dumps(self.source, sort_keys=True).encode()).hexdigest()},
            "request": request.model_dump(), "goal": request.action, "session_timeout_s": 1800,
            "note": "Operator spatial workflow; controller arrival is not independently scored scenario success. Preparation before the first home request is outside this recording."}

    async def start(self):
        def attach(sim):
            if self.worker.recorder is not None:
                raise ValueError("Another recording already owns this worker")
            self.manifest["home_map"] = map_provenance(self.worker)
            self.worker.recorder = self.recorder
            self.recorder.capture(self.worker)
        try:
            await asyncio.to_thread(write_recording_json, self.directory / "experiment.json", self.manifest)
            await self.worker.call(attach)
            await asyncio.to_thread(write_recording_json, self.directory / "experiment.json", self.manifest)
            await asyncio.to_thread(self.publish)
        except BaseException:
            await self.worker.call(lambda sim: setattr(self.worker, "recorder", None) if self.worker.recorder is self.recorder else None)
            self.recorder.stream.close()
            raise
        self.writer = asyncio.create_task(self.watch())

    def publish(self, score=None):
        home = self.recorder.home_telemetry or {}
        task = home.get("task") or {}
        report = {"case_id": self.case_id, "challenge": self.case_id, "evidence": self.manifest["evidence"],
            "environment": self.manifest["cases"][0]["environment"], "physics_success": False,
            "verification_eligible": False, "initial_goal": self.request.action, "final_goal": self.request.action,
            "run_status": "finished" if score is not None else "running", "updated_at": time.time(),
            "termination_reason": self.reason or task.get("status") or "running", "phase": self.reason or task.get("status") or "mapping",
            "rendering": self.worker.rendering, "recording_error": self.error,
            "request_error": self.request_error,
            "evaluation_elapsed_s": self.recorder.clock() - self.recorder.started,
            "home_map": home.get("map"), "input_tokens": 0, "output_tokens": 0,
            "recording_scorecard": score or {"complete_recording": False, "samples": self.recorder.sample_sequence,
                "dropped_records": self.recorder.dropped, "spatial_recording": {"initial": None, "final": home.get("coverage"),
                    "final_task": task, "snapshots": self.recorder.home_snapshot_count, "complete": False}}}
        write_recording_json(self.directory / self.case_id / "report.json", report)

    async def watch(self):
        try:
            while not self.done.is_set():
                try:
                    await asyncio.wait_for(self.done.wait(), .5)
                except TimeoutError:
                    pass
                if self.done.is_set():
                    break
                await self.worker.call(lambda sim: self.recorder.capture(self.worker))
                await asyncio.to_thread(self.recorder.flush)
                await asyncio.to_thread(self.publish)
                mission = self.worker.home_mission
                awaiting_room = mission and mission.workflow and mission.workflow["status"] == "awaiting_room_report"
                terminal = self.request.action in {"navigate_to", "explore", "explore_frontier", "start_exploration"} and mission and not mission.active and not awaiting_room
                if not self.executing and (self.worker.sim.cancel.is_set() or terminal or self.recorder.clock() - self.recorder.started >= 1800):
                    await self.finish("interrupted" if self.worker.sim.cancel.is_set() else "recording_limit" if not terminal else None)
                    break
        except (OSError, RuntimeError, ValueError) as error:
            self.error = str(error)
            self.worker.stop()
            self.worker.latest = {**self.worker.latest, "recording_error": self.error}
            await self.finish("recording_error")

    async def finish(self, reason=None):
        self.done.set()
        if self.writer and self.writer is not asyncio.current_task():
            await self.writer
        async with self.finalize_lock:
            if self.finished:
                return
            self.reason = reason
            def detach(sim):
                if self.worker.recorder is self.recorder:
                    self.recorder.capture(self.worker)
                    self.worker.recorder = None
                return self.worker.camera_frames.get((self.worker.latest.get("camera") or {}).get("frame_ref"))
            image = await self.worker.call(detach)
            def finalize():
                try:
                    score = self.recorder.finish({"evidence": self.manifest["evidence"], "real_model": False, "recording_error": self.error})
                    if self.error:
                        score["complete_recording"] = False
                        score["spatial_recording"]["complete"] = False
                    self.publish(score)
                    write_recording_json(self.directory / self.case_id / "recording" / "scorecard.json", score)
                    if image:
                        (self.directory / self.case_id / "terminal.png").write_bytes(image)
                    self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                    self.manifest["source_changed_during_run"] = any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in self.source.items())
                    write_recording_json(self.directory / "experiment.json", self.manifest)
                finally:
                    self.recorder.stream.close()
            try:
                await asyncio.to_thread(finalize)
            except OSError as error:
                self.error = str(error)
                self.worker.stop()
                self.worker.latest = {**self.worker.latest, "recording_error": self.error}
                score = {"complete_recording": False, "dropped_records": self.recorder.dropped,
                    "samples": self.recorder.sample_sequence, "spatial_recording": {"complete": False}}
                self.reason = "recording_error"
                await asyncio.to_thread(self.publish, score)
            finally:
                self.finished = True


def create_session(controller, worker, settings, profile):
    from backend.agent import ConfiguredModel
    evidence = controller.recording_evidence
    if evidence == "real_model" and controller.model_factory is not ConfiguredModel:
        evidence = "unknown"
    if getattr(settings, "mission_local_only", False):
        evidence = "scripted_test"
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
            manifest["home_map"] = map_provenance(worker)
            worker.recorder = recorder
            recorder.capture(worker)

        await worker.call(attach)
        await asyncio.to_thread(write_recording_json, directory / "experiment.json", manifest)
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