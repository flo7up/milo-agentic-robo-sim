import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image
import pybullet as bullet

from backend.challenges import get_challenge
from backend.navigation import NavigationRuntime
from backend.recording import RunRecorder
from backend.worker import SimulationWorker
from scripts.navigation_policy import apply


class PhysicsTestBatch:
    def __init__(self, directory=None, label="scripted-physics-recording-v1"):
        from scripts.evaluate_supervised import ROOT, design_snapshot
        self.root = ROOT
        self.directory = Path(directory) if directory else ROOT / ".runtime/performance" / (
            f"scripted-physics-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}")
        self.directory.mkdir(parents=True, exist_ok=False)
        design = design_snapshot(label)
        design["code_sha256"].update({path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "tests").glob("*.py"))})
        design["source_sha256"] = hashlib.sha256(json.dumps(design["code_sha256"], sort_keys=True).encode()).hexdigest()
        self.manifest = {"schema_version": 2, "experiment_id": str(uuid4()), "design": design,
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "mode": "scripted_physics", "stage": "challenges", "evidence": "scripted_test",
            "camera_history": "not_applicable", "cases": [],
            "note": "Known-geometry pytest rollouts, not model autonomy. Wall time includes recording overhead. Setup may prescribe pose and orientation."}
        self.results = []
        self.write()

    def write(self):
        from scripts.evaluate_supervised import performance_rows
        for name, value in (("experiment.json", self.manifest), ("summary.json", self.results),
                            ("performance.json", performance_rows(self.manifest, self.results))):
            path = self.directory / name
            pending = path.with_suffix(".tmp")
            pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
            pending.replace(path)

    def record(self, sim, case_id, setup):
        if not case_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in case_id):
            raise ValueError("Invalid recorded test case identifier")
        if any(case["case_id"] == case_id for case in self.manifest["cases"]):
            raise ValueError("Recorded test case already exists")
        challenge = sim.challenge_progress.challenge
        self.manifest["cases"].append({"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
            "initial_xy": challenge.initial_xy, "setup": setup,
            "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()})
        self.write()
        return PhysicsTestRecording(self, sim, case_id, setup)

    def finish(self):
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["source_changed_during_run"] = any(not (self.root / name).is_file()
            or hashlib.sha256((self.root / name).read_bytes()).hexdigest() != digest
            for name, digest in self.manifest["design"]["code_sha256"].items())
        self.write()


class PhysicsTestRecording:
    def __init__(self, batch, sim, case_id, setup):
        self.batch, self.sim, self.case_id, self.setup = batch, sim, case_id, setup
        self.directory = batch.directory / case_id
        self.recorder = RunRecorder(self.directory / "recording", context=lambda: {"phase": "scripted_test"})
        self.source = SimpleNamespace(sim=sim, continuous=None, navigation=None, latest={},
            camera_frames={}, spatial_frames={}, stop_revision=0)
        self.previous_tick = sim.on_tick
        self.finalized = False
        sim.on_tick = self.tick
        self.capture(force=True)

    def capture(self, force=False):
        now = time.monotonic()
        self.source.latest["proximity"] = self.sim.proximity_sensors().model_dump()
        if force or now - self.recorder.last_media_at >= .5:
            sequence = self.recorder.sample_sequence + 1
            reference = f"scripted-{sequence}"
            image = self.sim.capture()
            self.source.camera_frames = {reference: image}
            self.source.latest["camera"] = {"frame_ref": reference, "seq": sequence, "simulated_time_s": self.sim.ticks / 240}
            if force:
                self.recorder.last_media_at = -float("inf")
        self.recorder.capture(self.source)
        self.recorder.flush()

    def tick(self):
        if self.previous_tick:
            self.previous_tick()
        self.capture()

    def finish(self, failure=None):
        if self.finalized:
            return
        self.finalized = True
        self.sim.on_tick = self.previous_tick
        self.capture(force=True)
        score = self.recorder.finish({"challenge": self.sim.challenge_progress.challenge.id,
            "real_model": False, "evidence": "scripted_test", "setup": self.setup,
            "source": "Existing pytest control code and real PyBullet mechanics; recording is observational.",
            "test_passed": failure is None, "harness_error": failure.__name__ if failure else None})
        self.directory.joinpath("terminal.png").write_bytes(next(iter(self.source.camera_frames.values())))
        status = self.sim.challenge_status()
        report = {"case_id": self.case_id, "challenge": status["id"], "challenge_status": status,
            "environment": status["environment"], "evidence": "scripted_test", "real_luna": False,
            "scripted_known_map_reference": True, "physics_success": score["final_physics_success"],
            "test_passed": failure is None, "error": failure.__name__ if failure else None,
            "termination_reason": "test_failed" if failure else "completed" if score["final_physics_success"] else "test_finished",
            "evaluation_elapsed_s": score["elapsed_wall_s"], "rendering": self.sim.rendering,
            "recording_scorecard": score, "setup": self.setup}
        self.directory.joinpath("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self.batch.results.append(report)
        self.batch.write()


async def record_scenario(identifier, directory):
    worker = SimulationWorker(challenge=get_challenge(identifier), pace=True)
    recorder = None
    try:
        await asyncio.wrap_future(worker.ready)

        def execute(sim):
            nonlocal recorder
            runtime = NavigationRuntime()
            worker.navigation = runtime
            recorder = RunRecorder(directory, context=lambda: {"phase": "scripted_test"})
            worker.recorder = recorder
            publish = sim.on_tick
            last_render = -1000
            spectator = None

            def capture():
                nonlocal last_render, spectator
                publish()
                if sim.ticks - last_render >= 24:
                    view = bullet.computeViewMatrix([2.9, -4., 2.8], [.9, 0, .65], [0, 0, 1])
                    projection = bullet.computeProjectionMatrixFOV(52, 4 / 3, .05, 20)
                    pixels = bullet.getCameraImage(440, 330, view, projection,
                        renderer=bullet.ER_TINY_RENDERER, physicsClientId=sim.client)[2]
                    image = Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(330, 440, 4)[:, :, :3])
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    spectator = {"path": f"media/spectator-{sim.ticks}.png", "simulated_s": sim.ticks / 240}
                    (recorder.directory / spectator["path"]).write_bytes(buffer.getvalue())
                    last_render = sim.ticks
                if recorder.pending:
                    recorder.pending[-1][0]["spectator"] = spectator
                recorder.flush()

            sim.on_tick = capture
            capture()
            apply(runtime, sim, "begin_local_subgoal", {"goal": "Scripted safety approach"})
            for _ in range(240):
                if runtime.status in {"failed", "cancelled"}:
                    break
                if identifier == "bench" and sim.odometry[0] < -.2:
                    break
                if sum(entry[1] for entry in runtime.buffer) < 120:
                    apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive",
                        "linear_mps": -.1 if identifier == "bench" else .15, "angular_radps": 0., "duration_s": 1.}]})
                runtime.tick(sim)
            if identifier == "pedestrian_crossing":
                if "CLEARANCE_STOP" not in runtime.reason or sim.pedestrian_crossing["contact"]:
                    raise AssertionError("Crossing must cause precontact clearance braking")
                sim.cancel.clear()
                sim.hold_current()
                sim._ticks(4 * 240)
                if not sim.pedestrian_crossing["yielded"] or sim.pedestrian_crossing["contact"]:
                    raise AssertionError("Robot must yield through the entire crossing without contact")
                apply(runtime, sim, "begin_local_subgoal", {"goal": "Scripted fresh authorization after crossing"})
                for _ in range(280):
                    if sim.odometry[0] >= 1.8 or runtime.status == "failed":
                        break
                    if sum(entry[1] for entry in runtime.buffer) < 120:
                        apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive",
                            "linear_mps": .15, "angular_radps": 0., "duration_s": 1.}]})
                    runtime.tick(sim)
            runtime.cancel(sim, "Scripted recording finished")
            sim.hold_current()
            sim._ticks(240)
            capture()
            if identifier == "pedestrian_crossing" and sim.challenge_status()["status"] != "completed":
                raise AssertionError("Scripted crossing did not finish physically parked")
            if identifier == "bench" and sim.odometry[0] >= -.18:
                raise AssertionError("Bench smoke test did not move")
            worker.recorder = None
            return recorder.finish({"challenge": identifier,
                "label": "Pedestrian Crossing / scripted safety test" if identifier == "pedestrian_crossing" else "Practice Bench / scripted smoke test",
                "real_model": False, "source": "Scripted buffered motion through the validated physics worker; harness authorizes a new route after yielding.",
                "spectator_interval_simulated_s": .1})

        return await worker.call(execute)
    finally:
        worker.recorder = None
        await worker.close()
        if recorder and not recorder.stream.closed:
            recorder.finish({"challenge": identifier, "real_model": False, "harness_error": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record isolated scripted physics tests; no model inference or live-session control")
    parser.add_argument("--scenario", choices=["bench", "pedestrian_crossing"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    print(asyncio.run(record_scenario(options.scenario, options.output)))