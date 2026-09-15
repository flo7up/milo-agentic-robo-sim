import argparse
import asyncio
from collections import Counter
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import time

import numpy as np
from dotenv import load_dotenv

from backend.agent import AgentController, AgentStart, ConfiguredModel, FoundryConfig, NavigationEvaluationBudget
from backend.challenges import get_challenge
from backend.home_mapping import MapStore
from backend.home_mission import HomeMission
from backend.simulation import TIMESTEP
from backend import session_recording
from backend.worker import SimulationWorker
from scripts.benchmark_household import PhysicsMeasurements, changed_sources
from scripts.evaluate_supervised import ROOT, design_snapshot, workload_snapshot


GOAL = "Explore the home using observed safe routes. Find and identify the bathroom from visible fixtures, then enter it and stop safely. Do not claim arrival from a room label alone."
SETTINGS = {"mission_budget_s": 300., "reasoning": "low", "max_model_requests": 30,
    "max_model_tokens": 100000, "feedback_interval_s": .25, "images_per_request": 2,
    "context_tokens": 8192, "max_turns": 80, "unified_mission": True, "map_context": True,
    "execution_mode": "luna_continuous", "model_id": "luna", "compact_arms": True}


ROOM_PROTOCOL = {
    "schema": "mission-review-room-evaluation-v1", "evaluator_only": True,
    "results": {name: {"value": "boolean or null", "status": "measured or not_measured", "evidence": "separate source records"}
                for name in ("discovery", "correct_identification", "arrival")},
    "discovery": "Positive sampled fixture pixels in a captured head RGB image, first-hit Bullet rays in the calibrated frustum AND paired sensor depth agreement. Initial visibility must be measured; no proximity/room-label/wall-through inference. Initially visible is reported separately from newly discovered.",
    "fixtures": {"toilet": ["toilet_"], "bathtub": ["tub_"], "basin": ["bathroom_basin", "basin_rim_"]},
    "minimum_pixels": 3, "pixel_stride": 4, "capture_interval_wall_s": .5, "depth_tolerance_m": .03,
    "initial_frame": "Evaluator-only head RGB-D capture before starting control, separately named room-initial.png and excluded from report matching; subsequent samples use production capture sequences. Initial render overhead is recorded separately.",
    "source_pairing": "Review observation seq -> selected spatial seq -> exact PNG SHA256; run/epoch/calibration/sim-time agreement and <=15 s report age. Every link is evaluator-side only.",
    "identification": "bounded-current-report-v1: only report_observation.evidence_text from completed guide_mission responses. Explicit current bathroom assertion plus currently seen toilet/bathtub/basin; every mentioned fixture must match independently verified visible pixels of the exact exposed source image. Basin alone is insufficient. Goals, plans, reason and history never count. Unsupported language, negation, hypothetical or future claims are unknown, not semantic failure.",
    "arrival": "Unmodified kitchen_bathroom challenge_status completed, continuously observed at <=0.051 simulated-second gaps for >=0.5 simulated seconds with no active contacts; no mission receipt or semantic report required. Whole-run contacts reported separately.",
    "limits": ["Sparse pixel-center geometric visibility, not robust semantic segmentation; small fixtures may be missed.",
               "Bullet collision surfaces and enhanced rounded visuals differ; paired depth agreement is required but not a perfect visual instance mask.",
               "Only sampled captures are evaluated; unseen intervals do not establish absence. Sampling overhead is measured, not assumed free.",
               "Keyword rubric is deliberately narrow, uncalibrated and not robust semantic grading; unknown is not false.",
               "Original baseline lacks this exact paired visibility/dwell instrumentation; do not reconstruct from current scene or transfer new metrics."]}


def fixture_group(name):
    return next((group for group, prefixes in ROOM_PROTOCOL["fixtures"].items() if name.startswith(tuple(prefixes))), None)


class RoomEvaluator:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.began = clock()
        self.last_capture = -math.inf
        self.initial_sequence = None
        self.pending = OrderedDict()
        self.frames = []
        self.images = {}
        self.errors = []
        self.arrival_samples = []
        self.review_sources = []
        self.capture_timings = []
        self.initial_capture_s = None

    def capture_initial(self, sim):
        started = time.perf_counter()
        original_pose = sim.camera_pose
        captured = False

        def measured_pose():
            nonlocal captured
            eye, matrix = original_pose()
            if not captured:
                captured = True
                self.capture(sim, 1, eye, matrix, initial=True)
            return eye, matrix

        sim.camera_pose = measured_pose
        try:
            sensor, image = sim.capture_spatial(1)
            self.receive(sensor, image)
        except Exception as error:
            self.errors.append({"stage": "initial_capture", "type": type(error).__name__})
        finally:
            sim.camera_pose = original_pose
            self.initial_capture_s = time.perf_counter()-started

    def capture(self, sim, sequence, eye, matrix, *, initial=False):
        import pybullet as bullet
        from backend.spatial import calibration
        now = self.clock()
        if now - self.last_capture < ROOM_PROTOCOL["capture_interval_wall_s"]:
            return
        self.last_capture = now
        if initial:
            self.initial_sequence = sequence
        started = time.perf_counter()
        intrinsics = calibration(160, 120)
        targets = {item["id"]: (item["name"], fixture_group(item["name"])) for item in sim.objects
                   if fixture_group(item["name"])}
        valid = bool(sim.challenge_progress and sim.challenge_progress.challenge.id == "kitchen_bathroom" and targets)
        record = {"sequence": sequence, "initial": initial, "run_id": sim.run_id, "episode_epoch": sim.epoch,
            "wall_s": now-self.began, "simulated_s": sim.ticks*TIMESTEP, "annotation_available": valid,
            "camera_eye": np.asarray(eye).tolist(), "camera_rotation": np.asarray(matrix).tolist(),
            "calibration": intrinsics.model_dump(), "rays": []}
        if valid:
            pixels = [(column, row) for row in range(8, 112, ROOM_PROTOCOL["pixel_stride"])
                      for column in range(8, 152, ROOM_PROTOCOL["pixel_stride"])]
            directions = np.asarray([[1., -(column+.5-intrinsics.cx)/intrinsics.fx,
                                       -(row+.5-intrinsics.cy)/intrinsics.fy] for column, row in pixels]) @ matrix.T
            starts = eye + directions * intrinsics.near_m
            ends = eye + directions * intrinsics.usable_range_m
            hits = bullet.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=sim.client)
            for pixel, hit in zip(pixels, hits):
                if hit[0] in targets:
                    name, group = targets[hit[0]]
                    record["rays"].append({"pixel": pixel, "fixture": name, "group": group,
                        "axial_depth_m": intrinsics.near_m + hit[2]*(intrinsics.usable_range_m-intrinsics.near_m)})
            record["tested_pixels"] = len(pixels)
        record["raycast_s"] = time.perf_counter()-started
        self.capture_timings.append(record["raycast_s"])
        self.pending[sequence] = record
        while len(self.pending) > 64:
            self.pending.popitem(last=False)

    def receive(self, sensor, image):
        record = self.pending.pop(sensor.sequence, None)
        if record is None:
            return
        started = time.perf_counter()
        from PIL import Image
        with Image.open(BytesIO(image)) as decoded:
            image_size = decoded.size
        paired = (sensor.run_id == record["run_id"] and sensor.episode_epoch == record["episode_epoch"]
                  and sensor.simulated_time_s == record["simulated_s"]
                  and sensor.calibration.model_dump() == record["calibration"] and image_size == (160, 120)
                  and abs(sensor.captured_at - (record["wall_s"] + self.began)) <= .1)
        depth = np.asarray(sensor.depth_m, dtype=float).reshape(sensor.calibration.height, sensor.calibration.width)
        matches = []
        for ray in record.pop("rays"):
            column, row = ray["pixel"]
            measured = depth[row, column]
            if paired and math.isfinite(measured) and abs(measured-ray["axial_depth_m"]) <= ROOM_PROTOCOL["depth_tolerance_m"]:
                matches.append({**ray, "sensor_depth_m": float(measured)})
        counts = Counter(match["group"] for match in matches)
        digest = hashlib.sha256(image).hexdigest()
        record.update(paired=paired, image_sha256=digest,
            image_file="room-initial.png" if record["initial"] else f"room-frame-{sensor.sequence:06d}.png",
            captured_at=sensor.captured_at, head_rad=list(sensor.head_rad), odometry_m_rad=list(sensor.odometry_m_rad),
            pixel_matches=matches, visible_groups=[group for group, count in counts.items() if count >= ROOM_PROTOCOL["minimum_pixels"]],
            association_s=time.perf_counter()-started)
        self.frames.append(record)
        self.images[record["image_file"]] = image

    def sample_arrival(self, sim, physical, wall_s):
        status = sim.challenge_status()
        applicable = bool(sim.challenge_progress and sim.challenge_progress.challenge.id == "kitchen_bathroom")
        self.arrival_samples.append({"wall_s": wall_s, "simulated_s": sim.ticks/240.,
            "eligible": status.get("status") == "completed" if applicable and status else None,
            "contact": bool(physical.contact_active), "contact_episodes": physical.contacts})

    def artifact(self):
        return {"protocol": ROOM_PROTOCOL, "frames": self.frames, "arrival_samples": self.arrival_samples,
            "errors": self.errors, "initial_sequence": self.initial_sequence, "review_sources": self.review_sources,
            "timing": {"raycast_total_s": sum(self.capture_timings),
                "initial_capture_s": self.initial_capture_s,
                "raycast_max_s": max(self.capture_timings, default=0.), "selected_captures": len(self.capture_timings),
                "association_total_s": sum(frame["association_s"] for frame in self.frames),
                "sample_count": len(self.frames), "unevaluated_pending": len(self.pending),
                "scope": "Ray timing includes selected captures even if subsequently rejected/expired. Visibility covers accepted selected captures only; between-frame visibility is not measured."}}


class EvaluationWorker(SimulationWorker):
    room_evaluator = None

    async def mission_feedback(self, mission=None):
        sensor, image, observation = await super().mission_feedback(mission)
        if self.room_evaluator is not None:
            self.room_evaluator.review_sources.append({"observation_seq": observation.seq, "spatial_sequence": sensor.sequence,
                "run_id": sensor.run_id, "episode_epoch": sensor.episode_epoch,
                "image_sha256": hashlib.sha256(image).hexdigest(), "reviewed_at_unix_s": time.time(),
                "sensor_age_s": time.monotonic()-sensor.captured_at})
        return sensor, image, observation

    def _sample_spatial(self, force=False):
        evaluator = self.room_evaluator
        if evaluator is None:
            return super()._sample_spatial(force=force)
        original_pose = self.sim.camera_pose

        def measured_pose():
            eye, matrix = original_pose()
            try:
                evaluator.capture(self.sim, self.spatial_sequence, eye, matrix)
            except Exception as error:
                evaluator.errors.append({"stage": "capture", "type": type(error).__name__})
            return eye, matrix

        self.sim.camera_pose = measured_pose
        try:
            return super()._sample_spatial(force=force)
        finally:
            self.sim.camera_pose = original_pose

    def _receive_spatial(self, wait=False):
        result = super()._receive_spatial(wait=wait)
        if self.room_evaluator is not None:
            for sensor, image, _ in list(self.spatial_frames.values()):
                try:
                    self.room_evaluator.receive(sensor, image)
                except Exception as error:
                    self.room_evaluator.errors.append({"stage": "receive", "type": type(error).__name__})
        return result


def assess_report(text, frame):
    lowered = text.lower().strip()
    uncertain = re.search(r"\b(no|not|isn't|can't|cannot|without|maybe|possibly|may|might|could|would|should|will|find|search|goal|target|if|earlier|previously)\b", lowered)
    room_claim = bool(re.search(r"\b(this is|i am in|i'm in|i see|i can see) (?:a |the )?bathroom\b", lowered))
    mentioned = [group for group, pattern in {"toilet": r"\btoilet\b", "bathtub": r"\b(bathtub|tub)\b", "basin": r"\b(basin|sink)\b"}.items()
                 if re.search(pattern, lowered)]
    current = bool(re.search(r"\b(see|visible|in view)\b", lowered))
    semantic = True if room_claim and current and mentioned and not uncertain else None
    valid = bool(frame and frame.get("paired") and frame.get("annotation_available") and frame.get("image_verified"))
    match = bool(valid and set(mentioned) <= set(frame["visible_groups"]) and set(mentioned) & {"toilet", "bathtub"})
    return {"rubric": "bounded-current-report-v1", "text": text, "semantic_claim": semantic,
            "mentioned_groups": mentioned, "evidence_valid": valid,
            "report_fixture_match": match if semantic and valid else None,
            "correct_identification": match if semantic and valid else None,
            "limit": "Vocabulary/evidence match only, not calibrated semantic understanding; unsupported assertions remain unknown."}


def room_results(artifact=None, trace=None, source=None):
    missing = {"value": None, "status": "not_measured", "reason": "No exact paired evaluator evidence recorded."}
    result = {name: dict(missing) for name in ("discovery", "correct_identification", "arrival")}
    result["protocol"] = ROOM_PROTOCOL
    if not artifact or artifact.get("protocol", {}).get("schema") != ROOM_PROTOCOL["schema"]:
        return result
    frames = []
    for original in artifact.get("frames", []):
        frame = dict(original)
        path = source / frame["image_file"] if source else None
        frame["image_verified"] = bool(path and path.parent.resolve() == source.resolve() and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == frame["image_sha256"])
        frames.append(frame)
    valid = [frame for frame in frames if frame.get("paired") and frame.get("annotation_available") and frame["image_verified"]]
    initial = next((frame for frame in valid if frame.get("initial")), None)
    visible = [frame for frame in valid if frame["visible_groups"]]
    if initial is not None:
        result["discovery"] = {"value": bool(visible), "status": "measured", "scope": "sampled head frames only",
            "initial_visible": bool(initial["visible_groups"]),
            "newly_discovered": bool(visible) and not bool(initial["visible_groups"]),
            "first_visible_sequence": visible[0]["sequence"] if visible else None,
            "first_visible_wall_s": visible[0]["wall_s"] if visible else None}
    else:
        result["discovery"].update(reason="Initial visibility unavailable; later exposure does not establish discovery.",
            later_visible_sequences=[frame["sequence"] for frame in visible])
    reports, feedback = [], None
    for event in (trace or {}).get("events", []):
        payload = event.get("payload", {})
        if event.get("kind") == "feedback":
            feedback = event
        if event.get("kind") != "response" or payload.get("status") != "completed":
            continue
        for call in payload.get("calls", []):
            if call.get("name") != "guide_mission":
                continue
            try:
                decision = json.loads(call["arguments"])
            except (ValueError, TypeError, KeyError):
                continue
            if decision.get("action") != "report_observation" or not isinstance(decision.get("evidence_text"), str):
                continue
            matched = None
            if feedback and source and "current_head" == (feedback["payload"].get("image_roles") or [None])[0]:
                path = source / f"frame-{feedback['id']:04d}.png"
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                observation = feedback["payload"].get("observation", {})
                review = next((item for item in artifact.get("review_sources", [])
                    if item["observation_seq"] == observation.get("seq") and item["image_sha256"] == digest
                    and item["run_id"] == observation.get("run_id") and item["episode_epoch"] == observation.get("episode_epoch")), None)
                fresh = bool(review and 0 <= review["sensor_age_s"] <= 1.
                    and 0 <= event.get("timestamp", -math.inf)-review["reviewed_at_unix_s"]+review["sensor_age_s"] <= 15.)
                matched = next((frame for frame in reversed(valid) if fresh and not frame.get("initial")
                    and frame["sequence"] == review["spatial_sequence"]
                    and frame["image_sha256"] == digest
                    and frame["run_id"] == observation.get("run_id") and frame["episode_epoch"] == observation.get("episode_epoch")
                    and frame["simulated_s"] == observation.get("simulated_time_s")
                    and frame.get("odometry_m_rad") == observation.get("odometry_m_rad")
                    and frame.get("head_rad") == observation.get("head_rad")), None)
            reports.append({**assess_report(decision["evidence_text"], matched), "response_event_id": event["id"],
                "feedback_event_id": feedback["id"] if feedback else None,
                "source_sequence": matched["sequence"] if matched else None,
                "image_sha256": matched["image_sha256"] if matched else None})
    measured_reports = [report for report in reports if report["correct_identification"] is not None]
    if measured_reports:
        result["correct_identification"] = {"value": any(report["correct_identification"] for report in measured_reports),
            "status": "measured", "scope": "bounded-current-report-v1 only; not robust semantic grading"}
    result["reports"] = reports
    previous, dwell, maximum, first = None, 0., 0., None
    arrivals = artifact.get("arrival_samples", [])
    for sample in arrivals:
        eligible = sample.get("eligible") is True and sample.get("contact") is False
        elapsed = sample["simulated_s"] - previous["simulated_s"] if previous else 0.
        continuous = previous and previous.get("eligible") is True and previous.get("contact") is False and 0 <= elapsed <= .051
        dwell = dwell + elapsed if eligible and continuous else 0.
        maximum = max(maximum, dwell)
        if dwell >= .5-1e-9 and first is None:
            first = sample["wall_s"]
        previous = sample
    if any(sample.get("eligible") is not None for sample in arrivals):
        result["arrival"] = {"value": first is not None, "status": "measured", "first_wall_s": first,
            "maximum_contact_free_dwell_sim_s": maximum, "final_dwell_sim_s": dwell,
            "contact_episodes": max((sample.get("contact_episodes", 0) for sample in arrivals), default=0),
            "scope": "Physical scorer and contact-free dwell, independent of receipts and identification."}
    result["sampling"] = artifact.get("timing")
    result["errors"] = artifact.get("errors", [])
    return result


def analyze(source, output, *, overwrite=False):
    source, output = source.resolve(), output.resolve()
    target = output / "analysis.json"
    if source == output or source in output.parents:
        raise ValueError("Analysis output must be outside the immutable source run")
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if not overwrite or old.get("schema") != "mission-review-analysis-v1" or old.get("source") != str(source):
            raise FileExistsError("Refusing to overwrite an existing or unowned analysis")
    raw = {}
    hashes = {}
    for name in ("plan", "measurements", "inference", "trace", "result", "room_evaluation"):
        path = source / f"{name}.json"
        if name == "room_evaluation" and not path.exists():
            continue
        data = path.read_bytes()
        hashes[path.name] = hashlib.sha256(data).hexdigest()
        raw[name] = json.loads(data)
    state = raw["result"].get("controller", {})
    report = {"schema": "mission-review-analysis-v1", "source": str(source), "raw_sha256": hashes,
        "evidence": raw["plan"].get("evidence"), "design": raw["plan"].get("design"),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": summarize(raw["measurements"], raw["inference"]),
        "room_results": room_results(raw.get("room_evaluation"), raw["trace"], source),
        "raw_outcome": {key: raw["result"].get(key) for key in ("travel_m", "contact_episodes", "task_completion", "error", "source_unchanged")},
        "controller_outcome": state.get("outcome"), "inference_budget": state.get("inference_budget"),
        "input_tokens": state.get("input_tokens"), "output_tokens": state.get("output_tokens"),
        "request_windows": len(raw["inference"]),
        "completed_response_events": sum(event.get("kind") == "response" and event.get("payload", {}).get("status") == "completed"
                                         for event in raw["trace"].get("events", [])),
        "analysis_only": True, "model_calls": 0,
        "limitations": ["Raw baseline results retained, including obsolete velocity-based durations. This analysis supersedes timing only.",
            "Controller phase completed is not milestone completion; retain limited/error/blocked outcomes.",
            "No offline scene reconstruction: original baseline room milestones are not measured.",
            "New sampler changes harness overhead; matched candidate settings do not make old visibility metrics comparable."]}
    output.mkdir(parents=True, exist_ok=True)
    with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    snapshot = output / f"probe_mission_review-{report['analyzer_sha256']}.py"
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
        raise RuntimeError("Raw input changed during analysis")
    return report


def inference_windows(inference):
    windows = []
    for item in sorted(inference, key=lambda item: item["start_s"]):
        start, end = float(item["start_s"]), float(item["end_s"])
        if not math.isfinite(start + end) or end < start:
            raise ValueError("Invalid inference interval")
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(end, windows[-1][1])
        else:
            windows.append([start, end])
    return windows


def interval_overlap(start, end, windows):
    return sum(max(0., min(end, finish) - max(start, begin)) for begin, finish in windows)


def stop_category(sample, authorized_before, during_inference):
    reason = str(sample.get("stop_reason", "")).lower()
    if any(word in reason for word in ("collision", "contact", "clearance")):
        return "collision_related_guard"
    if any(word in reason for word in ("checkpoint", "review_interval", "periodic_review")):
        return "periodic_checkpoint"
    if during_inference and not sample.get("route_active"):
        return "inference_without_active_authorization" if authorized_before else "initial_authorization"
    return "other_stop"


def summarize(samples, inference, *, gap_limit_s=.5):
    durations = Counter()
    categories = Counter()
    windows = inference_windows(inference)
    authorized_before = False
    stops = []
    gaps = []
    for previous, current in zip(samples, samples[1:]):
        start, end = previous["wall_s"], current["wall_s"]
        elapsed = end - start
        if elapsed < 0:
            raise ValueError("Measurements must be time ordered")
        if not elapsed:
            continue
        overlap = interval_overlap(start, end, windows)
        distance = math.dist(previous["pose"][:2], current["pose"][:2])
        angle = abs(math.atan2(math.sin(current["pose"][2] - previous["pose"][2]), math.cos(current["pose"][2] - previous["pose"][2])))
        translating = distance > .015 * elapsed
        moving = translating or angle > .04 * elapsed
        durations["moving_s" if moving else "stationary_s"] += elapsed
        durations["translating_s"] += elapsed if translating else 0.
        durations["moving_during_inference_s" if moving else "stationary_during_inference_s"] += overlap
        durations["pose_displacement_m"] += distance
        uncertain = elapsed > gap_limit_s
        if uncertain:
            gaps.append({"start_s": start, "end_s": end, "duration_s": elapsed, "inference_overlap_s": overlap,
                         "pose_changed": distance > 1e-6 or angle > 1e-6})
            durations["uncertain_s"] += elapsed
            durations["uncertain_inference_s"] += overlap
        else:
            durations["well_sampled_moving_s" if moving else "well_sampled_stationary_s"] += elapsed
        if not moving:
            reason = previous.get("stop_reason")
            category = stop_category(previous, authorized_before, bool(overlap))
            categories[category] += overlap
            if stops and stops[-1]["reason"] == reason and stops[-1]["category"] == category and abs(stops[-1]["end_s"] - previous["wall_s"]) < .001:
                stops[-1]["end_s"] = current["wall_s"]
                stops[-1]["inference_overlap_s"] += overlap
            else:
                stops.append({"start_s": start, "end_s": end, "reason": reason, "category": category,
                              "inference_overlap_s": overlap})
        authorized_before |= bool(previous.get("route_active"))
    known = [sample["known_cells"] for sample in samples if sample["known_cells"] is not None]
    free = [sample["free_cells"] for sample in samples if sample.get("free_cells") is not None]
    covered = interval_overlap(samples[0]["wall_s"], samples[-1]["wall_s"], windows) if samples else 0.
    inference_s = sum(end - start for start, end in windows)
    return {**{key: float(durations[key]) for key in ("moving_s", "stationary_s", "translating_s",
        "moving_during_inference_s", "stationary_during_inference_s", "pose_displacement_m", "uncertain_s",
        "uncertain_inference_s", "well_sampled_moving_s", "well_sampled_stationary_s")},
        "inference_s": inference_s, "inference_outside_measurements_s": max(0., inference_s - covered),
        "stationary_inference_by_category_s": dict(categories), "sampling_gaps": gaps,
        "timing_protocol": {"gap_limit_s": gap_limit_s, "translation_threshold_mps": .015, "rotation_threshold_radps": .04,
            "overlap": "Exact intersection with union of recorded request windows; not sampled pending flags.",
            "motion": "Interval estimates from pose displacement, never frozen Bullet velocity. Endpoint changes prove some motion, not its duration or exact subinterval. Large gaps remain uncertain; no unnecessary-stop inference.",
            "collision_related_guard": "Reason category only; a clearance stop does not prove physical contact."},
        "newly_observed_area_m2": (known[-1] - known[0]) * .01 if known else None,
        "net_known_area_m2": (known[-1] - known[0]) * .01 if known else None,
        "net_free_area_m2": (free[-1] - free[0]) * .01 if free else None,
        "area_protocol": "Known includes occupied and free cells; net last-minus-first at 0.1 m resolution, not cumulative unique discovery. Legacy newly_observed_area_m2 aliases net_known_area_m2.",
        "stationary_intervals": [{**stop, "duration_s": stop["end_s"] - stop["start_s"]}
            for stop in stops if stop["end_s"] - stop["start_s"] >= .25],
        "max_sampling_gap_s": max((current["wall_s"] - previous["wall_s"] for previous, current in zip(samples, samples[1:])), default=0.)}


def self_check():
    common = {"linear_speed_mps": 0., "angular_speed_radps": 0., "inference_pending": True,
        "known_cells": 10, "stop_reason": "inference_pending_no_route"}
    samples = [{**common, "wall_s": 0., "pose": [0., 0., 0.], "linear_speed_mps": .2},
        {**common, "wall_s": 2., "pose": [0., 0., 0.], "linear_speed_mps": .1},
        {**common, "wall_s": 5., "pose": [.3, 0., 0.], "known_cells": 30}]
    result = summarize(samples, [{"start_s": 0., "end_s": 5.}])
    assert result["moving_during_inference_s"] == 3.
    assert result["stationary_during_inference_s"] == 2.
    assert result["newly_observed_area_m2"] == .2
    assert result["stationary_intervals"][0]["duration_s"] == 2.
    partial = summarize(samples, [{"start_s": 1., "end_s": 3.}, {"start_s": 2.5, "end_s": 4.}])
    assert partial["inference_s"] == 3.
    assert partial["moving_during_inference_s"] == 2.
    assert partial["stationary_during_inference_s"] == 1.
    assert partial["uncertain_s"] == 5.
    AgentStart(run_id="probe", episode_epoch=1, goal=GOAL, **SETTINGS)
    return {"status": "passed", "model_calls": 0}


async def run(output, design_label, start_offset):
    self_check()
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(ROOT / "backend/.env", override=False)
    load_dotenv(ROOT / ".env", override=False)
    config = FoundryConfig.from_environment()
    profile = next(profile for profile in config.models if profile.id == "luna" and profile.provider == "foundry")
    if not config.configured(profile):
        raise RuntimeError("Existing Luna profile is not configured")
    challenge = get_challenge("kitchen_bathroom").model_copy(deep=True)
    challenge.initial_xy = [challenge.initial_xy[0] + start_offset, challenge.initial_xy[1]]
    design = design_snapshot(design_label)
    plan = {"goal": GOAL, "settings": SETTINGS, "start_offset_m": start_offset, "challenge": challenge.model_dump(),
        "evidence": "real_model_diagnostic", "design": design, "started_at": datetime.now(timezone.utc).isoformat(),
        "map_reuse": False, "workload_before": workload_snapshot(), "model_identity_verified": False,
        "evaluation_protocol": ROOM_PROTOCOL}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    for name in design["code_sha256"]:
        if name.startswith("backend/") or name == "scripts/probe_mission_review.py":
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    session_recording.SESSION_RESULTS_ROOT = output
    samples, inference = [], []
    pending = False
    began = time.monotonic()

    class MeasuredModel(ConfiguredModel):
        async def respond(self, *args, **kwargs):
            nonlocal pending
            entry = {"start_s": time.monotonic() - began}
            pending = True
            try:
                return await super().respond(*args, **kwargs)
            finally:
                pending = False
                entry["end_s"] = time.monotonic() - began
                inference.append(entry)

    worker = EvaluationWorker(challenge=challenge, rendering="enhanced", pace=True)
    evaluator = RoomEvaluator()
    controller = AgentController(config, model_factory=MeasuredModel, evaluation_budget=NavigationEvaluationBudget(timeout_s=300.))
    controller.record_sessions = True
    controller.recording_evidence = "real_model"
    physical = PhysicsMeasurements(challenge.initial_xy)
    error = None
    final_mission = None

    def sample():
        elapsed = time.monotonic() - began
        if samples and elapsed - samples[-1]["wall_s"] < .025:
            return
        physical.sample(worker)
        evaluator.sample_arrival(worker.sim, physical, elapsed)
        mapped = worker.home_mission
        task = mapped.task or {}
        control = worker.continuous
        reason = (control.reason if control and control.status == "blocked" else None) or task.get("reason")
        if not reason:
            reason = "inference_pending_no_route" if pending and not mapped.active else controller.state["phase"]
        samples.append({"wall_s": elapsed, "simulated_s": worker.sim.ticks / 240., "pose": physical.pose,
            "linear_speed_mps": physical.linear_speed, "angular_speed_radps": physical.angular_speed,
            "inference_pending": pending, "phase": controller.state["phase"], "stop_reason": reason,
            "route_active": bool(mapped.active), "route_status": control.status if control else None,
            "contacts": physical.contacts, "known_cells": int(np.count_nonzero(mapped.home.cells != -1)) if mapped.home else None,
            "free_cells": int(np.count_nonzero(mapped.home.cells == 0)) if mapped.home else None})

    try:
        async with asyncio.timeout(90.):
            await asyncio.wrap_future(worker.ready)
            worker.home_mission = HomeMission(worker, MapStore(output / "maps.sqlite3"))
        began = time.monotonic()
        evaluator.began = began
        def attach(sim):
            evaluator.capture_initial(sim)
            worker.room_evaluator = evaluator
            previous = sim.on_tick
            def tick():
                previous()
                sample()
            sim.on_tick = tick
            sample()
        await worker.call(attach)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, goal=GOAL, **SETTINGS))
        last_turn = -1
        while not controller.task.done():
            await asyncio.sleep(.1)
            await worker.call(lambda sim: sample())
            if controller.state["turns"] != last_turn:
                last_turn = controller.state["turns"]
                print(json.dumps({"phase": controller.state["phase"], "turns": last_turn,
                    "elapsed_s": round(time.monotonic() - began, 2), "travel_m": physical.travel}), flush=True)
        await controller.task
        await worker.call(lambda sim: sample())
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    finally:
        final_state = dict(controller.state)
        final_mission = controller.mission.state() if controller.mission else None
        (output / "trace.json").write_text(json.dumps(controller.trace(), indent=2), encoding="utf-8")
        for event_id, image in list(controller.trace_images.items()):
            (output / f"frame-{event_id:04d}.png").write_bytes(image)
        for name, image in evaluator.images.items():
            (output / name).write_bytes(image)
        (output / "room_evaluation.json").write_text(json.dumps(evaluator.artifact(), indent=2), encoding="utf-8")
        await controller.halt()
        worker.stop()
        await worker.close()
    result = {**summarize(samples, inference), "travel_m": physical.travel, "contact_episodes": physical.contacts,
        "task_completion": final_mission, "controller": final_state, "error": error,
        "source_unchanged": not changed_sources(design["code_sha256"]), "samples": len(samples),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "limitations": "Real-model diagnostic, not the fixed household benchmark. Sampled stationary inference is overlap, not proof of an unnecessary stop. Room discovery/identification/arrival require separate hidden evaluation. Token threshold can overshoot by one response; cancelled usage may be unknown."}
    for name, value in (("measurements", samples), ("inference", inference), ("result", result)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"controller", "stationary_intervals"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen real-Luna mission continuity diagnostic")
    parser.add_argument("--stage", choices=["check", "run", "analyze"], default="check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--overwrite-analysis", action="store_true")
    parser.add_argument("--design", default="mission-review-baseline-v1")
    parser.add_argument("--start-offset", type=float, choices=[0., .1, -.1], default=0.)
    arguments = parser.parse_args()
    if arguments.stage == "check":
        print(json.dumps(self_check()))
    elif arguments.stage == "analyze":
        if arguments.source is None or arguments.output is None:
            parser.error("--source and --output are required for analyze")
        analysis = analyze(arguments.source, arguments.output, overwrite=arguments.overwrite_analysis)
        print(json.dumps({"analysis": str(arguments.output / "analysis.json"), "model_calls": 0,
            "timing": {key: analysis["timing"][key] for key in ("moving_s", "stationary_s", "inference_s", "net_known_area_m2", "net_free_area_m2")},
            "room_results": {key: analysis["room_results"][key] for key in ("discovery", "correct_identification", "arrival")}}, indent=2))
    elif arguments.output is None:
        parser.error("--output is required for run")
    else:
        asyncio.run(run(arguments.output, arguments.design, arguments.start_offset))