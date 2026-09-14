from collections import Counter, deque
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
import time


recording_json_lock = Lock()


def read_recording_json(path):
    with recording_json_lock:
        contents = path.read_text(encoding="utf-8")
    return json.loads(contents)


def write_recording_json(path, value):
    contents = json.dumps(value, indent=2)
    with recording_json_lock:
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(contents)
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(.02 * 2 ** attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def score_samples(samples, *, dropped=0):
    durations = Counter()
    distance = 0.
    stops = []
    stopped_since = None
    has_moved = False
    contacts = 0
    previous_contact = False
    first_success = None
    maximum_gap = 0.
    for index, sample in enumerate(samples):
        if sample["physics_status"] == "completed" and first_success is None:
            first_success = sample["wall_s"]
        contact = bool(sample["collisions"])
        contacts += int(contact and not previous_contact)
        previous_contact = contact
        moving = sample["linear_speed_mps"] > .015 or abs(sample["angular_speed_radps"]) > .04
        if moving:
            has_moved = True
            if stopped_since is not None:
                beginning, reason = stopped_since
                if sample["wall_s"] - beginning >= .5:
                    stops.append({"start_s": beginning, "duration_s": sample["wall_s"] - beginning, "reason": reason})
                stopped_since = None
        elif has_moved and stopped_since is None:
            stopped_since = sample["wall_s"], sample["activity"]
        if index:
            previous = samples[index - 1]
            elapsed = max(0., sample["wall_s"] - previous["wall_s"])
            durations[previous["activity"]] += elapsed
            maximum_gap = max(maximum_gap, elapsed)
            distance += math.dist(previous["position_m"][:2], sample["position_m"][:2])
    if stopped_since and samples[-1]["wall_s"] - stopped_since[0] >= .5:
        stops.append({"start_s": stopped_since[0], "duration_s": samples[-1]["wall_s"] - stopped_since[0], "reason": stopped_since[1]})
    final_success = bool(samples and samples[-1]["physics_status"] == "completed")
    return {"schema_version": 1, "complete_recording": dropped == 0, "dropped_records": dropped,
        "samples": len(samples), "final_physics_success": final_success,
        "first_verified_completion_s": first_success,
        "completion_time_s": first_success if final_success and not dropped else None,
        "elapsed_wall_s": samples[-1]["wall_s"] if samples else 0.,
        "simulated_elapsed_s": samples[-1]["simulated_s"] - samples[0]["simulated_s"] if samples else 0.,
        "actual_distance_m": distance, "contact_episodes": contacts,
        "activity_wall_s": dict(durations), "stationary_intervals": stops,
        "inference_stationary_s": durations["inference_wait"], "maximum_sample_gap_s": maximum_gap,
        "timing_note": "Monotonic wall time includes preparation and inference; first success is sampled physics state, not a model claim.",
        "stop_note": "Stationary intervals include intentional stops; reason labels are diagnostic, not a judgment of necessity."}


class RunRecorder:
    def __init__(self, directory, *, context=None, clock=time.monotonic, max_pending=512):
        if max_pending < 1:
            raise ValueError("Recording buffer must retain at least one sample")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "media").mkdir()
        self.context = context or (lambda: {})
        self.clock = clock
        self.started = clock()
        self.pending = deque(maxlen=max_pending)
        self.dropped = 0
        self.sample_sequence = 0
        self.last_camera = None
        self.last_spatial = None
        self.camera_media = []
        self.spatial_media = []
        self.last_media_at = -math.inf
        self.scene = None
        self.scene_written = False
        self.stream = (self.directory / "trajectory.jsonl").open("w", encoding="utf-8")

    def capture(self, worker):
        import pybullet as bullet
        sim = worker.sim
        if self.scene is None:
            self.scene = {"schema_version": 1, "evaluation_only": True,
                "coordinate_frame": "recorded_world_xy_m", "robot_body_id": sim.robot,
                "geometry": sim.geometry(), "snapshot": sim.snapshot()}
        position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        linear, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
        wheels = [bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[1] for joint in sim.wheels]
        head = [bullet.getJointState(sim.robot, sim.joints[name], physicsClientId=sim.client) for name in ("head_yaw", "head_pitch")]
        phase = self.context().get("phase", "idle")
        continuous = worker.continuous
        navigation = worker.navigation
        speed = math.hypot(linear[0], linear[1])
        reason = continuous.reason if continuous else navigation.reason if navigation else ""
        if speed > .015:
            activity = "driving"
        elif abs(angular[2]) > .04:
            activity = "turning"
        elif any(abs(state[1]) > .02 for state in head):
            activity = "head_scan"
        elif phase == "thinking":
            activity = "inference_wait"
        elif sim.cancel.is_set():
            activity = "stopped"
        elif continuous and continuous.status == "blocked":
            activity = "blocked"
        elif continuous and continuous.status == "arrived":
            activity = "arrival"
        else:
            activity = "preparing" if worker.latest.get("busy") else phase
        now = self.clock()
        self.sample_sequence += 1
        sample = {"index": self.sample_sequence, "wall_s": now - self.started, "simulated_s": sim.ticks / 240,
            "run_id": sim.run_id, "episode_epoch": sim.epoch, "position_m": list(position),
            "yaw_rad": bullet.getEulerFromQuaternion(orientation)[2], "odometry_m_rad": sim.odometry.tolist(),
            "linear_speed_mps": speed, "angular_speed_radps": angular[2], "wheel_radps": wheels,
            "head_rad": [state[0] for state in head], "phase": phase, "activity": activity, "reason": reason,
            "physics_status": (sim.challenge_status() or {}).get("status", "unscored"),
            "collisions": worker.latest.get("proximity", {}).get("collisions", []),
            "manual_placements": worker.latest.get("manual_placements", 0),
            "command_velocity": navigation.velocity.tolist() if navigation else None,
            "requested_segment": navigation.buffer[0][0].model_dump() if navigation and navigation.buffer else None,
            "path_odometry_m": continuous.path.tolist() if continuous else [],
            "route_status": continuous.status if continuous else None, "stop_revision": worker.stop_revision}
        if now - self.last_media_at >= .5:
            camera = worker.latest.get("camera")
            if camera and (image := worker.camera_frames.get(camera["frame_ref"])):
                filename = f"media/camera-{camera['seq']}.png"
                self.last_camera = {"path": filename, "simulated_s": camera["simulated_time_s"], "sequence": camera["seq"]}
                self.camera_media = [(filename, image)]
            spatial = next(reversed(worker.spatial_frames.values()), None)
            if spatial:
                sensor, rgb, depth = spatial
                prefix = f"media/spatial-{sensor.sequence}"
                self.last_spatial = {"rgb": prefix + ".png", "depth": prefix + "-depth.png", "data": prefix + ".json",
                    "sequence": sensor.sequence, "captured_at": sensor.captured_at, "simulated_s": sensor.simulated_time_s}
                self.spatial_media = [(prefix + ".png", rgb), (prefix + "-depth.png", depth),
                    (prefix + ".json", sensor.model_dump_json().encode("utf-8"))]
            self.last_media_at = now
        sample.update(camera=self.last_camera, spatial=self.last_spatial)
        if len(self.pending) == self.pending.maxlen:
            self.dropped += 1
        self.pending.append((sample, self.camera_media + self.spatial_media))

    def flush(self):
        if self.scene is not None and not self.scene_written:
            (self.directory / "scene.json").write_text(json.dumps(self.scene, allow_nan=False), encoding="utf-8")
            self.scene_written = True
        for _ in range(len(self.pending)):
            sample, media = self.pending.popleft()
            for filename, content in media:
                path = self.directory / filename
                if not path.exists():
                    path.write_bytes(content)
            self.stream.write(json.dumps(sample, separators=(",", ":"), allow_nan=False) + "\n")
        self.stream.flush()

    def finish(self, metadata):
        self.flush()
        self.stream.close()
        with (self.directory / "trajectory.jsonl").open(encoding="utf-8") as source:
            samples = [json.loads(line) for line in source]
        score = score_samples(samples, dropped=self.dropped)
        score["operator_assisted"] = any(sample["manual_placements"] for sample in samples)
        score["autonomous_success"] = score["final_physics_success"] and not score["operator_assisted"] and not self.dropped and metadata.get("real_model", False)
        manifest = {"schema_version": 1, "evaluation_only": True,
            "privacy": "Contains privileged physics measurements. Never send this recording or replay to the robot policy; curate sensor/action training data separately.",
            "media_interval_s": .5, **metadata}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (self.directory / "scorecard.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
        template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
        data = json.dumps({"manifest": manifest, "score": score, "samples": samples}, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        (self.directory / "replay.html").write_text(template.replace("__RECORDING_DATA__", data), encoding="utf-8")
        return score