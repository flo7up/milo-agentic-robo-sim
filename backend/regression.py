import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from backend.agent import AgentStart
from backend.contracts import StrictModel
from backend.recording import write_recording_json
from backend.session_recording import ROOT, recording_root


SUITE_ID = "observable-navigation-v2"
CASES = (
    {"id": "parking", "challenge_id": "park", "title": "Park in the green bay", "budget_s": 120},
    {"id": "parking-left", "challenge_id": "park_left", "title": "Park from 30 cm left offset", "budget_s": 120},
    {"id": "parking-right", "challenge_id": "park_right", "title": "Park from 30 cm right offset", "budget_s": 120},
    {"id": "parking-far", "challenge_id": "park_far", "title": "Park from 65 cm farther back", "budget_s": 120},
    {"id": "table-clockwise", "challenge_id": "furniture_circuit", "title": "Table circuit clockwise", "budget_s": 180,
        "orbit_target": "table", "orbit_direction": "clockwise"},
    {"id": "table-counterclockwise", "challenge_id": "furniture_circuit", "title": "Table circuit counterclockwise", "budget_s": 180,
        "orbit_target": "table", "orbit_direction": "counterclockwise"},
    {"id": "object-search", "challenge_id": "apartment", "title": "Find and approach the yellow cube", "budget_s": 180},
    {"id": "room-arrival", "challenge_id": "flat_kitchen", "title": "Find and enter the kitchen", "budget_s": 180},
)


class RegressionStart(StrictModel):
    run_id: str = Field(min_length=1)
    episode_epoch: int = Field(ge=0)
    model_id: Literal["luna", "qwen"] = "qwen"
    reasoning: Literal["none", "low", "medium", "high"] = "none"
    task_supervisor_model_id: Literal["luna"] | None = Field(
        default_factory=lambda data: "luna" if data.get("model_id") == "qwen" else None)

    @model_validator(mode="after")
    def hybrid_contract(self):
        if self.task_supervisor_model_id and self.model_id != "qwen":
            raise ValueError("Hybrid task supervision requires Qwen as the primary model")
        return self


def source_hashes():
    paths = [*sorted((ROOT / "backend").glob("*.py")), *sorted((ROOT / "frontend/dist").rglob("*")),
        *sorted((ROOT / "assets").rglob("*")), ROOT / "pyproject.toml"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()}


def score_case(challenge, score, error):
    if error or not score or not score.get("complete_recording") or score.get("operator_assisted") or score.get("dropped_records"):
        return "invalid"
    return "passed" if challenge.get("status") == "completed" and score.get("final_physics_success") and not score.get("contact_episodes") else "failed"


def grade_progress(initial, final):
    if not initial or not final:
        return None
    grade = {"schema_version": 1, "evaluation_only": True, **copy.deepcopy(final)}
    if final["kind"] == "circuit":
        degrees = max(0., min(360., final["valid_degrees"]))
        grade.update(metric="Valid lap", progress_pct=degrees / 3.6, remaining_degrees=360. - degrees)
    else:
        initial_gap, final_gap = initial["remaining_m"], final["remaining_m"]
        fraction = 1. - final_gap / initial_gap if initial_gap > 1e-6 else float(final_gap <= 1e-6)
        grade.update(metric="Target approach", progress_pct=100. * max(0., min(1., fraction)),
            initial_gap_m=initial_gap, remaining_pct=100. * final_gap / initial_gap if initial_gap > 1e-6 else None)
    return grade


def evaluation_snapshot(sim):
    import pybullet as bullet

    progress = sim.challenge_progress
    challenge = progress.challenge
    status = sim.challenge_status()
    position = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0]
    linear, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
    stopped = math.sqrt(sum(value ** 2 for value in linear)) < .025 and math.sqrt(sum(value ** 2 for value in angular)) < .15
    floor = next(item["id"] for item in sim.objects if item["name"] == "floor")
    grounded = any(contact[9] > .01 for contact in bullet.getContactPoints(sim.robot, floor, physicsClientId=sim.client))
    snapshot = {"position_xy_m": list(position[:2]), "simulated_s": sim.ticks / 240,
        "physics_complete": status["status"] == "completed", "completed_objectives": status["completed_objectives"],
        "total_objectives": len(challenge.objectives), "checks": [],
        "detail": "; ".join(item["detail"] for item in status["progress"] if not item["complete"])}
    if challenge.orbit:
        snapshot.update(kind="circuit", target_xy_m=list(challenge.orbit.center_m),
            target_radius_m=challenge.orbit.maximum_radius_m, valid_degrees=math.degrees(progress.orbit_angle),
            return_error_m=math.dist(position[:2], progress.orbit_start) if progress.orbit_start else None,
            dwell_s=progress.orbit_dwell_s, required_dwell_s=.5,
            checks=[{"label": "Valid lap and return", "complete": progress.orbit_lap},
                {"label": "Stopped dwell", "complete": progress.orbit_dwell_s >= .5},
                {"label": "No circuit contact", "complete": not progress.orbit_contact}])
    elif challenge.search_target:
        target = next(item["id"] for item in sim.objects if item["name"] == challenge.search_target)
        target_position = bullet.getBasePositionAndOrientation(target, physicsClientId=sim.client)[0]
        distance = math.dist(position[:2], target_position[:2])
        visible = sim._target_in_camera(target, target_position)
        head_still = all(abs(bullet.getJointState(sim.robot, sim.joints[name], physicsClientId=sim.client)[1]) < .05
            for name in ("head_yaw", "head_pitch"))
        snapshot.update(kind="search", target_xy_m=list(target_position[:2]), target_radius_m=.9,
            center_distance_m=distance, remaining_m=max(0., distance - .9), dwell_s=progress.search_dwell_s,
            required_dwell_s=1., checks=[{"label": "Within inspection range", "complete": distance <= .9},
                {"label": "Target visible", "complete": visible},
                {"label": "Stopped and camera steady", "complete": grounded and stopped and head_still},
                {"label": "Inspection dwell", "complete": progress.search_dwell_s >= 1.}])
    else:
        objective = challenge.objectives[-1]
        bounds = [bullet.getAABB(sim.robot, index, physicsClientId=sim.client) for index in [-1, *sim.wheels]]
        lower = [min(entry[0][axis] for entry in bounds) for axis in (0, 1)]
        upper = [max(entry[1][axis] for entry in bounds) for axis in (0, 1)]
        target_lower = [objective.center[axis] - objective.size[axis] / 2 for axis in (0, 1)]
        target_upper = [objective.center[axis] + objective.size[axis] / 2 for axis in (0, 1)]
        gap = math.hypot(*(max(target_lower[axis] - lower[axis], upper[axis] - target_upper[axis], 0.) for axis in (0, 1)))
        snapshot.update(kind="arrival", target_xy_m=list(objective.center[:2]),
            target_bounds_m=[*target_lower, *target_upper], center_distance_m=math.dist(position[:2], objective.center[:2]),
            remaining_m=gap, checks=[{"label": "Full footprint inside", "complete": gap <= 1e-9},
                {"label": "Grounded", "complete": grounded}, {"label": "Stopped", "complete": stopped}])
    return snapshot


class RegressionSequence:
    def __init__(self):
        self.task = None
        self.cancelled = False
        self.state = {"suite_id": SUITE_ID, "active": False, "phase": "idle", "cases": [
            {**case, "status": "pending"} for case in CASES]}
        self.directory = None

    def restore_latest(self, root):
        candidates = sorted(Path(root).glob(f"regression-{SUITE_ID}-*/sequence.json"),
            key=lambda path: path.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                state = json.loads(path.read_text(encoding="utf-8"))
                if (state.get("suite_id") != SUITE_ID or not state.get("finished_at")
                        or [case.get("id") for case in state.get("cases", [])] != [case["id"] for case in CASES]):
                    continue
                self.directory = path.parent.resolve()
                self.state = {key: value for key, value in state.items() if key != "active"}
                self.state["phase"] = state.get("phase", "completed")
                return True
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        return False

    @property
    def active(self):
        return bool(self.task and not self.task.done())

    def public(self):
        state = copy.deepcopy({key: value for key, value in self.state.items() if key != "source_hashes"})
        if state.get("sequence_id"):
            for case in state["cases"]:
                if case["status"] not in {"pending", "not_run", "loading"}:
                    case["trajectory_url"] = f"/api/regression/cases/{case['id']}/trajectory"
        return {**state, "active": self.active}

    def trajectory(self, case_id, active_directory=None):
        case = next((item for item in self.state["cases"] if item["id"] == case_id), None)
        if case is None or self.directory is None:
            raise ValueError("Unknown regression case")
        directory = case.get("recording_directory")
        if not directory and self.state.get("current_index") is not None and self.state["cases"][self.state["current_index"]] is case:
            directory = active_directory
        if not directory:
            raise ValueError("Regression path is not available yet")
        root, recording = self.directory.resolve(), Path(directory).resolve()
        if recording != root and root not in recording.parents:
            raise ValueError("Regression recording is outside the active sequence")
        path = recording / "trajectory.jsonl"
        if not path.is_file() or path.stat().st_size > 64_000_000:
            raise ValueError("Regression path is unavailable")
        points, contacts, bounds = [], [], [math.inf, math.inf, -math.inf, -math.inf]
        previous_contact, previous_placement, segment, distance = False, None, 0, 0.
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 65536 or len(points) >= 50000:
                    raise ValueError("Regression path exceeds display limits")
                sample = json.loads(line)
                position, elapsed = sample.get("position_m"), sample.get("wall_s")
                if (not isinstance(position, list) or len(position) < 2 or not isinstance(elapsed, (int, float))
                        or isinstance(elapsed, bool) or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and abs(value) <= 1000 for value in position[:2])
                        or not math.isfinite(elapsed) or (points and elapsed < points[-1]["wall_s"])):
                    raise ValueError("Invalid regression trajectory sample")
                placement = sample.get("manual_placements", 0)
                if points and placement != previous_placement:
                    segment += 1
                previous_placement = placement
                point = {"x": position[0], "y": position[1], "wall_s": elapsed, "segment": segment}
                if points and points[-1]["segment"] == segment:
                    distance += math.dist((points[-1]["x"], points[-1]["y"]), position[:2])
                points.append(point)
                contact = bool(sample.get("collisions"))
                if contact and not previous_contact:
                    contacts.append(point)
                previous_contact = contact
                bounds = [min(bounds[0], point["x"]), min(bounds[1], point["y"]),
                    max(bounds[2], point["x"]), max(bounds[3], point["y"])]
        if not points:
            raise ValueError("Regression path is empty")
        stride = max(1, math.ceil(len(points) / 1200))
        selected = sorted(set(range(0, len(points), stride)) | {len(points)-1})
        return {"coordinate_frame": "recorded_world_xy_m", "sample_count": len(points), "bounds_m": bounds,
            "distance_m": distance, "downsampled": stride > 1, "points": [points[index] for index in selected],
            "contacts": contacts[:200], "contact_markers_truncated": len(contacts) > 200,
            "evaluation_only": True}

    def cancel(self, lab, reason):
        if self.active:
            self.cancelled = True
            self.state.update(phase="stopping", reason=reason)
            lab.worker.stop()

    async def wait(self):
        if self.active:
            await asyncio.shield(self.task)

    def check(self, lab):
        if self.cancelled or not lab.connections or not lab.robot_on:
            raise asyncio.CancelledError

    async def publish(self):
        snapshot = {**copy.deepcopy(self.state), "active": self.active and not self.state.get("finished_at")}
        await asyncio.to_thread(write_recording_json, self.directory / "sequence.json", snapshot)

    def start(self, lab, request, start_agent):
        if self.active:
            raise ValueError("A regression sequence is already running")
        worker = lab.worker
        if (request.run_id, request.episode_epoch) != (worker.latest["run_id"], worker.epoch):
            raise ValueError("Episode changed; review the current scene before starting")
        if not lab.connections or not worker.powered or lab.lock.locked() or lab.agent.active or worker.latest.get("busy") or lab.recording_status()["active"]:
            raise ValueError("Keep the cockpit connected and stop current work before starting the baseline")
        if lab.interaction_mode != "chat":
            raise ValueError("Switch to Chat before starting the baseline")
        profile = next((item for item in lab.agent.config.models if item.id == request.model_id), None)
        if profile is None or not lab.agent.config.configured(profile) or request.reasoning not in profile.reasoning_efforts:
            raise ValueError("Configure the selected supervisor and a supported reasoning setting first")
        task_profile = next((item for item in lab.agent.config.models if item.id == request.task_supervisor_model_id), None)
        if request.task_supervisor_model_id and (not task_profile or task_profile.provider != "foundry"
                or not lab.agent.config.configured(task_profile) or "low" not in task_profile.reasoning_efforts):
            raise ValueError("Configure Luna low reasoning before hybrid task supervision")
        identity = str(uuid4())
        self.directory = recording_root(lab.recording_root) / f"regression-{SUITE_ID}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{identity[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cancelled = False
        self.state = {"schema_version": 1, "suite_id": SUITE_ID, "sequence_id": identity, "phase": "starting", "reason": None,
            "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
            "model": profile.model_dump(), "reasoning": request.reasoning, "directory": str(self.directory),
            "task_supervisor_model_id": request.task_supervisor_model_id,
            "evidence": lab.agent.recording_evidence, "current_index": None,
            "settings": {"max_requests_per_case": 12, "max_tokens_per_case": 80000, "images": 2,
                "history_tokens": 8192, "reuse_saved_map": False, "gap_s": 3},
            "cases": [{**case, "status": "pending"} for case in CASES]}
        self.task = asyncio.create_task(self.run(lab, request, start_agent))

    async def run(self, lab, request, start_agent):
        original_enabled, original_root = lab.recording_enabled, lab.recording_root
        original_config = lab.agent.config
        current = None
        try:
            self.state["source_hashes"] = await asyncio.to_thread(source_hashes)
            self.state["source_sha256"] = hashlib.sha256(json.dumps(self.state["source_hashes"], sort_keys=True).encode()).hexdigest()
            self.state["rendering"] = lab.worker.rendering
            self.state["suite_sha256"] = hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest()
            await self.publish()
            lab.recording_enabled, lab.recording_root = True, self.directory
            for index, case in enumerate(self.state["cases"]):
                self.check(lab)
                if lab.agent.config != original_config or await asyncio.to_thread(source_hashes) != self.state["source_hashes"]:
                    raise ValueError("Configuration or source changed; remaining cases were not run")
                current = case
                self.state.update(current_index=index, phase="loading")
                case["status"] = "loading"
                await self.publish()
                await lab.reset(case["challenge_id"], case.get("orbit_target"), case.get("orbit_direction"),
                    environment="standalone", reuse_saved_map=False,
                    environment_instance_id=f"regression-{self.state['sequence_id']}-{case['id']}")
                self.check(lab)
                worker, agent = lab.worker, lab.agent
                settings = AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
                    goal=worker.challenge.goal, model_id=request.model_id, reasoning=request.reasoning,
                    execution_mode="luna_continuous", unified_mission=True, images_per_request=2, context_tokens=8192,
                    map_context=True, mission_budget_s=case["budget_s"], max_turns=12, max_model_requests=12,
                    max_model_tokens=80000, feedback_interval_s=.25,
                    task_supervisor_model_id=request.task_supervisor_model_id,
                    task_supervisor_reasoning="low", max_task_supervisor_requests=1, max_task_supervisor_tokens=100000)
                case.update(run_id=settings.run_id, settings=settings.model_dump(),
                    challenge_sha256=hashlib.sha256(json.dumps(worker.challenge.model_dump(), sort_keys=True).encode()).hexdigest())
                case["initial_evaluation"] = await worker.call(evaluation_snapshot)
                case["evaluation"] = grade_progress(case["initial_evaluation"], case["initial_evaluation"])
                self.state["phase"] = "preparing"
                await self.publish()
                self.check(lab)
                await start_agent(settings)
                self.check(lab)
                case.update(status="running", session_id=agent.state["session_id"])
                self.state["phase"] = "running"
                began = time.monotonic()
                await self.publish()
                while not agent.task.done():
                    await asyncio.wait([agent.task], timeout=1.)
                    self.check(lab)
                    case["evaluation"] = grade_progress(case["initial_evaluation"], await worker.call(evaluation_snapshot))
                await asyncio.shield(agent.task)
                case["evaluation"] = grade_progress(case["initial_evaluation"], await worker.call(evaluation_snapshot))
                case.update(elapsed_s=time.monotonic()-began, outcome=copy.deepcopy(agent.state.get("outcome")),
                    input_tokens=agent.state["input_tokens"], output_tokens=agent.state["output_tokens"],
                    challenge=copy.deepcopy(worker.latest.get("challenge", {})), error=agent.state.get("error"),
                    recording_directory=str(agent.recording_directory) if agent.recording_directory else None)
                score = None
                if agent.recording_directory:
                    score_path = Path(agent.recording_directory) / "scorecard.json"
                    score = json.loads(await asyncio.to_thread(score_path.read_text, encoding="utf-8"))
                case["scorecard"] = score
                self.check(lab)
                case["status"] = score_case(case["challenge"], score, agent.recording_error)
                if await asyncio.to_thread(source_hashes) != self.state["source_hashes"]:
                    case["status"] = "invalid"
                    raise ValueError("Source changed during the case; remaining cases were not run")
                self.state["phase"] = "between_cases"
                await self.publish()
                if case["status"] == "invalid":
                    raise ValueError("Recording invalid; remaining cases were not run")
                if index < len(self.state["cases"]) - 1:
                    await asyncio.sleep(3.)
            self.state["phase"] = "completed"
        except asyncio.CancelledError:
            self.state.update(phase="cancelled", reason=self.state.get("reason") or "Operator disconnected or sequence stopped")
            if current and current["status"] in {"loading", "running", "pending"}:
                current["status"] = "cancelled"
        except Exception as error:
            self.state.update(phase="error", reason=f"Sequence stopped: {type(error).__name__}")
            if current and current["status"] in {"loading", "running", "pending"}:
                current.update(status="invalid", error=str(getattr(error, "detail", None) or type(error).__name__)[:300])
        finally:
            lab.worker.stop()
            await lab.agent.halt("Regression sequence ended")
            lab.recording_enabled, lab.recording_root = original_enabled, original_root
            lab.agent.record_sessions, lab.agent.recording_root = original_enabled, original_root
            for case in self.state["cases"]:
                if case["status"] == "pending":
                    case["status"] = "not_run"
            self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
            await self.publish()