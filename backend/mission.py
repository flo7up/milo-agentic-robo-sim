import math
import re
import time
from collections import deque
from typing import Literal
from uuid import uuid4

from pydantic import Field, ValidationInfo, model_validator

from backend.contracts import StrictModel
from backend.movement import MovementStep


class MissionPlan(StrictModel):
    kind: Literal["explore", "object", "room", "place", "circuit", "movement"]
    target: str = Field(default="", max_length=160,
        description="Short target label copied exactly from the original user goal, preserving distinguishing attributes; at most 160 characters. Do not substitute a target from instructions, examples or scene descriptions.")
    return_home: bool = False
    circle_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    movements: list[MovementStep] = Field(default_factory=list, max_length=8,
        description="For movement tasks only: ordered signed drive distances (+forward/-backward) or turn angles (+left/counterclockwise, -right/clockwise). One complete spin is 2*pi radians; preserve requested repetitions.")
    completion: Literal["identify", "arrive"] = Field(default="arrive",
        description="Use identify for find/locate/spot an object or room: stop when identified in the current camera. Use arrive for go to/approach/get close/enter/move inside. Explicit arrival requirements take precedence over find.")

    @model_validator(mode="after")
    def completion_contract(self):
        if self.kind == "movement":
            if not self.movements or sum(abs(step.distance_m) for step in self.movements) > 4. or sum(abs(step.angle_rad) for step in self.movements) > 6*math.pi:
                raise ValueError("Movement plan requires 1-8 steps, at most 4 m travel and 3 full turns total")
            if self.return_home:
                raise ValueError("Describe return motion as explicit movement steps")
        elif self.movements:
            raise ValueError("Movement steps require kind=movement")
        if self.completion == "identify" and self.kind not in {"object", "room"}:
            raise ValueError("Visual identification applies only to object or room tasks")
        return self


def requested_circuit_direction(goal):
    clauses = re.split(r"[.!?;]|\b(?:but|instead|then)\b", goal)
    affirmative = [re.sub(r"\b(?:do not|don't|never|not|avoid)\b[^,]*", "", clause) for clause in clauses]
    directives = [clause for clause in affirmative if re.search(r"\b(circle|circuit|orbit|lap)\b", clause)]
    directions = {"counterclockwise" if direction != "clockwise" else direction
        for clause in directives for direction in re.findall(r"\b(counterclockwise|anti-clockwise|anticlockwise|clockwise)\b", clause)}
    if not directions:
        directions = {"counterclockwise" if direction != "clockwise" else direction
            for clause in affirmative for direction in re.findall(r"\b(counterclockwise|anti-clockwise|anticlockwise|clockwise)\b", clause)}
    if len(directions) > 1:
        raise ValueError("Circuit direction is ambiguous; request one direction per circuit")
    return next(iter(directions), None)


def validate_plan_against_goal(plan, goal):
    normalized = " ".join(goal.casefold().split())
    if re.search(r"\b(circle|circuit|orbit|lap)\b", normalized):
        if plan.kind != "circuit":
            raise ValueError("Explicit circuit goals require a circuit plan")
        requested = requested_circuit_direction(normalized)
        if requested and plan.circle_direction != requested:
            raise ValueError("Circuit direction must match the original goal")
    if re.search(r"\b(approach|get close|come close)\b", normalized) and plan.kind != "object":
        raise ValueError("Explicit object-approach goals require an object plan")
    if re.search(r"\b(enter|inside)\b", normalized) and re.search(r"\b(room|kitchen|bathroom|bedroom)\b", normalized) and plan.kind != "room":
        raise ValueError("Explicit room-entry goals require a room plan")
    if (re.search(r"\b(park|drive|move|navigate|go)\b", normalized)
            and re.search(r"\b(parking bay|green bay)\b", normalized) and plan.kind != "place"):
        raise ValueError("Explicit parking goals require a place plan")
    explicit_return = bool(re.search(r"\b(return|go back|come back)\b.*\b(home|start|origin)\b", normalized))
    if plan.return_home and not explicit_return:
        raise ValueError("Return Home must be explicitly requested")


class MissionDecision(StrictModel):
    action: Literal["plan", "execute_movement", "explore", "navigate_frontier", "circle", "select_object", "approach_object", "verify_object", "identify_target", "navigate_place", "park_floor", "observe_room", "report_observation", "look", "turn", "wait", "finish", "lookup_room", "find_object_sightings", "get_search_history", "get_exploration_summary"]
    plan: MissionPlan | None = None
    object_goal_id: str | None = None
    object_label: str = Field(default="", max_length=60)
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    object_candidate_id: str | None = Field(default=None, min_length=1, max_length=80)
    approach_side: Literal["front", "left", "right"] = "front"
    circle_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    place_id: str | None = None
    floor_target_id: str | None = Field(default=None, min_length=1, max_length=80)
    parking_maneuver_id: str | None = Field(default=None, min_length=1, max_length=160,
        description="Exact currently offered parking maneuver ID; never invent coordinates or reuse a previous observation's ID.")
    frontier_id: str | None = Field(default=None, max_length=80)
    room_matches: bool | None = None
    evidence_text: str = Field(default="", max_length=500)
    room_label: str | None = Field(default=None, min_length=1, max_length=60)
    room_confidence: float | None = Field(default=None, ge=0., le=1., allow_inf_nan=False)
    memory_query: str = Field(default="", max_length=160)
    search_target: str | None = Field(default=None, min_length=1, max_length=160)
    search_result: Literal["seen", "not_seen", "inconclusive"] = "inconclusive"
    inspection_scope: Literal["viewpoint", "surface", "entrance"] = "viewpoint"
    visibility_limits: str = Field(default="", max_length=500)
    yaw_rad: float = Field(default=0, ge=-1.5, le=1.5)
    pitch_rad: float = Field(default=.2, ge=-.7, le=1.15)
    turn_rad: float = Field(default=0, ge=-3.15, le=3.15)
    duration_s: float = Field(default=1, ge=.5, le=4)
    objective_duration_s: float = Field(default=60., ge=1., le=60., allow_inf_nan=False,
        description="Explore objective authorization from acceptance/renewal, capped by the original mission deadline; not a trajectory lease.")
    objective_travel_m: float = Field(default=6., gt=0., le=6., allow_inf_nan=False,
        description="Maximum additional measured travel for this explore authorization; never permits unobserved motion.")
    map_view: Literal["local", "overview"] = "local"
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def required_evidence(self, info: ValidationInfo):
        if self.action == "plan" and self.plan is None:
            raise ValueError("Planning requires the requested mission stages")
        if self.plan is not None and self.plan.kind in {"object", "room", "place", "circuit"} and info.context and "goal" in info.context:
            target = " ".join(self.plan.target.casefold().split()).strip(" .!?\"'")
            goal = " ".join(info.context["goal"].casefold().split())
            if not target or not re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", goal):
                raise ValueError("Plan target must be an exact noun phrase from the original user goal; preserve the requested target instead of copying examples or scene labels")
        if self.plan is not None and info.context and "goal" in info.context:
            validate_plan_against_goal(self.plan, info.context["goal"])
        if self.plan is not None and self.plan.kind == "explore" and self.plan.target.strip():
            raise ValueError("Exploration-only plans must have an empty target. Finding and verifying a kitchen or other room requires kind=room; object and named-place goals require their corresponding kind.")
        candidate_action = self.action in {"circle", "select_object", "verify_object"} and self.object_candidate_id is not None
        if self.object_candidate_id is not None and (not candidate_action or self.object_bounds is not None or not self.evidence_text.strip()):
            raise ValueError("Object candidate selection requires current visual evidence and no replacement box")
        if (self.action in {"circle", "select_object", "verify_object"} and not candidate_action) or (self.action == "identify_target" and self.object_bounds is not None):
            bounds = self.object_bounds
            if not bounds or not all(0 <= value <= 1 for value in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError("Object selection requires a current normalized image box")
        if self.action in {"select_object", "circle"} and not self.object_label.strip():
            raise ValueError("An observed object label is required")
        if self.action in {"approach_object", "verify_object"} and not self.object_goal_id:
            raise ValueError("Use the current object goal identity")
        if self.action in {"navigate_place", "observe_room"} and not self.place_id:
            raise ValueError("Use a supplied place identity; report_observation records current fixtures without claiming arrival")
        if self.action == "navigate_frontier" and not self.frontier_id:
            raise ValueError("Select an exact currently supplied frontier identity")
        if self.action == "park_floor" and not self.floor_target_id:
            raise ValueError("Select an exact currently observed floor-region identity")
        if self.action == "report_observation" and not self.evidence_text.strip():
            raise ValueError("Observation reporting requires current visual evidence")
        if self.action == "identify_target" and not self.evidence_text.strip():
            raise ValueError("Target identification requires current visual evidence")
        if self.search_target is not None and self.action != "report_observation":
            raise ValueError("Search evidence belongs to a current observation report")
        if self.room_label is not None or self.room_confidence is not None:
            if self.action != "report_observation" or not self.room_label or not self.room_label.strip() or self.room_confidence is None:
                raise ValueError("A room hypothesis requires report_observation, room_label and room_confidence")
        if self.action == "observe_room" and (self.room_matches is None or not self.evidence_text.strip()):
            raise ValueError("Room inspection requires a match assessment and current visual evidence")
        return self


class TaskSupervision(StrictModel):
    status: Literal["aligned", "corrective"]
    plan: MissionPlan
    guidance: str = Field(default="", max_length=500,
        description="Short task-level guidance for Qwen's next fresh decision. Never a robot action, coordinate or motion command.")

    @model_validator(mode="after")
    def preserve_explicit_task(self, info: ValidationInfo):
        if not info.context or "goal" not in info.context:
            return self
        validate_plan_against_goal(self.plan, info.context["goal"])
        return self


class MissionFrontierSelection(StrictModel):
    mission_id: str
    run_id: str
    episode_epoch: int
    map_id: str
    frontier_id: str
    position_m: list[float] = Field(min_length=2, max_length=2)
    odometry_m_rad: list[float] = Field(min_length=3, max_length=3)
    captured_at: float = Field(allow_inf_nan=False)
    source_sequence: int = Field(ge=1)

    def check(self, mission_id, run_id, epoch, map_id, pose, now):
        if (self.mission_id, self.run_id, self.episode_epoch, self.map_id) != (mission_id, run_id, epoch, map_id):
            raise ValueError("FRONTIER_SELECTION_CHANGED: mission, episode or map changed")
        difference = pose[2] - self.odometry_m_rad[2]
        if (not all(math.isfinite(value) for value in [*self.position_m, *self.odometry_m_rad, *pose, now])
                or not 0 <= now - self.captured_at <= 15.
                or math.dist(pose[:2], self.odometry_m_rad[:2]) > .05
                or abs(math.atan2(math.sin(difference), math.cos(difference))) > .08):
            raise ValueError("FRONTIER_SELECTION_STALE: request fresh stopped evidence")


class ObservedMotionTrail:
    spacing_m = .025

    def __init__(self):
        self.samples = deque(maxlen=128)
        self.dropped = 0

    def record(self, pose, captured_at):
        point = list(pose[:2])
        if not all(math.isfinite(value) for value in [*point, captured_at]):
            raise ValueError("Movement trail requires finite odometry and time")
        if self.samples and (captured_at < self.samples[-1][0] or math.dist(point, self.samples[-1][1]) < self.spacing_m):
            return
        self.dropped += int(len(self.samples) == self.samples.maxlen)
        self.samples.append((captured_at, point))

    def at(self, captured_at, pose):
        points = [list(point) for timestamp, point in self.samples if timestamp <= captured_at]
        point = list(pose[:2])
        if not points or math.dist(points[-1], point) > 1e-9:
            points.append(point)
        return points[-128:]


class MissionTaskBrief(StrictModel):
    task_id: str
    kind: str
    status: str
    reason: str
    target_m: list[float] | None = None
    frontier_id: str | None = None
    retries: int = 0
    segments: int = 0
    visited_frontiers: int = 0
    completion_verified: bool = False
    recent_failures: list[dict] = Field(default_factory=list, max_length=2)


class MissionInferenceTimeout(TimeoutError):
    pass


class ObjectiveExpired(ValueError):
    pass


class MissionObjective:
    def __init__(self, operation_id, decision, pose, now, deadline, clock=time.monotonic):
        self.clock = clock
        self.issued_at = now
        self.renewed_at = None
        self.last_renewal = None
        self.history = deque(maxlen=16)
        self.stop = None
        self.operation_id = operation_id
        self.action = decision.action
        self.expires_at = min(deadline, now + decision.objective_duration_s)
        self.travel_limit_m = decision.objective_travel_m
        self.travel_m = 0.
        self.pose = list(pose[:2])
        self.revision = 0
        self.status = "active"
        self.reason = "Bounded objective authorized"
        self.history.append({"event": "issued", "at_s": now, "expires_at_s": self.expires_at})

    def record_renewal(self, result, reason=None):
        self.last_renewal = {"at_s": self.clock(), "result": result, "rejection_reason": reason}
        self.history.append({"event": "renewal", **self.last_renewal, "expires_at_s": self.expires_at})

    def revoke(self, reason, initiator="controller"):
        if self.status == "active":
            self.status, self.reason = "ended", reason
            self.stop = {"at_s": self.clock(), "initiator": initiator, "reason": reason}
            self.history.append({"event": "stopped", **self.stop})

    def check(self, pose, now):
        if self.status == "active":
            if not all(math.isfinite(value) for value in pose):
                self.revoke("Objective odometry is invalid")
            else:
                self.travel_m += math.dist(self.pose, pose[:2])
                self.pose = list(pose[:2])
                if now >= self.expires_at:
                    self.revoke(f"OBJECTIVE_TIME_EXPIRED: Exploration time authorization expired; travelled {self.travel_m:.3f} of {self.travel_limit_m:.3f} m",
                        "authorization_watchdog")
                elif self.travel_m >= self.travel_limit_m:
                    self.revoke(f"OBJECTIVE_DISTANCE_LIMIT: Exploration travel limit reached ({self.travel_m:.3f} >= {self.travel_limit_m:.3f} m)",
                        "authorization_watchdog")
        if self.status != "active":
            raise ObjectiveExpired(self.reason)

    def renew(self, decision, pose, now, deadline):
        self.check(pose, now)
        if decision.action != self.action:
            raise ValueError("Only the same objective can renew active authorization")
        self.expires_at = min(deadline, now + decision.objective_duration_s)
        self.travel_limit_m = self.travel_m + decision.objective_travel_m
        self.revision += 1
        self.renewed_at = now
        self.record_renewal("accepted")

    def state(self, now):
        return {"operation_id": self.operation_id, "action": self.action, "status": self.status,
            "remaining_s": max(0., self.expires_at - now), "travel_m": self.travel_m,
            "remaining_travel_m": max(0., self.travel_limit_m - self.travel_m), "revision": self.revision,
            "reason": self.reason,
            "authorization": {"id": self.operation_id, "scope": "exploration_objective_not_trajectory",
                "issuer": "mission_supervisor", "renewal_owner": "mission_supervisor", "validator": "simulation_worker",
                "clock": "monotonic" if self.clock is time.monotonic else "test",
                "issued_at_s": self.issued_at, "renewed_at_s": self.renewed_at,
                "expires_at_s": self.expires_at, "last_renewal": self.last_renewal,
                "stop": self.stop, "recent_events": list(self.history)}}


class Mission:
    terminal = {"completed", "blocked", "failed", "cancelled"}

    def __init__(self, run_id, epoch, stop_revision, task_revision, deadline, clock=time.monotonic):
        self.identity = str(uuid4())
        self.authority = (run_id, epoch, stop_revision, task_revision)
        self.deadline = deadline
        self.clock = clock
        self.phase = "preparing"
        self.plan = None
        self.operation = None
        self.objective = None
        self.receipts = {}
        self.reason = "Preparing observed mission"
        self.rejections = 0
        self.home_id = None
        self.target_id = None

    def check(self, authority):
        if tuple(authority) != self.authority or self.phase in self.terminal:
            raise ValueError("Mission authority changed or mission ended")
        if self.clock() >= self.deadline:
            raise TimeoutError("Mission time budget exhausted")

    def configure(self, plan):
        if self.plan is not None:
            raise ValueError("Mission plan is already established; update the user instruction to replace it")
        if plan.kind != "explore" and not plan.target.strip():
            raise ValueError("A target description is required")
        self.plan = plan
        self.phase = "interpreting"

    def begin(self, phase, authority):
        self.check(authority)
        if self.operation is not None:
            raise ValueError("A mission operation is already active")
        if phase not in {"preparing", "exploring", "navigating", "inspecting", "returning", "recovering"}:
            raise ValueError("Unknown mission phase")
        self.phase = phase
        self.operation = str(uuid4())
        self.objective = None
        return self.operation

    def authorize_objective(self, identity, decision, authority, pose, *, renew=False):
        try:
            return self._authorize_objective(identity, decision, authority, pose, renew=renew)
        except (ValueError, TimeoutError) as error:
            if renew and self.objective and self.objective.operation_id == identity:
                self.objective.record_renewal("rejected", str(error))
            raise

    def _authorize_objective(self, identity, decision, authority, pose, *, renew=False):
        self.check(authority)
        if identity != self.operation or decision.action != "explore":
            raise ValueError("Objective requires the active exploration operation")
        if renew:
            self.check_objective(identity, authority, pose)
            self.objective.renew(decision, pose, self.clock(), self.deadline)
        elif self.objective is not None:
            raise ValueError("Objective already authorized; expired authorization cannot restart")
        else:
            self.objective = MissionObjective(identity, decision, pose, self.clock(), self.deadline, self.clock)
        return self.objective.state(self.clock())

    def check_objective(self, identity, authority, pose):
        try:
            self.check(authority)
            if identity != self.operation or self.objective is None or self.objective.operation_id != identity:
                raise ValueError("Late or mismatched objective operation")
            self.objective.check(pose, self.clock())
        except (ValueError, TimeoutError):
            if self.objective:
                self.objective.revoke("Objective authority invalidated")
            raise

    def end_operation(self, identity, authority, receipt=None):
        self.check(authority)
        if identity != self.operation:
            raise ValueError("Late or mismatched mission operation")
        if self.objective:
            self.objective.revoke("Mission operation ended", "supervisor")
        self.operation = None
        if receipt:
            self.receipts[receipt] = {"operation_id": identity, "verified_at": self.clock()}
        self.phase = "interpreting"

    def complete(self, authority):
        self.check(authority)
        if not self.plan or self.operation:
            raise ValueError("Mission cannot finish before planning and execution")
        required = "exploration" if self.plan.kind == "explore" else "target"
        if required not in self.receipts or (self.plan.return_home and "return" not in self.receipts):
            raise ValueError("Requested mission stages are not verified")
        self.phase = "completed"
        self.reason = ("Target identified in the current camera; no approach or entry requested"
            + ("; returned Home" if self.plan.return_home else "")
            if self.plan.completion == "identify" else "Requested mission stages verified; independent evaluation remains separate")

    def finish(self, phase, reason):
        if phase not in self.terminal:
            raise ValueError("Unknown terminal mission phase")
        if self.phase not in self.terminal:
            if self.objective:
                self.objective.revoke(reason)
            self.phase, self.reason, self.operation = phase, reason, None

    def state(self):
        return {"mission_id": self.identity, "phase": self.phase, "plan": self.plan.model_dump() if self.plan else None,
            "remaining_s": max(0., self.deadline - self.clock()), "operation_id": self.operation,
            "deadline_clock": "monotonic" if self.clock is time.monotonic else "test", "deadline_at_s": self.deadline,
            "objective": self.objective.state(self.clock()) if self.objective else None,
            "home_id": self.home_id, "target_id": self.target_id, "receipts": self.receipts.copy(),
            "reason": self.reason, "rejections": self.rejections}