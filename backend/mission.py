import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from backend.contracts import StrictModel


class MissionPlan(StrictModel):
    kind: Literal["explore", "object", "room", "place"]
    target: str = Field(default="", max_length=160)
    return_home: bool = False


class MissionDecision(StrictModel):
    action: Literal["plan", "explore", "select_object", "approach_object", "verify_object", "navigate_place", "observe_room", "look", "turn", "wait", "finish"]
    plan: MissionPlan | None = None
    object_goal_id: str | None = None
    object_label: str = Field(default="", max_length=60)
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    approach_side: Literal["front", "left", "right"] = "front"
    place_id: str | None = None
    frontier_id: str | None = None
    room_matches: bool | None = None
    evidence_text: str = Field(default="", max_length=500)
    yaw_rad: float = Field(default=0, ge=-1.5, le=1.5)
    pitch_rad: float = Field(default=.2, ge=-.7, le=1.15)
    turn_rad: float = Field(default=0, ge=-3.15, le=3.15)
    duration_s: float = Field(default=1, ge=.5, le=4)
    map_view: Literal["local", "overview"] = "local"
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def required_evidence(self):
        if self.action == "plan" and self.plan is None:
            raise ValueError("Planning requires the requested mission stages")
        if self.action in {"select_object", "verify_object"}:
            bounds = self.object_bounds
            if not bounds or not all(0 <= value <= 1 for value in bounds) or bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError("Object selection requires a current normalized image box")
        if self.action == "select_object" and not self.object_label.strip():
            raise ValueError("An observed object label is required")
        if self.action in {"approach_object", "verify_object"} and not self.object_goal_id:
            raise ValueError("Use the current object goal identity")
        if self.action in {"navigate_place", "observe_room"} and not self.place_id:
            raise ValueError("Use a supplied place identity")
        if self.action == "observe_room" and (self.room_matches is None or not self.evidence_text.strip()):
            raise ValueError("Room inspection requires a match assessment and current visual evidence")
        return self


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
        return self.operation

    def end_operation(self, identity, authority, receipt=None):
        self.check(authority)
        if identity != self.operation:
            raise ValueError("Late or mismatched mission operation")
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
        self.reason = "Requested mission stages verified; independent evaluation remains separate"

    def finish(self, phase, reason):
        if phase not in self.terminal:
            raise ValueError("Unknown terminal mission phase")
        if self.phase not in self.terminal:
            self.phase, self.reason, self.operation = phase, reason, None

    def state(self):
        return {"mission_id": self.identity, "phase": self.phase, "plan": self.plan.model_dump() if self.plan else None,
            "remaining_s": max(0., self.deadline - self.clock()), "operation_id": self.operation,
            "home_id": self.home_id, "target_id": self.target_id, "receipts": self.receipts.copy(),
            "reason": self.reason, "rejections": self.rejections}