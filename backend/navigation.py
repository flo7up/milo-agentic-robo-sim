from collections import OrderedDict, deque
import math
import time
from typing import Annotated, Literal

import numpy as np
import pybullet as bullet
from pydantic import Field, model_validator

from backend.contracts import StrictModel
from backend.simulation import MotionError, TIMESTEP


CONTINUOUS_SPEED_MPS = .5
LINEAR_ACCELERATION_MPS2 = .3


class Skill(StrictModel):
    skill: Literal["inspect_room", "locate_doorway", "approach", "cross"]
    goal: str = Field(min_length=1, max_length=240)


class NavigationPlan(StrictModel):
    expected_revision: int = Field(ge=0)
    steps: list[Skill] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def ordered_skills(self):
        order = ["inspect_room", "locate_doorway", "approach", "cross"]
        if any(step.skill != order[index % 4] for index, step in enumerate(self.steps)):
            raise ValueError("Plan skills must follow inspect_room, locate_doorway, approach, cross in order.")
        return self


class HeadSegment(StrictModel):
    kind: Literal["head"]
    yaw_rad: float = Field(ge=-1.2, le=1.2)
    pitch_rad: float = Field(ge=-.5, le=1)
    duration_s: float = Field(ge=.2, le=1)


class DriveSegment(StrictModel):
    kind: Literal["drive"]
    linear_mps: float = Field(ge=-.15, le=.15)
    angular_radps: float = Field(ge=-.5, le=.5)
    duration_s: float = Field(ge=.2, le=1)


class ReplaceMotionBuffer(StrictModel):
    expected_revision: int = Field(ge=0)
    segments: list[Annotated[HeadSegment | DriveSegment, Field(discriminator="kind")]] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def bounded_horizon(self):
        if sum(segment.duration_s for segment in self.segments) > 2:
            raise ValueError("Motion buffer must not exceed two simulated seconds.")
        if len({segment.kind for segment in self.segments}) != 1:
            raise ValueError("A buffer must control only the head or only the base.")
        return self


class CompleteNavigationSkill(StrictModel):
    expected_revision: int = Field(ge=0)
    evidence: str = Field(min_length=10, max_length=500)


class LocalSubgoal(StrictModel):
    expected_revision: int = Field(ge=0)
    goal: str = Field(min_length=1, max_length=240)


NAVIGATION_TOOLS = {
    "set_navigation_plan": NavigationPlan,
    "replace_motion_buffer": ReplaceMotionBuffer,
    "complete_navigation_skill": CompleteNavigationSkill,
    "begin_local_subgoal": LocalSubgoal,
}
NAVIGATION_DESCRIPTIONS = {
    "begin_local_subgoal": "Begin a fresh, stopped local-policy subgoal after supervisor feedback.",
    "set_navigation_plan": "Replace the task plan, brake, and cancel its old pending skills. Order: inspect_room, locate_doorway, approach, cross. Use the current navigation revision.",
    "replace_motion_buffer": "Replace only future motion at a control-slice boundary. Up to four segments, two seconds total. Inspection is head-only; locating permits head scans or base turns, not translation. Refill before remaining_s reaches zero. This returns acceptance, not motion completion.",
    "complete_navigation_skill": "Brake and complete the active skill after reviewing fresh camera/sensor feedback. Inspection/locating require a measured scan; approach/cross require movement. Explain visual evidence. Doorway/room identity is a model judgment, not an automatic semantic score.",
}


class NavigationRuntime:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.revision = 0
        self.steps = []
        self.index = 0
        self.status = "idle"
        self.reason = ""
        self.buffer = deque()
        self.velocity = np.zeros(2)
        self.tickets = OrderedDict()
        self.expires_at = 0
        self.sensed_at = 0
        self.skill_deadline = 0
        self.skill_started_at = 0
        self.recover_deadline = False
        self.start_odometry = np.zeros(3)
        self.scan_min = self.scan_max = 0
        self.travel = 0
        self.authorized_travel_m = None
        self.recover_clearance = False
        self.continuous_motion = False

    def state(self):
        return {"revision": self.revision, "status": self.status, "reason": self.reason,
            "active_step": self.index if self.index < len(self.steps) else None,
            "steps": [dict(step) for step in self.steps],
            "remaining_s": round(sum(entry[1] for entry in self.buffer) * TIMESTEP, 3),
            "expires_in_s": round(max(0, self.expires_at - self.clock()), 3) if self.buffer else 0,
            "skill_deadline_in_s": round(max(0, self.skill_deadline - self.clock()), 3),
            "effective_skill_budget_s": round(max(0, self.skill_deadline - self.skill_started_at), 3),
            "deadline_recovery_enabled": self.recover_deadline,
            "velocity_mps_radps": self.velocity.tolist(), "travel_m": round(self.travel, 3),
            "scan_span_rad": round(self.scan_max - self.scan_min, 3)}

    def observe(self, sim, observation):
        self.tickets[observation.seq] = (self.revision, self.clock(), sim.odometry.copy(), bool(self.buffer))
        while len(self.tickets) > 16:
            self.tickets.popitem(last=False)
        return self.state()

    def validate_ticket(self, sim, sequence, revision):
        ticket = self.tickets.get(sequence)
        if sim.cancel.is_set():
            raise MotionError("CANCELLED", "Navigation was stopped.")
        if not ticket or revision != self.revision or ticket[0] != self.revision:
            raise MotionError("STALE_PLAN", "Navigation revision changed. Observe again before replacing motion.")
        if self.clock() - ticket[1] > (2 if ticket[3] else 45):
            raise MotionError("STALE_OBSERVATION", "Navigation observation expired.")
        delta = sim.odometry - ticket[2]
        if np.linalg.norm(delta[:2]) > .25 or abs(delta[2]) > .35:
            raise MotionError("STALE_OBSERVATION", "Robot moved too far since the planning observation.")

    def brake(self, sim, status, reason):
        self.buffer.clear()
        self.velocity[:] = 0
        self.expires_at = 0
        self.revision += 1
        self.tickets.clear()
        self.status, self.reason = status, reason
        sim.hold_current()

    def cancel(self, sim, reason="Navigation cancelled"):
        self.brake(sim, "cancelled", reason)
        for step in self.steps:
            if step["status"] in {"pending", "running"}:
                step["status"] = "cancelled"

    def fail(self, sim, reason):
        self.brake(sim, "failed", reason)
        for step in self.steps:
            if step["status"] == "running":
                step["status"] = "failed"
            elif step["status"] == "pending":
                step["status"] = "cancelled"
        sim.stop()

    def begin_skill(self, sim):
        self.start_odometry = sim.odometry.copy()
        self.authorized_travel_m = None
        head = bullet.getJointState(sim.robot, sim.joints["head_yaw"], physicsClientId=sim.client)[0]
        self.scan_min = self.scan_max = head + sim.odometry[2]
        self.travel = 0
        self.skill_started_at = self.clock()
        self.skill_deadline = self.skill_started_at + 60
        self.steps[self.index]["status"] = "running"
        self.status, self.reason = "awaiting_feedback", "Review the camera and supply the next skill buffer."

    def apply(self, sim, name, arguments, sequence):
        self.validate_ticket(sim, sequence, arguments.expected_revision)
        if name == "begin_local_subgoal":
            if self.buffer or sim.proximity_sensors().collisions:
                raise MotionError("SKILL_PRECONDITION", "Stop with clear contacts before changing a local subgoal.")
            self.cancel(sim, "Supervisor issued a fresh local subgoal")
            self.steps = [{"skill": "approach", "goal": arguments.goal, "status": "pending", "evidence": "Supervisor instruction; not task success"}]
            self.index = 0
            self.begin_skill(sim)
            return
        if name == "set_navigation_plan":
            self.cancel(sim, "Plan replaced")
            self.steps = [{**step.model_dump(), "status": "pending", "evidence": ""} for step in arguments.steps]
            self.index = 0
            self.begin_skill(sim)
            return
        if self.index >= len(self.steps) or self.steps[self.index]["status"] != "running":
            raise MotionError("NO_ACTIVE_SKILL", "Set a navigation plan before executing skills.")
        skill = self.steps[self.index]["skill"]
        if name == "complete_navigation_skill":
            if sim.proximity_sensors().collisions:
                raise MotionError("SKILL_PRECONDITION", "Clear contact before completing a navigation skill.")
            if skill in {"inspect_room", "locate_doorway"} and self.scan_max - self.scan_min < .4:
                raise MotionError("SKILL_PRECONDITION", "Inspect more directions before completing this scan.")
            if skill in {"approach", "cross"} and np.linalg.norm(sim.odometry[:2] - self.start_odometry[:2]) < .1:
                raise MotionError("SKILL_PRECONDITION", "At least 0.1 m of measured movement is required.")
            self.brake(sim, "awaiting_feedback", "Skill completed; review the next objective.")
            self.steps[self.index].update(status="completed", evidence=arguments.evidence)
            self.index += 1
            if self.index == len(self.steps):
                self.status, self.reason = "completed", "Plan finished; semantic evidence is model-reported."
            else:
                self.begin_skill(sim)
            return
        for segment in arguments.segments:
            limit = CONTINUOUS_SPEED_MPS if self.continuous_motion else .15
            if segment.kind == "drive" and abs(segment.linear_mps) > limit:
                raise MotionError("VELOCITY_LIMIT", "Motion exceeds this controller's speed limit.")
            if skill == "inspect_room" and segment.kind != "head":
                raise MotionError("SKILL_PRECONDITION", "Room inspection controls the head only.")
            if skill == "locate_doorway" and segment.kind == "drive" and segment.linear_mps != 0:
                raise MotionError("SKILL_PRECONDITION", "Locate the opening before approaching it.")
        if sim.held:
            raise MotionError("OBJECT_HELD", "Release carried objects before buffered navigation.")
        self.buffer = deque([segment, math.ceil(segment.duration_s / TIMESTEP), False] for segment in arguments.segments)
        self.revision += 1
        self.tickets.clear()
        self.status, self.reason = "running", "Executing a bounded skill buffer."
        self.expires_at = self.clock() + sum(segment.duration_s for segment in arguments.segments) + .5
        self.sensed_at = self.clock()

    def check_clearance(self, sim, linear, angular):
        sensors = sim.proximity_sensors()
        if sensors.collisions:
            raise MotionError("CONTACT_STOP", "Contact detected during buffered motion.")
        directions = {"front", "front_left", "front_right"} if linear >= 0 else {"rear", "rear_left", "rear_right"}
        stopping_distance = linear * linear / (2 * LINEAR_ACCELERATION_MPS2) + abs(linear) * .05
        if any(reading.direction in directions and reading.status == "hit" and reading.distance_m < max(.18, stopping_distance + .08)
               for reading in sensors.distances) and abs(linear) > .001:
            raise MotionError("CLEARANCE_STOP", "Insufficient distance-sensor clearance.")
        sim._sync_planner()
        position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        roll, pitch, yaw = bullet.getEulerFromQuaternion(orientation)
        if abs(roll) > .25 or abs(pitch) > .25:
            raise MotionError("UNSTABLE_BASE", "Base tilt exceeds the buffered-navigation limit.")
        floor = next(item for item in sim.objects if item["name"] == "floor")
        if not any(contact[9] > .01 for contact in bullet.getContactPoints(sim.robot, floor["id"], physicsClientId=sim.client)):
            raise MotionError("UNSTABLE_BASE", "No supporting floor contact during navigation.")
        floor_lower, floor_upper = bullet.getAABB(floor["id"], physicsClientId=sim.planner)
        for item in sim.objects:
            if item["name"] == "floor" or item.get("marker"):
                continue
            velocity = sim.kinematic_velocities.get(item["id"])
            if velocity is None:
                velocity = np.array(bullet.getBaseVelocity(item["id"], physicsClientId=sim.client)[0])
            speed = float(np.linalg.norm(velocity))
            if speed < .02:
                continue
            obstacle_position, obstacle_rotation = bullet.getBasePositionAndOrientation(item["id"], physicsClientId=sim.client)
            try:
                prediction_s = max(.8, abs(linear) / LINEAR_ACCELERATION_MPS2 + .05)
                for horizon in np.linspace(0, prediction_s, math.ceil(prediction_s / .1) + 1):
                    heading = yaw + angular * horizon
                    destination = [position[0] + linear * horizon * math.cos(yaw + angular * horizon / 2),
                                   position[1] + linear * horizon * math.sin(yaw + angular * horizon / 2), position[2]]
                    bullet.resetBasePositionAndOrientation(sim.shadow, destination,
                        bullet.getQuaternionFromEuler([roll, pitch, heading]), physicsClientId=sim.planner)
                    bullet.resetBasePositionAndOrientation(item["id"], np.array(obstacle_position) + velocity * horizon,
                        obstacle_rotation, physicsClientId=sim.planner)
                    if bullet.getClosestPoints(sim.shadow, item["id"], .08 + speed * .05, physicsClientId=sim.planner):
                        raise MotionError("CLEARANCE_STOP", "Moving obstacle approaches the robot's predicted swept volume.")
            finally:
                bullet.resetBasePositionAndOrientation(item["id"], obstacle_position, obstacle_rotation, physicsClientId=sim.planner)
        lookahead_s = max(.4, abs(linear) / (2 * LINEAR_ACCELERATION_MPS2) + .05)
        for horizon in np.linspace(0, lookahead_s, math.ceil(lookahead_s / .1) + 1):
            heading = yaw + angular * horizon
            destination = [position[0] + linear * horizon * math.cos(yaw + angular * horizon / 2),
                           position[1] + linear * horizon * math.sin(yaw + angular * horizon / 2), position[2]]
            bullet.resetBasePositionAndOrientation(sim.shadow, destination,
                bullet.getQuaternionFromEuler([roll, pitch, heading]), physicsClientId=sim.planner)
            for index in [-1, *sim.joints.values()]:
                lower, upper = bullet.getAABB(sim.shadow, index, physicsClientId=sim.planner)
                if any(lower[axis] < floor_lower[axis] or upper[axis] > floor_upper[axis] for axis in (0, 1)):
                    raise MotionError("CLEARANCE_STOP", "Buffered motion would leave the floor.")
            for item in sim.objects:
                if item["name"] == "floor" or item.get("marker"):
                    continue
                if bullet.getClosestPoints(sim.shadow, item["id"], .015, physicsClientId=sim.planner):
                    raise MotionError("CLEARANCE_STOP", "Buffered motion lacks whole-robot clearance.")

    def expire_skill(self, sim):
        if sim.cancel.is_set():
            self.cancel(sim, "Navigation interrupted by Stop.")
            return
        elapsed = max(0., self.clock() - self.skill_started_at)
        budget = max(0., self.skill_deadline - self.skill_started_at)
        details = f"Local skill deadline expired at {elapsed:.3f} s since start (effective deadline {budget:.3f} s)."
        if self.recover_deadline:
            self.cancel(sim, "MOTION_LEASE_EXPIRED: " + details + " Motion discarded; fresh stopped feedback required.")
        else:
            self.fail(sim, "SKILL_TIMEOUT: " + details)

    def tick(self, sim):
        if (self.status in {"running", "awaiting_feedback"} and self.index < len(self.steps) and
            self.steps[self.index]["status"] == "running" and self.clock() >= self.skill_deadline):
            self.expire_skill(sim)
            return
        if not self.buffer:
            return
        try:
            if self.clock() - self.sensed_at > .75:
                raise MotionError("FEEDBACK_EXPIRED", "Local motion feedback expired.")
            segment = self.buffer[0][0]
            linear, angular = (segment.linear_mps, segment.angular_radps) if segment.kind == "drive" else (0, 0)
            self.check_clearance(sim, linear, angular)
            if np.max(np.abs(self.velocity - [linear, angular])) > .01:
                self.check_clearance(sim, *self.velocity)
            self.sensed_at = self.clock()
            for _ in range(12):
                if not self.buffer:
                    break
                segment, remaining, initialized = self.buffer[0]
                if not initialized and segment.kind == "head":
                    sim.targets[sim.joints["head_yaw"]] = (segment.yaw_rad, 5)
                    sim.targets[sim.joints["head_pitch"]] = (segment.pitch_rad, 5)
                    sim._hold()
                self.buffer[0][2] = True
                def control(tick):
                    if self.clock() >= self.skill_deadline:
                        raise MotionError("SKILL_TIMEOUT", "Local skill deadline expired.")
                    if self.clock() >= self.expires_at:
                        raise MotionError("BUFFER_EXPIRED", "Motion buffer expired.")
                    horizon = sum(entry[1] for entry in self.buffer) * TIMESTEP
                    desired = np.array([segment.linear_mps, segment.angular_radps]) if segment.kind == "drive" else np.zeros(2)
                    acceleration = np.array([LINEAR_ACCELERATION_MPS2, 1.0])
                    desired = np.clip(desired, -acceleration * horizon, acceleration * horizon)
                    self.velocity += np.clip(desired - self.velocity, -acceleration * TIMESTEP, acceleration * TIMESTEP)
                    speeds = np.array([self.velocity[0] - self.velocity[1] * .19, self.velocity[0] + self.velocity[1] * .19]) / .09
                    for index, speed in zip(sim.wheels, speeds):
                        bullet.setJointMotorControl2(sim.robot, index, bullet.VELOCITY_CONTROL,
                                                    targetVelocity=speed, force=5, physicsClientId=sim.client)
                prior = sim.path_length
                sim._ticks(1, control)
                self.travel += sim.path_length - prior
                head = bullet.getJointState(sim.robot, sim.joints["head_yaw"], physicsClientId=sim.client)[0] + sim.odometry[2]
                self.scan_min, self.scan_max = min(self.scan_min, head), max(self.scan_max, head)
                travel_limit = (self.authorized_travel_m if self.continuous_motion and self.authorized_travel_m is not None
                    else 2 if self.steps[self.index]["skill"] == "approach" else 1.5)
                if self.travel > travel_limit:
                    raise MotionError("SKILL_DISTANCE_LIMIT", "Review the plan before travelling further.")
                self.buffer[0][1] -= 1
                if remaining == 1:
                    self.buffer.popleft()
            if not self.buffer:
                self.brake(sim, "awaiting_feedback", "Buffer exhausted. Fresh feedback is required; old replacements are invalid.")
        except MotionError as error:
            if error.code == "CANCELLED":
                self.cancel(sim, "Navigation interrupted by Stop.")
            elif error.code == "SKILL_TIMEOUT":
                self.expire_skill(sim)
            elif error.code == "BUFFER_EXPIRED":
                self.brake(sim, "awaiting_feedback", "Motion buffer expired; braked and discarded pending motion. Fresh feedback is required.")
            elif error.code == "CLEARANCE_STOP" and self.recover_clearance:
                self.brake(sim, "awaiting_feedback", f"CLEARANCE_STOP: {error}")
            else:
                self.fail(sim, f"{error.code}: {error}")