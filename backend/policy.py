import asyncio
import base64
from contextlib import suppress
import math
from collections import deque
import time
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
import httpx

from backend.contracts import StrictModel
from backend.robot import ARM_LIMITS


JOINT_NAMES = [f"left_joint_{number}" for number in range(1, 7)]
ACTION_NAMES = [*JOINT_NAMES, "left_gripper_opening_m"]


class PolicyConfig(StrictModel):
    endpoint: str = "http://127.0.0.1:8085"

    @field_validator("endpoint")
    @classmethod
    def loopback_only(cls, value):
        parsed = urlsplit(value)
        if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or
                parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Policy endpoint must be a loopback HTTP origin")
        return value.rstrip("/")


class PolicyMetadata(StrictModel):
    backend: Literal["smolvla"]
    checkpoint: str = Field(min_length=1, max_length=500)
    embodiment: Literal["milo-left-arm-v1"]
    action_names: list[str]
    camera_key: Literal["observation.images.head"]
    fps: Literal[20]
    trained_for_milo: bool
    skills: list[Literal["pick_place"]] = Field(min_length=1, max_length=1)

    @field_validator("action_names")
    @classmethod
    def exact_action_order(cls, values):
        if values != ACTION_NAMES:
            raise ValueError("Checkpoint action order must match six left-arm radians and gripper opening in meters")
        return values


class StartSkill(StrictModel):
    expected_revision: int = Field(ge=0)
    skill: Literal["pick_place"]
    instruction: str = Field(min_length=1, max_length=500)
    timeout_s: float = Field(default=30, ge=2, le=60)


class EndSkill(StrictModel):
    expected_revision: int = Field(ge=0)
    reason: str = Field(min_length=1, max_length=500)


SKILL_TOOLS = {"start_skill": StartSkill, "cancel_skill": EndSkill, "complete_skill": EndSkill}
SKILL_DESCRIPTIONS = {
    "start_skill": "Start a trained left-arm pick/place policy with a concise task instruction. Base, head and right arm stay fixed. Acceptance is not task success.",
    "cancel_skill": "Cancel the active skill, brake, and invalidate pending policy actions.",
    "complete_skill": "Stop the skill after reviewing camera and gripper feedback. Completion is your semantic assessment, not a verified task score.",
}


class PolicyTicket(StrictModel):
    run_id: str
    episode_epoch: int = Field(ge=0)
    revision: int = Field(ge=0)
    observation_seq: int = Field(ge=0)


ActionVector = Annotated[list[float], Field(min_length=7, max_length=7)]


class PolicyChunk(StrictModel):
    ticket: PolicyTicket
    actions: list[ActionVector] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def bounded_values(self):
        for action in self.actions:
            if any(not low <= value <= high for value, (low, high) in zip(action[:6], ARM_LIMITS)):
                raise ValueError("Policy action exceeds an arm joint limit")
            if not 0 <= action[6] <= .11:
                raise ValueError("Policy gripper opening exceeds limits")
        return self


def future_actions(chunk, expected_ticket, captured_at, now=None):
    now = time.monotonic() if now is None else now
    age = now - captured_at
    if chunk.ticket != expected_ticket or age < 0 or age > 1:
        raise ValueError("Stale policy response")
    skipped = math.ceil(age * 20)
    actions = chunk.actions[skipped:skipped + 20]
    if not actions:
        raise ValueError("Policy chunk contains no future actions")
    return actions


class SkillRuntime:
    def __init__(self, metadata, clock=time.monotonic):
        self.metadata = metadata
        self.clock = clock
        self.revision = 0
        self.motion_revision = 0
        self.status = "idle"
        self.skill = None
        self.instruction = ""
        self.reason = "No skill active"
        self.buffer = deque()
        self.pending = None
        self.deadline = 0
        self.expires_at = 0
        self.policy_requests = 0
        self.rejected_chunks = 0
        self.policy_latency_s = None
        self.completion_source = None
        self.segment_tick = 0
        self.last_tick = None

    @property
    def active(self):
        return self.status in {"running", "awaiting_policy"}

    def state(self):
        return {"revision": self.revision, "motion_revision": self.motion_revision, "status": self.status, "skill": self.skill,
                "instruction": self.instruction, "reason": self.reason,
                "remaining_s": max(0, (len(self.buffer) * 12 - self.segment_tick) / 240),
                "checkpoint": self.metadata.checkpoint, "policy_requests": self.policy_requests,
                "rejected_chunks": self.rejected_chunks, "policy_latency_s": self.policy_latency_s,
                "completion_source": self.completion_source}

    def brake(self, sim, status, reason):
        import numpy as np
        import pybullet as bullet
        self.buffer.clear()
        self.segment_tick = 0
        self.pending = None
        self.motion_revision += 1
        self.status, self.reason = status, reason
        self.velocity = np.zeros(7)
        sim.hold_current()
        if hasattr(self, "position"):
            self.position = np.array([bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0]
                                      for joint in sim.arms["left"]] + [sim.gripper_sensors()["left"]["aperture_m"]])
            self.position[6] = np.clip(self.position[6], 0, .11)

    def cancel(self, sim, reason="Skill cancelled"):
        self.revision += 1
        self.brake(sim, "cancelled", reason)

    def fail(self, sim, reason):
        self.revision += 1
        self.brake(sim, "failed", reason)
        sim.stop()

    def apply(self, sim, name, arguments):
        import numpy as np
        import pybullet as bullet
        from backend.simulation import MotionError
        if sim.cancel.is_set():
            raise MotionError("CANCELLED", "Skill execution was stopped")
        if arguments.expected_revision != self.revision:
            raise MotionError("STALE_SKILL", "Skill revision changed; observe again")
        if name != "start_skill":
            if not self.active:
                raise MotionError("NO_ACTIVE_SKILL", "No skill is running")
            self.revision += 1
            self.brake(sim, "completed" if name == "complete_skill" else "cancelled", arguments.reason)
            self.completion_source = "supervisor" if name == "complete_skill" else None
            return
        if not self.metadata.trained_for_milo:
            raise MotionError("POLICY_NOT_TRAINED", "A Milo-trained checkpoint is required; base-model actions are disabled")
        if self.active:
            raise MotionError("CONTROL_CONFLICT", "Cancel the active skill before replacing its instruction")
        if arguments.skill not in self.metadata.skills:
            raise MotionError("UNSUPPORTED_SKILL", "Checkpoint does not support this skill")
        linear, angular = bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)
        if np.linalg.norm(linear) > .03 or np.linalg.norm(angular) > .05 or sim.proximity_sensors().collisions:
            raise MotionError("SKILL_PRECONDITION", "Park the base and clear contacts before manipulation")
        if sim.held:
            raise MotionError("SKILL_PRECONDITION", "Release assisted grasps before switching to policy control")
        self.revision += 1
        self.brake(sim, "awaiting_policy", "Waiting for a fresh policy chunk")
        self.skill, self.instruction = arguments.skill, arguments.instruction.strip()
        self.deadline = self.clock() + arguments.timeout_s
        self.completion_source = None
        self.position = np.array([bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0]
                                  for joint in sim.arms["left"]] + [sim.gripper_sensors()["left"]["aperture_m"]])
        self.position[6] = np.clip(self.position[6], 0, .11)
        self.fixed_base = np.array(bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0])
        self.fixed_orientation = np.array(bullet.getEulerFromQuaternion(
            bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[1]))
        self.fixed_joints = {joint: bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0]
                             for joint in sim.arms["right"] + [sim.joints["head_yaw"], sim.joints["head_pitch"]]}
        self.last_tick = self.clock()

    def ticket(self, sim, observation, captured_at=None):
        from backend.simulation import MotionError
        if not self.active or sim.cancel.is_set():
            raise MotionError("CANCELLED", "Skill no longer accepts policy requests")
        ticket = PolicyTicket(run_id=sim.run_id, episode_epoch=sim.epoch,
                              revision=self.motion_revision, observation_seq=observation.seq)
        self.pending = (ticket, self.clock() if captured_at is None else captured_at)
        self.policy_requests += 1
        return ticket

    def accept(self, sim, chunk):
        import numpy as np
        from backend.simulation import MotionError
        chunk = PolicyChunk.model_validate(chunk.model_dump())
        if not self.active or sim.cancel.is_set() or not self.pending:
            raise MotionError("STALE_POLICY", "Policy request was cancelled or expired")
        ticket, captured_at = self.pending
        self.policy_latency_s = self.clock() - captured_at
        try:
            values = future_actions(chunk, ticket, captured_at, self.clock())
        except ValueError as error:
            raise MotionError("STALE_POLICY", str(error)) from error
        position, velocity = self.position.copy(), self.velocity.copy()
        speed_limit = np.array([.8] * 6 + [.06])
        acceleration_limit = np.array([4.] * 6 + [.3])
        for value in values:
            target = np.array(value)
            next_velocity = (target - position) / .05
            if np.any(np.abs(next_velocity) > speed_limit + 1e-6):
                raise MotionError("POLICY_SPEED_LIMIT", "Policy chunk exceeds joint/gripper speed limits")
            if np.any(np.abs(next_velocity - velocity) > acceleration_limit * .05 + 1e-6):
                raise MotionError("POLICY_ACCELERATION_LIMIT", "Policy chunk transition is not smooth")
            sim._check_trajectory("left", position[:6], target[:6], opening=target[6])
            position, velocity = target, next_velocity
        if sim.cancel.is_set():
            raise MotionError("CANCELLED", "Policy chunk was invalidated during validation")
        if self.clock() > captured_at + (math.ceil(self.policy_latency_s * 20) + 1) / 20:
            raise MotionError("STALE_POLICY", "Policy chunk became stale during collision validation")
        self.buffer = deque(np.array(value) for value in values)
        self.segment_tick = 0
        self.segment_start = self.position.copy()
        self.expires_at = self.clock() + len(values) * .05 + .15
        self.pending = None
        self.status, self.reason = "running", "Executing validated policy chunk"

    def tick(self, sim):
        import numpy as np
        import pybullet as bullet
        from backend.simulation import MotionError
        if not self.active:
            return
        if sim.cancel.is_set():
            self.cancel(sim, "Interrupted by Stop")
            return
        if self.clock() >= self.deadline:
            self.fail(sim, "SKILL_TIMEOUT: Skill deadline reached")
            return
        if not self.buffer:
            return
        if self.clock() >= self.expires_at:
            self.brake(sim, "awaiting_policy", "Chunk expired; waiting for fresh feedback")
            return
        try:
            if self.segment_tick == 0:
                if sim.proximity_sensors().collisions:
                    raise MotionError("CONTACT_STOP", "Unexpected robot contact")
                base = np.array(bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0])
                orientation = np.array(bullet.getEulerFromQuaternion(
                    bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[1]))
                if np.linalg.norm(base - self.fixed_base) > .02 or np.max(np.abs(orientation - self.fixed_orientation)) > .05:
                    raise MotionError("BASE_MOVED", "Policy manipulation requires a stationary base")
                if any(abs(bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0] - value) > .08
                       for joint, value in self.fixed_joints.items()):
                    raise MotionError("FIXED_JOINT_MOVED", "Head and right arm must remain fixed during policy manipulation")
                actual = np.array([bullet.getJointState(sim.robot, joint, physicsClientId=sim.client)[0]
                                   for joint in sim.arms["left"]])
                if np.max(np.abs(actual - self.position[:6])) > .10:
                    raise MotionError("TRACKING_ERROR", "Arm did not track the bounded policy trajectory")
                sim._check_trajectory("left", self.position[:6], self.buffer[0][:6], opening=self.buffer[0][6])
            target = self.buffer[0]
            self.segment_tick += 1
            prior = self.position.copy()
            self.position = self.segment_start + (target - self.segment_start) * (self.segment_tick / 12)
            self.velocity = (self.position - prior) * 240
            for joint, value in zip(sim.arms["left"], self.position[:6]):
                sim.targets[joint] = (float(value), 20)
            for joint in sim.fingers["left"]:
                sim.targets[joint] = (float(self.position[6] / 2), 20)
            sim._hold()
            sim._brake()
            sim._ticks(1)
            if self.segment_tick == 12:
                self.buffer.popleft()
                self.segment_start = self.position.copy()
                self.segment_tick = 0
            if not self.buffer:
                self.brake(sim, "awaiting_policy", "Chunk exhausted; fresh feedback required")
        except MotionError as error:
            self.fail(sim, f"{error.code}: {error}")


class PolicyConnectionError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class LocalPolicyClient:
    def __init__(self, config):
        self.client = httpx.AsyncClient(base_url=config.endpoint, timeout=5, trust_env=False, follow_redirects=False)

    async def describe(self):
        try:
            response = await self.client.get("/policy")
            response.raise_for_status()
            return PolicyMetadata.model_validate(response.json())
        except httpx.RequestError as error:
            raise PolicyConnectionError("unavailable",
                "The SmolVLA left-arm server is offline or unreachable. This mode requires a milo-left-arm-v1 pick/place checkpoint, not a navigation checkpoint. Fine-tuned navigation checkpoints are currently CLI-only. Single step and Navigation plan use the selected LLM and do not require this server.") from error
        except (httpx.HTTPStatusError, ValueError) as error:
            raise PolicyConnectionError("incompatible",
                "This endpoint is not a compatible milo-left-arm-v1 pick/place service. Navigation checkpoints cannot be used in Luna + SmolVLA arm mode.") from error

    async def predict(self, ticket, observation, image, instruction):
        response = await self.client.post("/predict", json={"ticket": ticket.model_dump(),
            "observation": observation.model_dump(), "image": base64.b64encode(image).decode("ascii"),
            "instruction": instruction})
        response.raise_for_status()
        return PolicyChunk.model_validate(response.json())

    async def close(self):
        await self.client.aclose()


async def check_policy_readiness(config, factory=LocalPolicyClient):
    client = factory(config)
    try:
        async with asyncio.timeout(3):
            metadata = await client.describe()
        if not metadata.trained_for_milo:
            return {"ready": False, "status": "untrained", "message":
                "The loaded checkpoint is not declared trained for Milo. Base-model execution is disabled.", "metadata": None}
        return {"ready": True, "status": "ready", "message": "Compatible policy server connected; task success is not verified.",
                "metadata": metadata.model_dump()}
    except PolicyConnectionError as error:
        return {"ready": False, "status": error.status, "message": str(error), "metadata": None}
    except (TimeoutError, httpx.RequestError):
        return {"ready": False, "status": "unavailable", "message":
            "The SmolVLA left-arm server did not respond. This mode requires a milo-left-arm-v1 pick/place service; fine-tuned navigation checkpoints are currently CLI-only.", "metadata": None}
    except ValueError:
        return {"ready": False, "status": "incompatible", "message": "Policy metadata is incompatible with Milo.", "metadata": None}
    finally:
        await client.close()


class PolicyRunner:
    def __init__(self, worker, config, factory=LocalPolicyClient, trace=None):
        self.worker = worker
        self.client = factory(config)
        self.trace = trace
        self.task = None
        self.latency = .25

    async def start(self, stop_revision):
        metadata = await self.client.describe()
        if not metadata.trained_for_milo:
            raise ValueError("SmolVLA requires a Milo-trained checkpoint; the base model cannot execute robot actions")
        await self.worker.begin_skill_mode(metadata, stop_revision)
        self.task = asyncio.create_task(self.run())

    def event(self, title, payload, image=None):
        if self.trace:
            self.trace("policy", title, payload, image=image)

    async def request(self, ticket, observation, image, instruction):
        pending = asyncio.create_task(self.client.predict(ticket, observation, image, instruction))
        try:
            while not pending.done():
                state = self.worker.latest.get("skill")
                if (self.worker.closed or self.worker.latest.get("stopped") or not state or
                        state["motion_revision"] != ticket.revision or state["status"] not in {"running", "awaiting_policy"}):
                    return None
                await asyncio.wait({pending}, timeout=.025)
            return await pending
        finally:
            if not pending.done():
                pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending

    async def run(self):
        from backend.simulation import MotionError
        rejected = 0
        try:
            while not self.worker.closed and not self.worker.latest.get("stopped"):
                state = self.worker.latest.get("skill")
                if not state or state["status"] not in {"running", "awaiting_policy"}:
                    await asyncio.sleep(.025)
                    continue
                if state["remaining_s"] > min(.9, max(.4, self.latency + .15)):
                    await asyncio.sleep(.025)
                    continue
                ticket, observation, image = await self.worker.policy_feedback()
                self.event("SmolVLA observation", {"ticket": ticket.model_dump(),
                    "observation": observation.model_dump(), "instruction": state["instruction"]}, image)
                started = time.monotonic()
                chunk = await self.request(ticket, observation, image, state["instruction"])
                self.latency = time.monotonic() - started
                if chunk is None:
                    self.event("Policy response discarded", {"reason": "Skill or motion revision changed"})
                    continue
                try:
                    await self.worker.accept_policy(chunk)
                except MotionError as error:
                    rejected += 1
                    self.event("Policy chunk rejected", {"error": error.code, "message": str(error)})
                    if rejected >= 3 or error.code != "STALE_POLICY":
                        raise
                    continue
                rejected = 0
                self.event("Policy chunk accepted", {"ticket": chunk.ticket.model_dump(),
                    "actions": chunk.actions, "latency_s": self.latency,
                    "note": "Only unexpired future actions execute; acceptance is not task success"})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self.worker.closed:
                message = str(error) if isinstance(error, (MotionError, ValueError)) else "Local policy inference failed"
                await self.worker.fail_policy(message)
                self.event("Policy stopped", {"reason": message})

    async def close(self):
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
        await self.client.close()