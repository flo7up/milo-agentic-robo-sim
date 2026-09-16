from enum import Enum
from typing import Annotated, Literal, Protocol
import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Arm(str, Enum):
    left = "left"
    right = "right"


Duration = Annotated[float, Field(gt=0, le=2, description="Motion duration in seconds, at most 2 s.")]
SixJoints = Annotated[list[float], Field(min_length=6, max_length=6)]


class Empty(StrictModel):
    pass


class Stop(StrictModel):
    task_status: Literal["completed", "unachievable"] | None = None
    reason: str = Field(default="", max_length=1000)


class Wait(StrictModel):
    duration_s: Duration


class Drive(StrictModel):
    linear_mps: float = Field(ge=-.3, le=.3, description="Forward speed in m/s.")
    angular_radps: float = Field(ge=-.8, le=.8, description="Left-turn speed in rad/s.")
    duration_s: Duration


class Head(StrictModel):
    yaw_rad: float = Field(ge=-1.5, le=1.5)
    pitch_rad: float = Field(ge=-.7, le=1.15, description="Positive tilts down.")
    duration_s: Duration


class ArmJoints(StrictModel):
    arm: Arm
    joint_positions_rad: SixJoints
    duration_s: Duration


class EndEffector(StrictModel):
    arm: Arm
    position_m: Annotated[list[float], Field(min_length=3, max_length=3)]
    orientation_xyzw: Annotated[list[float], Field(min_length=4, max_length=4)]
    frame: Literal["base"]
    duration_s: Duration

    @field_validator("orientation_xyzw")
    @classmethod
    def unit_quaternion(cls, values):
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1) > .02:
            raise ValueError("Quaternion norm must be within 0.02 of one")
        return [value / norm for value in values]


class Gripper(StrictModel):
    arm: Arm
    opening_m: float = Field(ge=0, le=.11)
    max_force_n: float = Field(gt=0, le=35)


class Finish(StrictModel):
    summary: str = Field(max_length=1000)


class Answer(StrictModel):
    answer: str = Field(max_length=100)


TOOL_MODELS = {"observe": Empty, "wait": Wait, "drive_base": Drive, "set_head": Head,
               "set_arm_joints": ArmJoints, "move_end_effector": EndEffector,
               "set_gripper": Gripper, "stop": Stop, "finish_task": Finish,
               "submit_answer": Answer}
DESCRIPTIONS = {
    "observe": "Capture a fresh head RGB image and proprioception. No motion.",
    "wait": "Brake and hold the current joints for up to 2 simulated seconds, then observe. Allows a survey, charging, or settling to advance without commanding movement.",
    "drive_base": "Drive bounded differential wheels, then brake and observe.",
    "set_head": "Servo the head yaw and pitch, then observe.",
    "set_arm_joints": "Move six joints in radians through a collision-checked trajectory.",
    "move_end_effector": "Move the palm origin in meters in the base frame captured at acceptance. IK and collision checks may reject the target.",
    "set_gripper": "Set jaw opening in meters and bounded force. Assistance requires opposing physical finger contacts; no remote attachment.",
    "stop": "Interrupt motion and hold joints with wheel brakes. When the task has ended, report task_status as completed or unachievable with a brief reason.",
    "finish_task": "Request hidden evaluation. The summary does not establish success.",
    "submit_answer": "Submit the visible marker text for hidden exact evaluation."}


def tool_schemas(include_answer=False):
    return [{"type": "function", "name": name, "description": DESCRIPTIONS[name],
             "parameters": model.model_json_schema()} for name, model in TOOL_MODELS.items()
            if name != "submit_answer" or include_answer]


class Command(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    action_id: str = Field(min_length=1, max_length=120)
    observation_seq: int = Field(ge=0)
    tool: str
    arguments: dict = Field(default_factory=dict)


class ManualPlacement(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    observation_seq: int = Field(ge=0)
    xy_m: Annotated[list[float], Field(min_length=2, max_length=2)]


class JointSensor(StrictModel):
    name: str
    position: float
    velocity: float


class GripperSensor(StrictModel):
    aperture_m: float
    contact: list[bool]
    load_n: float


class BatterySensor(StrictModel):
    charge_pct: float = Field(ge=0, le=100)
    low: bool
    charging: bool


class DistanceReading(StrictModel):
    direction: str
    bearing_rad: float
    distance_m: float | None = Field(default=None, ge=0, le=2)
    status: Literal["hit", "clear", "occluded"]
    origin_base_m: list[float] | None = Field(default=None, min_length=3, max_length=3)


class CollisionReading(StrictModel):
    direction: Literal["front", "left", "rear", "right"]
    force_n: float = Field(ge=0)


class ProximitySensors(StrictModel):
    simulated_time_s: float
    max_range_m: float = 2
    distances: list[DistanceReading]
    collisions: list[CollisionReading]
    coverage: str = "Eight individual rays, not sectors; clear means no detected hit within 2 m of each ray origin, not a free body corridor."


class NavigationStepState(StrictModel):
    skill: Literal["inspect_room", "locate_doorway", "approach", "cross"]
    goal: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    evidence: str


class NavigationDiagnostics(StrictModel):
    clock: Literal["monotonic", "simulation", "test"] = "monotonic"
    authorization_id: str | None = None
    scope: str = "short_motion_buffer"
    issuer: str = "navigation_runtime"
    renewal_owner: str = "controller"
    task_id: str | None = None
    objective_id: str | None = None
    issued_at_s: float | None = None
    renewed_at_s: float | None = None
    expires_at_s: float | None = None
    last_renewal: dict | None = None
    last_controller_tick_at_s: float | None = None
    maximum_recent_tick_gap_s: float = 0.
    tick_window_s: float = 5.
    sensor_at_last_command: dict | None = None
    stop: dict | None = None
    recent_events: list[dict] = Field(default_factory=list, max_length=16)


class NavigationFeedback(StrictModel):
    revision: int
    status: Literal["idle", "running", "awaiting_feedback", "completed", "failed", "cancelled"]
    reason: str
    active_step: int | None
    steps: list[NavigationStepState]
    remaining_s: float
    expires_in_s: float
    skill_deadline_in_s: float = Field(default=0., ge=0)
    effective_skill_budget_s: float = Field(default=0., ge=0)
    deadline_recovery_enabled: bool = False
    velocity_mps_radps: list[float]
    travel_m: float
    scan_span_rad: float
    diagnostics: NavigationDiagnostics | None = None


class SkillFeedback(StrictModel):
    revision: int
    motion_revision: int
    status: Literal["idle", "running", "awaiting_policy", "completed", "cancelled", "failed"]
    skill: str | None
    instruction: str
    reason: str
    remaining_s: float
    checkpoint: str
    policy_requests: int
    rejected_chunks: int
    policy_latency_s: float | None
    completion_source: Literal["supervisor"] | None = None


class ObservedMapContext(StrictModel):
    version: Literal[1] = 1
    map_id: str
    revision: str
    run_id: str
    episode_epoch: int
    source_sequence: int
    captured_at: float
    age_s: float
    capture_clock: Literal["monotonic"] = "monotonic"
    age_basis: str = "Age of the paired current camera/depth capture; not age of all accumulated geometry."
    snapshot_built_at_monotonic_s: float | None = None
    geometry_updated_unix_s: float | None = None
    geometry_age_s: float | None = None
    geometry_age_basis: str = "Time since last map integration, not oldest cell observation; per-cell ages are unavailable."
    geometry_source: Literal["rolling_sensor", "accumulated_sensor_map"] = "rolling_sensor"
    frame: Literal["map", "wheel_odometry"]
    localization: str
    scope: Literal["local", "overview"] = "local"
    width: int = Field(ge=1, le=96)
    height: int = Field(ge=1, le=96)
    resolution_m: float = Field(gt=0)
    origin_m: list[float] = Field(min_length=2, max_length=2)
    cells: list[Literal[-1, 0, 100]] = Field(max_length=9216)
    robot_pose_m_rad: list[float] = Field(min_length=3, max_length=3)
    camera_yaw_rad: float
    destinations: list[dict] = Field(default_factory=list, max_length=24)
    trail_m: list[list[float]] = Field(default_factory=list, max_length=128)
    trail_source: Literal["review_samples", "worker_odometry"] = "review_samples"
    trail_truncated: bool = False
    trail_spacing_m: float | None = Field(default=None, gt=0)
    route_m: list[list[float]] = Field(default_factory=list, max_length=128)
    note: str = "Observed geometry only; unknown is not traversable. Labels are hypotheses. Revalidate all motion."

    @model_validator(mode="after")
    def grid_shape(self):
        if len(self.cells) != self.width * self.height:
            raise ValueError("Observed map dimensions must match its cells")
        return self


class AgentObservation(StrictModel):
    run_id: str
    episode_epoch: int
    seq: int
    wall_timestamp: float
    simulated_time_s: float
    frame_ref: str
    joints: list[JointSensor]
    grippers: dict[str, GripperSensor]
    head_rad: list[float]
    odometry_m_rad: list[float]
    bumpers: list[str]
    battery: BatterySensor | None = None
    proximity: ProximitySensors | None = None
    navigation: NavigationFeedback | None = None
    skill: SkillFeedback | None = None
    spatial: dict | None = None
    observed_map: ObservedMapContext | None = None
    sensor_profile: Literal["rgb_proprioception"] = "rgb_proprioception"


class SpatialSettings(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    enabled: bool


class DepthCalibration(StrictModel):
    width: int = Field(ge=16, le=640)
    height: int = Field(ge=16, le=480)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float
    near_m: float = Field(default=.015, gt=0)
    far_m: float = Field(default=12., gt=0)
    usable_range_m: float = Field(default=8., gt=0)

    @model_validator(mode="after")
    def validate_range(self):
        if not self.near_m < self.usable_range_m <= self.far_m:
            raise ValueError("Invalid depth clipping range")
        if not 0 <= self.cx <= self.width or not 0 <= self.cy <= self.height:
            raise ValueError("Invalid camera principal point")
        return self


class SpatialObservation(StrictModel):
    run_id: str
    episode_epoch: int = Field(ge=0)
    sequence: int = Field(ge=1)
    captured_at: float
    simulated_time_s: float = Field(ge=0)
    calibration: DepthCalibration
    head_rad: list[float] = Field(min_length=2, max_length=2)
    odometry_m_rad: list[float] = Field(min_length=3, max_length=3)
    depth_m: list[float | None]

    @model_validator(mode="after")
    def validate_depth(self):
        import math
        if len(self.depth_m) != self.calibration.width * self.calibration.height:
            raise ValueError("Depth dimensions do not match calibration")
        if not all(math.isfinite(value) for value in [self.captured_at, *self.head_rad, *self.odometry_m_rad]):
            raise ValueError("Spatial observation requires finite measured pose and timestamp")
        if any(value is not None and (not math.isfinite(value) or not self.calibration.near_m <= value <= self.calibration.usable_range_m)
               for value in self.depth_m):
            raise ValueError("Depth values must be valid axial meters or null")
        return self


class EvaluatorState(StrictModel):
    simulated_time_s: float
    robot_position: list[float]
    bodies: list[dict]
    retained_bodies: list[int]


class ToolResult(StrictModel):
    action_id: str
    status: Literal["ok", "error", "cancelled"]
    error: str | None = None
    message: str = ""
    actual_duration_s: float = 0
    interruption_reason: str | None = None
    sensor_deltas: dict = Field(default_factory=dict)
    observation: AgentObservation


class SimulationBackend(Protocol):
    def execute(self, command: Command) -> ToolResult: ...
    def close(self) -> None: ...


class RobotController(Protocol):
    def stop(self) -> None: ...


class ObservationProvider(Protocol):
    def observe(self) -> AgentObservation: ...
    def frame(self, reference: str) -> bytes: ...


class ChallengeEvaluator(Protocol):
    def evaluate(self, state: EvaluatorState) -> dict: ...


class RunRecorder(Protocol):
    def append(self, kind: str, payload: dict) -> None: ...