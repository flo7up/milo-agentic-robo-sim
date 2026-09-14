import math
import time
from collections import OrderedDict
from typing import Literal
from uuid import uuid4

from pydantic import Field

from backend.contracts import StrictModel
from backend.continuous_navigation import ContinuousMotionBuffer
from backend.navigation import LocalSubgoal, NavigationRuntime
from backend.simulation import MotionError


ROS_SPEED_MPS = .35


def capture_laser(sim):
    import numpy as np
    import pybullet as bullet
    position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
    rotation = np.array(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    laser_offset = np.array([.19, 0., .15])
    origin = np.array(position) + rotation @ laser_offset
    angles = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles)))) @ rotation.T
    hits = bullet.rayTestBatch([origin.tolist()] * len(angles), (origin + directions * 8.).tolist(),
                              physicsClientId=sim.client)
    ranges = [None if hit[0] == sim.robot else 8.01 if hit[0] < 0 else float(hit[2] * 8.) for hit in hits]
    return {"frame": "laser", "origin_m": [.19, 0., .305],
        "angle_min": -math.pi, "angle_increment": 2 * math.pi / len(angles),
        "range_min": .03, "range_max": 8., "ranges_m": ranges}


def capture_sensors(sim, sequence):
    import base64
    from backend.spatial import camera_in_odometry
    captured_unix_s = time.time()
    sensor, image = sim.capture_spatial(sequence)
    laser = capture_laser(sim)
    eye, optical_rotation = camera_in_odometry(sensor.head_rad, [0., 0., 0.])
    return {"run_id": sim.run_id, "episode_epoch": sim.epoch, "sequence": sequence,
        "captured_at": sensor.captured_at, "captured_unix_s": captured_unix_s,
        "simulated_time_s": sensor.simulated_time_s, "odometry_m_rad": sensor.odometry_m_rad,
        "footprint": sim.robot_footprint(), "calibration": sensor.calibration.model_dump(),
        "head_rgb_png": base64.b64encode(image).decode("ascii"), "depth_m": sensor.depth_m,
        "camera_origin_m": eye.tolist(), "camera_rotation": optical_rotation.tolist(),
        "laser": laser,
        "source": "simulated_sensors", "frame": "wheel_odometry"}


class RosStart(StrictModel):
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    target_m_rad: list[float] = Field(min_length=3, max_length=3)


class RosVelocity(StrictModel):
    session_id: str = Field(min_length=1, max_length=80)
    sensor_sequence: int = Field(ge=1)
    command_sequence: int = Field(ge=1)
    linear_mps: float = Field(ge=-ROS_SPEED_MPS, le=ROS_SPEED_MPS)
    angular_radps: float = Field(ge=-.5, le=.5)


class RosBridgeStatus(StrictModel):
    bridge_id: str = Field(min_length=1, max_length=80)
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    ready: bool
    message: str = Field(default="", max_length=200)


class RosGoalResult(StrictModel):
    session_id: str = Field(min_length=1, max_length=80)
    status: Literal["succeeded", "aborted", "cancelled"]
    message: str = Field(default="", max_length=200)


class RosNavigationSession:
    def __init__(self, sim, request, stop_revision, clock=time.monotonic, *, task_revision=0, owner="operator"):
        if sim.cancel.is_set():
            raise MotionError("CANCELLED", "Resume manual control before starting ROS navigation")
        if request.run_id != sim.run_id or request.episode_epoch != sim.epoch:
            raise MotionError("STALE_STATE", "ROS goal belongs to another episode")
        if not all(math.isfinite(value) for value in request.target_m_rad):
            raise ValueError("ROS goal must be finite")
        if not .15 <= math.dist(request.target_m_rad[:2], sim.odometry[:2]) <= 1.8:
            raise ValueError("ROS local goal must be between 0.15 and 1.8 m away")
        self.identity = str(uuid4())
        self.run_id, self.epoch = sim.run_id, sim.epoch
        self.stop_revision = stop_revision
        self.task_revision = task_revision
        self.owner = owner
        self.target = list(request.target_m_rad)
        self.clock = clock
        self.runtime = NavigationRuntime(clock)
        self.runtime.continuous_motion = True
        self.runtime.recover_clearance = True
        observation = sim.observe(render=False)
        self.runtime.observe(sim, observation)
        self.runtime.apply(sim, "begin_local_subgoal", LocalSubgoal(expected_revision=0,
            goal="Operator-selected ROS 2 local goal"), observation.seq)
        self.status, self.reason = "running", "Waiting for Nav2 commands"
        self.command_sequence = 0
        self.sensors = OrderedDict()
        self.command_deadline = clock() + 5.
        self.last_sensor_at = clock()

    @property
    def active(self):
        return self.status == "running"

    def state(self):
        return {"session_id": self.identity, "run_id": self.run_id, "episode_epoch": self.epoch,
            "status": self.status, "reason": self.reason, "target_m_rad": self.target,
            "command_sequence": self.command_sequence, "controller": "nav2_external",
            "task_revision": self.task_revision,
            "owner": self.owner,
            "speed_limit_mps": ROS_SPEED_MPS}

    def finish(self, sim, status, reason):
        self.status, self.reason = status, reason
        self.sensors.clear()
        self.runtime.cancel(sim, reason)

    def guard(self, sim, stop_revision, task_revision=None):
        if (not self.active or sim.cancel.is_set() or stop_revision != self.stop_revision
                or sim.run_id != self.run_id or sim.epoch != self.epoch
                or (task_revision is not None and task_revision != self.task_revision)):
            raise MotionError("CANCELLED", "ROS session was invalidated")
        if self.clock() >= self.command_deadline:
            self.finish(sim, "cancelled", "ROS_WATCHDOG: Command stream expired")
            raise MotionError("CANCELLED", self.reason)
        if self.clock() >= self.runtime.skill_deadline:
            self.finish(sim, "cancelled", "ROS local goal deadline expired")
            raise MotionError("CANCELLED", self.reason)

    def observe(self, sequence, captured_at):
        self.sensors[sequence] = captured_at
        self.last_sensor_at = captured_at
        while len(self.sensors) > 8:
            self.sensors.popitem(last=False)

    def apply(self, sim, command, stop_revision, *, task_revision=None):
        self.guard(sim, stop_revision, task_revision)
        if command.session_id != self.identity or command.command_sequence <= self.command_sequence:
            raise MotionError("STALE_PLAN", "ROS command is from an old session or was already consumed")
        captured_at = self.sensors.get(command.sensor_sequence)
        if captured_at is None or not 0 <= self.clock() - captured_at <= .5:
            raise MotionError("STALE_OBSERVATION", "ROS command requires a recent sensor packet")
        if self.runtime.status in {"failed", "cancelled"}:
            raise MotionError("CANCELLED", "ROS motion runtime is no longer active")
        observation = sim.observe(render=False)
        self.runtime.observe(sim, observation)
        self.runtime.apply(sim, "replace_motion_buffer", ContinuousMotionBuffer(
            expected_revision=self.runtime.revision, segments=[{"kind": "drive",
                "linear_mps": command.linear_mps, "angular_radps": command.angular_radps,
                "duration_s": 1.}] * 2), observation.seq)
        self.command_sequence = command.command_sequence
        self.command_deadline = self.clock() + .5
        self.runtime.expires_at = self.command_deadline

    def tick(self, sim, stop_revision, *, task_revision=None):
        try:
            self.guard(sim, stop_revision, task_revision)
            if not self.command_sequence:
                return
            if self.clock() - self.last_sensor_at > .75:
                self.finish(sim, "cancelled", "ROS_SENSOR_STALE: Sensor stream expired")
                return
            heading = math.atan2(math.sin(sim.odometry[2] - self.target[2]),
                                 math.cos(sim.odometry[2] - self.target[2]))
            if math.dist(sim.odometry[:2], self.target[:2]) < .04 and abs(heading) < .08:
                self.finish(sim, "arrived", "Reached operator-selected local goal; task success is not inferred")
                return
            self.runtime.tick(sim)
            if self.runtime.status in {"failed", "cancelled"}:
                self.status, self.reason = "blocked", self.runtime.reason
            elif not self.runtime.buffer:
                sim.hold_current()
                sim._ticks(12)
        except MotionError as error:
            self.finish(sim, "cancelled", str(error))

    def complete(self, sim, result, stop_revision, task_revision):
        self.guard(sim, stop_revision, task_revision)
        if result.session_id != self.identity:
            raise MotionError("STALE_PLAN", "Nav2 result belongs to an old goal")
        heading = math.atan2(math.sin(sim.odometry[2] - self.target[2]), math.cos(sim.odometry[2] - self.target[2]))
        arrived = result.status == "succeeded" and math.dist(sim.odometry[:2], self.target[:2]) < .04 and abs(heading) < .08
        self.finish(sim, "arrived" if arrived else "blocked",
            "Nav2 destination verified by odometry; mission success not inferred" if arrived else
            f"NAV2_{result.status.upper()}: {result.message or 'Destination was not reached within local tolerance'}")