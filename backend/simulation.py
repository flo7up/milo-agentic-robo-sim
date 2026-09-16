from collections import OrderedDict
from io import BytesIO
import math
import threading
import time
from uuid import uuid4

import numpy as np
from PIL import Image
import pybullet as bullet
from pydantic import ValidationError
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from backend.contracts import AgentObservation, Command, EvaluatorState, JointSensor, ProximitySensors, TOOL_MODELS, ToolResult
from backend.challenges import ChallengeProgress
from backend.materials import ensure_textures, scene_material
from backend.robot import ARM_LIMITS, NEUTRAL, ensure_asset

TIMESTEP = 1 / 240


class MotionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class BulletSimulation:
    def __init__(self, run_id=None, epoch=0, width=640, height=480, scene=None, challenge=None, rendering=None):
        import os
        self.rendering = rendering or os.environ.get("MILO_RENDERER", "tiny")
        if self.rendering not in {"tiny", "enhanced"}:
            raise ValueError("MILO_RENDERER must be enhanced or tiny")
        self.camera_renderer = None
        self.owns_camera_renderer = True
        self.run_id = run_id or str(uuid4())
        self.epoch = epoch
        self.width, self.height = width, height
        self.client = bullet.connect(bullet.DIRECT)
        self.planner = bullet.connect(bullet.DIRECT)
        self.cancel = threading.Event()
        self.seq = 0
        self.ticks = 0
        self.frames = OrderedDict()
        self.results = {}
        self.held = {}
        self.targets = {}
        self.odometry = np.zeros(3)
        self.path_length = 0.0
        self.collisions = 0
        self.grasp_attempts = 0
        self.drops = 0
        self.objects = []
        self.visuals = []
        self.on_tick = None
        self.action_contacts = None
        self.challenge_progress = None
        self.pedestrian_crossing = None
        self.kinematic_velocities = {}
        self.scene = challenge.scene() if challenge else scene or [
            {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.77, .80, .79, 1]},
            {"name": "back_wall", "size": [.1, 6, 1], "position": [2.5, 0, .5], "color": [.50, .57, .56, 1]},
            {"name": "cube", "size": [.06, .06, .06], "position": [.36, .25, .032], "color": [.86, .17, .27, 1], "mass": .08}]
        self.scene = [{**item, "texture": scene_material(item, bool(challenge and challenge.id in {"apartment", "kitchen_bathroom", "clinic_delivery"}))} for item in self.scene]
        textures = {name: bullet.loadTexture(str(path), physicsClientId=self.client) for name, path in ensure_textures().items()}
        for client in (self.client, self.planner):
            bullet.setGravity(0, 0, -9.81, physicsClientId=client)
            bullet.setTimeStep(TIMESTEP, physicsClientId=client)
            bullet.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1, physicsClientId=client)
            for item in self.scene:
                half = np.array(item["size"]) / 2
                cylinder = item.get("shape") == "cylinder"
                shape_type = bullet.GEOM_CYLINDER if cylinder else bullet.GEOM_BOX
                collision_dimensions = {"radius": half[0], "height": item["size"][2]} if cylinder else {"halfExtents": half}
                visual_dimensions = {"radius": half[0], "length": item["size"][2]} if cylinder else {"halfExtents": half}
                shape = -1 if item.get("marker") else bullet.createCollisionShape(shape_type, **collision_dimensions, physicsClientId=client)
                visual = bullet.createVisualShape(shape_type, **visual_dimensions, rgbaColor=item["color"], physicsClientId=client)
                body = bullet.createMultiBody(item.get("mass", 0), shape, visual, item["position"], physicsClientId=client)
                bullet.changeDynamics(body, -1, lateralFriction=.8, restitution=0, physicsClientId=client)
                if client == self.client:
                    if item["texture"]:
                        bullet.changeVisualShape(body, -1, textureUniqueId=textures[item["texture"]], physicsClientId=client)
                    self.objects.append({**item, "id": body})
            robot = bullet.loadURDF(str(ensure_asset()), [*(challenge.initial_xy if challenge else [0, 0]), .155], useFixedBase=client == self.planner,
                                    flags=bullet.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
            if client == self.client:
                self.robot = robot
            else:
                self.shadow = robot
        self.joints = {bullet.getJointInfo(self.robot, index, physicsClientId=self.client)[1].decode(): index
                       for index in range(bullet.getNumJoints(self.robot, physicsClientId=self.client))}
        self.arms = {side: [self.joints[f"{side}_joint_{number}"] for number in range(1, 7)] for side in ("left", "right")}
        self.fingers = {side: [self.joints[f"{side}_{suffix}_finger"] for suffix in ("inner", "outer")] for side in self.arms}
        self.palms = {side: self.joints[f"{side}_palm_fixed"] for side in self.arms}
        self.wheels = [self.joints[f"{side}_wheel"] for side in self.arms]
        for name, index in self.joints.items():
            info = bullet.getJointInfo(self.robot, index, physicsClientId=self.client)
            bullet.changeDynamics(self.robot, index, lateralFriction=1.1, restitution=0, physicsClientId=self.client)
            if "support" in name:
                bullet.changeDynamics(self.robot, index, lateralFriction=.001, rollingFriction=0, spinningFriction=0, physicsClientId=self.client)
            if info[2] != bullet.JOINT_FIXED and index not in self.wheels:
                value = NEUTRAL[int(name[-1]) - 1] if "_joint_" in name else .055 if "finger" in name else 0
                if name == "head_pitch" and challenge:
                    value = challenge.initial_head_pitch
                self.targets[index] = (value, 24 if "joint" in name else 5 if "head" in name else 35)
                bullet.resetJointState(self.robot, index, value, physicsClientId=self.client)
        self._hold()
        self._brake()
        self._ticks(120)
        self.ticks = 0
        self.odometry[:] = 0
        self.path_length = 0
        self.last_encoders = self._encoders()
        self.challenge_progress = ChallengeProgress(challenge) if challenge else None
        if challenge and challenge.id == "pedestrian_crossing":
            self.pedestrian_crossing = {"started_at": None, "elapsed_s": 0., "contact": False, "yield_s": 0., "yielded": False, "scored_at": 0}
        self._update_challenge()

    def _update_challenge(self):
        if not self.challenge_progress:
            return
        measurements = {}
        held_bodies = {grasp["body"] for grasp in self.held.values()}
        floor = next(item["id"] for item in self.objects if item["name"] == "floor")
        challenge = self.challenge_progress.challenge
        for objective in challenge.objectives:
            body = self.robot if objective.body == "robot" else next(item["id"] for item in self.objects if item["name"] == objective.body)
            indices = [-1, *self.wheels] if objective.body == "robot" else [-1]
            bounds = [bullet.getAABB(body, index, physicsClientId=self.client) for index in indices]
            velocity, angular = bullet.getBaseVelocity(body, physicsClientId=self.client)
            measurements[objective.body] = {"bounds": [np.min([entry[0] for entry in bounds], axis=0),
                                                       np.max([entry[1] for entry in bounds], axis=0)],
                                            "speed": float(np.linalg.norm(velocity)), "angular_speed": float(np.linalg.norm(angular)),
                                            "grounded": any(contact[9] > .01 for contact in bullet.getContactPoints(body, floor, physicsClientId=self.client))}
            if objective.require_lift:
                for fingers in self.fingers.values():
                    contacts = [[contact for contact in bullet.getContactPoints(self.robot, body, linkIndexA=finger, physicsClientId=self.client)
                                 if contact[9] > .02] for finger in fingers]
                    if all(contacts) and any(np.dot(inner[7], outer[7]) < -.7 for inner in contacts[0] for outer in contacts[1]):
                        held_bodies.add(body)
        if self.pedestrian_crossing is not None:
            crossing = self.pedestrian_crossing
            position = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)[0]
            measurement = measurements["robot"]
            waiting = (crossing["started_at"] is not None and 1. <= crossing["elapsed_s"] <= 3. and
                       .25 <= position[0] < .75 and abs(position[1]) < .4 and
                       measurement["grounded"] and measurement["speed"] < .025 and measurement["angular_speed"] < .15)
            elapsed = (self.ticks - crossing["scored_at"]) * TIMESTEP
            crossing["scored_at"] = self.ticks
            crossing["yield_s"] = crossing["yield_s"] + elapsed if waiting else 0.
            crossing["yielded"] |= crossing["yield_s"] >= .4
            measurement.update(pedestrian_contact=crossing["contact"], yielded=crossing["yielded"],
                               pedestrian_passed=crossing["elapsed_s"] >= 4.)
        if challenge.orbit:
            position = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)[0]
            measurements["robot"].update(position_xy=list(position[:2]), contact=bool(self.proximity_sensors().collisions))
        if challenge.search_target:
            target = next(item["id"] for item in self.objects if item["name"] == challenge.search_target)
            target_position = np.array(bullet.getBasePositionAndOrientation(target, physicsClientId=self.client)[0])
            robot_position = np.array(bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)[0])
            measurements["robot"].update({
                "target_nearby": np.linalg.norm(target_position[:2] - robot_position[:2]) <= .9,
                "target_visible": self._target_in_camera(target, target_position),
                "head_stationary": all(abs(bullet.getJointState(self.robot, self.joints[name], physicsClientId=self.client)[1]) < .05
                                       for name in ("head_yaw", "head_pitch")),
            })
        self.challenge_progress.update(measurements, {item["name"] for item in self.objects if item["id"] in held_bodies},
                                       simulated_time_s=self.ticks * TIMESTEP, travel_m=self.path_length)

    def _target_in_camera(self, target, position):
        eye, matrix = self.camera_pose()
        samples = [position + offset for offset in (np.zeros(3), matrix[:, 1] * .04, -matrix[:, 1] * .04,
                                                    matrix[:, 2] * .04, -matrix[:, 2] * .04)]
        vertical = math.tan(math.radians(65 / 2)) * .95
        horizontal = vertical * self.width / self.height
        hits = bullet.rayTestBatch([eye.tolist()] * len(samples), [sample.tolist() for sample in samples], physicsClientId=self.client)
        visible = 0
        for sample, hit in zip(samples, hits):
            local = matrix.T @ (sample - eye)
            if .015 < local[0] < 12 and abs(local[1]) < local[0] * horizontal and abs(local[2]) < local[0] * vertical and hit[0] == target:
                visible += 1
        return visible >= 3

    def challenge_status(self):
        return self.challenge_progress.status if self.challenge_progress else None

    def battery_sensor(self):
        return self.challenge_progress.battery if self.challenge_progress else None

    def _encoders(self):
        return np.array([bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] for index in self.wheels])

    def _hold(self):
        for index, (target, force) in self.targets.items():
            bullet.setJointMotorControl2(self.robot, index, bullet.POSITION_CONTROL, targetPosition=target,
                                        force=force, maxVelocity=.15 if index in sum(self.fingers.values(), []) else 1.8,
                                        positionGain=.25, velocityGain=1, physicsClientId=self.client)

    def _brake(self):
        for index in self.wheels:
            bullet.setJointMotorControl2(self.robot, index, bullet.VELOCITY_CONTROL, targetVelocity=0, force=5, physicsClientId=self.client)

    def stop(self):
        self.cancel.set()

    def hold_current(self):
        for index, (_, force) in self.targets.items():
            self.targets[index] = (bullet.getJointState(self.robot, index, physicsClientId=self.client)[0], force)
        self._hold()
        self._brake()

    def _advance_pedestrian(self):
        crossing = self.pedestrian_crossing
        if crossing is None:
            return
        now = self.ticks * TIMESTEP
        if crossing["started_at"] is None:
            velocity = bullet.getBaseVelocity(self.robot, physicsClientId=self.client)[0]
            if self.odometry[0] < .30 or velocity[0] < .03:
                return
            crossing["started_at"] = now
        elapsed = now - crossing["started_at"]
        crossing["elapsed_s"] = elapsed
        displacement = min(2.6, .65 * elapsed)
        walking = displacement < 2.6
        for item in self.objects:
            if not item.get("pedestrian"):
                continue
            position = list(item["position"])
            swing = item.get("gait", 0) * math.sin(elapsed * 2 * math.pi) if walking else 0.
            position[1] += displacement + .06 * swing
            rotation = bullet.getQuaternionFromEuler([.18 * swing if "foot" not in item["name"] else 0., 0., 0.])
            previous = bullet.getBasePositionAndOrientation(item["id"], physicsClientId=self.client)[0]
            self.kinematic_velocities[item["id"]] = (np.array(position) - previous) / TIMESTEP
            bullet.resetBasePositionAndOrientation(item["id"], position, rotation, physicsClientId=self.client)

    def _ticks(self, count, callback=None, allow_depleted=False):
        for tick in range(count):
            if self.cancel.is_set():
                self.hold_current()
                raise MotionError("CANCELLED", "Motion interrupted")
            if callback:
                callback(tick)
            self._advance_pedestrian()
            before = self._encoders()
            bullet.stepSimulation(physicsClientId=self.client)
            self.ticks += 1
            if self.pedestrian_crossing is not None and not self.pedestrian_crossing["contact"]:
                self.pedestrian_crossing["contact"] = any(
                    bullet.getContactPoints(self.robot, item["id"], physicsClientId=self.client)
                    for item in self.objects if item.get("pedestrian"))
            delta = (self._encoders() - before) * .09
            distance, heading = float(np.mean(delta)), float((delta[1] - delta[0]) / .38)
            self.odometry[:2] += distance * np.array([math.cos(self.odometry[2] + heading / 2), math.sin(self.odometry[2] + heading / 2)])
            self.odometry[2] += heading
            self.path_length += abs(distance)
            if self.ticks % 12 == 0:
                self._update_challenge()
                if self.action_contacts is not None:
                    for contact in self.proximity_sensors().collisions:
                        self.action_contacts[contact.direction] = max(self.action_contacts.get(contact.direction, 0), contact.force_n)
                if not allow_depleted and self.battery_sensor() and self.battery_sensor().charge_pct <= 0:
                    self.hold_current()
                    raise MotionError("BATTERY_EMPTY", "Battery depleted. Recharge while parked or reset the episode.")
            if self.on_tick and self.ticks % 12 == 0:
                self.on_tick()
            for side, grasp in list(self.held.items()):
                load = np.linalg.norm(bullet.getConstraintState(grasp["constraint"], physicsClientId=self.client)[:3])
                palm_pos = bullet.getLinkState(self.robot, self.palms[side], physicsClientId=self.client)[4]
                object_pos = bullet.getBasePositionAndOrientation(grasp["body"], physicsClientId=self.client)[0]
                if load > 18 or np.linalg.norm(np.array(palm_pos) - object_pos) > .12:
                    self._release(side)
                    self.drops += 1

    def _release(self, side):
        grasp = self.held.pop(side, None)
        if grasp:
            bullet.removeConstraint(grasp["constraint"], physicsClientId=self.client)

    def _sync_planner(self):
        pose = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        bullet.resetBasePositionAndOrientation(self.shadow, *pose, physicsClientId=self.planner)
        for index in self.joints.values():
            value = bullet.getJointState(self.robot, index, physicsClientId=self.client)[0]
            bullet.resetJointState(self.shadow, index, value, physicsClientId=self.planner)
        for item in self.objects:
            pose = bullet.getBasePositionAndOrientation(item["id"], physicsClientId=self.client)
            bullet.resetBasePositionAndOrientation(item["id"], *pose, physicsClientId=self.planner)

    def reposition(self, placement):
        if placement.run_id != self.run_id or placement.episode_epoch != self.epoch or placement.observation_seq != self.seq:
            raise MotionError("STALE_STATE", "Robot state changed. Select and place the robot again.")
        if self.cancel.is_set():
            raise MotionError("CANCELLED", "Resume manual control before positioning the robot.")
        if self.held:
            raise MotionError("OBJECT_HELD", "Release held objects before positioning the robot.")
        position, orientation = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        destination = [*placement.xy_m, position[2]]
        self._sync_planner()
        bullet.resetBasePositionAndOrientation(self.shadow, destination, orientation, physicsClientId=self.planner)
        bullet.performCollisionDetection(physicsClientId=self.planner)
        floor = next(item for item in self.objects if item["name"] == "floor")
        floor_lower, floor_upper = bullet.getAABB(floor["id"], physicsClientId=self.planner)
        for index in [-1, *self.joints.values()]:
            lower, upper = bullet.getAABB(self.shadow, index, physicsClientId=self.planner)
            if any(lower[axis] < floor_lower[axis] or upper[axis] > floor_upper[axis] for axis in (0, 1)):
                raise MotionError("OUT_OF_BOUNDS", "Place the entire robot on the floor.")
        for item in self.objects:
            if item["name"] == "floor" or item.get("marker"):
                continue
            if any(contact[8] < -.001 for contact in bullet.getClosestPoints(self.shadow, item["id"], 0, physicsClientId=self.planner)):
                raise MotionError("COLLISION_BLOCKED", "That position overlaps an obstacle. Choose a clear floor location.")
        if self.cancel.is_set():
            raise MotionError("CANCELLED", "Placement interrupted.")
        self.hold_current()
        bullet.resetBasePositionAndOrientation(self.robot, destination, orientation, physicsClientId=self.client)
        bullet.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)
        for index in self.joints.values():
            value = bullet.getJointState(self.robot, index, physicsClientId=self.client)[0]
            bullet.resetJointState(self.robot, index, value, targetVelocity=0, physicsClientId=self.client)
        bullet.performCollisionDetection(physicsClientId=self.client)
        self.results.clear()
        self._update_challenge()
        return self.observe()

    def _shadow_pose(self, side, values):
        for index, value in zip(self.arms[side], values):
            bullet.resetJointState(self.shadow, index, value, physicsClientId=self.planner)
        state = bullet.getLinkState(self.shadow, self.palms[side], computeForwardKinematics=True, physicsClientId=self.planner)
        return np.array(state[4]), Rotation.from_quat(state[5])

    def inverse_kinematics(self, side, position, orientation, seed=None):
        self._sync_planner()
        base_pose = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        target_position, target_orientation = bullet.multiplyTransforms(*base_pose, position, orientation)
        desired_rotation = Rotation.from_quat(target_orientation)
        current = [bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] for index in self.arms[side]]

        def residual(values):
            actual_position, actual_rotation = self._shadow_pose(side, values)
            return np.concatenate([(actual_position - target_position) * 5, (desired_rotation.inv() * actual_rotation).as_rotvec()])

        for initial in (seed if seed is not None else current, [0, .2, 1.2, 0, -1.4, 0], NEUTRAL):
            solution = least_squares(residual, np.clip(initial, np.array(ARM_LIMITS)[:, 0] + 1e-5, np.array(ARM_LIMITS)[:, 1] - 1e-5),
                                     bounds=np.array(ARM_LIMITS).T, diff_step=.001, max_nfev=160, xtol=1e-7)
            error = residual(solution.x)
            if np.linalg.norm(error[:3]) < .008 * 5 and np.linalg.norm(error[3:]) < .08:
                return solution.x
        raise MotionError("UNREACHABLE_TARGET", "IK could not satisfy position/orientation tolerances")

    def _check_trajectory(self, side, current, target, opening=None):
        self._sync_planner()
        moving = self.arms[side] + [self.palms[side]] + self.fingers[side]
        for fraction in np.linspace(0, 1, max(2, math.ceil(max(abs(target - current)) / .025))):
            if opening is not None:
                for joint in self.fingers[side]:
                    initial = bullet.getJointState(self.robot, joint, physicsClientId=self.client)[0]
                    bullet.resetJointState(self.shadow, joint, initial + (opening / 2 - initial) * fraction,
                                          physicsClientId=self.planner)
            self._shadow_pose(side, current + (target - current) * fraction)
            bullet.performCollisionDetection(physicsClientId=self.planner)
            for item in self.objects:
                for contact in bullet.getClosestPoints(self.shadow, item["id"], 0, physicsClientId=self.planner):
                    if contact[3] in moving and contact[8] < -.001:
                        if contact[3] in self.fingers[side] and item.get("mass", 0) > 0:
                            continue
                        raise MotionError("COLLISION_BLOCKED", "Arm trajectory intersects a contact surface")
            other_side = "right" if side == "left" else "left"
            blocked = [-1, self.joints["torso_fixed"], self.joints["head_pitch"], self.joints[f"{other_side}_shoulder_fixed"]] + self.arms[other_side] + self.fingers[other_side]
            for link in moving:
                mounting = [] if link in self.arms[side][:2] else [self.joints[f"{side}_shoulder_fixed"]]
                for other in blocked + mounting:
                    if bullet.getClosestPoints(self.shadow, self.shadow, -.001, linkIndexA=link, linkIndexB=other, physicsClientId=self.planner):
                        raise MotionError("COLLISION_BLOCKED", "Arm trajectory intersects the robot")

    def move_cartesian(self, side, position, orientation, duration):
        self.inverse_kinematics(side, position, orientation)
        self._sync_planner()
        base_pose = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        palm = bullet.getLinkState(self.robot, self.palms[side], physicsClientId=self.client)
        local_position, local_orientation = bullet.multiplyTransforms(*bullet.invertTransform(*base_pose), palm[4], palm[5])
        current = np.array([bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] for index in self.arms[side]])
        rotations = Slerp([0, 1], Rotation.from_quat([local_orientation, orientation]))
        segments = max(4, math.ceil(np.linalg.norm(np.array(position) - local_position) / .015))
        waypoints = [current]
        for fraction in np.linspace(0, 1, segments + 1)[1:]:
            point = np.array(local_position) + (np.array(position) - local_position) * fraction
            target = self.inverse_kinematics(side, point, rotations(fraction).as_quat(), waypoints[-1])
            if max(abs(target - waypoints[-1])) / (duration / segments) > 1.8:
                raise MotionError("INVALID_ARGUMENT", "Cartesian path exceeds joint velocity limits")
            self._check_trajectory(side, waypoints[-1], target)
            waypoints.append(target)
        count = math.ceil(duration / TIMESTEP)

        def servo(tick):
            progress = (tick + 1) / count * segments
            index = min(int(progress), segments - 1)
            values = waypoints[index] + (waypoints[index + 1] - waypoints[index]) * (progress - index)
            for joint, value in zip(self.arms[side], values):
                self.targets[joint] = (value, 24)
            self._hold()

        self._ticks(count, servo)
        actual = bullet.getLinkState(self.robot, self.palms[side], physicsClientId=self.client)
        expected = bullet.multiplyTransforms(*base_pose, position, orientation)
        if np.linalg.norm(np.array(actual[4]) - expected[0]) > .015 or (Rotation.from_quat(actual[5]).inv() * Rotation.from_quat(expected[1])).magnitude() > .1:
            raise MotionError("COLLISION_BLOCKED", "Servo did not achieve Cartesian tolerance under bounded force")

    def move_arm(self, side, target, duration):
        target = np.array(target)
        if any(not low <= value <= high for value, (low, high) in zip(target, ARM_LIMITS)):
            raise MotionError("JOINT_LIMIT", "A target exceeds a joint limit")
        current = np.array([bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] for index in self.arms[side]])
        if max(abs(target - current)) / duration > 1.8:
            raise MotionError("INVALID_ARGUMENT", "Requested trajectory exceeds 1.8 rad/s; increase duration or use an intermediate target")
        self._check_trajectory(side, current, target)
        count = math.ceil(duration / TIMESTEP)

        def servo(tick):
            values = current + (target - current) * (tick + 1) / count
            for index, value in zip(self.arms[side], values):
                self.targets[index] = (value, 24)
            self._hold()

        self._ticks(count, servo)
        error = max(abs(bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] - value) for index, value in zip(self.arms[side], target))
        if error > .10:
            raise MotionError("COLLISION_BLOCKED", "Servo failed to reach target under bounded motor force")

    def set_gripper(self, side, opening, force):
        if opening > .065:
            self._release(side)
        for index in self.fingers[side]:
            self.targets[index] = (opening / 2, force)
        self._hold()
        self._ticks(180)
        if opening > .01 or side in self.held:
            return
        self.grasp_attempts += 1
        contacts = [bullet.getContactPoints(self.robot, linkIndexA=index, physicsClientId=self.client) for index in self.fingers[side]]
        for item in self.objects:
            body = item["id"]
            if not 0 < item.get("mass", 0) <= .35 or max(item["size"]) > .11:
                continue
            opposing = [[contact for contact in finger_contacts if contact[2] == body and contact[9] > .02] for finger_contacts in contacts]
            if not all(opposing):
                continue
            if np.dot(opposing[0][0][7], opposing[1][0][7]) > -.7:
                continue
            palm = bullet.getLinkState(self.robot, self.palms[side], physicsClientId=self.client)
            object_pose = bullet.getBasePositionAndOrientation(body, physicsClientId=self.client)
            local = bullet.multiplyTransforms(*bullet.invertTransform(palm[4], palm[5]), *object_pose)
            if abs(local[0][0]) > .045 or abs(local[0][1]) > .03 or abs(local[0][2]) > .03:
                continue
            parent_local = bullet.multiplyTransforms(*bullet.invertTransform(palm[0], palm[1]), *object_pose)
            constraint = bullet.createConstraint(self.robot, self.palms[side], body, -1, bullet.JOINT_FIXED,
                                                  [0, 0, 0], parent_local[0], [0, 0, 0],
                                                  parentFrameOrientation=parent_local[1], physicsClientId=self.client)
            bullet.changeConstraint(constraint, maxForce=18, erp=.2, physicsClientId=self.client)
            self.held[side] = {"body": body, "constraint": constraint}
            break

    def camera_pose(self):
        head = bullet.getLinkState(self.robot, self.joints["head_pitch"], computeForwardKinematics=True, physicsClientId=self.client)
        eye, rotation = bullet.multiplyTransforms(head[4], head[5], [.095, 0, .005], [0, 0, 0, 1])
        matrix = np.array(bullet.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        return np.array(eye), matrix

    def capture(self):
        if self.rendering == "enhanced":
            return self._enhanced_capture(self.width, self.height, depth=False)[0]
        eye, matrix = self.camera_pose()
        view = bullet.computeViewMatrix(eye, eye + matrix[:, 0], matrix[:, 2])
        projection = bullet.computeProjectionMatrixFOV(65, self.width / self.height, .015, 12)
        pixels = bullet.getCameraImage(self.width, self.height, view, projection, renderer=bullet.ER_TINY_RENDERER,
                                       flags=bullet.ER_NO_SEGMENTATION_MASK, physicsClientId=self.client)[2]
        output = BytesIO()
        Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)[:, :, :3]).save(output, format="PNG")
        return output.getvalue()

    def _enhanced_capture(self, width, height, depth=True):
        from backend.camera import EnhancedRenderer, scene_snapshot
        try:
            if self.camera_renderer is None:
                self.camera_renderer = EnhancedRenderer()
            return self.camera_renderer.capture(scene_snapshot(self, width, height), depth=depth)
        except Exception:
            self.stop()
            raise

    def robot_footprint(self):
        position, orientation = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        rotation = np.array(bullet.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        corners = []
        if not hasattr(self, "footprint_geometry"):
            self.footprint_geometry = {}
        for index in [-1, *self.joints.values()]:
            if index not in self.footprint_geometry:
                points = []
                mesh_margin = 0.
                for shape in bullet.getCollisionShapeData(self.robot, index, physicsClientId=self.client):
                    shape_type, dimensions = shape[2:4]
                    if shape_type == bullet.GEOM_MESH:
                        vertices = bullet.getMeshData(self.robot, index, flags=bullet.MESH_DATA_SIMULATION_MESH,
                            physicsClientId=self.client)[1]
                        if not vertices:
                            raise ValueError("Robot collision mesh has no footprint geometry")
                        points.extend(vertices)
                        mesh_margin = max(mesh_margin, bullet.getDynamicsInfo(self.robot, index, physicsClientId=self.client)[11])
                        continue
                    if shape_type == bullet.GEOM_BOX:
                        extent = np.asarray(dimensions) / 2
                    elif shape_type == bullet.GEOM_SPHERE:
                        extent = np.repeat(dimensions[0], 3)
                    elif shape_type in {bullet.GEOM_CYLINDER, bullet.GEOM_CAPSULE}:
                        extent = np.array([dimensions[1], dimensions[1], dimensions[0] / 2])
                        if shape_type == bullet.GEOM_CAPSULE:
                            extent[2] += dimensions[1]
                    else:
                        raise ValueError("Unsupported robot collision shape for footprint")
                    local_rotation = np.array(bullet.getMatrixFromQuaternion(shape[6])).reshape(3, 3)
                    for horizontal in (-extent[0], extent[0]):
                        for lateral in (-extent[1], extent[1]):
                            for height in (-extent[2], extent[2]):
                                points.append(np.array([horizontal, lateral, height]) @ local_rotation.T + shape[5])
                self.footprint_geometry[index] = (np.asarray(points).reshape(-1, 3), mesh_margin)
            local_points, margin = self.footprint_geometry[index]
            if not len(local_points):
                continue
            link_position, link_orientation = (position, orientation) if index == -1 else bullet.getLinkState(
                self.robot, index, computeForwardKinematics=True, physicsClientId=self.client)[:2]
            link_rotation = np.array(bullet.getMatrixFromQuaternion(link_orientation)).reshape(3, 3)
            points = (local_points @ link_rotation.T + link_position - position) @ rotation
            lower, upper = points.min(axis=0) - margin, points.max(axis=0) + margin
            for horizontal in (lower[0], upper[0]):
                for lateral in (lower[1], upper[1]):
                    corners.append([horizontal, lateral, 0.])
        corners = np.array(corners)
        return {"lower_xy_m": corners[:, :2].min(axis=0).tolist(), "upper_xy_m": corners[:, :2].max(axis=0).tolist(),
            "radius_m": float(np.linalg.norm(corners[:, :2], axis=1).max()), "frame": "robot_base",
            "method": "Conservative union of robot-frame collision-shape bounds including margins"}

    def capture_spatial(self, sequence):
        from scipy.ndimage import binary_dilation
        from backend.contracts import SpatialObservation
        from backend.spatial import calibration, metric_depth
        captured_at = time.monotonic()
        head = [bullet.getJointState(self.robot, self.joints[name], physicsClientId=self.client)[0]
                for name in ("head_yaw", "head_pitch")]
        odometry = self.odometry.tolist()
        eye, matrix = self.camera_pose()
        intrinsics = calibration(160, 120)
        if self.rendering == "enhanced":
            image, depth = self._enhanced_capture(intrinsics.width, intrinsics.height)
            depth[(depth < intrinsics.near_m) | (depth > intrinsics.usable_range_m)] = np.nan
            observation = SpatialObservation(run_id=self.run_id, episode_epoch=self.epoch, sequence=sequence,
                captured_at=captured_at, simulated_time_s=self.ticks * TIMESTEP, calibration=intrinsics,
                head_rad=head, odometry_m_rad=odometry, depth_m=np.where(np.isfinite(depth), depth, None).ravel().tolist())
            return observation, image
        view = bullet.computeViewMatrix(eye, eye + matrix[:, 0], matrix[:, 2])
        projection = bullet.computeProjectionMatrixFOV(65, 4 / 3, intrinsics.near_m, intrinsics.far_m)
        rendered = bullet.getCameraImage(intrinsics.width, intrinsics.height, view, projection,
            renderer=bullet.ER_TINY_RENDERER, flags=bullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=self.client)
        segments = np.asarray(rendered[4], dtype=np.int64).reshape(intrinsics.height, intrinsics.width)
        self_mask = (segments >= 0) & ((segments & ((1 << 24) - 1)) == self.robot)
        self_mask = binary_dilation(self_mask, iterations=1)
        depth = metric_depth(np.asarray(rendered[3]).reshape(intrinsics.height, intrinsics.width), self_mask, intrinsics)
        output = BytesIO()
        Image.fromarray(np.asarray(rendered[2], dtype=np.uint8).reshape(intrinsics.height, intrinsics.width, 4)[:, :, :3]).save(output, format="PNG")
        observation = SpatialObservation(run_id=self.run_id, episode_epoch=self.epoch, sequence=sequence,
            captured_at=captured_at, simulated_time_s=self.ticks * TIMESTEP, calibration=intrinsics,
            head_rad=head, odometry_m_rad=odometry, depth_m=np.where(np.isfinite(depth), depth, None).ravel().tolist())
        return observation, output.getvalue()

    def gripper_sensors(self):
        sensors = {}
        for side, indices in self.fingers.items():
            contacts = [bullet.getContactPoints(self.robot, linkIndexA=index, physicsClientId=self.client) for index in indices]
            sensors[side] = {"aperture_m": sum(bullet.getJointState(self.robot, index, physicsClientId=self.client)[0] for index in indices),
                             "contact": [any(contact[9] > .02 for contact in entries) for entries in contacts],
                             "load_n": float(np.linalg.norm(bullet.getConstraintState(self.held[side]["constraint"], physicsClientId=self.client)[:3])) if side in self.held else 0.0}
        return sensors

    def proximity_sensors(self):
        position, rotation = bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        matrix = np.array(bullet.getMatrixFromQuaternion(rotation)).reshape(3, 3)
        names = ("front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right")
        starts, ends, origins = [], [], []
        for index in range(8):
            angle = index * math.pi / 4
            direction = np.array([math.cos(angle), math.sin(angle), 0])
            edge = min(.18 / max(abs(direction[0]), 1e-9), .25 / max(abs(direction[1]), 1e-9))
            origin = direction * edge + [0, 0, -.115]
            origins.append(origin.tolist())
            starts.append((position + matrix @ origin).tolist())
            ends.append((position + matrix @ (origin + direction * 2)).tolist())
        hits = bullet.rayTestBatch(starts, ends, physicsClientId=self.client)
        distances = [{"direction": names[index], "bearing_rad": index * math.pi / 4,
                      "origin_base_m": origins[index],
                      "distance_m": round(hit[2] * 2, 3) if hit[0] not in (-1, self.robot) else None,
                      "status": "clear" if hit[0] == -1 else "occluded" if hit[0] == self.robot else "hit"}
                     for index, hit in enumerate(hits)]
        floor = next(item["id"] for item in self.objects if item["name"] == "floor")
        forces = {}
        fingers = {index for values in self.fingers.values() for index in values}
        graspable = {item["id"] for item in self.objects if item.get("mass", 0) > 0}
        for contact in bullet.getContactPoints(self.robot, physicsClientId=self.client):
            if contact[2] in (floor, self.robot) or (contact[3] in fingers and contact[2] in graspable) or contact[9] <= .05:
                continue
            local = matrix.T @ (np.array(contact[5]) - position)
            sector = ("front", "left", "rear", "right")[round(math.atan2(local[1], local[0]) / (math.pi / 2)) % 4]
            forces[sector] = forces.get(sector, 0) + contact[9]
        return ProximitySensors(simulated_time_s=self.ticks * TIMESTEP, distances=distances,
                                collisions=[{"direction": direction, "force_n": round(force, 3)} for direction, force in sorted(forces.items())])

    def observe(self, render=True):
        self.seq += 1
        reference = f"{self.epoch}-{self.seq}.png"
        if render:
            self.frames[reference] = self.capture()
            while len(self.frames) > 16:
                self.frames.popitem(last=False)
        joints = [JointSensor(name=name, position=state[0], velocity=state[1]) for name, index in self.joints.items()
                  if (state := bullet.getJointState(self.robot, index, physicsClientId=self.client)) and
                  bullet.getJointInfo(self.robot, index, physicsClientId=self.client)[2] != bullet.JOINT_FIXED]
        proximity = self.proximity_sensors()
        bumpers = [contact.direction for contact in proximity.collisions]
        return AgentObservation(run_id=self.run_id, episode_epoch=self.epoch, seq=self.seq, wall_timestamp=time.time(),
                                simulated_time_s=self.ticks * TIMESTEP, frame_ref=reference, joints=joints,
                                grippers=self.gripper_sensors(), head_rad=[bullet.getJointState(self.robot, self.joints[name], physicsClientId=self.client)[0] for name in ("head_yaw", "head_pitch")],
                                odometry_m_rad=self.odometry.tolist(), bumpers=bumpers, battery=self.battery_sensor(), proximity=proximity)

    def frame(self, reference):
        return self.frames[reference]

    def execute(self, command: Command, *, drive_guard=None):
        started = self.ticks
        prior = self.odometry.copy()
        status, code, message = "ok", None, ""
        if command.action_id in self.results and command.run_id == self.run_id and command.episode_epoch == self.epoch:
            return self.results[command.action_id]
        self.action_contacts = {}
        try:
            if command.run_id != self.run_id or command.episode_epoch != self.epoch:
                raise MotionError("CANCELLED", "Episode no longer active")
            if command.tool not in TOOL_MODELS:
                raise MotionError("INVALID_ARGUMENT", "Unknown tool")
            arguments = TOOL_MODELS[command.tool].model_validate(command.arguments)
            if command.tool not in ("observe", "stop") and command.observation_seq != self.seq:
                raise MotionError("STALE_OBSERVATION", "Observe again before moving")
            if self.cancel.is_set() and command.tool not in ("observe", "stop"):
                raise MotionError("CANCELLED", "Run is stopped")
            if self.battery_sensor() and self.battery_sensor().charge_pct <= 0 and command.tool not in ("observe", "stop", "wait"):
                raise MotionError("BATTERY_EMPTY", "Battery depleted. Recharge while parked or reset the episode.")
            if command.tool == "wait":
                self.hold_current()
                self._ticks(math.ceil(arguments.duration_s / TIMESTEP), allow_depleted=True)
            elif command.tool == "drive_base":
                desired = np.array([arguments.linear_mps - arguments.angular_radps * .19, arguments.linear_mps + arguments.angular_radps * .19]) / .09
                count = math.ceil(arguments.duration_s / TIMESTEP)

                def drive(tick):
                    if drive_guard is not None and tick % 12 == 0:
                        drive_guard(arguments)
                    ramp = min(1, (tick + 1) * TIMESTEP / .3, (count - tick) * TIMESTEP / .3)
                    for index, speed in zip(self.wheels, desired * ramp):
                        bullet.setJointMotorControl2(self.robot, index, bullet.VELOCITY_CONTROL, targetVelocity=speed, force=5, physicsClientId=self.client)

                self._ticks(count, drive)
            elif command.tool == "set_head":
                self.targets[self.joints["head_yaw"]] = (arguments.yaw_rad, 5)
                self.targets[self.joints["head_pitch"]] = (arguments.pitch_rad, 5)
                self._hold()
                self._ticks(math.ceil(arguments.duration_s / TIMESTEP))
            elif command.tool == "set_arm_joints":
                self.move_arm(arguments.arm.value, arguments.joint_positions_rad, arguments.duration_s)
            elif command.tool == "move_end_effector":
                self.move_cartesian(arguments.arm.value, arguments.position_m, arguments.orientation_xyzw, arguments.duration_s)
            elif command.tool == "set_gripper":
                self.set_gripper(arguments.arm.value, arguments.opening_m, arguments.max_force_n)
            elif command.tool == "stop":
                self.stop()
                self.hold_current()
            elif command.tool in ("finish_task", "submit_answer"):
                raise MotionError("INVALID_ARGUMENT", "Evaluation requires a challenge session")
        except ValidationError:
            status, code, message = "error", "INVALID_ARGUMENT", "Arguments do not match the published tool schema"
        except MotionError as error:
            status, code, message = "cancelled" if error.code == "CANCELLED" else "error", error.code, str(error)
            self.hold_current()
        finally:
            self._brake()
            contacts_during_action = self.action_contacts
            self.action_contacts = None
        self._update_challenge()
        observation = self.observe()
        odometry_delta = self.odometry - prior
        elapsed = (self.ticks - started) * TIMESTEP
        if command.tool == "drive_base" and status == "ok" and observation.bumpers:
            meaningful_request = (abs(arguments.linear_mps) * elapsed >= .02 or
                                  abs(arguments.angular_radps) * elapsed >= .08)
            if meaningful_request and np.linalg.norm(odometry_delta[:2]) < .005 and abs(odometry_delta[2]) < .02:
                status, code = "error", "NO_PROGRESS"
                message = ("Drive had contact and no measurable encoder progress. Do not repeat it unchanged. "
                           "Inspect with set_head, then choose a safe different motion or stop.")
        sensor_deltas = {"odometry_m_rad": odometry_delta.tolist()}
        if contacts_during_action:
            sensor_deltas["contacts_during_action"] = [
                {"direction": direction, "force_n": force} for direction, force in sorted(contacts_during_action.items())]
        result = ToolResult(action_id=command.action_id, status=status, error=code, message=message,
                            actual_duration_s=elapsed, interruption_reason=code,
                            sensor_deltas=sensor_deltas, observation=observation)
        if command.run_id == self.run_id and command.episode_epoch == self.epoch:
            self.results[command.action_id] = result
        return result

    def evaluator_state(self):
        return EvaluatorState(simulated_time_s=self.ticks * TIMESTEP,
                              robot_position=list(bullet.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)[0]),
                              bodies=[{**item, "position": list(bullet.getBasePositionAndOrientation(item["id"], physicsClientId=self.client)[0]),
                                       "velocity": list(bullet.getBaseVelocity(item["id"], physicsClientId=self.client)[0])} for item in self.objects],
                              retained_bodies=[grasp["body"] for grasp in self.held.values()])

    def snapshot(self):
        bodies = []
        for body in [item["id"] for item in self.objects] + [self.robot]:
            for link in range(-1, bullet.getNumJoints(body, physicsClientId=self.client)):
                if link < 0:
                    position, rotation = bullet.getBasePositionAndOrientation(body, physicsClientId=self.client)
                else:
                    state = bullet.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=self.client)
                    position, rotation = state[4], state[5]
                bodies.append({"key": f"{body}:{link}", "position": list(position), "quaternion": list(rotation)})
        return {"simulated_time_s": self.ticks * TIMESTEP, "poses": bodies}

    def geometry(self):
        geometry = []
        textures = {item["id"]: item.get("texture") for item in self.objects}
        names = {item["id"]: item["name"] for item in self.objects}
        for body in [item["id"] for item in self.objects] + [self.robot]:
            for visual in bullet.getVisualShapeData(body, physicsClientId=self.client):
                geometry.append({"key": f"{body}:{visual[1]}", "type": visual[2], "dimensions": list(visual[3]),
                                 "position": list(visual[5]), "quaternion": list(visual[6]), "color": list(visual[7]),
                                 "name": names.get(body, "robot"),
                                 "texture": f"/api/textures/{textures[body]}.png" if textures.get(body) else None})
        return geometry

    def close(self):
        if self.camera_renderer and self.owns_camera_renderer:
            self.camera_renderer.close()
        for client in (self.client, self.planner):
            if bullet.isConnected(client):
                bullet.disconnect(client)