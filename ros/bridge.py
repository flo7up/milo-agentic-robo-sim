import argparse
import base64
import math
import time
from uuid import uuid4

import cv2
import numpy as np
import requests
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point32, Polygon, PoseStamped, Quaternion, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from lifecycle_msgs.srv import GetState
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def stamp(seconds):
    nanoseconds = round(seconds * 1_000_000_000)
    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def quaternion(matrix):
    vector, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))
    angle = float(np.linalg.norm(vector))
    axis = vector.flatten() * (math.sin(angle / 2) / angle if angle else 0.)
    return Quaternion(x=float(axis[0]), y=float(axis[1]), z=float(axis[2]), w=math.cos(angle / 2))


class MiloBridge(Node):
    def __init__(self, backend):
        super().__init__("milo_bridge")
        self.backend = backend.rstrip("/")
        self.http = requests.Session()
        self.http.trust_env = False
        self.command_http = requests.Session()
        self.command_http.trust_env = False
        self.odometry_http = requests.Session()
        self.odometry_http.trust_env = False
        self.sensor_packets = ()
        self.episode = None
        self.session_id = None
        self.session_started = math.inf
        self.command = None
        self.command_sequence = 0
        self.goal_future = None
        self.goal_handle = None
        self.result_future = None
        self.cancel_future = None
        self.bridge_id = str(uuid4())
        self.heartbeat_at = 0.
        self.finished = False
        self.previous = None
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.laser_transform_sent = False
        self.odom = self.create_publisher(Odometry, "odom", 5)
        self.scan = self.create_publisher(LaserScan, "scan", qos_profile_sensor_data)
        self.laser_cloud = self.create_publisher(PointCloud2, "laser/points", qos_profile_sensor_data)
        self.cloud = self.create_publisher(PointCloud2, "head/points", qos_profile_sensor_data)
        self.rgb = self.create_publisher(CompressedImage, "head/image/compressed", qos_profile_sensor_data)
        self.depth = self.create_publisher(Image, "head/depth", qos_profile_sensor_data)
        self.intrinsics = self.create_publisher(CameraInfo, "head/camera_info", qos_profile_sensor_data)
        self.sim_clock = self.create_publisher(Clock, "simulation_clock", 5)
        self.sensor_clock = self.create_publisher(Clock, "clock", 5)
        self.footprints = [self.create_publisher(Polygon, f"{name}/footprint", 5)
                           for name in ("local_costmap", "global_costmap")]
        self.create_subscription(TwistStamped, "cmd_vel", self.receive_velocity, 1)
        self.navigator = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.lifecycle = {name: {"client": self.create_client(GetState, f"/{name}/get_state"), "future": None, "active": False}
            for name in ("bt_navigator", "planner_server", "controller_server")}
        self.io_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(.1, self.update, callback_group=self.io_group)
        self.command_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(.05, self.send_velocity, callback_group=self.command_group)
        self.odometry_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(.05, self.update_odometry, callback_group=self.odometry_group)

    def receive_velocity(self, message):
        issued = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if self.session_id and issued >= self.session_started and 0 <= time.time() - issued <= .25:
            self.command = (self.session_id, issued, message.twist.linear.x, message.twist.angular.z)

    def send_velocity(self):
        session_id, goal_handle = self.session_id, self.goal_handle
        packets, command = self.sensor_packets, self.command
        if self.finished or not session_id or not goal_handle or not goal_handle.accepted or not packets:
            return
        packet = packets[-1]
        navigation = packet["navigation"]
        if not navigation or navigation["session_id"] != session_id or navigation["status"] != "running":
            return
        linear, angular = 0., 0.
        if command and command[0] == session_id and command[1] >= self.session_started and 0 <= time.time() - command[1] <= .25:
            paired = [candidate for candidate in packets if candidate["captured_unix_s"] <= command[1] + 1e-6]
            if paired:
                packet = paired[-1]
                _, _, linear, angular = command
        try:
            self.command_sequence += 1
            response = self.command_http.post(f"{self.backend}/api/ros/velocity", json={"session_id": session_id,
                "sensor_sequence": packet["sequence"], "command_sequence": self.command_sequence,
                "linear_mps": linear, "angular_radps": angular}, timeout=.5)
            if response.status_code != 409:
                response.raise_for_status()
        except requests.RequestException as error:
            self.get_logger().error(f"Command bridge stopped: {type(error).__name__}: {error}")
            self.finished = True

    def transform(self, parent, child, position, rotation, timestamp):
        message = TransformStamped(header=Header(stamp=timestamp, frame_id=parent), child_frame_id=child)
        message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, position)
        message.transform.rotation = rotation
        return message

    def update_odometry(self):
        if self.finished:
            return
        try:
            response = self.odometry_http.get(f"{self.backend}/api/ros/odometry", timeout=.5)
            response.raise_for_status()
            packet = response.json()
            episode = (packet["run_id"], packet["episode_epoch"])
            if self.episode is not None and episode != self.episode:
                raise RuntimeError("Episode changed; restart ROS to discard old maps and goals")
            if not 0 <= time.time() - packet["captured_unix_s"] <= .5:
                return
            self.publish_odometry(packet)
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as error:
            self.get_logger().error(f"Odometry bridge stopped: {type(error).__name__}: {error}")
            self.finished = True

    def publish_odometry(self, packet):
        timestamp = stamp(packet["captured_unix_s"])
        horizontal, lateral, heading = packet["odometry_m_rad"]
        orientation = Quaternion(z=math.sin(heading / 2), w=math.cos(heading / 2))
        self.tf.sendTransform(self.transform("odom", "base_link", [horizontal, lateral, 0.], orientation, timestamp))
        if self.previous and packet["captured_unix_s"] <= self.previous["captured_unix_s"]:
            return
        odom = Odometry(header=Header(stamp=timestamp, frame_id="odom"), child_frame_id="base_link")
        odom.pose.pose.position.x, odom.pose.pose.position.y = horizontal, lateral
        odom.pose.pose.orientation = orientation
        if self.previous:
            elapsed = packet["captured_unix_s"] - self.previous["captured_unix_s"]
            if elapsed > 0:
                delta = np.array(packet["odometry_m_rad"]) - self.previous["odometry_m_rad"]
                odom.twist.twist.linear.x = float((delta[0] * math.cos(heading) + delta[1] * math.sin(heading)) / elapsed)
                odom.twist.twist.angular.z = float(delta[2] / elapsed)
        self.previous = packet
        self.odom.publish(odom)
        self.sensor_clock.publish(Clock(clock=timestamp))

    def publish_sensors(self, packet):
        timestamp = stamp(packet["captured_unix_s"])
        self.tf.sendTransform(self.transform("base_link", "head_optical", packet["camera_origin_m"],
            quaternion(packet["camera_rotation"]), timestamp))
        if not self.laser_transform_sent:
            self.static_tf.sendTransform(self.transform("base_link", "laser", packet["laser"]["origin_m"], Quaternion(w=1.), timestamp))
            self.laser_transform_sent = True
        self.tf.sendTransform(self.transform("odom", "base_link", [*packet["odometry_m_rad"][:2], 0.],
            Quaternion(z=math.sin(packet["odometry_m_rad"][2] / 2), w=math.cos(packet["odometry_m_rad"][2] / 2)), timestamp))
        laser = packet["laser"]
        scan = LaserScan(header=Header(stamp=timestamp, frame_id="laser"), angle_min=laser["angle_min"],
            angle_max=laser["angle_min"] + (len(laser["ranges_m"]) - 1) * laser["angle_increment"],
            angle_increment=laser["angle_increment"], range_min=laser["range_min"], range_max=laser["range_max"])
        scan.ranges = [math.nan if value is None else min(value, 7.875)
                       for value in laser["ranges_m"]]
        self.scan.publish(scan)
        ranges = np.asarray(scan.ranges)
        angles = laser["angle_min"] + np.arange(len(ranges)) * laser["angle_increment"]
        laser_points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles), np.zeros(len(ranges))))
        self.laser_cloud.publish(point_cloud2.create_cloud_xyz32(scan.header, laser_points[np.isfinite(ranges)]))
        calibration = packet["calibration"]
        depth = np.asarray(packet["depth_m"], dtype=np.float32).reshape(calibration["height"], calibration["width"])
        rows, columns = np.mgrid[0:calibration["height"]:2, 0:calibration["width"]:2]
        sampled = depth[::2, ::2]
        points = np.stack(((columns + .5 - calibration["cx"]) * sampled / calibration["fx"],
                           (rows + .5 - calibration["cy"]) * sampled / calibration["fy"], sampled), axis=-1)
        header = Header(stamp=timestamp, frame_id="head_optical")
        self.cloud.publish(point_cloud2.create_cloud_xyz32(header, points[np.isfinite(sampled)]))
        self.rgb.publish(CompressedImage(header=header, format="png", data=base64.b64decode(packet["head_rgb_png"])))
        self.depth.publish(Image(header=header, height=calibration["height"], width=calibration["width"],
            encoding="32FC1", is_bigendian=0, step=calibration["width"] * 4, data=depth.astype("<f4").tobytes()))
        info = CameraInfo(header=header, height=calibration["height"], width=calibration["width"], distortion_model="plumb_bob")
        info.k = [calibration["fx"], 0., calibration["cx"], 0., calibration["fy"], calibration["cy"], 0., 0., 1.]
        info.p = [calibration["fx"], 0., calibration["cx"], 0., 0., calibration["fy"], calibration["cy"], 0., 0., 0., 1., 0.]
        info.r = np.eye(3).flatten().tolist()
        info.d = [0.] * 5
        self.intrinsics.publish(info)
        lower, upper = packet["footprint"]["lower_xy_m"], packet["footprint"]["upper_xy_m"]
        polygon = Polygon(points=[Point32(x=float(horizontal), y=float(lateral), z=0.) for horizontal, lateral in
            [(lower[0], lower[1]), (upper[0], lower[1]), (upper[0], upper[1]), (lower[0], upper[1])]])
        for publisher in self.footprints:
            publisher.publish(polygon)
        self.sim_clock.publish(Clock(clock=stamp(packet["simulated_time_s"])))

    def update(self):
        if self.finished:
            return
        try:
            response = self.http.get(f"{self.backend}/api/ros/sensors", timeout=.5)
            response.raise_for_status()
            packet = response.json()
            episode = (packet["run_id"], packet["episode_epoch"])
            if self.episode is not None and episode != self.episode:
                raise RuntimeError("Episode changed; restart ROS to discard old maps and goals")
            self.episode = episode
            sensor_age = time.time() - packet["captured_unix_s"]
            if not -.1 <= sensor_age <= .5:
                if self.session_id is None and not (packet["navigation"] or {}).get("status") == "running":
                    self.get_logger().warning(f"Discarded idle sensor packet aged {sensor_age:.3f}s")
                    return
                raise RuntimeError(f"Sensor timestamp is stale or clocks are not synchronized: {sensor_age:.3f}s")
            self.publish_sensors(packet)
            self.sensor_packets = (*self.sensor_packets[-7:], packet)
            if time.monotonic() - self.heartbeat_at >= .5:
                for service in self.lifecycle.values():
                    if service["future"] is not None and service["future"].done():
                        service["active"] = service["future"].result().current_state.id == 3
                        service["future"] = None
                    if service["future"] is None and service["client"].service_is_ready():
                        service["future"] = service["client"].call_async(GetState.Request())
                    if not service["client"].service_is_ready():
                        service["active"] = False
                ready = self.navigator.server_is_ready() and all(service["active"] for service in self.lifecycle.values())
                heartbeat = self.http.post(f"{self.backend}/api/ros/heartbeat", json={"bridge_id": self.bridge_id,
                    "run_id": packet["run_id"], "episode_epoch": packet["episode_epoch"], "ready": ready,
                    "message": "Nav2 ready" if ready else "Nav2 lifecycle or map is still starting"}, timeout=.5)
                heartbeat.raise_for_status()
                self.heartbeat_at = time.monotonic()
            navigation = packet["navigation"]
            if self.session_id and (not navigation or navigation["session_id"] != self.session_id or navigation["status"] != "running"):
                self.command = None
                if not self.release_goal():
                    return
            if not navigation or navigation["status"] != "running":
                return
            if self.session_id is None:
                self.session_id = navigation["session_id"]
                self.session_started = time.time()
                self.command = None
            if self.goal_future is None and self.navigator.server_is_ready():
                goal = PoseStamped(header=Header(stamp=stamp(time.time()), frame_id="odom"))
                goal.pose.position.x, goal.pose.position.y, yaw = navigation["target_m_rad"]
                goal.pose.orientation = Quaternion(z=math.sin(yaw / 2), w=math.cos(yaw / 2))
                self.goal_future = self.navigator.send_goal_async(NavigateToPose.Goal(pose=goal))
            if self.goal_future and self.goal_future.done() and self.goal_handle is None:
                self.goal_handle = self.goal_future.result()
                if not self.goal_handle.accepted:
                    self.report_result("aborted", "Nav2 rejected the local goal")
                    return
                self.result_future = self.goal_handle.get_result_async()
                self.session_started = time.time()
                self.command = None
                self.get_logger().info("Nav2 accepted observed local goal")
            if self.result_future and self.result_future.done():
                status = self.result_future.result().status
                self.report_result("succeeded" if status == 4 else "cancelled" if status == 5 else "aborted",
                    f"Nav2 action ended with status {status}")
                return
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as error:
            self.get_logger().error(f"Bridge stopped: {type(error).__name__}: {error}")
            self.finished = True

    def release_goal(self):
        if self.goal_future is not None and not self.goal_future.done():
            return False
        if self.goal_future is not None and self.goal_handle is None:
            self.goal_handle = self.goal_future.result()
        if self.goal_handle is not None and self.goal_handle.accepted:
            if self.result_future is None:
                self.result_future = self.goal_handle.get_result_async()
            if not self.result_future.done():
                if self.cancel_future is None:
                    self.cancel_future = self.goal_handle.cancel_goal_async()
                return False
        self.goal_future = self.goal_handle = self.result_future = self.cancel_future = None
        self.session_id = None
        self.session_started = math.inf
        self.command = None
        return True

    def report_result(self, status, message):
        self.command = None
        response = self.http.post(f"{self.backend}/api/ros/result", json={"session_id": self.session_id,
            "status": status, "message": message}, timeout=.5)
        if response.status_code != 409:
            response.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://host.docker.internal:8012")
    args = parser.parse_args()
    rclpy.init()
    node = MiloBridge(args.backend)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=.1)
    finally:
        if node.goal_handle and node.goal_handle.accepted:
            node.goal_handle.cancel_goal_async()
        executor.shutdown()
        node.http.close()
        node.command_http.close()
        node.odometry_http.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()