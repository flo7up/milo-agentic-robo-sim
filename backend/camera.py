import multiprocessing
from queue import Empty, Full


class EnhancedRenderer:
    def __init__(self):
        import subprocess
        import threading
        from pathlib import Path
        from queue import Queue
        root = Path(__file__).resolve().parents[1]
        self.lock = threading.Lock()
        self.replies = Queue(maxsize=2)
        self.sequence = 0
        self.closed = False
        self.device = None
        self.process = subprocess.Popen(["node", str(root / "scripts/render_camera.mjs"), "--serve"], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True, name="enhanced-camera-replies")
        self.reader.start()
        try:
            if self.replies.get(timeout=60) != {"ready": True}:
                raise RuntimeError("Enhanced camera did not start; build the frontend and check Microsoft Edge")
        except Exception:
            self.close()
            raise

    def _read(self):
        import json
        try:
            for line in self.process.stdout:
                self.replies.put_nowait(json.loads(line))
        except Exception:
            pass
        finally:
            try:
                self.replies.put_nowait({"error": "Enhanced camera process exited"})
            except Full:
                pass

    def capture(self, packet, depth=True):
        import base64
        import json
        import numpy as np
        with self.lock:
            if self.closed:
                raise RuntimeError("Enhanced camera is closed")
            self.sequence += 1
            try:
                self.process.stdin.write(json.dumps({"id": self.sequence, "packet": packet, "depth": depth}, allow_nan=False) + "\n")
                self.process.stdin.flush()
                reply = self.replies.get(timeout=20 if self.device is None else 5)
                if reply.get("id") != self.sequence or reply.get("error"):
                    raise RuntimeError(reply.get("error", "Enhanced camera response mismatch"))
                frame = reply["frame"]
                identity = (packet["run_id"], packet["episode_epoch"], packet["observation_seq"], packet["snapshot"]["simulated_time_s"])
                if (frame["run_id"], frame["episode_epoch"], frame["observation_seq"], frame["simulated_time_s"]) != identity:
                    raise RuntimeError("Enhanced camera returned a different snapshot")
                width, height = packet["camera"]["width"], packet["camera"]["height"]
                if (frame["width"], frame["height"]) != (width, height):
                    raise RuntimeError("Enhanced camera dimensions differ")
                image = base64.b64decode(frame["rgb"], validate=True)
                if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise RuntimeError("Enhanced camera did not return PNG")
                values = np.frombuffer(base64.b64decode(frame["depth_f32"], validate=True), dtype="<f4").reshape(height, width).copy() if depth else None
                self.device = reply["renderer"]
                return image, values
            except Exception as error:
                self.close()
                raise RuntimeError(f"Enhanced camera failed: {error}") from error

    def close(self):
        import os
        import subprocess
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"close":true}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                else:
                    self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()


def scene_snapshot(sim, width=None, height=None):
    width = sim.width if width is None else width
    height = sim.height if height is None else height
    if any(type(value) is not int or not 1 <= value <= 2048 for value in (width, height)):
        raise ValueError("Render dimensions must be integers between 1 and 2048")
    eye, rotation = sim.camera_pose()
    return {"schema": "milo-render-v1", "run_id": sim.run_id, "episode_epoch": sim.epoch,
            "observation_seq": sim.seq, "robot_body_id": sim.robot, "snapshot": sim.snapshot(),
            "geometry": sim.geometry(), "camera": {"eye": eye.tolist(),
                "target": (eye + rotation[:, 0]).tolist(), "up": rotation[:, 2].tolist(),
                "vertical_fov_deg": 65, "near_m": .015, "far_m": 12,
                "width": width, "height": height}}


def prepare_spatial_processor():
    import backend.spatial
    from PIL import Image
    from scipy.ndimage import minimum_filter
    minimum_filter([[0., 0.], [0., 0.]], size=3)
    Image.init()
    return True


def process_spatial(observed_map, observation, image):
    from copy import copy
    from io import BytesIO
    import numpy as np
    from PIL import Image
    mapped = copy(observed_map)
    mapped.observation = None
    for name in ("cells", "floor_colors", "obstacle_low", "obstacle_high", "clear_votes", "origin"):
        setattr(mapped, name, getattr(observed_map, name).copy())
    mapped.update(observation, rgb=np.asarray(Image.open(BytesIO(image)).convert("RGB")))
    depth = np.array(observation.depth_m, dtype=float).reshape(observation.calibration.height, observation.calibration.width)
    pixels = np.zeros((*depth.shape, 4), dtype=np.uint8)
    valid = np.isfinite(depth)
    intensity = np.zeros(depth.shape, dtype=np.uint8)
    intensity[valid] = np.clip(255 * (1 - depth[valid] / observation.calibration.usable_range_m), 1, 255).astype(np.uint8)
    pixels[:, :, :3] = intensity[:, :, None]
    pixels[:, :, 3] = valid * 255
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return mapped, observation, image, output.getvalue()


class EnhancedResources:
    def __init__(self):
        from concurrent.futures import ProcessPoolExecutor
        self.processor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        self.renderer = None
        try:
            self.processor.submit(prepare_spatial_processor).result(timeout=60)
            self.renderer = EnhancedRenderer()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.renderer:
            self.renderer.close()
        self.processor.shutdown(wait=True, cancel_futures=True)


class EnhancedSnapshotRenderer:
    def __init__(self, renderer):
        from concurrent.futures import ThreadPoolExecutor
        self.renderer = renderer
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enhanced-preview")
        self.pending = None

    def submit(self, snapshot):
        if self.pending is not None:
            return
        def render():
            image, _ = self.renderer.capture(snapshot["packet"], depth=False)
            return snapshot["observation"], image, snapshot["captured_at"]
        self.pending = self.executor.submit(render)

    def latest(self):
        if self.pending is None or not self.pending.done():
            return None
        pending, self.pending = self.pending, None
        return pending.result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def snapshot_renderer(sim):
    if sim.rendering == "enhanced":
        if sim.camera_renderer is None:
            sim.capture()
        return EnhancedSnapshotRenderer(sim.camera_renderer)
    return SnapshotRenderer(sim.scene)


def render_snapshots(scene, requests, responses):
    import pybullet as bullet
    from backend.simulation import BulletSimulation
    sim = None
    try:
        sim = BulletSimulation(scene=scene, width=320, height=240, rendering="tiny")
        responses.put(("ready", None))
        while True:
            request = requests.get()
            if request is None:
                return
            for body, position, orientation in request["bodies"]:
                bullet.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=sim.client)
            for name, position in request["joints"]:
                bullet.resetJointState(sim.robot, sim.joints[name], position, physicsClientId=sim.client)
            sim.width, sim.height = request.get("dimensions", (320, 240))
            result = (request["observation"], sim.capture(), request["captured_at"])
            try:
                responses.put_nowait(("frame", result))
            except Full:
                try:
                    responses.get_nowait()
                except Empty:
                    pass
                try:
                    responses.put_nowait(("frame", result))
                except Full:
                    pass
    finally:
        if sim:
            sim.close()


class SnapshotRenderer:
    def __init__(self, scene):
        context = multiprocessing.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.responses = context.Queue(maxsize=1)
        self.process = context.Process(target=render_snapshots, args=(scene, self.requests, self.responses), daemon=True)
        self.process.start()
        try:
            if self.responses.get(timeout=20)[0] != "ready":
                raise RuntimeError("Snapshot renderer could not start")
        except Exception:
            self.close()
            raise

    def submit(self, snapshot):
        try:
            self.requests.put_nowait(snapshot)
        except Full:
            try:
                self.requests.get_nowait()
            except Empty:
                pass
            try:
                self.requests.put_nowait(snapshot)
            except Full:
                pass

    def latest(self):
        latest = None
        while True:
            try:
                kind, result = self.responses.get_nowait()
                if kind == "frame":
                    latest = result
            except Empty:
                return latest

    def close(self):
        try:
            self.requests.put_nowait(None)
        except Full:
            pass
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for queue in (self.requests, self.responses):
            queue.cancel_join_thread()
            queue.close()