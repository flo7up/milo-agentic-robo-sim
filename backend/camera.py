import multiprocessing
from queue import Empty, Full


def render_snapshots(scene, requests, responses):
    import pybullet as bullet
    from backend.simulation import BulletSimulation
    sim = None
    try:
        sim = BulletSimulation(scene=scene, width=320, height=240)
        responses.put(("ready", None))
        while True:
            request = requests.get()
            if request is None:
                return
            for body, position, orientation in request["bodies"]:
                bullet.resetBasePositionAndOrientation(body, position, orientation, physicsClientId=sim.client)
            for name, position in request["joints"]:
                bullet.resetJointState(sim.robot, sim.joints[name], position, physicsClientId=sim.client)
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