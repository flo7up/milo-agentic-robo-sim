import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import time

import uvicorn


class ServerActivity:
    def __init__(self, app, idle_timeout_s=300., clock=time.monotonic):
        if not math.isfinite(idle_timeout_s) or idle_timeout_s < 0:
            raise ValueError("Idle timeout must be finite and nonnegative")
        self.app = app
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self.last_activity = None
        self.connections = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        self.connections += 1
        self.last_activity = self.clock()
        try:
            await self.app(scope, receive, send)
        finally:
            self.connections -= 1
            self.last_activity = self.clock()

    def expired(self, busy=False):
        now = self.clock()
        if self.last_activity is None or self.connections or busy:
            self.last_activity = now
        return bool(self.idle_timeout_s and not self.connections and not busy
            and now - self.last_activity >= self.idle_timeout_s)


def lab_is_busy(lab):
    worker = lab.worker
    return bool(lab.connections or lab.lock.locked() or lab.agent.active or lab.agent.recording_active
    or (getattr(lab, "regression", None) and lab.regression.active)
        or (lab.home_recording is not None and not getattr(lab.home_recording, "finished", False))
        or (worker and (worker.latest.get("busy") or worker.power_transition
            or (worker.home_mission and worker.home_mission.active)
            or (worker.skill and worker.skill.active)
            or (worker.continuous and worker.continuous.active)
            or (worker.ros_navigation and worker.ros_navigation.active)
            or worker.recorder is not None)))


class IdleServer(uvicorn.Server):
    def __init__(self, config, activity, busy, preserve=None):
        super().__init__(config)
        self.activity = activity
        self.busy = busy
        self.preserve = preserve

    async def on_tick(self, counter):
        if await super().on_tick(counter):
            return True
        if self.activity.expired(self.busy()):
            previous_activity = self.activity.last_activity
            if self.preserve is not None:
                try:
                    await self.preserve()
                except Exception:
                    logging.getLogger("uvicorn.error").exception("Could not preserve unused simulator; postponing shutdown.")
                    self.activity.last_activity = self.activity.clock()
                    return False
            if self.activity.last_activity != previous_activity or not self.activity.expired(self.busy()):
                return False
            logging.getLogger("uvicorn.error").info("No connected clients or active work for %.0f seconds; shutting down simulator.",
                self.activity.idle_timeout_s)
            return True
        return False


async def preserve_idle_state(lab, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1] / ".runtime" / "idle-sessions"
    worker = lab.worker
    if worker is None or lab_is_busy(lab):
        return
    state = lab.state()
    trace = lab.agent.trace()

    def capture(sim):
        home = worker.home_mission
        return {"state": state, "trace": trace,
            "home_map": home.home.document() if home and home.home else None,
            "home_state": home.state() if home else None}

    snapshot = await worker.call(capture)
    image = next(reversed(worker.camera_frames.values()), None)

    def write():
        directory = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + state["run_id"])
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
        if image is not None:
            (directory / "camera.png").write_bytes(image)

    await asyncio.to_thread(write)


def main():
    parser = argparse.ArgumentParser(description="Run the local simulator with unused-server cleanup.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--idle-timeout", type=float, default=300., help="Unused seconds before shutdown; 0 explicitly keeps the server alive.")
    options = parser.parse_args()
    if not math.isfinite(options.idle_timeout) or options.idle_timeout < 0:
        parser.error("--idle-timeout must be finite and nonnegative")
    from backend.app import app, lab
    activity = ServerActivity(app, options.idle_timeout)
    server = IdleServer(uvicorn.Config(activity, host=options.host, port=options.port, access_log=False),
        activity, lambda: lab_is_busy(lab), lambda: preserve_idle_state(lab))
    server.run()


if __name__ == "__main__":
    main()