"""Operator-only call locations. Never include this history in model observations."""
from collections import deque
import json
import math
import time


class ModelCallHistory:
    capacity = 256

    def __init__(self):
        self.calls = deque(maxlen=self.capacity)
        self.count = 0

    def begin(self, worker, profile, session_id, turn, kind):
        latest = worker.latest
        snapshot = latest.get("snapshot", {})
        base = next((pose for pose in snapshot.get("poses", [])
                     if pose["key"] == f"{latest.get('robot_body_id')}:-1"), None)
        xy = list(base["position"][:2]) if base else None
        if xy is not None and not all(math.isfinite(value) for value in xy):
            xy = None
        self.count += 1
        call = {"id": f"{session_id}:{self.count}", "number": self.count, "session_id": session_id,
                "run_id": latest.get("run_id"), "episode_epoch": latest.get("episode_epoch"),
                "turn": turn, "model": profile.label, "kind": kind,
                "timestamp": time.time(), "simulated_time_s": snapshot.get("simulated_time_s"),
                "position_world_m": xy, "status": "thinking", "summary": "Waiting for model response."}
        self.calls.append(call)
        return call

    @staticmethod
    def finish(call, response=None, status="responded"):
        call["status"] = status
        if response is None:
            call["summary"] = "Model request cancelled." if status == "cancelled" else "Model request failed."
            return
        summaries = []
        text = getattr(response, "output_text", "")
        if text:
            summaries.append(text)
        for item in getattr(response, "output", []):
            # Only public function arguments, never provider reasoning blocks.
            item = item.model_dump(exclude_none=True)
            if item.get("type") != "function_call":
                continue
            try:
                arguments = json.loads(item.get("arguments", ""))
            except (TypeError, ValueError):
                continue
            if not isinstance(arguments, dict):
                continue
            reason = arguments.get("reason") or arguments.get("guidance")
            action = arguments.get("action") or arguments.get("intent") or item.get("name", "")
            summary = reason if isinstance(reason, str) and reason.strip() else str(action).replace("_", " ")
            if arguments.get("frontier_id"):
                summary += f" / target {arguments['frontier_id']}"
            if summary:
                summaries.append(summary)
        call["summary"] = "\n".join(dict.fromkeys(summaries))[:2000] or "Response received; no explanation reported."
