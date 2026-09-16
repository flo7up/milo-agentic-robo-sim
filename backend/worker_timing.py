from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import time


class WorkerTiming:
    def __init__(self, clock=time.monotonic, capacity=512, incident_capacity=8):
        if capacity < 1 or incident_capacity < 1:
            raise ValueError("Timing buffers must be bounded and nonempty")
        self.clock = clock
        self.clock_name = "monotonic" if clock is time.monotonic else "test"
        self.events = deque(maxlen=capacity)
        self.incidents = deque(maxlen=incident_capacity)
        self.stack = []
        self.sequence = 0
        self.span_sequence = 0
        self.incident_sequence = 0
        self.events_evicted = 0
        self.incidents_dropped = 0

    def event(self, kind, **values):
        self.sequence += 1
        if len(self.events) == self.events.maxlen:
            self.events_evicted += 1
        self.events.append({"sequence": self.sequence, "at_s": self.clock(), "kind": kind, **values})

    @contextmanager
    def measure(self, phase):
        self.span_sequence += 1
        span = {"span_id": self.span_sequence, "parent_id": self.stack[-1]["span_id"] if self.stack else None,
            "phase": phase, "started_at_s": self.clock(), "children_s": 0.}
        self.stack.append(span)
        error_type = None
        error_code = None
        try:
            yield
        except BaseException as error:
            error_type = type(error).__name__
            code = getattr(error, "code", None)
            error_code = code[:80] if isinstance(code, str) else None
            raise
        finally:
            ended = self.clock()
            self.stack.pop()
            elapsed = max(0., ended - span["started_at_s"])
            if self.stack:
                self.stack[-1]["children_s"] += elapsed
            self.event("span", **{key: value for key, value in span.items() if key != "children_s"},
                duration_s=elapsed, self_s=max(0., elapsed - span["children_s"]), error_type=error_type, error_code=error_code)

    def stop(self, diagnostics, run_id, epoch, simulated_s):
        now = self.clock()
        self.incident_sequence += 1
        if len(self.incidents) == self.incidents.maxlen:
            self.incidents_dropped += 1
        active = []
        for index, span in enumerate(self.stack):
            elapsed = max(0., now - span["started_at_s"])
            active_child = max(0., now - self.stack[index + 1]["started_at_s"]) if index + 1 < len(self.stack) else 0.
            active.append({key: value for key, value in span.items() if key != "children_s"} | {
                "duration_so_far_s": elapsed, "self_so_far_s": max(0., elapsed - span["children_s"] - active_child)})
        self.incidents.append({"schema_version": 1, "incident_id": self.incident_sequence,
            "run_id": run_id, "episode_epoch": epoch, "simulated_s": simulated_s,
            "clock": self.clock_name, "captured_at_s": now, "navigation": deepcopy(diagnostics),
            "events": deepcopy(list(self.events)), "active_spans": active,
            "events_evicted": self.events_evicted, "incidents_dropped": self.incidents_dropped,
            "note": "Worker wall time, not CPU time. self_s excludes measured child spans only. Queue waits overlap execution; active spans are incomplete."})

    def take_incidents(self):
        while self.incidents:
            yield self.incidents.popleft()


def timed(phase):
    def decorate(function):
        @wraps(function)
        def measured(owner, *args, **kwargs):
            timing = getattr(owner, "worker_timing", None)
            if timing is None:
                timing = getattr(getattr(owner, "worker", None), "worker_timing", None)
            if timing is None and args:
                timing = getattr(args[0], "worker_timing", None)
            if timing is None:
                return function(owner, *args, **kwargs)
            with timing.measure(phase):
                return function(owner, *args, **kwargs)
        return measured
    return decorate