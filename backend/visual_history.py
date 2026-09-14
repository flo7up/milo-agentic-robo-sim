import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, eq=False)
class CameraView:
    frame_id: str
    sequence: int
    captured_at: float
    simulated_time_s: float
    head_rad: tuple
    odometry_m_rad: tuple
    image: bytes
    feature: np.ndarray
    size: tuple

    @property
    def heading(self):
        angle = self.odometry_m_rad[2] + self.head_rad[0]
        return math.atan2(math.sin(angle), math.cos(angle))

    def public(self, now):
        return {"frame_id": self.frame_id, "sequence": self.sequence, "age_s": round(now - self.captured_at, 3),
                "simulated_time_s": self.simulated_time_s, "head_rad": list(self.head_rad),
                "odometry_m_rad": list(self.odometry_m_rad), "view_heading_rad": self.heading,
                "image_size": list(self.size), "historical": True, "authorizes_motion": False}


class CameraHistory:
    recent_limit = 50
    keyframe_limit = 6
    retained_limit = 24
    max_age_s = 300.
    max_image_bytes = 256_000

    def __init__(self, run_id, episode_epoch, clock=time.monotonic):
        self.run_id, self.episode_epoch, self.clock = run_id, episode_epoch, clock
        self.generation = uuid4().hex[:8]
        self.recent = OrderedDict()
        self.retained = OrderedDict()
        self.keyframes = []
        self.pending_sequence = None
        self.pending_keyframes = []
        self.pending_reference = None
        self.reference = None
        self.last_sequence = 0
        self.last_capture = -math.inf
        self.reviewed_sequence = 0

    @staticmethod
    def distance(first, second):
        yaw = first.heading - second.heading
        yaw = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
        visual = np.abs(first.feature - second.feature).reshape(6, 4, 8, 4, 3).mean(axis=(1, 3, 4)).max()
        return max(math.dist(first.odometry_m_rad[:2], second.odometry_m_rad[:2]) / .3,
                   yaw / .3, abs(first.head_rad[1] - second.head_rad[1]) / .2, float(visual) / .06)

    @classmethod
    def select(cls, frames, limit):
        if len(frames) <= limit:
            return list(frames)
        selected = [frames[0], frames[-1]]
        while len(selected) < limit:
            candidates = [frame for frame in frames if frame not in selected]
            selected.append(max(candidates, key=lambda frame: min(cls.distance(frame, other) for other in selected)))
        return sorted(selected, key=lambda frame: frame.sequence)

    def prune(self):
        now = self.clock()
        for store, limit in ((self.recent, self.recent_limit), (self.retained, self.retained_limit)):
            for identifier, frame in list(store.items()):
                if not 0 <= now - frame.captured_at <= self.max_age_s:
                    del store[identifier]
            while len(store) > limit:
                store.popitem(last=False)
        self.keyframes = [frame for frame in self.keyframes if 0 <= now - frame.captured_at <= self.max_age_s]
        self.pending_keyframes = [frame for frame in self.pending_keyframes if 0 <= now - frame.captured_at <= self.max_age_s]
        if self.pending_reference is not None and not 0 <= now - self.pending_reference.captured_at <= self.max_age_s:
            self.pending_reference = None
        if self.reference is not None and not 0 <= now - self.reference.captured_at <= self.max_age_s:
            self.reference = None

    def record(self, sensor, image):
        if (sensor.run_id != self.run_id or sensor.episode_epoch != self.episode_epoch
                or sensor.sequence <= self.last_sequence or sensor.captured_at < self.last_capture
                or not 0 <= self.clock() - sensor.captured_at <= 1. or len(image) > self.max_image_bytes
                or not all(math.isfinite(value) for value in [sensor.simulated_time_s, *sensor.head_rad, *sensor.odometry_m_rad])):
            return False
        with Image.open(BytesIO(image)) as decoded:
            if decoded.size != (sensor.calibration.width, sensor.calibration.height) or not (1 <= decoded.width <= 640 and 1 <= decoded.height <= 480):
                return False
            feature = np.asarray(decoded.convert("RGB").resize((32, 24)), dtype=np.float32) / 255.
            size = decoded.size
        frame = CameraView(f"view-{self.generation}-{sensor.sequence}", sensor.sequence, sensor.captured_at,
            sensor.simulated_time_s, tuple(sensor.head_rad), tuple(sensor.odometry_m_rad), bytes(image), feature, size)
        self.last_sequence, self.last_capture = frame.sequence, frame.captured_at
        self.recent[frame.frame_id] = frame
        self.prune()
        if self.pending_sequence is not None and frame.sequence > self.pending_sequence:
            pending_references = self.pending_keyframes or ([self.pending_reference] if self.pending_reference else [])
            if not pending_references or min(self.distance(frame, other) for other in pending_references) >= 1.:
                self.pending_keyframes = self.select([*self.pending_keyframes, frame], self.keyframe_limit)
        references = self.keyframes or ([self.reference] if self.reference else [])
        if not references or min(self.distance(frame, other) for other in references) >= 1.:
            self.keyframes = self.select([*self.keyframes, frame], self.keyframe_limit)
            self.reference = frame
        return True

    def review(self, current_sequence):
        self.prune()
        available = [frame for frame in self.keyframes if self.reviewed_sequence < frame.sequence < current_sequence]
        selected = self.select(available, 4)
        for frame in available:
            self.retained[frame.frame_id] = frame
            self.retained.move_to_end(frame.frame_id)
        if self.pending_sequence != current_sequence:
            self.pending_sequence = current_sequence
            self.pending_keyframes = []
            preceding = [frame for frame in self.recent.values() if frame.sequence <= current_sequence]
            self.pending_reference = preceding[-1] if preceding else self.reference
            for frame in self.recent.values():
                references = self.pending_keyframes or ([self.pending_reference] if self.pending_reference else [])
                if frame.sequence > current_sequence and (not references or min(self.distance(frame, other) for other in references) >= 1.):
                    self.pending_keyframes = self.select([*self.pending_keyframes, frame], self.keyframe_limit)
        self.prune()
        now = self.clock()
        metadata = {"generation": self.generation, "through_sequence": current_sequence,
                    "frames": [frame.public(now) for frame in selected],
                    "available_frames": [frame.public(now) for frame in available],
                    "layout": "2x2, oldest to newest, left to right then top to bottom",
                    "source": "paired_head_camera", "authorizes_motion": False}
        return metadata, self.contact_sheet(selected)

    def acknowledge(self, generation, sequence):
        if generation != self.generation or sequence > self.last_sequence:
            raise ValueError("Camera review belongs to an invalid history generation or sequence")
        self.reviewed_sequence = max(self.reviewed_sequence, sequence)
        remaining = {frame.sequence: frame for frame in [*self.keyframes, *self.pending_keyframes]
                     if frame.sequence > self.reviewed_sequence}
        self.keyframes = self.select(sorted(remaining.values(), key=lambda frame: frame.sequence), self.keyframe_limit)
        self.pending_sequence = None
        self.pending_keyframes = []
        self.pending_reference = None
        preceding = [frame for frame in self.recent.values() if frame.sequence <= self.reviewed_sequence]
        if preceding:
            self.reference = preceding[-1]

    def original(self, frame_id):
        self.prune()
        frame = self.retained.get(frame_id)
        if frame is None:
            raise ValueError("Historical frame expired or was not offered in this camera history")
        return frame.public(self.clock()), frame.image

    @staticmethod
    def contact_sheet(frames):
        if not frames:
            return None
        width = max(frame.size[0] for frame in frames)
        height = max(frame.size[1] for frame in frames)
        gap, caption, heading = 4, 30, 20
        sheet = Image.new("RGB", (width * 2 + gap * 3, (height + caption) * 2 + gap * 3 + heading), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        draw.text((gap, gap), "HISTORICAL VIEWS - NOT CURRENT", fill="black")
        for index, frame in enumerate(frames):
            left = gap + (index % 2) * (width + gap)
            top = heading + gap + (index // 2) * (height + caption + gap)
            draw.text((left, top), f"{frame.frame_id}\nsim {frame.simulated_time_s:.2f}s | {math.degrees(frame.heading):+.0f} deg", fill="black")
            with Image.open(BytesIO(frame.image)) as original:
                sheet.paste(original.convert("RGB"), (left, top + caption))
        encoded = BytesIO()
        sheet.save(encoded, format="PNG")
        return encoded.getvalue()