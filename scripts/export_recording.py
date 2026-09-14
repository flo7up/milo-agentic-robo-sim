import argparse
from bisect import bisect_right
from functools import lru_cache
import json
import hashlib
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class RecordingVideo:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.score = json.loads((self.directory / "scorecard.json").read_text(encoding="utf-8"))
        self.samples = [json.loads(line) for line in (self.directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.samples:
            raise ValueError("Recording has no samples")
        self.times = [sample["wall_s"] for sample in self.samples]
        if any(not math.isfinite(value) or value < 0 for value in self.times) or self.times != sorted(self.times):
            raise ValueError("Recording times must be finite, nonnegative and ordered")
        self.duration = self.times[-1]
        self.heading_font, self.text_font, self.small_font = font(28), font(20), font(16)
        positions = [sample["position_m"] for sample in self.samples]
        minimum = [min(position[axis] for position in positions) for axis in (0, 1)]
        maximum = [max(position[axis] for position in positions) for axis in (0, 1)]
        self.center = [(minimum[axis] + maximum[axis]) / 2 for axis in (0, 1)]
        self.span = max(2., maximum[0] - minimum[0] + 1., maximum[1] - minimum[1] + 1.)

    @lru_cache(maxsize=4)
    def camera(self, relative_path):
        path = (self.directory / relative_path).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Camera path escapes recording directory")
        with Image.open(path) as image:
            return image.convert("RGB").resize((768, 576), Image.Resampling.NEAREST)

    def point(self, position):
        return (1016 + (position[0] - self.center[0]) * 400 / self.span,
                336 - (position[1] - self.center[1]) * 400 / self.span)

    def frame(self, wall_s, speed=1.):
        index = max(0, bisect_right(self.times, wall_s) - 1)
        sample = self.samples[index]
        image = Image.new("RGB", (1280, 720), "#f7f4ef")
        draw = ImageDraw.Draw(image)
        title = self.manifest.get("label") or self.manifest.get("challenge", "Milo test")
        while draw.textlength(title, font=self.heading_font) > 1000:
            title = title[:-4] + "..."
        draw.text((24, 12), title, font=self.heading_font, fill="#242424")
        source = "REAL MODEL" if self.manifest.get("real_model") else "SCRIPTED TEST / NOT AUTONOMY"
        draw.text((24, 50), f"{source}  |  {speed:g}x playback  |  sampled head-camera footage", font=self.small_font, fill="#5c5c5c")
        draw.rectangle((24, 88, 791, 663), fill="#292929")
        camera = sample.get("camera")
        if camera:
            image.paste(self.camera(camera["path"]), (24, 88))
        else:
            draw.text((48, 110), "No camera sample available", font=self.text_font, fill="white")
        spectator = sample.get("spectator")
        draw.text((816, 88), "Measured spectator view" if spectator else "Measured physical route", font=self.text_font, fill="#242424")
        draw.rectangle((816, 128, 1256, 548), fill="white", outline="#dedede")
        route = [self.point(entry["position_m"]) for entry in self.samples[:index + 1]]
        if len(route) > 1:
            draw.line(route, fill="#b11f4b", width=3)
        horizontal, vertical = self.point(sample["position_m"])
        draw.ellipse((horizontal - 6, vertical - 6, horizontal + 6, vertical + 6), fill="#242424")
        draw.line((horizontal, vertical, horizontal + 22 * math.cos(sample["yaw_rad"]), vertical - 22 * math.sin(sample["yaw_rad"])), fill="#0078d4", width=4)
        if spectator:
            image.paste(self.camera(spectator["path"]).resize((440, 330)), (816, 150))
        draw.text((828, 522), "Evaluator only / never model input" if spectator else f"View spans {self.span:.1f} m / evaluator only", font=self.small_font, fill="#5c5c5c")
        draw.text((816, 568), f"State: {sample['activity'].replace('_', ' ')}", font=self.text_font, fill="#242424")
        draw.text((816, 602), f"Speed: {sample['linear_speed_mps']:.2f} m/s", font=self.text_font, fill="#242424")
        draw.text((816, 636), f"Physics: {sample['physics_status'].replace('_', ' ')}", font=self.text_font, fill="#242424")
        draw.text((24, 682), f"Wall {min(wall_s, self.duration):.1f} / {self.duration:.1f} s  |  Simulation {sample['simulated_s']:.1f} s", font=self.small_font, fill="#242424")
        final = "UNSCORED" if self.samples[-1]["physics_status"] == "unscored" else "PASS" if self.score["final_physics_success"] else "NOT COMPLETED"
        integrity = "" if self.score["complete_recording"] else " / INCOMPLETE RECORDING"
        draw.text((816, 682), f"Final: {final}{integrity}", font=self.small_font, fill="#166534" if final == "PASS" else "#b91c1c")
        return image


def export_video(directory, output, *, speed=4., fps=10):
    import av
    output = Path(output)
    if output.suffix.lower() != ".mp4" or output.exists():
        raise ValueError("Choose a new .mp4 output path")
    if not math.isfinite(speed) or not 0 < speed <= 16 or not 1 <= fps <= 30:
        raise ValueError("Playback speed must be in (0, 16] and FPS in [1, 30]")
    recording = RecordingVideo(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = math.ceil(recording.duration / speed * fps) + 1
    with av.open(str(output), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        stream.options = {"crf": "22", "preset": "fast"}
        for index in range(frames):
            frame = av.VideoFrame.from_image(recording.frame(min(recording.duration, index * speed / fps), speed))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = {"video": str(output.resolve()), "recording": str(Path(directory).resolve()),
        "scenario": recording.manifest.get("challenge"), "real_model": recording.manifest.get("real_model", False),
        "playback_speed": speed, "encoded_fps": fps, "camera_sample_interval_s": recording.manifest.get("media_interval_s"),
        "duration_s": frames / fps, "frames": frames, "final_physics_success": recording.score["final_physics_success"],
        "complete_recording": recording.score["complete_recording"], "audio": False}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def verify_videos(directory):
    import av
    paths = sorted(Path(directory).glob("*.mp4"))
    if not paths:
        raise ValueError("No MP4 files to verify")
    sheet = Image.new("RGB", (640, 360 * len(paths)), "white")
    results = []
    for row, path in enumerate(paths):
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        signatures = set()
        count = 0
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"Expected one video stream: {path.name}")
            for frame in container.decode(video=0):
                if (frame.width, frame.height) != (1280, 720):
                    raise ValueError(f"Unexpected video dimensions: {path.name}")
                if count % 10 == 0 or count == metadata["frames"] // 2:
                    image = frame.to_image()
                    camera = image.crop((24, 88, 792, 664)).resize((64, 48))
                    if len(camera.getcolors(3073)) <= 10:
                        raise ValueError(f"Blank camera footage: {path.name}")
                    signatures.add(hashlib.sha256(camera.tobytes()).hexdigest())
                    if count == metadata["frames"] // 2:
                        sheet.paste(image.resize((640, 360)), (0, row * 360))
                count += 1
        if count != metadata["frames"] or len(signatures) < 2:
            raise ValueError(f"Incomplete or unchanging footage: {path.name}")
        results.append({"file": path.name, "frames": count, "duration_s": metadata["duration_s"],
                        "changing_camera_samples": len(signatures), "decode": "passed"})
    sheet.save(Path(directory) / "video-contact-sheet.png")
    (Path(directory) / "video-verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export saved simulation evidence as labeled MP4; no inference or actuation")
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-directory", type=Path)
    parser.add_argument("--speed", type=float, default=4.)
    parser.add_argument("--fps", type=int, default=10)
    options = parser.parse_args()
    if options.verify_directory:
        print(json.dumps(verify_videos(options.verify_directory), indent=2))
    elif options.recording and options.output:
        print(json.dumps(export_video(options.recording, options.output, speed=options.speed, fps=options.fps), indent=2))
    else:
        parser.error("Provide --recording and --output, or --verify-directory")