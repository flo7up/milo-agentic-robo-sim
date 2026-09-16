import hashlib
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from backend.contracts import ObservedMapContext


def observed_context(cells, origin, resolution, pose, *, identity, run_id, epoch, sequence, captured_at,
                     now, frame, localization, camera_yaw, destinations=(), trail=(), route=(), overview=False):
    grid = np.asarray(cells)
    if grid.ndim != 2 or not np.isin(grid, [-1, 0, 100]).all() or not 0 <= now - captured_at <= 1.:
        raise ValueError("Map context requires fresh sensor evidence and a valid occupancy grid")
    if frame == "map" and localization != "localized":
        raise ValueError("Unlocalized saved geometry cannot be shown as the current map")
    center = np.floor((np.asarray(pose[:2]) - origin) / resolution).astype(int)
    if overview:
        known = np.argwhere(grid != -1)
        if not len(known):
            raise ValueError("No observed map cells")
        lower = np.minimum(known.min(axis=0)[::-1], center)
        upper = np.maximum(known.max(axis=0)[::-1] + 1, center + 1)
    else:
        lower, upper = center - 40, center + 40
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, grid.shape[::-1])
    if np.any(upper <= lower):
        raise ValueError("Robot pose is outside the observed map")
    crop = grid[lower[1]:upper[1], lower[0]:upper[0]].copy()
    scale = max(1, math.ceil(max(crop.shape) / 96))
    if scale > 1:
        rows, columns = (math.ceil(size / scale) for size in crop.shape)
        padded = np.full((rows * scale, columns * scale), -1, dtype=np.int16)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        blocks = padded.reshape(rows, scale, columns, scale)
        occupied = (blocks == 100).any(axis=(1, 3))
        free = (blocks == 0).all(axis=(1, 3))
        crop = np.where(occupied, 100, np.where(free, 0, -1))
    cropped_origin = (np.asarray(origin) + lower * resolution).tolist()
    revision = hashlib.sha256(crop.astype(np.int16).tobytes() + repr((identity, frame, cropped_origin, resolution * scale)).encode()).hexdigest()[:16]
    return ObservedMapContext(map_id=identity, revision=revision, run_id=run_id, episode_epoch=epoch,
        source_sequence=sequence, captured_at=captured_at, age_s=now-captured_at, frame=frame,
        snapshot_built_at_monotonic_s=now,
        localization=localization, scope="overview" if overview else "local", width=crop.shape[1], height=crop.shape[0],
        resolution_m=resolution * scale, origin_m=cropped_origin, cells=crop.ravel().tolist(),
        robot_pose_m_rad=list(pose), camera_yaw_rad=camera_yaw, destinations=list(destinations)[:24],
        trail_m=[list(point[:2]) for point in trail][-128:], route_m=[list(point[:2]) for point in route][:128])


def render_observed_map(context: ObservedMapContext):
    grid = np.asarray(context.cells).reshape(context.height, context.width)
    pixels = np.full((*grid.shape, 3), [142, 148, 153], dtype=np.uint8)
    pixels[grid == 0] = [246, 248, 246]
    pixels[grid == 100] = [43, 48, 49]
    image = Image.new("RGB", (512, 552), "white")
    scale = min(480 / context.width, 480 / context.height)
    width, height = round(context.width * scale), round(context.height * scale)
    left, top = (512 - width) // 2, 24 + (480 - height) // 2
    image.paste(Image.fromarray(pixels[::-1]).resize((width, height), Image.Resampling.NEAREST), (left, top))
    draw = ImageDraw.Draw(image)
    draw.text((12, 4), f"OBSERVED {context.scope.upper()} / {context.frame} / +Y up", fill="black")

    def point(position):
        horizontal = (position[0] - context.origin_m[0]) / context.resolution_m
        vertical = context.height - (position[1] - context.origin_m[1]) / context.resolution_m
        if not 0 <= horizontal < context.width or not 0 <= vertical <= context.height:
            return None
        return (left + horizontal * scale, top + vertical * scale)

    for path, color in ((context.trail_m, "#178266"), (context.route_m, "#af3c79")):
        for begin, end in zip(path, path[1:]):
            first, last = point(begin), point(end)
            if first and last:
                draw.line((first, last), fill=color, width=2)
    for index, destination in enumerate(context.destinations):
        location = point(destination["position_m"])
        if location:
            draw.ellipse((location[0]-3, location[1]-3, location[0]+3, location[1]+3), fill="#a05412")
            draw.text((location[0]+4, location[1]-10), str(index+1), fill="#803500")
    robot = point(context.robot_pose_m_rad)
    if robot:
        draw.ellipse((robot[0]-5, robot[1]-5, robot[0]+5, robot[1]+5), fill="#0078b4")
        for angle, length, color in ((context.robot_pose_m_rad[2], 20, "#0078b4"), (context.camera_yaw_rad, 32, "#d23535")):
            draw.line((robot, (robot[0]+length*math.cos(angle), robot[1]-length*math.sin(angle))), fill=color, width=3)
    draw.text((12, 509), "White free | Dark occupied | Gray unknown", fill="black")
    draw.text((12, 525), "Blue robot | Red view | Green travelled | Pink planned", fill="black")
    if context.trail_truncated:
        draw.text((12, 539), "Recent travelled segment only / older samples dropped", fill="black")
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    return encoded.getvalue()