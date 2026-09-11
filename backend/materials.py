from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


TEXTURE_NAMES = ("concrete", "plaster", "wood", "tile", "fabric", "stone")
TEXTURE_ROOT = Path(__file__).resolve().parents[1] / "assets" / "textures"


@lru_cache(maxsize=1)
def ensure_textures():
    TEXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    vertical, horizontal = np.indices((512, 512))
    random = np.random.default_rng(713)
    for name in TEXTURE_NAMES:
        path = TEXTURE_ROOT / f"{name}.png"
        noise = random.normal(0, 3, (512, 512))
        if name == "wood":
            boards = vertical // 64
            grain = np.sin(horizontal * .08 + np.sin(vertical * .28) * 2) * 4
            values = 218 + grain + noise + np.take([4, -8, 0, 10, -3, 6, -6, 2], boards)
            joints = (vertical % 64 < 2) | ((horizontal + (boards % 2) * 128) % 256 < 2)
            values[joints] = 140
            rgb = np.stack([values, values * .97, values * .91], axis=2)
        elif name in {"tile", "concrete"}:
            values = 224 + noise + np.sin(horizontal * .018) * np.sin(vertical * .026) * 4
            spacing = 64 if name == "tile" else 256
            values[(vertical % spacing < 2) | (horizontal % spacing < 2)] = 165 if name == "tile" else 186
            rgb = np.repeat(values[:, :, None], 3, axis=2)
        elif name == "fabric":
            values = 206 + noise * 1.8 + ((horizontal % 4 < 2) ^ (vertical % 4 < 2)) * 24
            rgb = np.repeat(values[:, :, None], 3, axis=2)
        elif name == "stone":
            values = 213 + noise * 2 + np.sin(horizontal * .05 + np.sin(vertical * .025) * 3) * 8
            rgb = np.repeat(values[:, :, None], 3, axis=2)
        else:
            values = 236 + noise * 1.6 + np.sin(horizontal * .04) * np.cos(vertical * .07) * 3
            rgb = np.repeat(values[:, :, None], 3, axis=2)
        if not path.exists():
            Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)
    return {name: TEXTURE_ROOT / f"{name}.png" for name in TEXTURE_NAMES}


def scene_material(item, apartment=False):
    name = item["name"]
    if item.get("material") in TEXTURE_NAMES:
        return item["material"]
    if name == "floor":
        return "wood" if apartment else "concrete"
    if name.endswith("_floor"):
        return "tile" if name.startswith(("kitchen", "bathroom")) else "wood"
    if item.get("marker"):
        return None
    if "sofa" in name:
        return "fabric"
    if "counter" in name or "pedestal" in name:
        return "stone"
    if any(part in name for part in ("wall", "partition", "divider", "screen", "door_end")):
        return "plaster"
    return None