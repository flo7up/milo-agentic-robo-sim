from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ARM_LIMITS = [(-1.8, 1.8), (-2.5, 2.5), (-2.7, 2.7), (-3.0, 3.0), (-2.5, 2.5), (-3.0, 3.0)]
NEUTRAL = [0.0, -0.6, 1.8, 0.0, -1.2, 0.0]
ARM_AXES = [(0, 0, 1), (0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (1, 0, 0)]


def numbers(values):
    return " ".join(str(value) for value in values)


def build_robot():
    root = ET.Element("robot", name="Milo-01")

    def link(name, size, mass, color, center=(0, 0, 0), shape="box", rotation=(0, 0, 0)):
        node = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(node, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(center), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(mass))
        if shape == "box":
            inertia = [mass * (size[1] ** 2 + size[2] ** 2) / 12,
                       mass * (size[0] ** 2 + size[2] ** 2) / 12,
                       mass * (size[0] ** 2 + size[1] ** 2) / 12]
        else:
            inertia = [mass * size[0] ** 2 * 0.5] * 3
        ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
        for kind in ("visual", "collision"):
            part = ET.SubElement(node, kind)
            ET.SubElement(part, "origin", xyz=numbers(center), rpy=numbers(rotation))
            geometry = ET.SubElement(part, "geometry")
            if shape == "box":
                ET.SubElement(geometry, "box", size=numbers(size))
            elif shape == "sphere":
                ET.SubElement(geometry, "sphere", radius=str(size[0]))
            else:
                ET.SubElement(geometry, "cylinder", radius=str(size[0]), length=str(size[1]))
            if kind == "visual":
                material = ET.SubElement(part, "material", name=name + "_paint")
                ET.SubElement(material, "color", rgba=numbers(color))

    def joint(name, parent, child, xyz, axis=(0, 0, 1), limits=None, kind="revolute", force=24, velocity=1.8):
        node = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(node, "parent", link=parent)
        ET.SubElement(node, "child", link=child)
        ET.SubElement(node, "origin", xyz=numbers(xyz), rpy="0 0 0")
        ET.SubElement(node, "axis", xyz=numbers(axis))
        if kind != "fixed":
            lower, upper = limits or (-100000, 100000)
            ET.SubElement(node, "limit", lower=str(lower), upper=str(upper), effort=str(force), velocity=str(velocity))
            ET.SubElement(node, "dynamics", damping="0.05", friction="0.01")

    shell, dark, accent = (0.86, 0.91, 0.89, 1), (0.13, 0.17, 0.18, 1), (0.76, 0.16, 0.30, 1)
    link("base", (.34, .31, .16), 12, shell)
    for side, sign in (("left", 1), ("right", -1)):
        link(side + "_wheel", (.09, .045), .45, dark, shape="cylinder", rotation=(math.pi / 2, 0, 0))
        joint(side + "_wheel", "base", side + "_wheel", (0, sign * .19, -.065), (0, 1, 0), kind="continuous", force=5, velocity=8)
    for name, forward in (("front", .135), ("rear", -.135)):
        link(name + "_support", (.035,), .1, dark, shape="sphere")
        joint(name + "_support", "base", name + "_support", (forward, 0, -.12), kind="fixed")
    link("torso", (.14, .18, .24), 1.1, accent, center=(-.04, 0, .12))
    joint("torso_fixed", "base", "torso", (0, 0, .08), kind="fixed")
    link("neck", (.065, .065, .10), .12, dark, center=(0, 0, .05))
    joint("head_yaw", "torso", "neck", (-.04, 0, .27), limits=(-1.5, 1.5), force=5)
    link("head", (.16, .22, .13), .35, shell)
    joint("head_pitch", "neck", "head", (0, 0, .10), (0, 1, 0), (-.7, 1.15), force=5)
    link("visor", (.012, .175, .065), .01, dark)
    joint("visor_fixed", "head", "visor", (.083, 0, .005), kind="fixed")
    for side, sign in (("left", 1), ("right", -1)):
        parent = side + "_shoulder_mount"
        link(parent, (.075, .18, .075), .12, dark, center=(0, -sign * .08, 0))
        joint(side + "_shoulder_fixed", "torso", parent, (0, sign * .25, .165), kind="fixed")
        offsets = [(0, 0, 0), (0, 0, 0), (.24, 0, 0), (.24, 0, 0), (0, 0, 0), (0, 0, 0)]
        for index in range(6):
            name = f"{side}_joint_{index + 1}"
            segment = f"{side}_link_{index + 1}"
            length = .24 if index in (1, 2) else .035
            if length == .24:
                link(segment, (length, .032, .032), .18, shell, center=(length / 2, 0, 0))
            else:
                link(segment, (.018,), .055, accent, shape="sphere")
            joint(name, parent, segment, offsets[index], ARM_AXES[index], ARM_LIMITS[index])
            parent = segment
        palm = side + "_palm"
        link(palm, (.028, .115, .045), .09, dark, center=(-.052, 0, 0))
        joint(side + "_palm_fixed", parent, palm, (.065, 0, 0), kind="fixed")
        for suffix, direction in (("inner", -1), ("outer", 1)):
            finger = f"{side}_{suffix}_finger"
            link(finger, (.075, .012, .045), .04, shell, center=(.008, 0, 0))
            joint(finger, palm, finger, (0, direction * .006, 0), (0, direction, 0), (0, .055), kind="prismatic", force=35, velocity=.15)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def ensure_asset():
    ASSETS.mkdir(exist_ok=True)
    content = build_robot()
    path = ASSETS / "milo.urdf"
    if not path.exists() or path.read_text() != content:
        path.write_text(content, encoding="utf-8")
    return path


def calibration():
    return {"name": "Milo-01", "asset_hash": hashlib.sha256(build_robot().encode()).hexdigest(),
            "frames": {"base": "x forward, y left, z up; origin chassis center 0.155 m above floor",
                       "camera_optical": "x right, y down, z forward", "quaternion": "xyzw",
                       "head_camera_offset_m": [.095, 0, .005]},
            "arms": {side: [{"name": f"{side}_joint_{index + 1}", "axis": ARM_AXES[index],
                             "limits_rad": limit, "neutral_rad": NEUTRAL[index], "velocity_radps": 1.8,
                             "force_nm": 24} for index, limit in enumerate(ARM_LIMITS)] for side in ("left", "right")},
            "head_limits_rad": {"yaw": [-1.5, 1.5], "pitch": [-.7, 1.15]},
            "gripper": {"max_opening_m": .11, "max_force_n": 35, "max_mass_kg": .35, "max_load_n": 18},
            "wheel": {"radius_m": .09, "track_m": .38}, "neutral_rad": NEUTRAL}


if __name__ == "__main__":
    ensure_asset()
    (ASSETS / "robot.json").write_text(json.dumps(calibration(), indent=2))