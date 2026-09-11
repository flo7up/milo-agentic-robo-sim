import xml.etree.ElementTree as ET

import pytest

from backend.robot import build_robot, calibration


def test_robot_has_two_six_axis_arms_and_bounded_fingers():
    root = ET.fromstring(build_robot())
    for side in ("left", "right"):
        joints = [joint for joint in root.findall("joint") if joint.attrib["name"].startswith(side + "_joint_")]
        assert len(joints) == 6
        assert all(joint.find("limit") is not None for joint in joints)
    assert calibration()["gripper"]["max_opening_m"] == .11


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_shoulder_mount_bridges_torso_and_unchanged_arm_pivot(side, sign):
    root = ET.fromstring(build_robot())
    mount = root.find(f"link[@name='{side}_shoulder_mount']")
    attachment = root.find(f"joint[@name='{side}_shoulder_fixed']")
    shoulder = root.find(f"joint[@name='{side}_joint_1']")
    torso = root.find("link[@name='torso']")
    assert mount is not None and attachment is not None
    assert attachment.attrib["type"] == "fixed"
    assert attachment.find("parent").attrib["link"] == "torso"
    assert attachment.find("child").attrib["link"] == mount.attrib["name"]
    assert shoulder.find("parent").attrib["link"] == mount.attrib["name"]

    def origin(element):
        return [float(value) for value in element.find("origin").attrib["xyz"].split()]

    torso_origin = origin(root.find("joint[@name='torso_fixed']"))
    mount_origin = [base + offset for base, offset in zip(torso_origin, origin(attachment))]
    pivot = [base + offset for base, offset in zip(mount_origin, origin(shoulder))]
    assert pivot == pytest.approx([0, sign * .25, .245])
    assert float(mount.find("inertial/mass").attrib["value"]) > 0

    for kind in ("visual", "collision"):
        def bounds(link, position):
            shape = link.find(kind)
            size = [float(value) for value in shape.find("geometry/box").attrib["size"].split()]
            center = [base + offset for base, offset in zip(position, origin(shape))]
            return [(value - length / 2, value + length / 2) for value, length in zip(center, size)]

        mount_bounds = bounds(mount, mount_origin)
        torso_bounds = bounds(torso, torso_origin)
        assert all(lower <= value <= upper for (lower, upper), value in zip(mount_bounds, pivot))
        assert all(min(mount_upper, torso_upper) > max(mount_lower, torso_lower)
                   for (mount_lower, mount_upper), (torso_lower, torso_upper) in zip(mount_bounds, torso_bounds))