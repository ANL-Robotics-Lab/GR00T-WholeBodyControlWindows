"""Canonical RBY1 joint and body ordering definitions.

This module deliberately has no Isaac Lab dependency so conversion tools,
training code, and simulator configuration can all share the same mappings.
Mapping semantics are always ``output[i] = input[mapping[i]]``.
"""

from __future__ import annotations


def _indices_for(output_names: list[str], input_names: list[str]) -> list[int]:
    """Return indices that reorder ``input_names`` into ``output_names``."""
    if len(input_names) != len(set(input_names)):
        raise ValueError("RBY1 input ordering contains duplicate names.")
    missing = [name for name in output_names if name not in input_names]
    if missing:
        raise ValueError(f"RBY1 ordering is missing names: {missing}")
    return [input_names.index(name) for name in output_names]


# Motion-library / MuJoCo active-joint order used by PKL ``dof`` arrays.
RBY1_MUJOCO_DOF_NAMES = [
    "right_wheel",
    "left_wheel",
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
    "head_0",
    "head_1",
]

# Filtered 24-DOF order observed from the loaded RBY1 Isaac articulation.
RBY1_ISAACLAB_DOF_NAMES = [
    "left_wheel",
    "torso_0",
    "right_wheel",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "left_arm_0",
    "right_arm_0",
    "head_0",
    "left_arm_1",
    "right_arm_1",
    "head_1",
    "left_arm_2",
    "right_arm_2",
    "left_arm_3",
    "right_arm_3",
    "left_arm_4",
    "right_arm_4",
    "left_arm_5",
    "right_arm_5",
    "left_arm_6",
    "right_arm_6",
]

RBY1_ISAACLAB_TO_MUJOCO_DOF = _indices_for(
    RBY1_MUJOCO_DOF_NAMES, RBY1_ISAACLAB_DOF_NAMES
)
RBY1_MUJOCO_TO_ISAACLAB_DOF = _indices_for(
    RBY1_ISAACLAB_DOF_NAMES, RBY1_MUJOCO_DOF_NAMES
)

# Axis-sign conversion from motion-library / clean-MJCF coordinates to the
# loaded Isaac Sim articulation.  The clean RBY1 MJCF used for FK has both
# wheel axes along -Y, while flatremovedRBY3.usd exposes both revolute joints
# along +Y.  All articulated upper-body axes use the same sign in both assets.
#
# This vector is deliberately in RBY1_MUJOCO_DOF_NAMES order so it can be
# applied directly to PKL ``dof`` positions and velocities before they are
# compared with, or written into, the Isaac articulation.
RBY1_MOTION_TO_ISAACLAB_DOF_SIGN = [
    -1.0 if name in {"right_wheel", "left_wheel"} else 1.0
    for name in RBY1_MUJOCO_DOF_NAMES
]


# Position-target scale used by the RBY1 policy action.  These values map the
# verified trainable dance's upper-body envelope to approximately [-1, 1]
# policy units. The articulation's physical limits remain authoritative.
# Keeping reference targets near unit scale is important for PPO: the previous
# effort/stiffness-derived arm scales required raw actions as large as 14.
#
# Wheels are velocity-controlled and therefore use a separate scale in the
# Isaac Lab robot configuration.
RBY1_POSITION_ACTION_SCALE = {
    "torso_0": 0.40,
    "torso_1": 1.20,
    "torso_2": 1.60,
    "torso_3": 1.30,
    "torso_4": 0.60,
    "torso_5": 0.75,
    "right_arm_0": 2.30,
    "right_arm_1": 3.20,
    "right_arm_2": 2.00,
    "right_arm_3": 2.70,
    "right_arm_4": 2.30,
    "right_arm_5": 2.20,
    "right_arm_6": 1.70,
    "left_arm_0": 2.30,
    "left_arm_1": 3.20,
    "left_arm_2": 2.00,
    "left_arm_3": 2.70,
    "left_arm_4": 2.30,
    "left_arm_5": 2.20,
    "left_arm_6": 1.70,
    "head_0": 1.60,
    "head_1": 1.60,
}


# Full 38-body order observed in the imported USD articulation.
RBY1_FULL_ISAACLAB_BODY_NAMES = [
    "base",
    "wheel_l",
    "link_torso_0",
    "Cylinder_01",
    "Cylinder_03",
    "wheel_r",
    "link_torso_1",
    "Cylinder",
    "Cylinder_02",
    "link_torso_2",
    "link_torso_3",
    "link_torso_4",
    "link_torso_5",
    "link_head_0",
    "link_left_arm_0",
    "link_right_arm_0",
    "link_head_1",
    "link_left_arm_1",
    "link_right_arm_1",
    "link_head_2",
    "link_left_arm_2",
    "link_right_arm_2",
    "link_left_arm_3",
    "link_right_arm_3",
    "link_left_arm_4",
    "link_right_arm_4",
    "link_left_arm_5",
    "link_right_arm_5",
    "link_left_arm_6",
    "link_right_arm_6",
    "FT_sensor_L",
    "FT_sensor_R",
    "ee_left",
    "ee_right",
    "ee_finger_l1",
    "ee_finger_l2",
    "ee_finger_r1",
    "ee_finger_r2",
]

# Collapsed body order used by the RBY1 motion-library MJCF.
RBY1_MUJOCO_BODY_NAMES = [
    "base",
    "wheel_r",
    "wheel_l",
    "link_torso_0",
    "link_torso_1",
    "link_torso_2",
    "link_torso_3",
    "link_torso_4",
    "link_torso_5",
    "link_right_arm_0",
    "link_right_arm_1",
    "link_right_arm_2",
    "link_right_arm_3",
    "link_right_arm_4",
    "link_right_arm_5",
    "link_right_arm_6",
    "link_left_arm_0",
    "link_left_arm_1",
    "link_left_arm_2",
    "link_left_arm_3",
    "link_left_arm_4",
    "link_left_arm_5",
    "link_left_arm_6",
    "link_head_1",
    "link_head_2",
]

# The same collapsed subset in full-articulation Isaac order.
RBY1_ISAACLAB_BODY_NAMES_25 = [
    "base",
    "wheel_l",
    "link_torso_0",
    "wheel_r",
    "link_torso_1",
    "link_torso_2",
    "link_torso_3",
    "link_torso_4",
    "link_torso_5",
    "link_left_arm_0",
    "link_right_arm_0",
    "link_head_1",
    "link_left_arm_1",
    "link_right_arm_1",
    "link_head_2",
    "link_left_arm_2",
    "link_right_arm_2",
    "link_left_arm_3",
    "link_right_arm_3",
    "link_left_arm_4",
    "link_right_arm_4",
    "link_left_arm_5",
    "link_right_arm_5",
    "link_left_arm_6",
    "link_right_arm_6",
]

RBY1_FULL_BODY_TO_COLLAPSED_25 = _indices_for(
    RBY1_ISAACLAB_BODY_NAMES_25, RBY1_FULL_ISAACLAB_BODY_NAMES
)
RBY1_FULL_ISAACLAB_TO_MUJOCO_BODY = _indices_for(
    RBY1_MUJOCO_BODY_NAMES, RBY1_FULL_ISAACLAB_BODY_NAMES
)
RBY1_ISAACLAB_TO_MUJOCO_BODY = _indices_for(
    RBY1_MUJOCO_BODY_NAMES, RBY1_ISAACLAB_BODY_NAMES_25
)
RBY1_MUJOCO_TO_ISAACLAB_BODY = _indices_for(
    RBY1_ISAACLAB_BODY_NAMES_25, RBY1_MUJOCO_BODY_NAMES
)


def _validate_inverse(forward: list[int], inverse: list[int]) -> None:
    if [forward[i] for i in inverse] != list(range(len(forward))):
        raise ValueError("RBY1 forward and inverse mappings are inconsistent.")


_validate_inverse(RBY1_ISAACLAB_TO_MUJOCO_DOF, RBY1_MUJOCO_TO_ISAACLAB_DOF)
_validate_inverse(RBY1_ISAACLAB_TO_MUJOCO_BODY, RBY1_MUJOCO_TO_ISAACLAB_BODY)
