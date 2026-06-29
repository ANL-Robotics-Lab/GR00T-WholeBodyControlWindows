"""Joint utility functions and constants for G1 and RBY1 robots.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

import torch

# G1 body joint names in IsaacLab order (29 DOF)
G1_ISAACLab_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

RBY1_BODY_JOINTS = [
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

RBY1_EXTRA_JOINTS = [
    "backwheel",
    "backwheel2",
    "gripper_finger_r1",
    "gripper_finger_r2",
]

# G1 hand joint names (14 DOF) - order from g1_43dof.yaml
G1_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# Caches for joint indices
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}


def _get_joint_indices_by_names(
    asset,
    joint_names: list[str],
    cache: dict,
    *,
    expected_count: int | None = None,
    label: str = "joints",
) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]

    robot_joint_names = asset.joint_names
    missing = [name for name in joint_names if name not in robot_joint_names]
    if missing:
        raise ValueError(
            f"Missing expected {label}: {missing}\n"
            f"Available robot joints: {robot_joint_names}"
        )

    indices = [robot_joint_names.index(name) for name in joint_names]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)

    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {label}, found {len(indices)}: {joint_names}"
        )

    cache[cache_key] = indices_tensor
    return indices_tensor


def _is_rby1(asset) -> bool:
    """Detect RBY1 from the presence of its core motion-library joints."""
    robot_joint_names = asset.joint_names
    return any(name in robot_joint_names for name in RBY1_BODY_JOINTS)


def get_body_joint_indices(asset) -> torch.Tensor:
    """Get indices of body joints tracked by the motion library."""
    if _is_rby1(asset):
        return _get_joint_indices_by_names(
            asset,
            RBY1_BODY_JOINTS,
            _body_joint_indices_cache,
            expected_count=24,
            label="RBY1 body joints",
        )

    return _get_joint_indices_by_names(
        asset,
        G1_ISAACLab_ORDER,
        _body_joint_indices_cache,
        expected_count=29,
        label="G1 body joints",
    )


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of extra joints absent from the motion library."""
    if _is_rby1(asset):
        return _get_joint_indices_by_names(
            asset,
            RBY1_EXTRA_JOINTS,
            _hand_joint_indices_cache,
            expected_count=4,
            label="RBY1 extra joints",
        )

    return _get_joint_indices_by_names(
        asset,
        G1_HAND_JOINTS,
        _hand_joint_indices_cache,
        expected_count=14,
        label="G1 hand joints",
    )