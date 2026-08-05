"""Joint utility functions and constants for G1 and RBY1 robots.

This module provides joint ordering constants and helper functions for mapping
between motion library data and robot joints.
"""

import torch

from gear_sonic.utils.rby1_order import (
    RBY1_ISAACLAB_DOF_NAMES as RBY1_ACTUAL_ISAACLAB_DOF_NAMES,
    RBY1_MUJOCO_DOF_NAMES as RBY1_MOTIONLIB_BODY_JOINTS,
)

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


# Backward-compatible alias if other files import RBY1_BODY_JOINTS.
RBY1_BODY_JOINTS = RBY1_MOTIONLIB_BODY_JOINTS


# ---------------------------------------------------------------------
# Actual selected RBY1 Isaac articulation order from your dynamic_control log
#
# This is the filtered 24-policy-DOF order seen in the loaded USD.
# It excludes Axil/backwheel/fixed/end-effector/gripper extras.
# ---------------------------------------------------------------------

RBY1_EXTRA_JOINTS = [
    "backwheel",
    "backwheel2",
    "gripper_finger_r1",
    "gripper_finger_r2",
]



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
    """Detect RBY1 from a conservative set of core joints."""
    robot_joint_names = set(asset.joint_names)
    required = {
        "left_wheel",
        "right_wheel",
        "torso_0",
        "right_arm_0",
        "left_arm_0",
        "head_0",
    }
    return required.issubset(robot_joint_names)


def get_body_joint_indices(asset) -> torch.Tensor:
    """
    Get body joint indices tracked by the motion library.

    For RBY1, this intentionally returns indices in motion-lib / MuJoCo order,
    not raw Isaac articulation order. That means:

        asset.data.joint_pos[:, get_body_joint_indices(asset)]

    should line up with:

        motion_lib_entry["dof"]

    which is stored as:
        right_wheel, left_wheel, torso_0..5,
        right_arm_0..6, left_arm_0..6, head_0, head_1
    """
    if _is_rby1(asset):
        return _get_joint_indices_by_names(
            asset,
            RBY1_MOTIONLIB_BODY_JOINTS,
            _body_joint_indices_cache,
            expected_count=24,
            label="RBY1 motion-lib body joints",
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
        available = set(asset.joint_names)
        existing_extra = [name for name in RBY1_EXTRA_JOINTS if name in available]

        # Do not hard-fail if a cleaned 24-DOF USD removes these joints.
        if not existing_extra:
            return torch.empty(0, dtype=torch.long, device=asset.device)

        return _get_joint_indices_by_names(
            asset,
            existing_extra,
            _hand_joint_indices_cache,
            expected_count=None,
            label="RBY1 extra joints",
        )

    return _get_joint_indices_by_names(
        asset,
        G1_HAND_JOINTS,
        _hand_joint_indices_cache,
        expected_count=14,
        label="G1 hand joints",
    )

def get_rby1_isaaclab_body_joint_indices(asset) -> torch.Tensor:
    """
    Get RBY1 policy DOF indices in the actual Isaac articulation order.

    Use this only for code that explicitly wants Isaac-order policy DOFs.
    Do not use this for direct comparison against a MuJoCo-order motion-lib PKL.
    """
    return _get_joint_indices_by_names(
        asset,
        RBY1_ACTUAL_ISAACLAB_DOF_NAMES,
        _body_joint_indices_cache,
        expected_count=24,
        label="RBY1 IsaacLab-order body joints",
    )
