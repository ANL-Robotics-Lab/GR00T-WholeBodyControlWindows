from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gear_sonic.data_process.rby1_retarget_seed_to_motionlib import load_rby1_csv
from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase, interpolate_linear_frames
from gear_sonic.utils.rby1_order import (
    RBY1_FULL_BODY_TO_COLLAPSED_25,
    RBY1_FULL_ISAACLAB_BODY_NAMES,
    RBY1_ISAACLAB_BODY_NAMES_25,
    RBY1_ISAACLAB_DOF_NAMES,
    RBY1_ISAACLAB_TO_MUJOCO_BODY,
    RBY1_ISAACLAB_TO_MUJOCO_DOF,
    RBY1_MOTION_TO_ISAACLAB_DOF_SIGN,
    RBY1_MUJOCO_BODY_NAMES,
    RBY1_MUJOCO_DOF_NAMES,
    RBY1_MUJOCO_TO_ISAACLAB_BODY,
    RBY1_MUJOCO_TO_ISAACLAB_DOF,
)


def test_rby1_name_derived_mappings_round_trip():
    isaac_dof = np.arange(24)
    mujoco_dof = isaac_dof[RBY1_ISAACLAB_TO_MUJOCO_DOF]
    np.testing.assert_array_equal(
        mujoco_dof[RBY1_MUJOCO_TO_ISAACLAB_DOF], isaac_dof
    )

    isaac_body = np.arange(25)
    mujoco_body = isaac_body[RBY1_ISAACLAB_TO_MUJOCO_BODY]
    np.testing.assert_array_equal(
        mujoco_body[RBY1_MUJOCO_TO_ISAACLAB_BODY], isaac_body
    )

    assert [RBY1_ISAACLAB_DOF_NAMES[i] for i in RBY1_ISAACLAB_TO_MUJOCO_DOF] == (
        RBY1_MUJOCO_DOF_NAMES
    )
    assert [RBY1_ISAACLAB_BODY_NAMES_25[i] for i in RBY1_ISAACLAB_TO_MUJOCO_BODY] == (
        RBY1_MUJOCO_BODY_NAMES
    )
    assert [RBY1_FULL_ISAACLAB_BODY_NAMES[i] for i in RBY1_FULL_BODY_TO_COLLAPSED_25] == (
        RBY1_ISAACLAB_BODY_NAMES_25
    )

    signs_by_name = dict(
        zip(RBY1_MUJOCO_DOF_NAMES, RBY1_MOTION_TO_ISAACLAB_DOF_SIGN, strict=True)
    )
    assert signs_by_name["right_wheel"] == -1.0
    assert signs_by_name["left_wheel"] == -1.0
    assert all(
        sign == 1.0
        for name, sign in signs_by_name.items()
        if name not in {"right_wheel", "left_wheel"}
    )


def test_raw_dof_interpolation_uses_fk_time_grid():
    source = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    result = interpolate_linear_frames(
        source,
        source_fps=30,
        target_fps=50,
        num_frames=5,
    )
    torch.testing.assert_close(
        result.squeeze(-1),
        torch.tensor([0.0, 0.6, 1.2, 1.8, 2.4]),
    )


def test_raw_dof_fallback_returns_mujoco_order():
    motion_lib = MotionLibBase.__new__(MotionLibBase)
    motion_lib.length_starts = torch.tensor([0], dtype=torch.long)
    motion_lib.dof_pos = torch.arange(24, dtype=torch.float32).unsqueeze(0)
    motion_lib.dof_vel = motion_lib.dof_pos + 100.0
    motion_lib.m_cfg = {"isaaclab_to_mujoco_dof": RBY1_ISAACLAB_TO_MUJOCO_DOF}
    motion_ids = torch.tensor([0], dtype=torch.long)
    motion_steps = torch.tensor([0], dtype=torch.long)

    expected_pos = motion_lib.dof_pos[:, RBY1_ISAACLAB_TO_MUJOCO_DOF]
    expected_vel = motion_lib.dof_vel[:, RBY1_ISAACLAB_TO_MUJOCO_DOF]
    torch.testing.assert_close(motion_lib.get_raw_dof_pos(motion_ids, motion_steps), expected_pos)
    torch.testing.assert_close(motion_lib.get_raw_dof_vel(motion_ids, motion_steps), expected_vel)


def test_direct_rby1_columns_auto_detect_meters_and_radians(tmp_path: Path):
    data = {
        "root_x": [1.0, 1.25],
        "root_y": [2.0, 2.50],
        "root_z": [0.1, 0.2],
        "root_quat_w": [1.0, 1.0],
        "root_quat_x": [0.0, 0.0],
        "root_quat_y": [0.0, 0.0],
        "root_quat_z": [0.0, 0.0],
    }
    expected_dof = np.empty((2, len(RBY1_MUJOCO_DOF_NAMES)), dtype=np.float32)
    for index, name in enumerate(RBY1_MUJOCO_DOF_NAMES):
        values = np.asarray([index / 100.0, (index + 1) / 100.0], dtype=np.float32)
        data[name] = values
        expected_dof[:, index] = values

    csv_path = tmp_path / "direct.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    motion = load_rby1_csv(
        csv_path,
        fps_source=30,
        input_joint_order="mujoco",
    )

    np.testing.assert_allclose(motion.root_pos_m, [[1.0, 2.0, 0.1], [1.25, 2.5, 0.2]])
    np.testing.assert_allclose(motion.dof_mj_rad, expected_dof, atol=1e-7)


def test_bones_style_rby1_columns_rebase_path_and_map_isaac_order(tmp_path: Path):
    data = {
        "root_translateX": [100.0, 100.0],
        "root_translateY": [200.0, 300.0],
        "root_translateZ": [50.0, 60.0],
        "root_rotateX": [0.0, 0.0],
        "root_rotateY": [0.0, 0.0],
        "root_rotateZ": [90.0, 90.0],
    }
    degrees_by_name = {}
    for index, name in enumerate(RBY1_MUJOCO_DOF_NAMES):
        values = np.asarray([index, index + 0.5], dtype=np.float32)
        data[f"{name}_dof"] = values
        degrees_by_name[name] = values

    csv_path = tmp_path / "bones_style.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    motion = load_rby1_csv(
        csv_path,
        fps_source=30,
        input_joint_order="isaaclab",
        root_mode="planar_yaw_relative",
        nominal_root_z=0.2,
    )

    expected_dof = np.stack(
        [np.deg2rad(degrees_by_name[name]) for name in RBY1_MUJOCO_DOF_NAMES], axis=1
    )
    np.testing.assert_allclose(motion.dof_mj_rad, expected_dof, atol=1e-7)
    np.testing.assert_allclose(motion.root_pos_m, [[0.0, 0.0, 0.2], [1.0, 0.0, 0.2]], atol=1e-6)
    np.testing.assert_allclose(motion.root_quat_wxyz, [[1.0, 0.0, 0.0, 0.0]] * 2, atol=1e-6)


def test_gem_planar_rby1_schema_auto_detects_meters_and_radians(tmp_path: Path):
    data = {
        "Frame": [0, 1],
        "root_translateX": [0.25, 0.50],
        "root_translateY": [-0.10, -0.20],
        "root_yaw": [0.0, np.pi / 2.0],
    }
    expected_dof = np.empty((2, len(RBY1_MUJOCO_DOF_NAMES)), dtype=np.float32)
    for index, name in enumerate(RBY1_MUJOCO_DOF_NAMES):
        values = np.asarray([index / 10.0, (index + 1) / 10.0], dtype=np.float32)
        data[f"{name}_dof"] = values
        expected_dof[:, index] = values

    # These upstream-only joints are intentionally ignored by the 24-DOF
    # SONIC motion representation.
    data["gripper_finger_r1_dof"] = [-0.025, -0.025]
    data["gripper_finger_r2_dof"] = [0.025, 0.025]
    data["gripper_finger_l1_dof"] = [-0.025, -0.025]
    data["gripper_finger_l2_dof"] = [0.025, 0.025]

    csv_path = tmp_path / "gem_planar.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    motion = load_rby1_csv(
        csv_path,
        fps_source=30,
        input_joint_order="mujoco",
        nominal_root_z=0.02,
    )

    np.testing.assert_allclose(
        motion.root_pos_m,
        [[0.25, -0.10, 0.02], [0.50, -0.20, 0.02]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        motion.root_quat_wxyz,
        [[1.0, 0.0, 0.0, 0.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]],
        atol=1e-6,
    )
    np.testing.assert_allclose(motion.dof_mj_rad, expected_dof, atol=1e-7)


def test_wheel_feasible_root_reconstructs_differential_drive_path(tmp_path: Path):
    data = {
        "root_x": [9.0, 8.0, 7.0, 6.0],
        "root_y": [4.0, 5.0, 6.0, 7.0],
        "root_z": [1.0] * 4,
        "root_quat_w": [1.0] * 4,
        "root_quat_x": [0.0] * 4,
        "root_quat_y": [0.0] * 4,
        "root_quat_z": [0.0] * 4,
    }
    for name in RBY1_MUJOCO_DOF_NAMES:
        data[name] = [0.0] * 4

    # GEM-X wheel angles are exported with sign -1. The corresponding GEM-X
    # deltas are: straight 0.1 m, rotate +0.4 rad, straight 0.1 m.
    data["right_wheel"] = [0.0, -1.0, -2.0, -3.0]
    data["left_wheel"] = [0.0, -1.0, 0.0, -1.0]
    csv_path = tmp_path / "wheel_path.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    motion = load_rby1_csv(
        csv_path,
        fps_source=10,
        input_joint_order="mujoco",
        root_mode="wheel_feasible_planar_yaw_relative",
        nominal_root_z=0.2,
        wheel_radius=0.1,
        wheel_track=0.5,
        wheel_sign=-1.0,
    )

    expected_xy = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.1, 0.0],
            [0.1 + 0.1 * np.cos(0.4), 0.1 * np.sin(0.4)],
        ]
    )
    np.testing.assert_allclose(motion.root_pos_m[:, :2], expected_xy, atol=1e-6)
    np.testing.assert_allclose(motion.root_pos_m[:, 2], 0.2, atol=1e-7)
    expected_quat = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(0.2), 0.0, 0.0, np.sin(0.2)],
            [np.cos(0.2), 0.0, 0.0, np.sin(0.2)],
        ]
    )
    np.testing.assert_allclose(motion.root_quat_wxyz, expected_quat, atol=1e-6)


def test_rby1_joint_smoothing_reduces_single_frame_spike(tmp_path: Path):
    data = {
        "root_x": [0.0] * 5,
        "root_y": [0.0] * 5,
        "root_z": [0.0] * 5,
        "root_quat_w": [1.0] * 5,
        "root_quat_x": [0.0] * 5,
        "root_quat_y": [0.0] * 5,
        "root_quat_z": [0.0] * 5,
    }
    for name in RBY1_MUJOCO_DOF_NAMES:
        data[name] = [0.0] * 5
    data["torso_0"] = [0.0, 0.0, 3.0, 0.0, 0.0]
    csv_path = tmp_path / "spike.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    motion = load_rby1_csv(
        csv_path,
        fps_source=10,
        input_joint_order="mujoco",
        joint_smoothing_window=3,
    )

    torso_index = RBY1_MUJOCO_DOF_NAMES.index("torso_0")
    np.testing.assert_allclose(
        motion.dof_mj_rad[:, torso_index],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        atol=1e-7,
    )
