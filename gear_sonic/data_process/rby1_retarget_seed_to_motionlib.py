#!/usr/bin/env python3
# noqa: EXE001
"""
Retarget / package BONES-SEED-like seed motions into SONIC motion-lib PKLs
for the 24-DOF RBY1 embodiment.

This script intentionally separates two use cases:

1) Canonical path: `rby1_csv`
   Convert trajectories that are already retargeted to the RBY1 joint space
   into SONIC's motion-library PKL format.

2) Practical smoke-test path: `bones_g1_csv_projection`
   Project BONES-SEED G1 CSV trajectories into a 24-DOF RBY1 trajectory using
   an explicit, documented heuristic:
     - root planar translation/yaw -> RBY1 floating base
     - planar base velocity -> wheel joint angles
     - G1 waist -> distributed RBY1 torso joints
     - G1 arms -> same-semantic RBY1 arm joint slots
     - head joints -> zero by default
   This is useful for testing the RBY1 SONIC plumbing, not as a high-fidelity
   final retargeter. Replace this projection with a proper SOMA-BVH -> RBY1 IK
   retargeter for production-quality reference motions.

Output PKL entry format:
    {
        "<motion_name>": {
            "root_trans_offset": (T, 3) float32,
            "pose_aa":           (T, 25, 3) float32,  # root + 24 bodies
            "dof":               (T, 24) float32,    # MuJoCo joint order
            "root_rot":          (T, 4) float32,     # configurable wxyz/xyzw
            "smpl_joints":       (T, 24, 3) float32, # zeros unless supplied later
            "fps":               int,
        }
    }

The RBY1 DOF order matches the clean 24-DOF MJCF used in this conversation:
    right_wheel, left_wheel,
    torso_0 ... torso_5,
    right_arm_0 ... right_arm_6,
    left_arm_0 ... left_arm_6,
    head_0, head_1

Example: project BONES-SEED G1 CSVs into RBY1 PKLs
    python gear_sonic/data_process/rby1_retarget_seed_to_motionlib.py \
        --input /path/to/bones_seed/g1/csv \
        --output data/rby1_seed_projection/robot \
        --input-format bones_g1_csv_projection \
        --fps-source 120 --fps 30 \
        --individual --recursive \
        --wheel-radius 0.10 --wheel-track 0.53

Example: convert already-retargeted RBY1 CSVs
    python gear_sonic/data_process/rby1_retarget_seed_to_motionlib.py \
        --input /path/to/rby1_csvs \
        --output data/rby1_motions/robot \
        --input-format rby1_csv \
        --fps-source 120 --fps 30 \
        --individual --recursive
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# RBY1 24-DOF embodiment constants
# ---------------------------------------------------------------------------

RBY1_MJ_DOF_NAMES: list[str] = [
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

RBY1_NUM_DOF = len(RBY1_MJ_DOF_NAMES)
RBY1_NUM_BODIES = RBY1_NUM_DOF + 1  # floating base/root + one body per DOF

# Joint axes in MuJoCo active-joint order, parsed from the clean RBY1 MJCF.
RBY1_DOF_AXIS = np.asarray(
    [
        [0.0, -1.0, 0.0],              # right_wheel
        [0.0, -1.0, 0.0],              # left_wheel
        [1.0, 0.0, 0.0],               # torso_0
        [0.0, 1.0, 0.0],               # torso_1
        [0.0, 1.0, 0.0],               # torso_2
        [0.0, 1.0, 0.0],               # torso_3
        [1.0, 0.0, 0.0],               # torso_4
        [0.0, 0.0, 1.0],               # torso_5
        [0.0, 0.939693, -0.342020],    # right_arm_0
        [1.0, 0.0, 0.0],               # right_arm_1
        [0.0, 0.0, 1.0],               # right_arm_2
        [0.0, 1.0, 0.0],               # right_arm_3
        [0.0, 0.0, 1.0],               # right_arm_4
        [0.0, 1.0, 0.0],               # right_arm_5
        [0.0, 0.0, 1.0],               # right_arm_6
        [0.0, 0.939693, 0.342020],     # left_arm_0
        [1.0, 0.0, 0.0],               # left_arm_1
        [0.0, 0.0, 1.0],               # left_arm_2
        [0.0, 1.0, 0.0],               # left_arm_3
        [0.0, 0.0, 1.0],               # left_arm_4
        [0.0, 1.0, 0.0],               # left_arm_5
        [0.0, 0.0, 1.0],               # left_arm_6
        [0.0, 0.0, 1.0],               # head_0
        [0.0, 1.0, 0.0],               # head_1
    ],
    dtype=np.float32,
)

# MuJoCo joint ranges from the clean 24-DOF MJCF. Wheels are unbounded.
RBY1_DOF_LIMITS: dict[str, tuple[float, float] | None] = {
    "right_wheel": None,
    "left_wheel": None,
    "torso_0": (-0.349066, 0.349066),
    "torso_1": (-1.0472, 1.52173),
    "torso_2": (-2.47837, 1.5708),
    "torso_3": (-0.785398, 1.5708),
    "torso_4": (-0.523599, 0.523599),
    "torso_5": (-1.5708, 1.5708),
    "right_arm_0": (-2.35619, 2.35619),
    "right_arm_1": (-3.14159, 0.05),
    "right_arm_2": (-2.0944, 2.0944),
    "right_arm_3": (-2.61799, 0.01),
    "right_arm_4": (-6.28319, 6.28319),
    "right_arm_5": (-1.74533, 2.00713),
    "right_arm_6": (-2.96706, 2.96706),
    "left_arm_0": (-2.35619, 2.35619),
    "left_arm_1": (-0.05, 3.14159),
    "left_arm_2": (-2.0944, 2.0944),
    "left_arm_3": (-2.61799, 0.01),
    "left_arm_4": (-6.28319, 6.28319),
    "left_arm_5": (-1.74533, 2.00713),
    "left_arm_6": (-2.96706, 2.96706),
    "head_0": (-1.57, 1.57),
    "head_1": (-1.57, 1.57),
}

# Optional support for RBY1 trajectories serialized in IsaacLab DOF order.
# Mapping semantics: output[:, i] = input[:, mapping[i]]
# These arrays must match gear_sonic.envs.manager_env.robots.rby1.
RBY1_ISAACLAB_TO_MUJOCO_DOF = np.asarray(
    [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 11, 14, 16, 18, 20, 22,
        9, 12, 15, 17, 19, 21, 23,
        10, 13,
    ],
    dtype=np.int64,
)
RBY1_MUJOCO_TO_ISAACLAB_DOF = np.asarray(
    [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 15, 22, 9, 16, 23, 10,
        17, 11, 18, 12, 19, 13, 20,
        14, 21,
    ],
    dtype=np.int64,
)

# BONES-SEED / G1 CSV columns used by NVIDIA's existing conversion script.
G1_BONES_DOF_COLUMNS = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]
G1_DOF_IDX = {name: i for i, name in enumerate(G1_BONES_DOF_COLUMNS)}


@dataclass(frozen=True)
class MotionArrays:
    """Canonical in-memory representation prior to motion-lib packaging."""

    name: str
    root_pos_m: np.ndarray          # (T, 3)
    root_quat_wxyz: np.ndarray      # (T, 4)
    dof_mj_rad: np.ndarray          # (T, 24)
    fps: int


@dataclass(frozen=True)
class ProjectionConfig:
    """Parameters for the BONES-SEED G1 -> RBY1 heuristic projection."""

    wheel_radius: float = 0.10
    wheel_track: float = 0.53
    wheel_sign: float = 1.0
    root_motion_scale: float = 1.0
    keep_root_z: bool = False
    nominal_root_z: float = 0.0
    planar_yaw_only: bool = True
    velocity_smoothing_window: int = 1
    torso_pitch_weights: tuple[float, float, float] = (0.35, 0.40, 0.25)
    torso_roll_weights: tuple[float, float] = (0.50, 0.50)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quaternion array shape (T, 4), got {q.shape}")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Encountered near-zero quaternion norm.")
    return (q / norm).astype(np.float32)


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.asarray(q_wxyz)[..., [1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray(q_xyzw)[..., [3, 0, 1, 2]]


def _yaw_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Extract Z-up yaw angle from wxyz quaternions."""
    q_xyzw = _quat_wxyz_to_xyzw(q_wxyz)
    euler_xyz = Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False)
    return np.unwrap(euler_xyz[:, 2]).astype(np.float64)


def _quat_wxyz_from_yaw(yaw: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_euler("z", yaw.reshape(-1, 1), degrees=False).as_quat()
    return _quat_xyzw_to_wxyz(q_xyzw).astype(np.float32)


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    if window % 2 == 0:
        raise ValueError("--velocity-smoothing-window must be odd when > 1.")
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def _clip_to_rby1_limits(dof: np.ndarray) -> np.ndarray:
    clipped = np.asarray(dof, dtype=np.float32).copy()
    for idx, name in enumerate(RBY1_MJ_DOF_NAMES):
        lim = RBY1_DOF_LIMITS[name]
        if lim is not None:
            clipped[:, idx] = np.clip(clipped[:, idx], lim[0], lim[1])
    return clipped


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _motion_name_from_path(path: Path, *, root: Path | None = None) -> str:
    stem = path.stem
    if root is None:
        return stem
    try:
        rel = path.relative_to(root)
    except ValueError:
        return stem
    rel_no_suffix = rel.with_suffix("")
    return "__".join(rel_no_suffix.parts)


def _downsample_motion(motion: MotionArrays, fps_target: int) -> MotionArrays:
    if motion.fps == fps_target:
        return motion
    if fps_target <= 0 or motion.fps <= 0:
        raise ValueError("FPS values must be positive.")
    ratio = motion.fps / fps_target
    stride = int(round(ratio))
    if not math.isclose(ratio, stride, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Source FPS {motion.fps} is not an integer multiple of target FPS {fps_target}. "
            "This script currently uses stride-based downsampling."
        )
    if stride <= 1:
        return motion
    return MotionArrays(
        name=motion.name,
        root_pos_m=motion.root_pos_m[::stride],
        root_quat_wxyz=motion.root_quat_wxyz[::stride],
        dof_mj_rad=motion.dof_mj_rad[::stride],
        fps=fps_target,
    )


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def load_rby1_csv(
    csv_path: Path,
    *,
    fps_source: int,
    input_joint_order: Literal["mujoco", "isaaclab"],
) -> MotionArrays:
    """
    Load an already-retargeted RBY1 CSV.

    Accepted root pose schemas:
      A) position + quaternion:
         root_x, root_y, root_z, root_quat_w, root_quat_x, root_quat_y, root_quat_z
      B) position + Euler XYZ degrees:
         root_x, root_y, root_z, root_roll_deg, root_pitch_deg, root_yaw_deg
      C) Bones-style root column names:
         root_translateX, root_translateY, root_translateZ,
         root_rotateX, root_rotateY, root_rotateZ
         Here translations are assumed to already be meters unless
         --input-format bones_g1_csv_projection is used.

    Accepted joint column names:
      - Direct RBY1 names: right_wheel, left_wheel, torso_0, ...
      - Or each with `_dof` suffix: right_wheel_dof, ...
    Joint values are interpreted as radians.
    """
    df = pd.read_csv(csv_path)
    T = len(df)
    if T == 0:
        raise ValueError(f"{csv_path}: CSV has no rows.")

    if {"root_x", "root_y", "root_z"}.issubset(df.columns):
        root_pos = df[["root_x", "root_y", "root_z"]].to_numpy(dtype=np.float32)
    elif {"root_translateX", "root_translateY", "root_translateZ"}.issubset(df.columns):
        root_pos = df[["root_translateX", "root_translateY", "root_translateZ"]].to_numpy(dtype=np.float32)
    else:
        raise ValueError(
            f"{csv_path}: expected root position columns "
            "root_x/root_y/root_z or root_translateX/root_translateY/root_translateZ."
        )

    quat_cols = ["root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"]
    euler_cols = ["root_roll_deg", "root_pitch_deg", "root_yaw_deg"]
    bones_euler_cols = ["root_rotateX", "root_rotateY", "root_rotateZ"]

    if set(quat_cols).issubset(df.columns):
        root_quat_wxyz = _normalize_quat_wxyz(df[quat_cols].to_numpy(dtype=np.float32))
    elif set(euler_cols).issubset(df.columns):
        q_xyzw = Rotation.from_euler(
            "xyz",
            df[euler_cols].to_numpy(dtype=np.float64),
            degrees=True,
        ).as_quat()
        root_quat_wxyz = _quat_xyzw_to_wxyz(q_xyzw).astype(np.float32)
    elif set(bones_euler_cols).issubset(df.columns):
        q_xyzw = Rotation.from_euler(
            "xyz",
            df[bones_euler_cols].to_numpy(dtype=np.float64),
            degrees=True,
        ).as_quat()
        root_quat_wxyz = _quat_xyzw_to_wxyz(q_xyzw).astype(np.float32)
    else:
        raise ValueError(
            f"{csv_path}: expected root quaternion columns {quat_cols} or Euler columns "
            f"{euler_cols} or {bones_euler_cols}."
        )

    direct_cols = RBY1_MJ_DOF_NAMES
    suffixed_cols = [f"{name}_dof" for name in RBY1_MJ_DOF_NAMES]
    if set(direct_cols).issubset(df.columns):
        dof = df[direct_cols].to_numpy(dtype=np.float32)
    elif set(suffixed_cols).issubset(df.columns):
        dof = df[suffixed_cols].to_numpy(dtype=np.float32)
    else:
        missing_direct = [c for c in direct_cols if c not in df.columns]
        missing_suffixed = [c for c in suffixed_cols if c not in df.columns]
        raise ValueError(
            f"{csv_path}: could not find a complete RBY1 joint set. "
            f"Missing direct-name columns: {missing_direct}. "
            f"Missing `_dof` columns: {missing_suffixed}."
        )

    if dof.shape != (T, RBY1_NUM_DOF):
        raise ValueError(f"{csv_path}: expected DOF shape {(T, RBY1_NUM_DOF)}, got {dof.shape}")

    if input_joint_order == "isaaclab":
        # Convert IsaacLab order -> MuJoCo order using the mapping semantics
        # output[:, i] = input[:, mapping[i]].
        dof = dof[:, RBY1_ISAACLAB_TO_MUJOCO_DOF]
    elif input_joint_order != "mujoco":
        raise ValueError(f"Unknown input_joint_order: {input_joint_order}")

    dof = _clip_to_rby1_limits(dof)
    return MotionArrays(
        name=csv_path.stem,
        root_pos_m=_as_float32(root_pos),
        root_quat_wxyz=_normalize_quat_wxyz(root_quat_wxyz),
        dof_mj_rad=_as_float32(dof),
        fps=fps_source,
    )


def load_bones_g1_csv_projection(
    csv_path: Path,
    *,
    fps_source: int,
    cfg: ProjectionConfig,
) -> MotionArrays:
    """
    Load a BONES-SEED G1 CSV and heuristically project it to RBY1.

    Expected format matches NVIDIA's existing BONES-SEED converter:
      Frame,
      root_translateX/Y/Z       [centimeters],
      root_rotateX/Y/Z          [degrees, Euler XYZ],
      29 G1 joint columns ending in `_dof` [degrees].
    """
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{csv_path}: CSV has no rows.")

    _require_columns(
        df,
        [
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
            *G1_BONES_DOF_COLUMNS,
        ],
        source=str(csv_path),
    )

    root_pos = (
        df[["root_translateX", "root_translateY", "root_translateZ"]]
        .to_numpy(dtype=np.float64)
        / 100.0
    )
    root_pos[:, :2] *= cfg.root_motion_scale

    q_xyzw = Rotation.from_euler(
        "xyz",
        df[["root_rotateX", "root_rotateY", "root_rotateZ"]].to_numpy(dtype=np.float64),
        degrees=True,
    ).as_quat()
    root_quat_wxyz = _quat_xyzw_to_wxyz(q_xyzw).astype(np.float32)

    # RBY1 floating base: keep planar path and yaw. Optionally discard vertical motion.
    yaw = _yaw_from_quat_wxyz(root_quat_wxyz)
    if cfg.planar_yaw_only:
        root_quat_wxyz = _quat_wxyz_from_yaw(yaw)

    if cfg.keep_root_z:
        root_pos_out = root_pos.copy()
    else:
        root_pos_out = root_pos.copy()
        root_pos_out[:, 2] = cfg.nominal_root_z

    # G1 joint motion in radians.
    g1_dof = np.deg2rad(df[G1_BONES_DOF_COLUMNS].to_numpy(dtype=np.float64))

    dof = np.zeros((len(df), RBY1_NUM_DOF), dtype=np.float64)

    # 1) Wheels from planar base trajectory.
    dt = 1.0 / float(fps_source)
    x = root_pos_out[:, 0]
    y = root_pos_out[:, 1]

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dyaw = np.diff(yaw, prepend=yaw[0])

    yaw_mid = yaw
    v_forward = (dx * np.cos(yaw_mid) + dy * np.sin(yaw_mid)) / dt
    omega = dyaw / dt

    v_forward = _moving_average_1d(v_forward, cfg.velocity_smoothing_window)
    omega = _moving_average_1d(omega, cfg.velocity_smoothing_window)

    if cfg.wheel_radius <= 0.0:
        raise ValueError("--wheel-radius must be positive.")
    if cfg.wheel_track <= 0.0:
        raise ValueError("--wheel-track must be positive.")

    qdot_right = cfg.wheel_sign * (v_forward + 0.5 * cfg.wheel_track * omega) / cfg.wheel_radius
    qdot_left = cfg.wheel_sign * (v_forward - 0.5 * cfg.wheel_track * omega) / cfg.wheel_radius
    dof[:, 0] = np.cumsum(qdot_right * dt)
    dof[:, 1] = np.cumsum(qdot_left * dt)

    # 2) G1 waist -> RBY1 torso.
    waist_yaw = g1_dof[:, G1_DOF_IDX["waist_yaw_joint_dof"]]
    waist_roll = g1_dof[:, G1_DOF_IDX["waist_roll_joint_dof"]]
    waist_pitch = g1_dof[:, G1_DOF_IDX["waist_pitch_joint_dof"]]

    torso_pitch_w = np.asarray(cfg.torso_pitch_weights, dtype=np.float64)
    if not np.isclose(torso_pitch_w.sum(), 1.0):
        torso_pitch_w = torso_pitch_w / torso_pitch_w.sum()
    torso_roll_w = np.asarray(cfg.torso_roll_weights, dtype=np.float64)
    if not np.isclose(torso_roll_w.sum(), 1.0):
        torso_roll_w = torso_roll_w / torso_roll_w.sum()

    dof[:, 2] = torso_roll_w[0] * waist_roll   # torso_0
    dof[:, 3] = torso_pitch_w[0] * waist_pitch # torso_1
    dof[:, 4] = torso_pitch_w[1] * waist_pitch # torso_2
    dof[:, 5] = torso_pitch_w[2] * waist_pitch # torso_3
    dof[:, 6] = torso_roll_w[1] * waist_roll   # torso_4
    dof[:, 7] = waist_yaw                        # torso_5

    # 3) Same-semantic G1 arm slots -> RBY1 arm slots.
    #    This is not an IK retarget; it is an explicit joint-space projection.
    right_src = [
        "right_shoulder_pitch_joint_dof",
        "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof",
        "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof",
        "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof",
    ]
    left_src = [
        "left_shoulder_pitch_joint_dof",
        "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof",
        "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof",
        "left_wrist_pitch_joint_dof",
        "left_wrist_yaw_joint_dof",
    ]
    for j, src_name in enumerate(right_src):
        dof[:, 8 + j] = g1_dof[:, G1_DOF_IDX[src_name]]
    for j, src_name in enumerate(left_src):
        dof[:, 15 + j] = g1_dof[:, G1_DOF_IDX[src_name]]

    # 4) Head stays neutral by default.
    dof[:, 22] = 0.0
    dof[:, 23] = 0.0

    dof = _clip_to_rby1_limits(dof)
    return MotionArrays(
        name=csv_path.stem,
        root_pos_m=_as_float32(root_pos_out),
        root_quat_wxyz=_normalize_quat_wxyz(root_quat_wxyz),
        dof_mj_rad=_as_float32(dof),
        fps=fps_source,
    )


# ---------------------------------------------------------------------------
# Motion-lib packaging
# ---------------------------------------------------------------------------

def build_motionlib_entry(
    motion: MotionArrays,
    *,
    root_rot_order: Literal["wxyz", "xyzw"],
) -> dict[str, np.ndarray | int]:
    """
    Convert canonical RBY1 arrays into SONIC motion-lib entry fields.

    `pose_aa` layout:
      body 0   = root/base rotvec
      body 1+  = each actuated RBY1 joint converted as axis * angle
    """
    T = motion.dof_mj_rad.shape[0]
    if motion.root_pos_m.shape != (T, 3):
        raise ValueError(f"{motion.name}: root_pos shape mismatch: {motion.root_pos_m.shape}")
    if motion.root_quat_wxyz.shape != (T, 4):
        raise ValueError(f"{motion.name}: root_quat shape mismatch: {motion.root_quat_wxyz.shape}")
    if motion.dof_mj_rad.shape != (T, RBY1_NUM_DOF):
        raise ValueError(f"{motion.name}: dof shape mismatch: {motion.dof_mj_rad.shape}")

    pose_aa = np.zeros((T, RBY1_NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 1:, :] = RBY1_DOF_AXIS[None, :, :] * motion.dof_mj_rad[:, :, None]

    root_quat_xyzw = _quat_wxyz_to_xyzw(motion.root_quat_wxyz)
    pose_aa[:, 0, :] = Rotation.from_quat(root_quat_xyzw).as_rotvec().astype(np.float32)

    if root_rot_order == "wxyz":
        root_rot = motion.root_quat_wxyz.astype(np.float32)
    elif root_rot_order == "xyzw":
        root_rot = root_quat_xyzw.astype(np.float32)
    else:
        raise ValueError(f"Unknown root_rot_order: {root_rot_order}")

    return {
        "root_trans_offset": motion.root_pos_m.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": motion.dof_mj_rad.astype(np.float32),
        "root_rot": root_rot,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(motion.fps),
    }


def save_motion_entry(
    motion_name: str,
    entry: dict[str, np.ndarray | int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_name: entry}, out_path, compress=True)


# ---------------------------------------------------------------------------
# Discovery / batch orchestration
# ---------------------------------------------------------------------------

def discover_csvs(input_path: Path, *, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {input_path}")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(input_path.glob(pattern))


def output_path_for_csv(
    csv_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    individual: bool,
) -> Path:
    if individual:
        if input_root.is_dir():
            rel = csv_path.relative_to(input_root).with_suffix(".pkl")
        else:
            rel = Path(csv_path.stem + ".pkl")
        return output_root / rel
    return output_root


def process_one_csv(
    csv_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    input_format: Literal["rby1_csv", "bones_g1_csv_projection"],
    fps_source: int,
    fps_target: int,
    input_joint_order: Literal["mujoco", "isaaclab"],
    projection_cfg: ProjectionConfig,
    root_rot_order: Literal["wxyz", "xyzw"],
    individual: bool,
) -> tuple[str, Path, int]:
    if input_format == "rby1_csv":
        motion = load_rby1_csv(
            csv_path,
            fps_source=fps_source,
            input_joint_order=input_joint_order,
        )
    elif input_format == "bones_g1_csv_projection":
        motion = load_bones_g1_csv_projection(
            csv_path,
            fps_source=fps_source,
            cfg=projection_cfg,
        )
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    motion = _downsample_motion(motion, fps_target)
    entry = build_motionlib_entry(motion, root_rot_order=root_rot_order)

    out_path = output_path_for_csv(
        csv_path,
        input_root=input_root,
        output_root=output_root,
        individual=individual,
    )
    motion_name = _motion_name_from_path(csv_path, root=input_root if input_root.is_dir() else None)

    if individual:
        save_motion_entry(motion_name, entry, out_path)
    return motion_name, out_path, motion.dof_mj_rad.shape[0]


def save_combined_entries(entries: dict[str, dict[str, np.ndarray | int]], out_path: Path) -> None:
    if not entries:
        raise ValueError("No entries to save.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(entries, out_path, compress=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retarget/package seed motions into SONIC motion-lib PKLs for 24-DOF RBY1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="CSV file or directory of CSV files.")
    p.add_argument("--output", required=True, type=Path, help="Output PKL file or output directory.")
    p.add_argument(
        "--input-format",
        required=True,
        choices=["rby1_csv", "bones_g1_csv_projection"],
        help="Input parser / retargeting mode.",
    )
    p.add_argument("--fps-source", type=int, default=120, help="Source CSV frame rate.")
    p.add_argument("--fps", type=int, default=30, help="Target output PKL frame rate.")
    p.add_argument("--recursive", action="store_true", help="Recursively search input directories for CSV files.")
    p.add_argument(
        "--individual",
        action="store_true",
        help="Write one PKL per motion and preserve subdirectory structure. "
             "If omitted, writes a single combined PKL at --output.",
    )
    p.add_argument(
        "--input-joint-order",
        choices=["mujoco", "isaaclab"],
        default="mujoco",
        help="Only used for rby1_csv input.",
    )
    p.add_argument(
        "--root-rot-order",
        choices=["wxyz", "xyzw"],
        default="wxyz",
        help="Quaternion order saved in PKL root_rot. Use wxyz for SONIC convention; "
             "xyzw is available for compatibility testing with scripts that store SciPy order.",
    )

    # Projection-only parameters.
    p.add_argument("--wheel-radius", type=float, default=0.10, help="Wheel radius in meters.")
    p.add_argument("--wheel-track", type=float, default=0.53, help="Wheel center-to-center track width in meters.")
    p.add_argument(
        "--wheel-sign",
        type=float,
        default=1.0,
        help="Wheel sign convention multiplier. Flip to -1.0 if positive wheel angle moves backward in your sim.",
    )
    p.add_argument(
        "--root-motion-scale",
        type=float,
        default=1.0,
        help="Scale planar root X/Y translation before converting to the RBY1 base path.",
    )
    p.add_argument(
        "--keep-root-z",
        action="store_true",
        help="Retain seed root Z translation. Default sets root Z to --nominal-root-z.",
    )
    p.add_argument(
        "--nominal-root-z",
        type=float,
        default=0.0,
        help="Root Z used when --keep-root-z is not set.",
    )
    p.add_argument(
        "--keep-full-root-rotation",
        action="store_true",
        help="Keep full seed root rotation instead of yaw-only base orientation in projection mode.",
    )
    p.add_argument(
        "--velocity-smoothing-window",
        type=int,
        default=1,
        help="Odd moving-average window applied to inferred base forward/yaw velocities.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.fps_source <= 0 or args.fps <= 0:
        raise ValueError("--fps-source and --fps must be positive.")

    csvs = discover_csvs(args.input, recursive=args.recursive)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under: {args.input}")

    projection_cfg = ProjectionConfig(
        wheel_radius=args.wheel_radius,
        wheel_track=args.wheel_track,
        wheel_sign=args.wheel_sign,
        root_motion_scale=args.root_motion_scale,
        keep_root_z=args.keep_root_z,
        nominal_root_z=args.nominal_root_z,
        planar_yaw_only=not args.keep_full_root_rotation,
        velocity_smoothing_window=args.velocity_smoothing_window,
    )

    combined: dict[str, dict[str, np.ndarray | int]] = {}
    converted = 0
    failed = 0

    for csv_path in csvs:
        try:
            name, out_path, n_frames = process_one_csv(
                csv_path,
                input_root=args.input,
                output_root=args.output,
                input_format=args.input_format,
                fps_source=args.fps_source,
                fps_target=args.fps,
                input_joint_order=args.input_joint_order,
                projection_cfg=projection_cfg,
                root_rot_order=args.root_rot_order,
                individual=args.individual,
            )
            if not args.individual:
                # Rebuild entry once for combined-output mode.
                if args.input_format == "rby1_csv":
                    motion = load_rby1_csv(
                        csv_path,
                        fps_source=args.fps_source,
                        input_joint_order=args.input_joint_order,
                    )
                else:
                    motion = load_bones_g1_csv_projection(
                        csv_path,
                        fps_source=args.fps_source,
                        cfg=projection_cfg,
                    )
                motion = _downsample_motion(motion, args.fps)
                combined[name] = build_motionlib_entry(motion, root_rot_order=args.root_rot_order)
            print(f"[OK] {csv_path} -> {out_path if args.individual else args.output} ({n_frames} frames)")
            converted += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {csv_path}: {exc}", file=sys.stderr)
            failed += 1

    if not args.individual:
        save_combined_entries(combined, args.output)

    print(f"Converted: {converted}; Failed: {failed}; Output: {args.output}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
