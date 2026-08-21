#!/usr/bin/env python3
"""Validate or replay a GR00T RBY1 motion-library PKL through rby1-sdk.

The default target is ``validate`` and never imports rby1-sdk or connects to a
robot.  Execution must be selected explicitly with ``--target simulator`` or
``--target robot``.  Real-hardware execution additionally requires the exact
confirmation token documented by ``--help``.

The expected PKL format is the one produced by
``gear_sonic/data_process/rby1_retarget_seed_to_motionlib.py``::

    {motion_name: {"dof": float[T, 24], "fps": number, ...}}

The 24 columns use the RBY1 Model A / motion-library order defined below.
Wheels are velocity controlled; their velocities are derived from the first
two wheel-angle columns.  The torso, arms, and head are position controlled.

This is a guarded test runner, not a safety-certified controller.  Keep a
trained operator at the physical E-stop and begin with wheels disabled.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np


LOGGER = logging.getLogger("rby1_sdk_replay")

# This is both the motion-library order and y1_model::A::kRobotJointNames.
RBY1_MODEL_A_JOINT_NAMES = (
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
)

WHEEL_INDICES = np.asarray([0, 1], dtype=np.int64)
TORSO_INDICES = np.arange(2, 8, dtype=np.int64)
RIGHT_ARM_INDICES = np.arange(8, 15, dtype=np.int64)
LEFT_ARM_INDICES = np.arange(15, 22, dtype=np.int64)
HEAD_INDICES = np.arange(22, 24, dtype=np.int64)
UPPER_BODY_INDICES = np.arange(2, 24, dtype=np.int64)

REAL_ROBOT_CONFIRMATION = "RUN_RBY1_MODEL_A"
SIMULATOR_RESET_POSE_TOLERANCE = 0.25
DEFAULT_URDF = (
    Path(__file__).resolve().parent
    / "assets"
    / "model_v1.2.urdf"
)


class ReplayError(RuntimeError):
    """Raised when replay cannot safely continue."""


@dataclass(frozen=True)
class JointLimit:
    lower: float | None
    upper: float | None
    velocity: float
    acceleration: float


@dataclass(frozen=True)
class Motion:
    name: str
    positions: np.ndarray
    fps: float
    source: Path

    @property
    def duration(self) -> float:
        return (self.positions.shape[0] - 1) / self.fps


@dataclass(frozen=True)
class ReplayTrajectory:
    times: np.ndarray
    source_frames: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.times[-1])


@dataclass(frozen=True)
class ValidationIssue:
    joint: str
    quantity: str
    observed: float
    allowed: float
    frame: int
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.joint}: {self.quantity} {self.observed:.5f} exceeds "
            f"{self.allowed:.5f} at output frame {self.frame} ({self.detail})"
        )


@dataclass(frozen=True)
class StateSnapshot:
    received_at: float
    positions: np.ndarray
    velocities: np.ndarray
    ready: np.ndarray
    temperatures: np.ndarray
    emo_pressed: bool
    control_manager_state: Any


class StateMonitor:
    """Thread-safe local copy of SDK state callbacks."""

    def __init__(self, rby_module: Any) -> None:
        self._rby = rby_module
        self._lock = threading.Lock()
        self._snapshot: StateSnapshot | None = None
        self._callback_error: BaseException | None = None

    def callback(self, state: Any, control_manager_state: Any) -> None:
        try:
            emo_pressed = any(
                emo.state == self._rby.EMOState.State.Pressed for emo in state.emo_states
            )
            snapshot = StateSnapshot(
                received_at=time.monotonic(),
                positions=np.asarray(state.position, dtype=np.float64).copy(),
                velocities=np.asarray(state.velocity, dtype=np.float64).copy(),
                ready=np.asarray(state.is_ready, dtype=bool).copy(),
                temperatures=np.asarray(state.temperature, dtype=np.float64).copy(),
                emo_pressed=emo_pressed,
                control_manager_state=control_manager_state.state,
            )
            with self._lock:
                self._snapshot = snapshot
        except BaseException as exc:  # callback failures must stop motion
            with self._lock:
                self._callback_error = exc

    def latest(self) -> StateSnapshot | None:
        with self._lock:
            if self._callback_error is not None:
                raise ReplayError(f"State callback failed: {self._callback_error}")
            snapshot = self._snapshot
        return snapshot

    def wait_for_first(self, timeout: float) -> StateSnapshot:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = self.latest()
            if snapshot is not None:
                return snapshot
            time.sleep(0.01)
        raise ReplayError(f"No SDK state callback received within {timeout:.1f}s")


def _finite_float(value: Any, field: str) -> float:
    result = float(np.asarray(value).item())
    if not math.isfinite(result):
        raise ReplayError(f"{field} must be finite, got {result}")
    return result


def load_motion(path: Path, motion_name: str | None = None) -> Motion:
    """Load one motion-library entry without importing the training stack."""
    if not path.is_file():
        raise ReplayError(f"Motion PKL does not exist: {path}")
    try:
        import joblib
    except ImportError as exc:
        raise ReplayError("joblib is required to read the generated PKL") from exc

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ReplayError(f"Could not load {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ReplayError(f"Expected top-level dict in {path}, got {type(payload).__name__}")

    # Also accept an already-unwrapped single entry for easier offline testing.
    if "dof" in payload and "fps" in payload:
        selected_name = motion_name or path.stem
        entry = payload
    else:
        available = list(payload.keys())
        if motion_name is None:
            if len(available) != 1:
                raise ReplayError(
                    "PKL contains multiple motions; select one with --motion: "
                    + ", ".join(map(str, available))
                )
            selected_name = str(available[0])
        else:
            if motion_name not in payload:
                raise ReplayError(
                    f"Motion {motion_name!r} not found; available: "
                    + ", ".join(map(str, available))
                )
            selected_name = motion_name
        entry = payload[selected_name]

    if not isinstance(entry, dict) or "dof" not in entry or "fps" not in entry:
        raise ReplayError(f"Motion {selected_name!r} must contain 'dof' and 'fps'")

    positions = np.asarray(entry["dof"], dtype=np.float64)
    fps = _finite_float(entry["fps"], "fps")
    if positions.ndim != 2 or positions.shape[1] != len(RBY1_MODEL_A_JOINT_NAMES):
        raise ReplayError(
            f"Motion 'dof' must have shape (frames, 24), got {positions.shape}"
        )
    if positions.shape[0] < 3:
        raise ReplayError("Motion must contain at least three frames")
    if fps <= 0.0:
        raise ReplayError(f"fps must be positive, got {fps}")
    if not np.isfinite(positions).all():
        bad = np.argwhere(~np.isfinite(positions))[0]
        raise ReplayError(
            f"Motion contains non-finite data at frame {int(bad[0])}, joint "
            f"{RBY1_MODEL_A_JOINT_NAMES[int(bad[1])]}"
        )

    return Motion(selected_name, positions, fps, path.resolve())


def load_model_a_limits(urdf_path: Path, head_acceleration: float = 5.0) -> tuple[JointLimit, ...]:
    """Read Model A v1.2 limits from the vendor SDK URDF."""
    if not urdf_path.is_file():
        raise ReplayError(f"RBY1 Model A URDF does not exist: {urdf_path}")
    if not math.isfinite(head_acceleration) or head_acceleration <= 0.0:
        raise ReplayError("--head-acceleration must be positive and finite")
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ReplayError(f"Could not parse URDF {urdf_path}: {exc}") from exc

    elements = {joint.attrib.get("name"): joint for joint in root.findall("joint")}
    limits: list[JointLimit] = []
    for name in RBY1_MODEL_A_JOINT_NAMES:
        joint = elements.get(name)
        limit = joint.find("limit") if joint is not None else None
        if joint is None or limit is None:
            raise ReplayError(f"URDF is missing limits for required joint {name}")
        attrs = limit.attrib
        try:
            velocity = float(attrs["velocity"])
            acceleration = float(attrs.get("acceleration", head_acceleration))
            lower = float(attrs["lower"]) if "lower" in attrs else None
            upper = float(attrs["upper"]) if "upper" in attrs else None
        except (KeyError, ValueError) as exc:
            raise ReplayError(f"Invalid URDF limits for {name}: {attrs}") from exc
        values = [velocity, acceleration]
        values.extend(value for value in (lower, upper) if value is not None)
        if not all(math.isfinite(value) for value in values):
            raise ReplayError(f"Non-finite URDF limit for {name}: {attrs}")
        if velocity <= 0.0 or acceleration <= 0.0:
            raise ReplayError(f"Non-positive velocity/acceleration limit for {name}")
        if lower is not None and upper is not None and lower >= upper:
            raise ReplayError(f"Invalid position interval for {name}: [{lower}, {upper}]")
        limits.append(JointLimit(lower, upper, velocity, acceleration))
    return tuple(limits)


def _slice_motion(motion: Motion, start_frame: int, end_frame: int | None) -> np.ndarray:
    n_frames = motion.positions.shape[0]
    stop = n_frames if end_frame is None else end_frame
    if start_frame < 0 or start_frame >= n_frames - 2:
        raise ReplayError(f"--start-frame must be in [0, {n_frames - 3}]")
    if stop <= start_frame + 2 or stop > n_frames:
        raise ReplayError(
            f"--end-frame must be in [{start_frame + 3}, {n_frames}] (exclusive)"
        )
    return motion.positions[start_frame:stop].copy()


def _resample_cubic_hermite(
    values: np.ndarray,
    source_fps: float,
    output_rate: float,
    speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothly resample keyframes while retaining every source endpoint."""
    if source_fps <= 0.0 or output_rate <= 0.0 or speed <= 0.0:
        raise ReplayError("source FPS, control rate, and speed must be positive")
    duration = (values.shape[0] - 1) / (source_fps * speed)
    count = max(3, int(math.ceil(duration * output_rate)) + 1)
    times = np.linspace(0.0, duration, count, dtype=np.float64)
    source_frames = np.clip(times * source_fps * speed, 0.0, values.shape[0] - 1)

    # Central finite-difference tangents in units per source frame.
    tangents = np.gradient(values, axis=0, edge_order=2)
    left = np.minimum(np.floor(source_frames).astype(np.int64), values.shape[0] - 2)
    right = left + 1
    u = (source_frames - left).reshape(-1, 1)
    u2 = u * u
    u3 = u2 * u

    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    result = (
        h00 * values[left]
        + h10 * tangents[left]
        + h01 * values[right]
        + h11 * tangents[right]
    )
    result[0] = values[0]
    result[-1] = values[-1]
    return times, source_frames, result


def prepare_trajectory(
    motion: Motion,
    *,
    control_rate: float,
    speed: float,
    reference_mode: str,
    start_positions: np.ndarray | None,
    torso_scale: float,
    arm_scale: float,
    head_scale: float,
    enable_wheels: bool,
    wheel_scale: float,
    start_frame: int = 0,
    end_frame: int | None = None,
    scale_absolute_pose: bool = False,
) -> ReplayTrajectory:
    """Create the exact command-rate trajectory that will be safety checked."""
    for label, value in (
        ("control rate", control_rate),
        ("speed", speed),
        ("torso scale", torso_scale),
        ("arm scale", arm_scale),
        ("head scale", head_scale),
        ("wheel scale", wheel_scale),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ReplayError(f"{label} must be finite and non-negative")
    if control_rate <= 0.0 or speed <= 0.0:
        raise ReplayError("control rate and speed must be greater than zero")
    if reference_mode not in {"relative", "absolute"}:
        raise ReplayError(f"Unknown reference mode: {reference_mode}")

    source = _slice_motion(motion, start_frame, end_frame)
    # Wheel coordinates are continuous. Unwrap accidental +/-pi representation
    # changes before differentiating them into velocity targets.
    source[:, WHEEL_INDICES] = np.unwrap(source[:, WHEEL_INDICES], axis=0)
    times, source_frames, resampled = _resample_cubic_hermite(
        source, motion.fps, control_rate, speed
    )
    source_frames += start_frame

    scales = np.zeros(len(RBY1_MODEL_A_JOINT_NAMES), dtype=np.float64)
    scales[TORSO_INDICES] = torso_scale
    scales[RIGHT_ARM_INDICES] = arm_scale
    scales[LEFT_ARM_INDICES] = arm_scale
    scales[HEAD_INDICES] = head_scale
    if enable_wheels:
        scales[WHEEL_INDICES] = wheel_scale

    if reference_mode == "relative":
        if start_positions is None:
            baseline = np.zeros(len(RBY1_MODEL_A_JOINT_NAMES), dtype=np.float64)
        else:
            baseline = np.asarray(start_positions, dtype=np.float64)
            if baseline.shape != (len(RBY1_MODEL_A_JOINT_NAMES),):
                raise ReplayError(f"Measured start pose must have shape (24,), got {baseline.shape}")
            if not np.isfinite(baseline).all():
                raise ReplayError("Measured start pose contains non-finite values")
        positions = baseline + (resampled - resampled[0]) * scales
    else:
        if scale_absolute_pose:
            # Collision-conscious absolute trial: scale the complete pose,
            # including frame zero, toward the model's zero configuration.
            positions = resampled * scales
        else:
            # Fidelity trial: scale variation while preserving the source's
            # first absolute pose.
            positions = resampled[0] + (resampled - resampled[0]) * scales

    velocities = np.gradient(positions, times, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, times, axis=0, edge_order=2)
    if not enable_wheels:
        velocities[:, WHEEL_INDICES] = 0.0
        accelerations[:, WHEEL_INDICES] = 0.0

    return ReplayTrajectory(times, source_frames, positions, velocities, accelerations)


def fit_trajectory_start_pose(
    trajectory: ReplayTrajectory,
    limits: Sequence[JointLimit],
    *,
    position_margin: float,
) -> tuple[ReplayTrajectory, np.ndarray]:
    """Minimally shift each upper-body joint so the trajectory fits its limits.

    A constant per-joint shift preserves the motion deltas, velocities, and
    accelerations.  It is useful when a simulator starts a one-sided joint at
    zero (for example RBY1's shoulder-roll joints), leaving no room for a
    relative trajectory in one direction.  When possible, corrections to the
    torso roll pair and three serial pitch joints sum to zero, preserving the
    source's aggregate torso angles.  Wheels are never shifted.
    """
    if position_margin < 0.0 or not math.isfinite(position_margin):
        raise ReplayError("position margin must be finite and non-negative")
    if len(limits) != len(RBY1_MODEL_A_JOINT_NAMES):
        raise ReplayError(f"Expected 24 joint limits, got {len(limits)}")

    positions = trajectory.positions.copy()
    offsets = np.zeros(len(RBY1_MODEL_A_JOINT_NAMES), dtype=np.float64)
    minimum_offsets = np.full(len(RBY1_MODEL_A_JOINT_NAMES), -math.inf)
    maximum_offsets = np.full(len(RBY1_MODEL_A_JOINT_NAMES), math.inf)
    # Keep the fitted trajectory just inside the later validation boundary to
    # avoid floating-point equality appearing as a limit violation.
    fit_padding = 1.0e-8

    for index in UPPER_BODY_INDICES:
        limit = limits[index]
        minimum = float(np.min(positions[:, index]))
        maximum = float(np.max(positions[:, index]))
        minimum_offset = -math.inf
        maximum_offset = math.inf
        if limit.lower is not None:
            minimum_offset = limit.lower + position_margin + fit_padding - minimum
        if limit.upper is not None:
            maximum_offset = limit.upper - position_margin - fit_padding - maximum
        if minimum_offset > maximum_offset:
            name = RBY1_MODEL_A_JOINT_NAMES[index]
            allowed_span = (
                math.inf
                if limit.lower is None or limit.upper is None
                else limit.upper - limit.lower - 2.0 * position_margin
            )
            raise ReplayError(
                f"{name} trajectory span {maximum - minimum:.5f} rad cannot fit "
                f"the available {allowed_span:.5f} rad position interval"
            )

        minimum_offsets[index] = minimum_offset
        maximum_offsets[index] = maximum_offset
        offsets[index] = min(max(0.0, minimum_offset), maximum_offset)

    # The clean motion model and vendor URDF differ most on torso_0 and
    # torso_1. Applying those corrections independently changes the robot's
    # overall lean. Redistribute each correction through joints with the same
    # nominal axis when their bounds permit a zero-sum solution. This keeps the
    # absolute dance posture much closer to IsaacLab's kinematic replay.
    for group in ((2, 6), (3, 4, 5)):
        lower = minimum_offsets[list(group)]
        upper = maximum_offsets[list(group)]
        if (
            np.isfinite(lower).all()
            and np.isfinite(upper).all()
            and float(np.sum(lower)) <= 0.0 <= float(np.sum(upper))
        ):
            search_lower = float(np.min(lower))
            search_upper = float(np.max(upper))
            for _ in range(80):
                midpoint = 0.5 * (search_lower + search_upper)
                candidate = np.clip(midpoint, lower, upper)
                if float(np.sum(candidate)) < 0.0:
                    search_lower = midpoint
                else:
                    search_upper = midpoint
            offsets[list(group)] = np.clip(
                0.5 * (search_lower + search_upper), lower, upper
            )

    positions[:, UPPER_BODY_INDICES] += offsets[UPPER_BODY_INDICES]

    fitted = ReplayTrajectory(
        trajectory.times,
        trajectory.source_frames,
        positions,
        trajectory.velocities,
        trajectory.accelerations,
    )
    return fitted, offsets


def _log_start_pose_offsets(offsets: np.ndarray) -> None:
    adjusted = [
        f"{name}={offsets[index]:+.3f}rad"
        for index, name in enumerate(RBY1_MODEL_A_JOINT_NAMES)
        if abs(float(offsets[index])) > 1.0e-6
    ]
    if adjusted:
        LOGGER.info("Fitted start-pose offsets: %s", ", ".join(adjusted))
    else:
        LOGGER.info("Fitted start pose already lies inside all position limits")


def validate_trajectory(
    trajectory: ReplayTrajectory,
    limits: Sequence[JointLimit],
    *,
    position_margin: float,
    dynamic_limit_scale: float,
    enable_wheels: bool,
) -> list[ValidationIssue]:
    """Return all limit violations; callers decide whether they are warnings or fatal."""
    if position_margin < 0.0 or not math.isfinite(position_margin):
        raise ReplayError("position margin must be finite and non-negative")
    if not 0.0 < dynamic_limit_scale <= 1.0:
        raise ReplayError("dynamic limit scale must be in (0, 1]")
    if len(limits) != len(RBY1_MODEL_A_JOINT_NAMES):
        raise ReplayError(f"Expected 24 joint limits, got {len(limits)}")

    issues: list[ValidationIssue] = []
    for index, (name, limit) in enumerate(zip(RBY1_MODEL_A_JOINT_NAMES, limits)):
        if index not in WHEEL_INDICES:
            values = trajectory.positions[:, index]
            if limit.lower is not None:
                allowed = limit.lower + position_margin
                frame = int(np.argmin(values))
                observed = float(values[frame])
                if observed < allowed:
                    issues.append(
                        ValidationIssue(name, "minimum position", observed, allowed, frame, "rad")
                    )
            if limit.upper is not None:
                allowed = limit.upper - position_margin
                frame = int(np.argmax(values))
                observed = float(values[frame])
                if observed > allowed:
                    issues.append(
                        ValidationIssue(name, "maximum position", observed, allowed, frame, "rad")
                    )

        if index in WHEEL_INDICES and not enable_wheels:
            continue
        velocity = np.abs(trajectory.velocities[:, index])
        velocity_allowed = limit.velocity * dynamic_limit_scale
        velocity_frame = int(np.argmax(velocity))
        if float(velocity[velocity_frame]) > velocity_allowed:
            issues.append(
                ValidationIssue(
                    name,
                    "peak absolute velocity",
                    float(velocity[velocity_frame]),
                    velocity_allowed,
                    velocity_frame,
                    "rad/s",
                )
            )
        acceleration = np.abs(trajectory.accelerations[:, index])
        acceleration_allowed = limit.acceleration * dynamic_limit_scale
        acceleration_frame = int(np.argmax(acceleration))
        if float(acceleration[acceleration_frame]) > acceleration_allowed:
            issues.append(
                ValidationIssue(
                    name,
                    "peak absolute acceleration",
                    float(acceleration[acceleration_frame]),
                    acceleration_allowed,
                    acceleration_frame,
                    "rad/s^2",
                )
            )
    return issues


def _print_issues(title: str, issues: Sequence[ValidationIssue], *, fatal: bool) -> None:
    level = logging.ERROR if fatal else logging.WARNING
    if not issues:
        LOGGER.info("%s: PASS", title)
        return
    LOGGER.log(level, "%s: %d issue(s)", title, len(issues))
    for issue in issues:
        LOGGER.log(level, "  %s", issue)


def _validate_sdk_model(model: Any) -> None:
    model_name = str(model.model_name).upper()
    actual_names = tuple(str(name) for name in model.robot_joint_names)
    if model_name != "A":
        raise ReplayError(f"Only RBY1 Model A is supported, SDK reported {model_name!r}")
    if actual_names != RBY1_MODEL_A_JOINT_NAMES:
        mismatch = next(
            (
                (index, expected, actual)
                for index, (expected, actual) in enumerate(
                    zip(RBY1_MODEL_A_JOINT_NAMES, actual_names)
                )
                if expected != actual
            ),
            None,
        )
        if mismatch is None:
            detail = f"expected 24 names, SDK returned {len(actual_names)}"
        else:
            detail = f"index {mismatch[0]}: expected {mismatch[1]}, got {mismatch[2]}"
        raise ReplayError(f"SDK Model A joint order does not match the PKL ({detail})")


def _state_safety_check(
    snapshot: StateSnapshot,
    *,
    now: float,
    rby_module: Any,
    state_timeout: float,
    max_temperature: float,
    enable_wheels: bool,
) -> None:
    if now - snapshot.received_at > state_timeout:
        raise ReplayError(
            f"SDK state is stale by {now - snapshot.received_at:.3f}s "
            f"(limit {state_timeout:.3f}s)"
        )
    for label, values in (
        ("position", snapshot.positions),
        ("velocity", snapshot.velocities),
        ("temperature", snapshot.temperatures),
    ):
        if values.shape != (len(RBY1_MODEL_A_JOINT_NAMES),):
            raise ReplayError(f"SDK {label} state has shape {values.shape}, expected (24,)")
        if not np.isfinite(values).all():
            raise ReplayError(f"SDK {label} state contains non-finite values")
    if snapshot.emo_pressed:
        raise ReplayError("An RBY1 emergency-stop button is pressed")
    if snapshot.control_manager_state != rby_module.ControlManagerState.State.Enabled:
        raise ReplayError(
            f"Control manager left Enabled state: {snapshot.control_manager_state}"
        )
    required = np.arange(24) if enable_wheels else UPPER_BODY_INDICES
    if snapshot.ready.shape != (24,):
        raise ReplayError(f"SDK ready state has shape {snapshot.ready.shape}, expected (24,)")
    not_ready = [RBY1_MODEL_A_JOINT_NAMES[i] for i in required if not snapshot.ready[i]]
    if not_ready:
        raise ReplayError("Commanded joints are not ready: " + ", ".join(not_ready))
    hottest = int(np.argmax(snapshot.temperatures))
    if float(snapshot.temperatures[hottest]) > max_temperature:
        raise ReplayError(
            f"{RBY1_MODEL_A_JOINT_NAMES[hottest]} temperature is "
            f"{snapshot.temperatures[hottest]:.1f} C (limit {max_temperature:.1f} C)"
        )


def _wait_for_stationary_state(
    monitor: StateMonitor,
    *,
    rby_module: Any,
    state_timeout: float,
    max_temperature: float,
    enable_wheels: bool,
    max_velocity: float,
    stable_duration: float,
    timeout: float,
    stop_requested: threading.Event | None = None,
) -> StateSnapshot:
    """Require continuously low measured velocity before beginning a new phase."""
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    last_peak = math.inf
    required = np.arange(24) if enable_wheels else UPPER_BODY_INDICES
    while time.monotonic() < deadline:
        if stop_requested is not None and stop_requested.is_set():
            raise ReplayError("Replay cancelled while waiting for a stationary state")
        now = time.monotonic()
        snapshot = monitor.latest()
        if snapshot is None:
            time.sleep(0.01)
            continue
        _state_safety_check(
            snapshot,
            now=now,
            rby_module=rby_module,
            state_timeout=state_timeout,
            max_temperature=max_temperature,
            enable_wheels=enable_wheels,
        )
        last_peak = float(np.max(np.abs(snapshot.velocities[required])))
        if last_peak <= max_velocity:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_duration:
                return snapshot
        else:
            stable_since = None
        time.sleep(0.01)
    raise ReplayError(
        f"Robot did not remain stationary for {stable_duration:.2f}s within {timeout:.2f}s; "
        f"latest peak joint velocity was {last_peak:.3f} rad/s "
        f"(limit {max_velocity:.3f} rad/s)"
    )


def _require_reset_pose_for_simulator_fit(
    positions: np.ndarray,
    *,
    target: str,
    fit_start_pose: bool,
) -> None:
    """Prevent fitting a new replay around a pose left by an aborted simulator run."""
    if target != "simulator" or not fit_start_pose:
        return
    offsets = np.abs(np.asarray(positions, dtype=np.float64)[UPPER_BODY_INDICES])
    if float(np.max(offsets)) <= SIMULATOR_RESET_POSE_TOLERANCE:
        return
    largest = np.argsort(offsets)[::-1][:4]
    detail = ", ".join(
        f"{RBY1_MODEL_A_JOINT_NAMES[int(UPPER_BODY_INDICES[index])]}="
        f"{positions[int(UPPER_BODY_INDICES[index])]:+.3f}rad"
        for index in largest
        if offsets[index] > SIMULATOR_RESET_POSE_TOLERANCE
    )
    raise ReplayError(
        "Isaac Sim is not at its reset zero pose while --fit-start-pose is active "
        f"({detail}; tolerance {SIMULATOR_RESET_POSE_TOLERANCE:.3f}rad). "
        "Press Backspace in the Isaac Sim window (or restart the simulator), wait "
        "for reset, and then rerun. This avoids fitting the motion around a pose "
        "left by an earlier abort."
    )


def _position_command_builder(
    rby_module: Any,
    positions: np.ndarray,
    limits: Sequence[JointLimit],
    indices: np.ndarray,
    *,
    minimum_time: float,
    hold_time: float,
    dynamic_limit_scale: float,
) -> Any:
    velocity_limits = np.asarray(
        [limits[index].velocity * dynamic_limit_scale for index in indices], dtype=np.float64
    )
    acceleration_limits = np.asarray(
        [limits[index].acceleration * dynamic_limit_scale for index in indices],
        dtype=np.float64,
    )
    return (
        rby_module.JointPositionCommandBuilder()
        .set_command_header(rby_module.CommandHeaderBuilder().set_control_hold_time(hold_time))
        .set_minimum_time(minimum_time)
        .set_position(np.asarray(positions[indices], dtype=np.float64))
        .set_velocity_limit(velocity_limits)
        .set_acceleration_limit(acceleration_limits)
    )


def build_command(
    rby_module: Any,
    positions: np.ndarray,
    wheel_velocities: np.ndarray,
    limits: Sequence[JointLimit],
    *,
    minimum_time: float,
    hold_time: float,
    dynamic_limit_scale: float,
    enable_wheels: bool,
) -> Any:
    """Build one component command in SDK Model A order."""
    body = (
        rby_module.BodyComponentBasedCommandBuilder()
        .set_torso_command(
            _position_command_builder(
                rby_module,
                positions,
                limits,
                TORSO_INDICES,
                minimum_time=minimum_time,
                hold_time=hold_time,
                dynamic_limit_scale=dynamic_limit_scale,
            )
        )
        .set_right_arm_command(
            _position_command_builder(
                rby_module,
                positions,
                limits,
                RIGHT_ARM_INDICES,
                minimum_time=minimum_time,
                hold_time=hold_time,
                dynamic_limit_scale=dynamic_limit_scale,
            )
        )
        .set_left_arm_command(
            _position_command_builder(
                rby_module,
                positions,
                limits,
                LEFT_ARM_INDICES,
                minimum_time=minimum_time,
                hold_time=hold_time,
                dynamic_limit_scale=dynamic_limit_scale,
            )
        )
    )
    components = rby_module.ComponentBasedCommandBuilder().set_body_command(body)
    components.set_head_command(
        _position_command_builder(
            rby_module,
            positions,
            limits,
            HEAD_INDICES,
            minimum_time=minimum_time,
            hold_time=hold_time,
            dynamic_limit_scale=dynamic_limit_scale,
        )
    )
    if enable_wheels:
        wheel_acceleration_limits = np.asarray(
            [limits[index].acceleration * dynamic_limit_scale for index in WHEEL_INDICES],
            dtype=np.float64,
        )
        components.set_mobility_command(
            rby_module.JointVelocityCommandBuilder()
            .set_command_header(
                rby_module.CommandHeaderBuilder().set_control_hold_time(hold_time)
            )
            .set_minimum_time(minimum_time)
            .set_velocity(np.asarray(wheel_velocities, dtype=np.float64))
            .set_acceleration_limit(wheel_acceleration_limits)
        )
    return rby_module.RobotCommandBuilder().set_command(components)


def _prepare_sdk_robot(args: argparse.Namespace, rby_module: Any) -> Any:
    robot = rby_module.create_robot(args.address, "a")
    try:
        if not robot.connect():
            raise ReplayError(f"Could not connect to RBY1 SDK server at {args.address}")
        _validate_sdk_model(robot.model())

        # Read-only checks happen before any optional power/servo operation.
        pre_enable_state = robot.get_state()
        if any(
            emo.state == rby_module.EMOState.State.Pressed
            for emo in pre_enable_state.emo_states
        ):
            raise ReplayError("An RBY1 emergency-stop button is pressed")
        temperatures = np.asarray(pre_enable_state.temperature, dtype=np.float64)
        if temperatures.shape == (24,) and np.isfinite(temperatures).all():
            hottest = int(np.argmax(temperatures))
            if float(temperatures[hottest]) > args.max_temperature:
                raise ReplayError(
                    f"{RBY1_MODEL_A_JOINT_NAMES[hottest]} temperature is "
                    f"{temperatures[hottest]:.1f} C before activation "
                    f"(limit {args.max_temperature:.1f} C)"
                )

        if args.auto_enable:
            if not robot.is_power_on(".*") and not robot.power_on(".*"):
                raise ReplayError("Failed to power on all RBY1 devices")
            if not robot.is_servo_on(".*") and not robot.servo_on(".*"):
                raise ReplayError("Failed to enable all RBY1 servos")

        if not robot.is_power_on(".*"):
            raise ReplayError("RBY1 power is off; prepare it manually or pass --auto-enable")
        if not robot.is_servo_on(".*"):
            raise ReplayError("RBY1 servos are off; prepare them manually or pass --auto-enable")

        manager_state = robot.get_control_manager_state()
        fault_states = {
            rby_module.ControlManagerState.State.MinorFault,
            rby_module.ControlManagerState.State.MajorFault,
        }
        if manager_state.state in fault_states:
            if not args.reset_faults:
                raise ReplayError(
                    "Control manager is faulted; inspect the robot before optionally using "
                    "--reset-faults"
                )
            if not robot.reset_fault_control_manager():
                raise ReplayError("Failed to reset the RBY1 control-manager fault")

        if robot.get_control_manager_state().state != rby_module.ControlManagerState.State.Enabled:
            if not args.auto_enable:
                raise ReplayError(
                    "Control manager is not enabled; enable it manually or pass --auto-enable"
                )
            if not robot.enable_control_manager(False):
                raise ReplayError("Failed to enable the RBY1 control manager")

        manager_state = robot.get_control_manager_state()
        if manager_state.state != rby_module.ControlManagerState.State.Enabled:
            raise ReplayError(f"Unexpected control-manager state: {manager_state.state}")
        if bool(manager_state.unlimited_mode_enabled):
            raise ReplayError("Refusing replay while the control manager is in unlimited mode")
        if manager_state.control_state != rby_module.ControlManagerState.ControlState.Idle:
            raise ReplayError(
                "Refusing to preempt a non-idle control-manager command: "
                f"{manager_state.control_state}"
            )
        return robot
    except BaseException:
        try:
            robot.disconnect()
        except Exception:
            pass
        raise


def _minimum_transition_time(
    current: np.ndarray,
    target: np.ndarray,
    limits: Sequence[JointLimit],
    dynamic_limit_scale: float,
) -> float:
    required = 0.0
    for index in UPPER_BODY_INDICES:
        distance = abs(float(target[index] - current[index]))
        velocity = limits[index].velocity * dynamic_limit_scale
        acceleration = limits[index].acceleration * dynamic_limit_scale
        # Exact peak factors for the quintic minimum-jerk blend used below:
        # max(ds/du) = 15/8 and max(abs(d2s/du2)) = 10/sqrt(3).
        required = max(
            required,
            1.875 * distance / velocity,
            math.sqrt((10.0 / math.sqrt(3.0)) * distance / acceleration),
        )
    return required


def _minimum_jerk_blend(progress: float) -> float:
    """Return a clamped quintic blend with zero endpoint velocity/acceleration."""
    u = min(1.0, max(0.0, float(progress)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def prepare_upper_body_transition(
    current: np.ndarray,
    target: np.ndarray,
    limits: Sequence[JointLimit],
    *,
    control_rate: float,
    minimum_seconds: float,
    dynamic_limit_scale: float,
) -> ReplayTrajectory:
    """Build a safety-checkable minimum-jerk upper-body transition.

    Wheel positions are deliberately held constant and wheel target velocities
    are zero. Returning the mobile base to its original location would require a
    separately planned collision-aware differential-drive path.
    """
    current = np.asarray(current, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    expected_shape = (len(RBY1_MODEL_A_JOINT_NAMES),)
    if current.shape != expected_shape or target.shape != expected_shape:
        raise ReplayError(
            f"Transition endpoints must both have shape {expected_shape}; "
            f"got {current.shape} and {target.shape}"
        )
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ReplayError("Transition endpoints contain non-finite values")
    if control_rate <= 0.0 or not math.isfinite(control_rate):
        raise ReplayError("Transition control rate must be positive and finite")
    if minimum_seconds <= 0.0 or not math.isfinite(minimum_seconds):
        raise ReplayError("Transition duration must be positive and finite")
    if not 0.0 < dynamic_limit_scale <= 1.0:
        raise ReplayError("Transition dynamic limit scale must be in (0, 1]")

    transition_time = max(
        minimum_seconds,
        1.1 * _minimum_transition_time(
            current,
            target,
            limits,
            dynamic_limit_scale,
        ),
    )
    sample_count = max(2, int(math.ceil(transition_time * control_rate)) + 1)
    times = np.linspace(0.0, transition_time, sample_count, dtype=np.float64)
    progress = times / transition_time
    blend = progress**3 * (10.0 + progress * (-15.0 + 6.0 * progress))
    blend_velocity = (
        30.0 * progress**2 - 60.0 * progress**3 + 30.0 * progress**4
    ) / transition_time
    blend_acceleration = (
        60.0 * progress - 180.0 * progress**2 + 120.0 * progress**3
    ) / (transition_time * transition_time)

    positions = np.repeat(current[np.newaxis, :], sample_count, axis=0)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    delta = target - current
    positions[:, UPPER_BODY_INDICES] += np.outer(
        blend,
        delta[UPPER_BODY_INDICES],
    )
    velocities[:, UPPER_BODY_INDICES] = np.outer(
        blend_velocity,
        delta[UPPER_BODY_INDICES],
    )
    accelerations[:, UPPER_BODY_INDICES] = np.outer(
        blend_acceleration,
        delta[UPPER_BODY_INDICES],
    )
    positions[0, UPPER_BODY_INDICES] = current[UPPER_BODY_INDICES]
    positions[-1, UPPER_BODY_INDICES] = target[UPPER_BODY_INDICES]
    source_frames = np.full(sample_count, np.nan, dtype=np.float64)
    return ReplayTrajectory(times, source_frames, positions, velocities, accelerations)


def _hold_target_until_settled(
    stream: Any,
    rby_module: Any,
    monitor: StateMonitor,
    target: np.ndarray,
    limits: Sequence[JointLimit],
    args: argparse.Namespace,
    stop_requested: threading.Event,
    *,
    target_name: str,
) -> StateSnapshot:
    """Refresh a finite-watchdog hold until the measured target is stationary."""
    settle_deadline = time.monotonic() + args.stationary_timeout
    next_command_time = time.monotonic()
    stable_since: float | None = None
    last_error = math.inf
    last_peak_velocity = math.inf
    velocity_indices = np.arange(24) if args.enable_wheels else UPPER_BODY_INDICES
    while time.monotonic() < settle_deadline:
        if stop_requested.is_set():
            raise ReplayError(f"Replay cancelled while settling at {target_name.lower()}")
        remaining = next_command_time - time.monotonic()
        if remaining > 0.0 and stop_requested.wait(remaining):
            raise ReplayError(f"Replay cancelled while settling at {target_name.lower()}")
        now = time.monotonic()
        snapshot = monitor.latest()
        if snapshot is None:
            raise ReplayError(f"No current SDK state is available while settling at {target_name.lower()}")
        _state_safety_check(
            snapshot,
            now=now,
            rby_module=rby_module,
            state_timeout=args.state_timeout,
            max_temperature=args.max_temperature,
            enable_wheels=args.enable_wheels,
        )
        command = build_command(
            rby_module,
            target,
            np.zeros(2, dtype=np.float64),
            limits,
            minimum_time=max(0.005, 1.02 / args.control_rate),
            hold_time=args.hold_time,
            dynamic_limit_scale=args.limit_scale,
            enable_wheels=args.enable_wheels,
        )
        stream.send_command(command, timeout_ms=args.command_timeout_ms)

        last_error = float(
            np.max(
                np.abs(
                    snapshot.positions[UPPER_BODY_INDICES]
                    - target[UPPER_BODY_INDICES]
                )
            )
        )
        last_peak_velocity = float(
            np.max(np.abs(snapshot.velocities[velocity_indices]))
        )
        if (
            last_error <= args.max_tracking_error
            and last_peak_velocity <= args.max_stationary_velocity
        ):
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= args.stationary_seconds:
                LOGGER.info(
                    "%s settled: max error %.3frad, peak velocity %.3frad/s",
                    target_name,
                    last_error,
                    last_peak_velocity,
                )
                return snapshot
        else:
            stable_since = None
        next_command_time += 1.0 / args.control_rate

    raise ReplayError(
        f"{target_name} did not settle within {args.stationary_timeout:.2f}s; "
        f"max error was {last_error:.3f}rad "
        f"(limit {args.max_tracking_error:.3f}rad), peak velocity was "
        f"{last_peak_velocity:.3f}rad/s "
        f"(limit {args.max_stationary_velocity:.3f}rad/s)"
    )


def _stream_upper_body_transition(
    stream: Any,
    rby_module: Any,
    monitor: StateMonitor,
    current: np.ndarray,
    target: np.ndarray,
    limits: Sequence[JointLimit],
    args: argparse.Namespace,
    stop_requested: threading.Event,
    *,
    minimum_seconds: float,
    phase_name: str,
    action_name: str,
    target_name: str,
) -> StateSnapshot:
    """Stream one minimum-jerk upper-body transition and verify convergence."""
    transition = prepare_upper_body_transition(
        current,
        target,
        limits,
        control_rate=args.control_rate,
        minimum_seconds=minimum_seconds,
        dynamic_limit_scale=args.limit_scale,
    )
    LOGGER.info(
        "Streaming %s over %.2fs (%d commands)",
        action_name,
        transition.duration,
        len(transition.times),
    )

    start_time = time.monotonic()
    for sample, target_time in enumerate(transition.times):
        if stop_requested.is_set():
            raise ReplayError(f"Replay cancelled during {phase_name}")
        deadline = start_time + float(target_time)
        remaining = deadline - time.monotonic()
        if remaining > 0.0 and stop_requested.wait(remaining):
            raise ReplayError(f"Replay cancelled during {phase_name}")
        now = time.monotonic()
        lag = now - deadline
        if lag > args.max_loop_lag:
            raise ReplayError(
                f"{phase_name.capitalize()} missed its deadline by {lag:.3f}s "
                f"(limit {args.max_loop_lag:.3f}s)"
            )

        snapshot = monitor.latest()
        if snapshot is None:
            raise ReplayError(f"No current SDK state is available during {phase_name}")
        _state_safety_check(
            snapshot,
            now=now,
            rby_module=rby_module,
            state_timeout=args.state_timeout,
            max_temperature=args.max_temperature,
            enable_wheels=args.enable_wheels,
        )

        if sample + 1 < len(transition.times):
            minimum_time = max(
                0.005,
                float(transition.times[sample + 1] - target_time) * 1.02,
            )
        else:
            minimum_time = max(0.005, 1.02 / args.control_rate)
        command = build_command(
            rby_module,
            transition.positions[sample],
            np.zeros(2, dtype=np.float64),
            limits,
            minimum_time=minimum_time,
            hold_time=args.hold_time,
            dynamic_limit_scale=args.limit_scale,
            enable_wheels=args.enable_wheels,
        )
        stream.send_command(command, timeout_ms=args.command_timeout_ms)

    return _hold_target_until_settled(
        stream,
        rby_module,
        monitor,
        target,
        limits,
        args,
        stop_requested,
        target_name=target_name,
    )


def _transition_to_first_pose(
    stream: Any,
    rby_module: Any,
    monitor: StateMonitor,
    current: np.ndarray,
    first_target: np.ndarray,
    limits: Sequence[JointLimit],
    args: argparse.Namespace,
    stop_requested: threading.Event,
) -> StateSnapshot:
    """Stream a smooth first-pose transition and wait for measured convergence.

    The vendor Isaac/SDK split backend accepts stream commands but does not
    reliably complete a one-shot position-command handler.  Keeping this phase
    on the same stream as replay also prevents a command-ownership gap between
    the transition and frame zero.
    """
    return _stream_upper_body_transition(
        stream,
        rby_module,
        monitor,
        current,
        first_target,
        limits,
        args,
        stop_requested,
        minimum_seconds=args.transition_seconds,
        phase_name="first-pose transition",
        action_name="the first upper-body target",
        target_name="First target",
    )


def _return_to_zero_pose(
    stream: Any,
    rby_module: Any,
    monitor: StateMonitor,
    final_target: np.ndarray,
    limits: Sequence[JointLimit],
    args: argparse.Namespace,
    stop_requested: threading.Event,
) -> StateSnapshot:
    """Stop mobility, settle the final target, then return the upper body to zero."""
    LOGGER.info("Motion frames complete; stopping wheels and settling the final target")
    final_snapshot = _hold_target_until_settled(
        stream,
        rby_module,
        monitor,
        final_target,
        limits,
        args,
        stop_requested,
        target_name="Final motion target",
    )
    if np.allclose(
        final_target[UPPER_BODY_INDICES],
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    ):
        LOGGER.info("Final upper-body target is already at the all-zero pose")
        return final_snapshot

    zero_target = final_target.copy()
    zero_target[UPPER_BODY_INDICES] = 0.0
    return_seconds = (
        args.transition_seconds if args.return_seconds is None else args.return_seconds
    )
    return _stream_upper_body_transition(
        stream,
        rby_module,
        monitor,
        final_target,
        zero_target,
        limits,
        args,
        stop_requested,
        minimum_seconds=return_seconds,
        phase_name="return-to-zero transition",
        action_name="the all-zero upper-body pose",
        target_name="All-zero pose",
    )


def _safe_stop(
    robot: Any,
    rby_module: Any,
    stream: Any | None,
    monitor: StateMonitor | None,
    limits: Sequence[JointLimit],
    args: argparse.Namespace,
) -> None:
    """Best-effort zero mobility and measured-position upper-body hold."""
    try:
        snapshot = monitor.latest() if monitor is not None else None
    except Exception:
        snapshot = None
    if snapshot is None:
        try:
            state = robot.get_state()
            hold_positions = np.asarray(state.position, dtype=np.float64).copy()
        except Exception:
            hold_positions = None
    else:
        hold_positions = snapshot.positions

    stop_command = None
    if hold_positions is not None and hold_positions.shape == (24,):
        try:
            stop_command = build_command(
                rby_module,
                hold_positions,
                np.zeros(2, dtype=np.float64),
                limits,
                minimum_time=0.25,
                hold_time=0.5,
                dynamic_limit_scale=args.limit_scale,
                enable_wheels=args.enable_wheels,
            )
            if stream is not None:
                stream.send_command(stop_command, timeout_ms=args.command_timeout_ms)
                time.sleep(0.05)
        except Exception as exc:
            LOGGER.error("Could not send final hold/zero command: %s", exc)
    if stream is not None:
        try:
            stream.cancel()
        except Exception as exc:
            LOGGER.error("Could not cancel command stream: %s", exc)
    if args.target == "simulator":
        # The hybrid Isaac/SDK backend can leave one-shot command handlers in
        # Executing indefinitely.  The measured hold above was already sent on
        # the owned stream; cancelling that stream cleanly returns it to Idle.
        return
    if stop_command is not None:
        try:
            # Reinforce the stop outside the stream in case the stream itself
            # ended because its transport failed.
            handler = robot.send_command(stop_command, args.command_priority + 1)
            if not handler.wait_for(1500):
                handler.cancel()
                LOGGER.error("Final direct hold/zero command timed out")
        except Exception as exc:
            LOGGER.error("Could not send direct final hold/zero command: %s", exc)


def _log_path(args: argparse.Namespace, motion: Motion) -> Path | None:
    if args.no_log:
        return None
    if args.log is not None:
        return args.log.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "replay_logs" / f"{motion.name}_{stamp}.csv"


def _csv_header() -> list[str]:
    header = [
        "phase",
        "elapsed_s",
        "source_frame",
        "state_age_s",
        "max_tracking_error_rad",
    ]
    header.extend(f"target_pos_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"target_vel_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"measured_pos_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"measured_vel_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    return header


def replay(
    args: argparse.Namespace,
    motion: Motion,
    limits: Sequence[JointLimit],
) -> Path | None:
    try:
        import rby1_sdk as rby
    except ImportError as exc:
        raise ReplayError(
            "rby1_sdk is not importable. Install the pinned SDK client with "
            "'python -m pip install -r rby1_hardware/requirements-sdk.txt'."
        ) from exc

    robot = None
    monitor = None
    stream = None
    log_file = None
    log_writer = None
    log_path = _log_path(args, motion)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = None
    if hasattr(signal, "SIGTERM"):
        old_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        robot = _prepare_sdk_robot(args, rby)
        monitor = StateMonitor(rby)
        robot.start_state_update(monitor.callback, args.state_rate)
        monitor.wait_for_first(args.state_timeout * 5.0)
        initial = _wait_for_stationary_state(
            monitor,
            rby_module=rby,
            state_timeout=args.state_timeout,
            max_temperature=args.max_temperature,
            enable_wheels=args.enable_wheels,
            max_velocity=args.max_stationary_velocity,
            stable_duration=args.stationary_seconds,
            timeout=args.stationary_timeout,
            stop_requested=stop_requested,
        )
        _require_reset_pose_for_simulator_fit(
            initial.positions,
            target=args.target,
            fit_start_pose=args.fit_start_pose,
        )

        trajectory = prepare_trajectory(
            motion,
            control_rate=args.control_rate,
            speed=args.speed,
            reference_mode=args.reference_mode,
            start_positions=initial.positions,
            torso_scale=args.torso_scale,
            arm_scale=args.arm_scale,
            head_scale=args.head_scale,
            enable_wheels=args.enable_wheels,
            wheel_scale=args.wheel_scale,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            scale_absolute_pose=args.scale_absolute_pose,
        )
        if args.fit_start_pose:
            trajectory, offsets = fit_trajectory_start_pose(
                trajectory,
                limits,
                position_margin=args.position_margin,
            )
            _log_start_pose_offsets(offsets)
        issues = validate_trajectory(
            trajectory,
            limits,
            position_margin=args.position_margin,
            dynamic_limit_scale=args.limit_scale,
            enable_wheels=args.enable_wheels,
        )
        _print_issues("Measured-start replay trajectory", issues, fatal=True)
        if issues:
            raise ReplayError("Prepared replay violates the configured hardware safety envelope")

        if args.return_to_zero:
            return_seconds = (
                args.transition_seconds
                if args.return_seconds is None
                else args.return_seconds
            )
            return_transition = prepare_upper_body_transition(
                trajectory.positions[-1],
                np.zeros(24, dtype=np.float64),
                limits,
                control_rate=args.control_rate,
                minimum_seconds=return_seconds,
                dynamic_limit_scale=args.limit_scale,
            )
            return_issues = validate_trajectory(
                return_transition,
                limits,
                position_margin=args.position_margin,
                dynamic_limit_scale=args.limit_scale,
                enable_wheels=args.enable_wheels,
            )
            _print_issues("Return-to-zero transition", return_issues, fatal=True)
            if return_issues:
                raise ReplayError(
                    "Prepared return-to-zero transition violates the configured "
                    "hardware safety envelope"
                )
            LOGGER.info(
                "Return-to-zero enabled: %.3fs upper-body transition; wheels stop in place",
                return_transition.duration,
            )

        LOGGER.info(
            "Prepared %d commands over %.3fs at approximately %.1f Hz",
            len(trajectory.times),
            trajectory.duration,
            args.control_rate,
        )
        LOGGER.info(
            "Mode=%s%s, speed=%.3f, scales: torso=%.3f arms=%.3f head=%.3f wheels=%s",
            args.reference_mode,
            "/scaled-pose" if args.scale_absolute_pose else "",
            args.speed,
            args.torso_scale,
            args.arm_scale,
            args.head_scale,
            f"{args.wheel_scale:.3f}" if args.enable_wheels else "DISABLED",
        )
        LOGGER.info(
            "Tracking watchdog: %.3frad sustained for %.2fs",
            args.max_tracking_error,
            args.tracking_error_duration,
        )

        if args.countdown > 0:
            LOGGER.warning(
                "First-pose transition begins in %d seconds; keep the E-stop in hand",
                args.countdown,
            )
            for countdown_remaining in range(args.countdown, 0, -1):
                LOGGER.info("Starting in %d...", countdown_remaining)
                if stop_requested.wait(1.0):
                    raise ReplayError("Replay cancelled during countdown")

        stream = robot.create_command_stream(args.command_priority)
        _transition_to_first_pose(
            stream,
            rby,
            monitor,
            initial.positions,
            trajectory.positions[0],
            limits,
            args,
            stop_requested,
        )

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            log_writer.writerow(_csv_header())
            LOGGER.info("Writing replay state log to %s", log_path)

        start_time = time.monotonic()
        tracking_error_since: float | None = None
        last_flush = start_time
        motion_completed = True
        for frame, target_time in enumerate(trajectory.times):
            if stop_requested.is_set():
                LOGGER.warning("Stop requested; ending replay")
                motion_completed = False
                break
            deadline = start_time + float(target_time)
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            now = time.monotonic()
            lag = now - deadline
            if lag > args.max_loop_lag:
                raise ReplayError(
                    f"Replay loop missed its deadline by {lag:.3f}s "
                    f"(limit {args.max_loop_lag:.3f}s)"
                )

            snapshot = monitor.latest()
            if snapshot is None:
                raise ReplayError("No current SDK state is available")
            _state_safety_check(
                snapshot,
                now=now,
                rby_module=rby,
                state_timeout=args.state_timeout,
                max_temperature=args.max_temperature,
                enable_wheels=args.enable_wheels,
            )

            tracking_errors = np.abs(
                snapshot.positions[UPPER_BODY_INDICES]
                - trajectory.positions[frame, UPPER_BODY_INDICES]
            )
            tracking_error_offset = int(np.argmax(tracking_errors))
            tracking_error_index = int(UPPER_BODY_INDICES[tracking_error_offset])
            tracking_error = float(tracking_errors[tracking_error_offset])
            if tracking_error > args.max_tracking_error:
                if tracking_error_since is None:
                    tracking_error_since = now
                elif now - tracking_error_since > args.tracking_error_duration:
                    raise ReplayError(
                        f"{RBY1_MODEL_A_JOINT_NAMES[tracking_error_index]} tracking error "
                        f"remained at {tracking_error:.3f}rad for more than "
                        f"{args.tracking_error_duration:.2f}s "
                        f"(target {trajectory.positions[frame, tracking_error_index]:+.3f}rad, "
                        f"measured {snapshot.positions[tracking_error_index]:+.3f}rad)"
                    )
            else:
                tracking_error_since = None

            if args.enable_wheels:
                wheel_error = float(
                    np.max(
                        np.abs(
                            snapshot.velocities[WHEEL_INDICES]
                            - trajectory.velocities[frame, WHEEL_INDICES]
                        )
                    )
                )
                if wheel_error > args.max_wheel_velocity_error:
                    raise ReplayError(
                        f"Wheel velocity tracking error is {wheel_error:.3f} rad/s "
                        f"(limit {args.max_wheel_velocity_error:.3f} rad/s)"
                    )

            if frame + 1 < len(trajectory.times):
                minimum_time = max(
                    0.005,
                    float(trajectory.times[frame + 1] - trajectory.times[frame]) * 1.02,
                )
            else:
                minimum_time = max(0.005, 1.02 / args.control_rate)
            command = build_command(
                rby,
                trajectory.positions[frame],
                trajectory.velocities[frame, WHEEL_INDICES],
                limits,
                minimum_time=minimum_time,
                hold_time=args.hold_time,
                dynamic_limit_scale=args.limit_scale,
                enable_wheels=args.enable_wheels,
            )
            stream.send_command(command, timeout_ms=args.command_timeout_ms)

            if log_writer is not None:
                log_writer.writerow(
                    [
                        "motion",
                        now - start_time,
                        float(trajectory.source_frames[frame]),
                        now - snapshot.received_at,
                        tracking_error,
                        *trajectory.positions[frame].tolist(),
                        *trajectory.velocities[frame].tolist(),
                        *snapshot.positions.tolist(),
                        *snapshot.velocities.tolist(),
                    ]
                )
                if now - last_flush >= 1.0 and log_file is not None:
                    log_file.flush()
                    last_flush = now

        LOGGER.info("Motion frames finished after %.3fs", time.monotonic() - start_time)
        if args.return_to_zero:
            if not motion_completed:
                LOGGER.warning("Skipping return-to-zero because replay did not finish normally")
            else:
                returned = _return_to_zero_pose(
                    stream,
                    rby,
                    monitor,
                    trajectory.positions[-1],
                    limits,
                    args,
                    stop_requested,
                )
                now = time.monotonic()
                return_error = float(
                    np.max(
                        np.abs(returned.positions[UPPER_BODY_INDICES])
                    )
                )
                if log_writer is not None:
                    return_log_target = trajectory.positions[-1].copy()
                    return_log_target[UPPER_BODY_INDICES] = 0.0
                    log_writer.writerow(
                        [
                            "return_to_zero",
                            now - start_time,
                            math.nan,
                            now - returned.received_at,
                            return_error,
                            *return_log_target.tolist(),
                            *np.zeros(24, dtype=np.float64).tolist(),
                            *returned.positions.tolist(),
                            *returned.velocities.tolist(),
                        ]
                    )
                    if log_file is not None:
                        log_file.flush()
                LOGGER.info(
                    "Return-to-zero complete: max upper-body error %.3frad",
                    return_error,
                )

        LOGGER.info("Replay sequence finished after %.3fs", time.monotonic() - start_time)
        return log_path
    finally:
        if robot is not None:
            _safe_stop(robot, rby, stream, monitor, limits, args)
            try:
                robot.stop_state_update()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
        if log_file is not None:
            log_file.close()
        signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)


def _loopback_address(address: str) -> bool:
    host = address.rsplit(":", 1)[0].strip("[]").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or safely replay a 24-DOF RBY1 motion-library PKL through rby1-sdk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pkl", type=Path, help="Generated motion-library PKL")
    parser.add_argument("--motion", help="Motion key when the PKL contains more than one entry")
    parser.add_argument(
        "--target",
        choices=("validate", "simulator", "robot"),
        default="validate",
        help="validate is strictly offline; simulator/robot send SDK commands",
    )
    parser.add_argument("--address", default="127.0.0.1:50051", help="rby1-sdk server address")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Bundled vendor Model A v1.2 URDF used for replay safety limits",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="relative applies PKL deltas about measured pose; absolute uses the PKL first pose",
    )
    parser.add_argument(
        "--fit-start-pose",
        action="store_true",
        help=(
            "minimally shift upper-body baselines to fit position limits while "
            "preserving motion deltas (useful for a zero-pose simulator)"
        ),
    )
    parser.add_argument(
        "--scale-absolute-pose",
        action="store_true",
        help=(
            "with absolute mode, apply torso/arm/head scales to the complete "
            "pose including frame zero; otherwise only pose variation is scaled"
        ),
    )
    parser.add_argument("--speed", type=_positive, default=0.25, help="1.0 is source timing; 0.25 is 4x slower")
    parser.add_argument("--control-rate", type=_positive, default=50.0, help="SDK command rate in Hz")
    parser.add_argument("--torso-scale", type=_nonnegative, default=0.10, help="Torso motion amplitude")
    parser.add_argument("--arm-scale", type=_nonnegative, default=0.25, help="Arm motion amplitude")
    parser.add_argument("--head-scale", type=_nonnegative, default=0.25, help="Head motion amplitude")
    parser.add_argument("--enable-wheels", action="store_true", help="Enable mobility commands (off by default)")
    parser.add_argument("--wheel-scale", type=_nonnegative, default=0.25, help="Wheel velocity amplitude")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame")
    parser.add_argument("--end-frame", type=int, help="Exclusive last source frame")
    parser.add_argument(
        "--return-to-zero",
        dest="return_to_zero",
        action="store_true",
        help=(
            "after normal completion, stop wheels and return every torso, arm, "
            "and head joint to 0 rad"
        ),
    )
    parser.add_argument(
        "--return-seconds",
        type=_positive,
        help="Minimum return-to-zero transition time; defaults to --transition-seconds",
    )

    safety = parser.add_argument_group("safety envelope")
    safety.add_argument(
        "--position-margin",
        type=_nonnegative,
        default=0.01,
        help="Margin inside each URDF position limit in rad",
    )
    safety.add_argument(
        "--limit-scale",
        type=_positive,
        default=0.50,
        help="Fraction of URDF velocity/acceleration limits",
    )
    safety.add_argument(
        "--head-acceleration",
        type=_positive,
        default=5.0,
        help="Conservative rad/s^2 fallback absent from v1.2 URDF",
    )
    safety.add_argument(
        "--transition-seconds",
        type=_positive,
        default=8.0,
        help="Minimum first-pose transition time",
    )
    safety.add_argument(
        "--state-rate", type=_positive, default=100.0, help="SDK state callback rate in Hz"
    )
    safety.add_argument(
        "--state-timeout",
        type=_positive,
        default=0.20,
        help="Abort if state is older than this many seconds",
    )
    safety.add_argument(
        "--max-stationary-velocity",
        type=_positive,
        default=0.10,
        help="Required pre-motion peak joint velocity in rad/s",
    )
    safety.add_argument(
        "--stationary-seconds",
        type=_positive,
        default=0.25,
        help="Required continuously stationary duration",
    )
    safety.add_argument(
        "--stationary-timeout",
        type=_positive,
        default=3.0,
        help="Time allowed to establish a stationary state",
    )
    safety.add_argument(
        "--max-temperature",
        type=_positive,
        default=80.0,
        help="Local deployment cutoff in degrees C",
    )
    safety.add_argument(
        "--max-tracking-error",
        type=_positive,
        default=0.35,
        help="Maximum sustained upper-body error in rad",
    )
    safety.add_argument(
        "--tracking-error-duration",
        type=_positive,
        default=0.50,
        help="Allowed tracking-error duration in seconds",
    )
    safety.add_argument(
        "--max-wheel-velocity-error",
        type=_positive,
        default=3.0,
        help="Immediate wheel error cutoff in rad/s",
    )
    safety.add_argument(
        "--max-loop-lag",
        type=_positive,
        default=0.10,
        help="Abort threshold for command-loop lag in seconds",
    )
    safety.add_argument(
        "--hold-time",
        type=_positive,
        default=0.10,
        help="SDK command hold/watchdog time in seconds",
    )
    safety.add_argument(
        "--command-timeout-ms",
        type=int,
        default=200,
        help="SDK stream-send timeout in milliseconds",
    )
    safety.add_argument("--command-priority", type=int, default=10, help="SDK command priority")
    safety.add_argument("--countdown", type=int, default=5, help="Seconds before streaming starts")

    activation = parser.add_argument_group("explicit activation")
    activation.add_argument(
        "--auto-enable",
        action="store_true",
        help="Explicitly power on, servo on, and enable the control manager",
    )
    activation.add_argument(
        "--reset-faults",
        action="store_true",
        help="Explicitly reset a faulted control manager after inspection",
    )
    activation.add_argument(
        "--confirm-real-robot",
        metavar="TOKEN",
        help=f"Required for --target robot; exact token: {REAL_ROBOT_CONFIRMATION}",
    )
    activation.add_argument(
        "--allow-remote-simulator",
        action="store_true",
        help="Permit simulator mode with a non-loopback address",
    )

    output = parser.add_argument_group("logging")
    output.add_argument("--log", type=Path, help="Replay CSV path; otherwise an automatic path is used")
    output.add_argument("--no-log", action="store_true", help="Disable execution-state CSV logging")
    output.add_argument("--verbose", action="store_true")
    return parser


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit_scale > 1.0:
        parser.error("--limit-scale may not exceed 1.0")
    if args.command_timeout_ms <= 0:
        parser.error("--command-timeout-ms must be positive")
    if args.command_priority < 1:
        parser.error("--command-priority must be positive")
    if args.countdown < 0:
        parser.error("--countdown must be non-negative")
    if args.scale_absolute_pose and args.reference_mode != "absolute":
        parser.error("--scale-absolute-pose requires --reference-mode absolute")
    if args.return_seconds is not None and not args.return_to_zero:
        parser.error("--return-seconds requires --return-to-zero")
    if args.target == "robot" and args.countdown < 3:
        parser.error("real hardware requires --countdown of at least 3 seconds")
    if args.log is not None and args.no_log:
        parser.error("--log and --no-log are mutually exclusive")
    if args.target == "robot" and args.confirm_real_robot != REAL_ROBOT_CONFIRMATION:
        parser.error(
            f"real hardware requires --confirm-real-robot {REAL_ROBOT_CONFIRMATION}"
        )
    if (
        args.target == "simulator"
        and not _loopback_address(args.address)
        and not args.allow_remote_simulator
    ):
        parser.error(
            "simulator mode only accepts a loopback address by default; use "
            "--allow-remote-simulator for a known remote simulator"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        motion = load_motion(args.pkl.expanduser().resolve(), args.motion)
        limits = load_model_a_limits(args.urdf.expanduser().resolve(), args.head_acceleration)
        LOGGER.info(
            "Loaded %s: %d frames, %.3f fps, %.3fs source duration",
            motion.name,
            motion.positions.shape[0],
            motion.fps,
            motion.duration,
        )

        # Always expose problems in the original full-amplitude motion. These are
        # diagnostics, while the transformed trajectory below is the execution gate.
        raw = prepare_trajectory(
            motion,
            control_rate=args.control_rate,
            speed=1.0,
            reference_mode="absolute",
            start_positions=None,
            torso_scale=1.0,
            arm_scale=1.0,
            head_scale=1.0,
            enable_wheels=True,
            wheel_scale=1.0,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
        raw_issues = validate_trajectory(
            raw,
            limits,
            position_margin=0.0,
            dynamic_limit_scale=1.0,
            enable_wheels=True,
        )
        _print_issues("Original full-amplitude PKL (diagnostic only)", raw_issues, fatal=False)

        if args.target == "validate":
            prepared = prepare_trajectory(
                motion,
                control_rate=args.control_rate,
                speed=args.speed,
                reference_mode=args.reference_mode,
                start_positions=None,
                torso_scale=args.torso_scale,
                arm_scale=args.arm_scale,
                head_scale=args.head_scale,
                enable_wheels=args.enable_wheels,
                wheel_scale=args.wheel_scale,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                scale_absolute_pose=args.scale_absolute_pose,
            )
            if args.fit_start_pose:
                prepared, offsets = fit_trajectory_start_pose(
                    prepared,
                    limits,
                    position_margin=args.position_margin,
                )
                _log_start_pose_offsets(offsets)
            issues = validate_trajectory(
                prepared,
                limits,
                position_margin=args.position_margin,
                dynamic_limit_scale=args.limit_scale,
                enable_wheels=args.enable_wheels,
            )
            if args.reference_mode == "relative":
                baseline = "zero-pose baseline"
            elif args.scale_absolute_pose:
                baseline = "scaled absolute PKL"
            else:
                baseline = "absolute PKL"
            _print_issues(f"Prepared offline trajectory ({baseline})", issues, fatal=True)
            return_issues: list[ValidationIssue] = []
            if args.return_to_zero:
                return_seconds = (
                    args.transition_seconds
                    if args.return_seconds is None
                    else args.return_seconds
                )
                return_transition = prepare_upper_body_transition(
                    prepared.positions[-1],
                    np.zeros(24, dtype=np.float64),
                    limits,
                    control_rate=args.control_rate,
                    minimum_seconds=return_seconds,
                    dynamic_limit_scale=args.limit_scale,
                )
                return_issues = validate_trajectory(
                    return_transition,
                    limits,
                    position_margin=args.position_margin,
                    dynamic_limit_scale=args.limit_scale,
                    enable_wheels=args.enable_wheels,
                )
                _print_issues("Prepared return-to-zero transition", return_issues, fatal=True)
                LOGGER.info(
                    "Prepared return duration %.3fs (%d commands); wheels stop in place",
                    return_transition.duration,
                    len(return_transition.times),
                )
            LOGGER.info(
                "Prepared duration %.3fs (%d commands); wheels %s",
                prepared.duration,
                len(prepared.times),
                "ENABLED" if args.enable_wheels else "disabled",
            )
            if args.reference_mode == "relative":
                LOGGER.info(
                    "Execution will repeat validation around the measured robot pose before commanding"
                )
            return 2 if issues or return_issues else 0

        if args.target == "robot":
            LOGGER.warning("REAL RBY1 HARDWARE TARGET SELECTED")
        log_path = replay(args, motion, limits)
        if log_path is not None:
            LOGGER.info("Replay log: %s", log_path)
        return 0
    except ReplayError as exc:
        LOGGER.error("Replay refused/stopped: %s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
