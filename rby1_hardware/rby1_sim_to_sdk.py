#!/usr/bin/env python3
"""Mirror a running Isaac Lab RBY1 checkpoint through the vendor Python SDK.

Isaac remains the policy and observation/history source.  This process receives
the simulator's realized 24-DOF state, applies an explicit Model A mapping and
safety envelope, and streams commands using the same rby1-sdk primitives as the
guarded PKL runner.

The default target is ``shadow``: packets are validated and logged, but the SDK
is not imported and no robot is contacted.  Real hardware is opt-in and
requires the same confirmation token as the PKL runner.
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
import socket
import sys
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.rby1_sim_bridge import (  # noqa: E402
    BridgePacket,
    MAX_DATAGRAM_BYTES,
    RBY1_SDK_JOINT_NAMES,
    SimBridgeError,
    decode_packet,
    encode_packet,
    is_loopback_host,
    parse_endpoint,
)
from rby1_hardware.rby1_sdk_replay import (  # noqa: E402
    DEFAULT_URDF,
    HEAD_INDICES,
    LEFT_ARM_INDICES,
    REAL_ROBOT_CONFIRMATION,
    RIGHT_ARM_INDICES,
    RBY1_MODEL_A_JOINT_NAMES,
    ReplayError,
    ReplayTrajectory,
    StateMonitor,
    TORSO_INDICES,
    UPPER_BODY_INDICES,
    WHEEL_INDICES,
    _hold_target_until_settled,
    _loopback_address,
    _prepare_sdk_robot,
    _return_to_zero_pose,
    _safe_stop,
    _state_safety_check,
    _transition_to_first_pose,
    _wait_for_stationary_state,
    build_command,
    load_model_a_limits,
    validate_trajectory,
)


LOGGER = logging.getLogger("rby1_sim_to_sdk")
DEFAULT_LISTEN = "127.0.0.1:50070"
LOCAL_SIM_PACKET_TIMEOUT = 5.0
LOCAL_SIM_MAX_LOOP_LAG = 1.0
LOCAL_SIM_POSITION_MARGIN = 0.0
REAL_ROBOT_PACKET_TIMEOUT = 0.20
REAL_ROBOT_MAX_LOOP_LAG = 0.10
REAL_ROBOT_POSITION_MARGIN = 0.01
RELAXED_LIMITS_CONFIRMATION = "WARN_ONLY_RBY1_MODEL_A"


class HardwareBridgeError(RuntimeError):
    """The live bridge cannot continue safely."""


@dataclass(frozen=True)
class ProjectedTarget:
    positions: np.ndarray
    velocities: np.ndarray


class TargetProjector:
    """Project canonical Isaac state into a conservative hardware target."""

    def __init__(
        self,
        sim_initial: np.ndarray,
        hardware_initial: np.ndarray,
        *,
        reference_mode: str,
        torso_scale: float,
        arm_scale: float,
        head_scale: float,
        wheel_scale: float,
        enable_wheels: bool,
    ) -> None:
        self.sim_initial = self._vector(sim_initial, "simulator initial position")
        self.hardware_initial = self._vector(
            hardware_initial, "hardware initial position"
        )
        if reference_mode not in {"relative", "absolute"}:
            raise HardwareBridgeError(f"Unknown reference mode: {reference_mode!r}")
        self.reference_mode = reference_mode
        self.enable_wheels = enable_wheels
        self.scales = np.zeros(24, dtype=np.float64)
        self.scales[WHEEL_INDICES] = wheel_scale if enable_wheels else 0.0
        self.scales[TORSO_INDICES] = torso_scale
        self.scales[RIGHT_ARM_INDICES] = arm_scale
        self.scales[LEFT_ARM_INDICES] = arm_scale
        self.scales[HEAD_INDICES] = head_scale

    @staticmethod
    def _vector(values: np.ndarray, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (24,) or not np.isfinite(vector).all():
            raise HardwareBridgeError(f"{label} must be a finite 24-vector")
        return vector.copy()

    def project(self, positions: np.ndarray, velocities: np.ndarray) -> ProjectedTarget:
        source_positions = self._vector(positions, "simulator position")
        source_velocities = self._vector(velocities, "simulator velocity")
        if self.reference_mode == "relative":
            target_positions = self.hardware_initial + self.scales * (
                source_positions - self.sim_initial
            )
        else:
            target_positions = self.scales * source_positions

        # Mobility is velocity-controlled.  Preserve a finite measured wheel
        # position in logs/builders, but never treat accumulated wheel angle as
        # a position target.
        target_positions[WHEEL_INDICES] = self.hardware_initial[WHEEL_INDICES]
        target_velocities = self.scales * source_velocities
        if not self.enable_wheels:
            target_velocities[WHEEL_INDICES] = 0.0
        return ProjectedTarget(target_positions, target_velocities)


class TargetRateLimiter:
    """Convert projected simulator targets into a feasible SDK command path.

    Upper-body SDK commands are position targets with velocity and acceleration
    limits. Isaac's measured joint velocities are therefore not the command
    trajectory, and can contain a one-frame initialization spike. This limiter
    advances the commanded positions at the same scaled URDF limits that are
    supplied to the SDK command builder. Mobility remains velocity-controlled.
    """

    _DYNAMIC_MARGIN = 0.999

    def __init__(
        self,
        initial_target: ProjectedTarget,
        limits: Sequence[Any],
        *,
        position_margin: float,
        limit_scale: float,
        enable_wheels: bool,
        clamp_position_limits: bool = False,
    ) -> None:
        if len(limits) != len(RBY1_MODEL_A_JOINT_NAMES):
            raise HardwareBridgeError(f"Expected 24 joint limits, got {len(limits)}")
        if not 0.0 < limit_scale <= 1.0:
            raise HardwareBridgeError("Dynamic limit scale must be in (0, 1]")
        if position_margin < 0.0 or not math.isfinite(position_margin):
            raise HardwareBridgeError("Position margin must be finite and non-negative")

        self._limits = tuple(limits)
        self._position_margin = position_margin
        self._enable_wheels = enable_wheels
        self._clamp_position_limits = bool(clamp_position_limits)
        self._warned_clamped_joints: set[int] = set()
        self._positions = self._constrain_positions(
            np.asarray(initial_target.positions, dtype=np.float64),
            "Initial projected target",
        )
        self._velocities = np.zeros(24, dtype=np.float64)
        self._sim_time = 0.0
        self._velocity_limits = np.asarray(
            [limit.velocity for limit in self._limits], dtype=np.float64
        ) * (limit_scale * self._DYNAMIC_MARGIN)
        self._acceleration_limits = np.asarray(
            [limit.acceleration for limit in self._limits], dtype=np.float64
        ) * (limit_scale * self._DYNAMIC_MARGIN)

    @property
    def current_target(self) -> ProjectedTarget:
        return ProjectedTarget(self._positions.copy(), self._velocities.copy())

    def _constrain_positions(self, positions: np.ndarray, label: str) -> np.ndarray:
        if positions.shape != (24,) or not np.isfinite(positions).all():
            raise HardwareBridgeError(f"{label} must be a finite 24-vector")
        constrained = positions.copy()
        for index in UPPER_BODY_INDICES:
            limit = self._limits[int(index)]
            lower = None if limit.lower is None else limit.lower + self._position_margin
            upper = None if limit.upper is None else limit.upper - self._position_margin
            if lower is not None and upper is not None and lower > upper:
                raise HardwareBridgeError(
                    f"Position margin leaves no valid range for "
                    f"{RBY1_MODEL_A_JOINT_NAMES[index]}"
                )
            value = float(constrained[index])
            if lower is not None and value < lower:
                if not self._clamp_position_limits:
                    raise HardwareBridgeError(
                        f"{label} puts {RBY1_MODEL_A_JOINT_NAMES[index]} at "
                        f"{value:.5f}rad, below the safe lower limit {lower:.5f}rad"
                    )
                constrained[index] = lower
                self._warn_position_clamp(int(index), value, lower, "lower")
            elif upper is not None and value > upper:
                if not self._clamp_position_limits:
                    raise HardwareBridgeError(
                        f"{label} puts {RBY1_MODEL_A_JOINT_NAMES[index]} at "
                        f"{value:.5f}rad, above the safe upper limit {upper:.5f}rad"
                    )
                constrained[index] = upper
                self._warn_position_clamp(int(index), value, upper, "upper")
        return constrained

    def _warn_position_clamp(
        self,
        index: int,
        requested: float,
        bounded: float,
        bound_name: str,
    ) -> None:
        if index in self._warned_clamped_joints:
            return
        self._warned_clamped_joints.add(index)
        LOGGER.warning(
            "Clamping local simulator target for %s from %.5frad to the vendor %s "
            "bound %.5frad; this checkpoint pose cannot be reproduced exactly",
            RBY1_MODEL_A_JOINT_NAMES[index],
            requested,
            bound_name,
            bounded,
        )

    def update(self, desired: ProjectedTarget, sim_time: float) -> ProjectedTarget:
        sim_time = float(sim_time)
        if not math.isfinite(sim_time):
            raise HardwareBridgeError("Simulator target time must be finite")
        dt = sim_time - self._sim_time
        if dt <= 0.0:
            raise HardwareBridgeError(
                f"Simulator target time did not increase: {self._sim_time:.6f} -> "
                f"{sim_time:.6f}"
            )

        desired_positions = self._constrain_positions(
            np.asarray(desired.positions, dtype=np.float64),
            "Projected simulator target",
        )
        desired_velocities = np.asarray(desired.velocities, dtype=np.float64)
        if desired_velocities.shape != (24,) or not np.isfinite(desired_velocities).all():
            raise HardwareBridgeError("Projected simulator velocity must be a finite 24-vector")

        # Position-controlled joints chase the current desired pose. Their
        # velocity changes are explicitly bounded before integration, so the
        # resulting target path is feasible even when Isaac reports a transient
        # measured-velocity spike.
        requested_velocities = (desired_positions - self._positions) / dt
        requested_velocities[WHEEL_INDICES] = (
            desired_velocities[WHEEL_INDICES] if self._enable_wheels else 0.0
        )
        requested_velocities = np.clip(
            requested_velocities,
            -self._velocity_limits,
            self._velocity_limits,
        )
        maximum_delta = self._acceleration_limits * dt
        next_velocities = self._velocities + np.clip(
            requested_velocities - self._velocities,
            -maximum_delta,
            maximum_delta,
        )
        next_positions = self._positions + next_velocities * dt

        # Wheel position is not an SDK command. Keep its measured baseline while
        # the optional velocity target follows the same acceleration envelope.
        next_positions[WHEEL_INDICES] = self._positions[WHEEL_INDICES]
        if not self._enable_wheels:
            next_velocities[WHEEL_INDICES] = 0.0

        self._positions = self._constrain_positions(
            next_positions,
            "Rate-limited SDK target",
        )
        self._velocities = next_velocities
        self._sim_time = sim_time
        return self.current_target


class LiveTargetValidator:
    """Apply the PKL runner's URDF position/velocity/acceleration envelope live."""

    def __init__(
        self,
        limits: Sequence[Any],
        *,
        position_margin: float,
        limit_scale: float,
        enable_wheels: bool,
        enforce: bool = True,
    ) -> None:
        self.limits = limits
        self.position_margin = position_margin
        self.limit_scale = limit_scale
        self.enable_wheels = enable_wheels
        self.enforce = bool(enforce)
        self._previous_velocity: np.ndarray | None = None
        self._previous_sim_time: float | None = None
        self._warned = False

    def validate(self, packet: BridgePacket, target: ProjectedTarget) -> None:
        if packet.sim_time is None:
            raise HardwareBridgeError("State packet is missing sim_time")
        if self._previous_velocity is None:
            acceleration = np.zeros(24, dtype=np.float64)
        else:
            assert self._previous_sim_time is not None
            dt = packet.sim_time - self._previous_sim_time
            if dt <= 0.0 or not math.isfinite(dt):
                raise HardwareBridgeError(
                    f"Simulator time did not increase: dt={dt!r} at sequence {packet.sequence}"
                )
            acceleration = (target.velocities - self._previous_velocity) / dt

        trajectory = ReplayTrajectory(
            times=np.asarray([packet.sim_time], dtype=np.float64),
            source_frames=np.asarray([float(packet.sequence)], dtype=np.float64),
            positions=target.positions[np.newaxis, :],
            velocities=target.velocities[np.newaxis, :],
            accelerations=acceleration[np.newaxis, :],
        )
        issues = validate_trajectory(
            trajectory,
            self.limits,
            position_margin=self.position_margin,
            dynamic_limit_scale=self.limit_scale,
            enable_wheels=self.enable_wheels,
        )
        if issues:
            details = "; ".join(str(issue) for issue in issues[:4])
            suffix = f"; plus {len(issues) - 4} more" if len(issues) > 4 else ""
            message = f"Simulator target violates the hardware envelope: {details}{suffix}"
            if self.enforce:
                raise HardwareBridgeError(message)
            if not self._warned:
                LOGGER.warning("%s; continuing under warning-only soft-limit policy", message)
                self._warned = True
        self._previous_velocity = target.velocities.copy()
        self._previous_sim_time = packet.sim_time


def _positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


def _nonnegative(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative and finite")
    return result


def _receive_hello(
    udp_socket: socket.socket,
    *,
    timeout: float,
    allow_remote_source: bool,
) -> tuple[BridgePacket, tuple[str, int]]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise HardwareBridgeError(
                f"No Isaac bridge hello packet arrived within {timeout:.1f}s"
            )
        udp_socket.settimeout(min(0.5, remaining))
        try:
            data, source = udp_socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
        except socket.timeout:
            continue
        if not allow_remote_source and not is_loopback_host(source[0]):
            LOGGER.warning("Ignoring bridge packet from non-loopback source %s:%d", *source)
            continue
        try:
            packet = decode_packet(data)
        except SimBridgeError as exc:
            LOGGER.warning("Ignoring invalid bridge hello datagram: %s", exc)
            continue
        if packet.kind == "hello":
            return packet, source


def _send_ready(
    udp_socket: socket.socket,
    source: tuple[str, int],
    session_id: str,
) -> None:
    udp_socket.sendto(encode_packet("ready", session_id, 0), source)


def _receive_session_packet(
    udp_socket: socket.socket,
    *,
    source: tuple[str, int],
    session_id: str,
    timeout: float,
    return_none_on_timeout: bool = False,
) -> BridgePacket | None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            if return_none_on_timeout:
                return None
            raise HardwareBridgeError(
                f"No simulator state packet arrived within {timeout:.3f}s"
            )
        udp_socket.settimeout(remaining)
        try:
            data, packet_source = udp_socket.recvfrom(MAX_DATAGRAM_BYTES + 1)
        except socket.timeout as exc:
            if return_none_on_timeout:
                return None
            raise HardwareBridgeError(
                f"No simulator state packet arrived within {timeout:.3f}s"
            ) from exc
        if packet_source != source:
            LOGGER.warning(
                "Ignoring bridge packet from unexpected source %s:%d", *packet_source
            )
            continue
        try:
            packet = decode_packet(data)
        except SimBridgeError as exc:
            raise HardwareBridgeError(f"Invalid packet from active Isaac source: {exc}") from exc
        if packet.session_id != session_id:
            raise HardwareBridgeError(
                "Isaac bridge session changed while commands were active; stopping"
            )
        if packet.kind == "hello":
            _send_ready(udp_socket, source, session_id)
            continue
        if packet.kind not in {"state", "stop"}:
            continue
        return packet


def _receive_session_packet_with_heartbeat(
    udp_socket: socket.socket,
    *,
    source: tuple[str, int],
    session_id: str,
    timeout: float,
    heartbeat_interval: float,
    heartbeat: Callable[[], None],
    stop_requested: threading.Event,
    enforce_timeout: bool = True,
) -> BridgePacket:
    """Wait for a packet while refreshing the finite SDK command-stream lease."""
    if heartbeat_interval <= 0.0 or not math.isfinite(heartbeat_interval):
        raise HardwareBridgeError("SDK hold heartbeat interval must be finite and positive")

    deadline = time.monotonic() + timeout
    next_heartbeat = time.monotonic() + heartbeat_interval
    timeout_warnings = 0
    while True:
        if stop_requested.is_set():
            raise HardwareBridgeError("Bridge cancelled while waiting for simulator state")

        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0.0:
            if enforce_timeout:
                raise HardwareBridgeError(
                    f"No simulator state packet arrived within {timeout:.3f}s"
                )
            timeout_warnings += 1
            LOGGER.warning(
                "No simulator state packet arrived within %.3fs; holding the latest "
                "upper-body target, commanding zero wheel velocity, and continuing "
                "under warning-only soft-limit policy (warning %d)",
                timeout,
                timeout_warnings,
            )
            deadline = now + timeout
            continue

        poll_timeout = min(
            deadline - time.monotonic(),
            max(0.001, next_heartbeat - time.monotonic()),
        )
        if poll_timeout <= 0.0:
            continue
        packet = _receive_session_packet(
            udp_socket,
            source=source,
            session_id=session_id,
            timeout=poll_timeout,
            return_none_on_timeout=True,
        )
        if packet is not None:
            return packet
        heartbeat()
        # Do not issue catch-up bursts if an SDK call or the OS scheduler ran late.
        next_heartbeat = time.monotonic() + heartbeat_interval


def _apply_target_defaults(args: argparse.Namespace) -> None:
    """Use relaxed local-test defaults without weakening real-hardware defaults."""
    if getattr(args, "soft_limit_policy", None) is None:
        args.soft_limit_policy = "enforce" if args.target == "robot" else "warn"
    enforce = args.soft_limit_policy == "enforce"
    if getattr(args, "packet_timeout", None) is None:
        args.packet_timeout = (
            REAL_ROBOT_PACKET_TIMEOUT
            if args.target == "robot" and enforce
            else LOCAL_SIM_PACKET_TIMEOUT
        )
    if getattr(args, "max_loop_lag", None) is None:
        args.max_loop_lag = (
            REAL_ROBOT_MAX_LOOP_LAG
            if args.target == "robot" and enforce
            else LOCAL_SIM_MAX_LOOP_LAG
        )
    if getattr(args, "position_margin", None) is None:
        args.position_margin = (
            REAL_ROBOT_POSITION_MARGIN
            if args.target == "robot" and enforce
            else LOCAL_SIM_POSITION_MARGIN
        )
    if getattr(args, "position_limit_mode", None) is None:
        args.position_limit_mode = "reject" if enforce else "clamp"


def _resolve_packet_clock(
    *,
    bridge_start: float,
    sim_time: float,
    now: float,
    max_loop_lag: float,
    target: str,
    enforce: bool,
) -> tuple[float, float, float]:
    """Enforce accumulated timing lag or warn and rebase the wall clock."""
    deadline = bridge_start + sim_time
    lag = now - deadline
    if lag <= max_loop_lag:
        return bridge_start, deadline, lag
    if enforce:
        raise HardwareBridgeError(
            f"SDK bridge missed simulator deadline by {lag:.3f}s "
            f"(limit {max_loop_lag:.3f}s)"
        )

    LOGGER.warning(
        "%s Isaac bridge is %.3fs behind its simulated clock; rebasing the "
        "wall-clock schedule under warning-only soft-limit policy",
        target,
        lag,
    )
    bridge_start = now - sim_time
    return bridge_start, now, 0.0


def _bridge_log_path(args: argparse.Namespace) -> Path | None:
    if args.no_log:
        return None
    if args.log is not None:
        return args.log.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "bridge_logs" / f"sim_to_{args.target}_{stamp}.csv"


def _csv_header(include_measured: bool) -> list[str]:
    header = [
        "sequence",
        "sim_time",
        "receive_time",
        "packet_lag",
        "tracking_error",
        "rate_limiter_lag",
    ]
    header.extend(f"sim_pos_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"sim_vel_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"target_pos_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    header.extend(f"target_vel_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    if include_measured:
        header.extend(f"measured_pos_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
        header.extend(f"measured_vel_{name}" for name in RBY1_MODEL_A_JOINT_NAMES)
    return header


def _validate_packet_rate(packet: BridgePacket, expected_rate: float) -> None:
    if packet.control_hz is None:
        raise HardwareBridgeError("Isaac hello packet did not include a control rate")
    tolerance = max(0.05, expected_rate * 0.005)
    if abs(packet.control_hz - expected_rate) > tolerance:
        raise HardwareBridgeError(
            f"Isaac publishes at {packet.control_hz:.3f} Hz but the SDK bridge is "
            f"configured for {expected_rate:.3f} Hz (tolerance {tolerance:.3f} Hz)"
        )


def _check_sequence(
    packet: BridgePacket,
    *,
    previous_sequence: int,
    max_sequence_gap: int,
) -> None:
    if packet.sequence <= previous_sequence:
        raise HardwareBridgeError(
            f"Simulator sequence did not increase ({previous_sequence} -> {packet.sequence})"
        )
    gap = packet.sequence - previous_sequence
    if gap > max_sequence_gap:
        raise HardwareBridgeError(
            f"Lost {gap - 1} consecutive simulator frame(s); maximum allowed gap is "
            f"{max_sequence_gap - 1}"
        )


def run_bridge(args: argparse.Namespace) -> None:
    _apply_target_defaults(args)
    if tuple(RBY1_MODEL_A_JOINT_NAMES) != RBY1_SDK_JOINT_NAMES:
        raise HardwareBridgeError("SDK replay and simulator bridge joint contracts disagree")

    limits = load_model_a_limits(args.urdf.expanduser().resolve(), args.head_acceleration)
    listen_address = parse_endpoint(args.listen)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(listen_address)
    LOGGER.info("Listening for paced Isaac RBY1 state on %s:%d", *listen_address)
    if args.target == "shadow":
        LOGGER.warning("SHADOW MODE: rby1-sdk will not be imported and no commands will be sent")

    robot = None
    monitor = None
    stream = None
    rby = None
    log_file = None
    log_writer = None
    stop_requested = threading.Event()
    last_target: ProjectedTarget | None = None
    normal_completion = False
    active_source: tuple[str, int] | None = None
    active_session: str | None = None

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = None
    if hasattr(signal, "SIGTERM"):
        old_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        hello, source = _receive_hello(
            udp_socket,
            timeout=args.source_timeout,
            allow_remote_source=args.allow_remote_source,
        )
        active_source = source
        active_session = hello.session_id
        _validate_packet_rate(hello, args.control_rate)
        assert hello.positions is not None and hello.velocities is not None
        LOGGER.info(
            "Accepted Isaac session %s from %s:%d at %.1f Hz",
            hello.session_id,
            source[0],
            source[1],
            hello.control_hz,
        )

        if args.target == "shadow":
            hardware_initial = np.zeros(24, dtype=np.float64)
        else:
            try:
                import rby1_sdk as rby_module
            except ImportError as exc:
                raise HardwareBridgeError(
                    "rby1_sdk is not importable. Install "
                    "rby1_hardware/requirements-sdk.txt in the SDK venv."
                ) from exc
            rby = rby_module
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
            hardware_initial = initial.positions.copy()

        projector = TargetProjector(
            hello.positions,
            hardware_initial,
            reference_mode=args.reference_mode,
            torso_scale=args.torso_scale,
            arm_scale=args.arm_scale,
            head_scale=args.head_scale,
            wheel_scale=args.wheel_scale,
            enable_wheels=args.enable_wheels,
        )
        desired_first_target = projector.project(hello.positions, hello.velocities)
        target_limiter = TargetRateLimiter(
            desired_first_target,
            limits,
            position_margin=args.position_margin,
            limit_scale=args.limit_scale,
            enable_wheels=args.enable_wheels,
            clamp_position_limits=args.position_limit_mode == "clamp",
        )
        validator = LiveTargetValidator(
            limits,
            position_margin=args.position_margin,
            limit_scale=args.limit_scale,
            enable_wheels=args.enable_wheels,
            enforce=args.soft_limit_policy == "enforce",
        )
        first_target = target_limiter.current_target
        validator.validate(hello, first_target)
        last_target = first_target

        LOGGER.info(
            "Mode=%s, scales: torso=%.3f arms=%.3f head=%.3f wheels=%s",
            args.reference_mode,
            args.torso_scale,
            args.arm_scale,
            args.head_scale,
            f"{args.wheel_scale:.3f}" if args.enable_wheels else "DISABLED",
        )
        LOGGER.info(
            "SDK command targets are rate-limited to %.1f%% of vendor velocity/acceleration limits",
            args.limit_scale * 100.0,
        )
        LOGGER.info(
            "Position envelope uses the vendor URDF bounds with a %.4frad inward margin "
            "and %s out-of-range targets",
            args.position_margin,
            args.position_limit_mode,
        )
        LOGGER.info("Soft position/dynamic/timing limit policy: %s", args.soft_limit_policy)
        if args.soft_limit_policy == "warn":
            LOGGER.warning(
                "WARNING-ONLY SOFT LIMITS: position targets will be clamped and timing "
                "gaps may hold/rebase instead of aborting"
            )

        if args.target != "shadow":
            assert robot is not None and monitor is not None and rby is not None
            if args.countdown > 0:
                LOGGER.warning(
                    "Live bridge activation begins in %d seconds; keep the E-stop in hand",
                    args.countdown,
                )
                for remaining in range(args.countdown, 0, -1):
                    LOGGER.info("Starting in %d...", remaining)
                    if stop_requested.wait(1.0):
                        raise HardwareBridgeError("Bridge cancelled during countdown")

            stream = robot.create_command_stream(args.command_priority)
            initial_target_error = float(
                np.max(
                    np.abs(
                        hardware_initial[UPPER_BODY_INDICES]
                        - first_target.positions[UPPER_BODY_INDICES]
                    )
                )
            )
            if initial_target_error <= args.max_tracking_error:
                _hold_target_until_settled(
                    stream,
                    rby,
                    monitor,
                    first_target.positions,
                    limits,
                    args,
                    stop_requested,
                    target_name="Initial bridge target",
                )
            else:
                _transition_to_first_pose(
                    stream,
                    rby,
                    monitor,
                    hardware_initial,
                    first_target.positions,
                    limits,
                    args,
                    stop_requested,
                )

        log_path = _bridge_log_path(args)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            log_writer.writerow(_csv_header(args.target != "shadow"))
            LOGGER.info("Writing bridge state log to %s", log_path)

        # Isaac remains frozen at reset until this acknowledgement.  If a
        # hardware transition was needed, it has completed above.
        _send_ready(udp_socket, source, hello.session_id)
        LOGGER.info("Receiver ready; Isaac policy episode may now advance")

        ready_time = time.monotonic()
        bridge_start: float | None = None
        previous_sequence = hello.sequence
        tracking_error_since: float | None = None
        peak_rate_limiter_lag = 0.0
        last_flush = ready_time

        hold_heartbeat_interval = min(
            1.0 / args.control_rate,
            args.hold_time * 0.5,
        )

        def refresh_last_target_hold() -> None:
            """Hold the latest accepted target while Isaac is between packets."""
            assert monitor is not None and rby is not None and stream is not None
            if last_target is None:
                raise HardwareBridgeError("No accepted SDK target is available to hold")
            snapshot = monitor.latest()
            if snapshot is None:
                raise HardwareBridgeError(
                    "No current SDK state is available while waiting for Isaac state"
                )
            _state_safety_check(
                snapshot,
                now=time.monotonic(),
                rby_module=rby,
                state_timeout=args.state_timeout,
                max_temperature=args.max_temperature,
                enable_wheels=args.enable_wheels,
            )
            command = build_command(
                rby,
                last_target.positions,
                # A delayed source packet must never leave mobility coasting.
                np.zeros(2, dtype=np.float64),
                limits,
                minimum_time=max(0.005, 1.02 / args.control_rate),
                hold_time=args.hold_time,
                dynamic_limit_scale=args.limit_scale,
                enable_wheels=args.enable_wheels,
            )
            stream.send_command(command, timeout_ms=args.command_timeout_ms)

        if args.target != "shadow":
            LOGGER.info(
                "Refreshing the latest target at %.1f Hz during packet gaps "
                "(first timeout %.1fs, active timeout %.1fs)",
                1.0 / hold_heartbeat_interval,
                args.first_packet_timeout,
                args.packet_timeout,
            )

        while not stop_requested.is_set():
            receive_timeout = (
                args.first_packet_timeout if bridge_start is None else args.packet_timeout
            )
            if args.target != "shadow":
                packet = _receive_session_packet_with_heartbeat(
                    udp_socket,
                    source=source,
                    session_id=hello.session_id,
                    timeout=receive_timeout,
                    heartbeat_interval=hold_heartbeat_interval,
                    heartbeat=refresh_last_target_hold,
                    stop_requested=stop_requested,
                    enforce_timeout=args.soft_limit_policy == "enforce",
                )
            else:
                packet = _receive_session_packet(
                    udp_socket,
                    source=source,
                    session_id=hello.session_id,
                    timeout=receive_timeout,
                )
                assert packet is not None
            if packet.kind == "stop":
                if not packet.normal_completion:
                    raise HardwareBridgeError(
                        f"Isaac publisher aborted the bridge: {packet.reason}"
                    )
                LOGGER.info("Isaac completed the bridged episode: %s", packet.reason)
                normal_completion = True
                break
            _check_sequence(
                packet,
                previous_sequence=previous_sequence,
                max_sequence_gap=args.max_sequence_gap,
            )
            _validate_packet_rate(packet, args.control_rate)
            assert packet.positions is not None and packet.velocities is not None
            assert packet.sim_time is not None
            desired_target = projector.project(packet.positions, packet.velocities)
            target = target_limiter.update(desired_target, packet.sim_time)
            validator.validate(packet, target)
            rate_limiter_lag = float(
                np.max(
                    np.abs(
                        desired_target.positions[UPPER_BODY_INDICES]
                        - target.positions[UPPER_BODY_INDICES]
                    )
                )
            )
            peak_rate_limiter_lag = max(peak_rate_limiter_lag, rate_limiter_lag)

            received_at = time.monotonic()
            if bridge_start is None:
                # Hold the settled first target while Isaac performs its first
                # policy/physics step. Anchor both sides' real-time clocks to
                # this first realized state; active-session watchdogs apply
                # to every packet after it.
                bridge_start = received_at - packet.sim_time
                LOGGER.info(
                    "First simulator state arrived %.3fs after ready; "
                    "steady-state %.3fs packet watchdog is now active",
                    received_at - ready_time,
                    args.packet_timeout,
                )

            deadline = bridge_start + packet.sim_time
            remaining = deadline - time.monotonic()
            if remaining > 0.0 and stop_requested.wait(remaining):
                break
            now = time.monotonic()
            bridge_start, deadline, lag = _resolve_packet_clock(
                bridge_start=bridge_start,
                sim_time=packet.sim_time,
                now=now,
                max_loop_lag=args.max_loop_lag,
                target=args.target,
                enforce=args.soft_limit_policy == "enforce",
            )

            tracking_error = 0.0
            snapshot = None
            if args.target != "shadow":
                assert monitor is not None and rby is not None and stream is not None
                snapshot = monitor.latest()
                if snapshot is None:
                    raise HardwareBridgeError("No current SDK state is available")
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
                    - target.positions[UPPER_BODY_INDICES]
                )
                tracking_offset = int(np.argmax(tracking_errors))
                tracking_index = int(UPPER_BODY_INDICES[tracking_offset])
                tracking_error = float(tracking_errors[tracking_offset])
                if tracking_error > args.max_tracking_error:
                    if tracking_error_since is None:
                        tracking_error_since = now
                    elif now - tracking_error_since > args.tracking_error_duration:
                        raise HardwareBridgeError(
                            f"{RBY1_MODEL_A_JOINT_NAMES[tracking_index]} tracking error "
                            f"remained at {tracking_error:.3f}rad for more than "
                            f"{args.tracking_error_duration:.2f}s"
                        )
                else:
                    tracking_error_since = None

                if args.enable_wheels:
                    wheel_error = float(
                        np.max(
                            np.abs(
                                snapshot.velocities[WHEEL_INDICES]
                                - target.velocities[WHEEL_INDICES]
                            )
                        )
                    )
                    if wheel_error > args.max_wheel_velocity_error:
                        raise HardwareBridgeError(
                            f"Wheel velocity tracking error is {wheel_error:.3f}rad/s "
                            f"(limit {args.max_wheel_velocity_error:.3f}rad/s)"
                        )

                command = build_command(
                    rby,
                    target.positions,
                    target.velocities[WHEEL_INDICES],
                    limits,
                    minimum_time=max(0.005, 1.02 / args.control_rate),
                    hold_time=args.hold_time,
                    dynamic_limit_scale=args.limit_scale,
                    enable_wheels=args.enable_wheels,
                )
                stream.send_command(command, timeout_ms=args.command_timeout_ms)

            if log_writer is not None:
                row: list[Any] = [
                    packet.sequence,
                    packet.sim_time,
                    now - bridge_start,
                    lag,
                    tracking_error,
                    rate_limiter_lag,
                    *packet.positions.tolist(),
                    *packet.velocities.tolist(),
                    *target.positions.tolist(),
                    *target.velocities.tolist(),
                ]
                if snapshot is not None:
                    row.extend(snapshot.positions.tolist())
                    row.extend(snapshot.velocities.tolist())
                log_writer.writerow(row)
                if now - last_flush >= 1.0 and log_file is not None:
                    log_file.flush()
                    last_flush = now

            previous_sequence = packet.sequence
            last_target = target

        if normal_completion:
            LOGGER.info(
                "Peak rate-limiter lag behind the projected simulator pose: %.3frad",
                peak_rate_limiter_lag,
            )

        if stop_requested.is_set():
            raise HardwareBridgeError("Bridge cancelled by operator")

        if (
            normal_completion
            and args.return_to_zero
            and args.target != "shadow"
            and last_target is not None
        ):
            assert stream is not None and rby is not None and monitor is not None
            _return_to_zero_pose(
                stream,
                rby,
                monitor,
                last_target.positions,
                limits,
                args,
                stop_requested,
            )
    except BaseException as exc:
        if active_source is not None and active_session is not None:
            try:
                abort_packet = encode_packet(
                    "stop",
                    active_session,
                    0,
                    reason=f"SDK receiver safety stop: {exc}",
                    normal_completion=False,
                )
                for _ in range(3):
                    udp_socket.sendto(abort_packet, active_source)
            except OSError:
                pass
        raise
    finally:
        if log_file is not None:
            log_file.flush()
            log_file.close()
        if robot is not None and rby is not None:
            _safe_stop(robot, rby, stream, monitor, limits, args)
            if monitor is not None:
                try:
                    robot.stop_state_update()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("Could not stop SDK state updates: %s", exc)
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Could not disconnect from SDK server: %s", exc)
        udp_socket.close()
        signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive a paced 24-joint RBY1 state from Isaac Lab and optionally "
            "stream it through the vendor SDK."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target",
        choices=("shadow", "simulator", "robot"),
        default="shadow",
        help="shadow only validates/logs; simulator and robot send SDK commands",
    )
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help="Isaac UDP listen HOST:PORT")
    parser.add_argument("--address", default="127.0.0.1:50051", help="rby1-sdk server address")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="Bundled vendor Model A v1.2 URDF used for safety limits",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("relative", "absolute"),
        default="relative",
        help=(
            "relative mirrors simulator deltas about the measured start; absolute mirrors "
            "scaled SDK-coordinate simulator poses"
        ),
    )
    parser.add_argument("--control-rate", type=_positive, default=50.0)
    parser.add_argument("--torso-scale", type=_nonnegative, default=0.10)
    parser.add_argument("--arm-scale", type=_nonnegative, default=0.25)
    parser.add_argument("--head-scale", type=_nonnegative, default=0.10)
    parser.add_argument("--enable-wheels", action="store_true", help="Enable mobility commands")
    parser.add_argument("--wheel-scale", type=_nonnegative, default=0.10)
    parser.add_argument(
        "--return-to-zero",
        action="store_true",
        help="After normal episode completion, return all 22 upper-body joints to 0 rad",
    )
    parser.add_argument(
        "--return-seconds",
        type=_positive,
        help="Minimum zero-return time; defaults to --transition-seconds",
    )

    network = parser.add_argument_group("bridge transport")
    network.add_argument(
        "--source-timeout",
        type=_positive,
        default=120.0,
        help="Seconds to wait for the initial Isaac hello",
    )
    network.add_argument(
        "--packet-timeout",
        type=_positive,
        default=argparse.SUPPRESS,
        help=(
            "Abort if an active Isaac session stops publishing; defaults to 5.0s for "
            "shadow/simulator and 0.2s for robot"
        ),
    )
    network.add_argument(
        "--first-packet-timeout",
        type=_positive,
        default=5.0,
        help=(
            "Startup-only wait for Isaac's first policy/physics state; the stricter "
            "--packet-timeout applies after it arrives"
        ),
    )
    network.add_argument(
        "--max-sequence-gap",
        type=int,
        default=2,
        help="Largest accepted sequence increment (2 tolerates one lost UDP frame)",
    )
    network.add_argument(
        "--allow-remote-source",
        action="store_true",
        help="Accept an Isaac publisher outside loopback on an isolated trusted network",
    )

    safety = parser.add_argument_group("safety envelope")
    safety.add_argument(
        "--soft-limit-policy",
        choices=("enforce", "warn"),
        default=argparse.SUPPRESS,
        help=(
            "Whether projected position/dynamic and bridge timing limit violations abort "
            "or warn/clamp/rebase; defaults to warn for shadow/simulator and enforce for robot"
        ),
    )
    safety.add_argument(
        "--position-margin",
        type=_nonnegative,
        default=argparse.SUPPRESS,
        help=(
            "Inward margin from each vendor URDF position bound; defaults to 0 for "
            "shadow/simulator and 0.01rad for robot"
        ),
    )
    safety.add_argument(
        "--position-limit-mode",
        choices=("reject", "clamp"),
        default=argparse.SUPPRESS,
        help=(
            "Handling for finite targets outside the URDF position range; defaults to "
            "clamp for shadow/simulator and reject for robot"
        ),
    )
    safety.add_argument("--limit-scale", type=_positive, default=0.50)
    safety.add_argument("--head-acceleration", type=_positive, default=5.0)
    safety.add_argument("--transition-seconds", type=_positive, default=8.0)
    safety.add_argument("--state-rate", type=_positive, default=100.0)
    safety.add_argument("--state-timeout", type=_positive, default=0.20)
    safety.add_argument("--max-stationary-velocity", type=_positive, default=0.10)
    safety.add_argument("--stationary-seconds", type=_positive, default=0.25)
    safety.add_argument("--stationary-timeout", type=_positive, default=6.0)
    safety.add_argument("--max-temperature", type=_positive, default=80.0)
    safety.add_argument("--max-tracking-error", type=_positive, default=0.35)
    safety.add_argument("--tracking-error-duration", type=_positive, default=0.50)
    safety.add_argument("--max-wheel-velocity-error", type=_positive, default=3.0)
    safety.add_argument(
        "--max-loop-lag",
        type=_positive,
        default=argparse.SUPPRESS,
        help=(
            "Maximum accumulated wall-clock lag; defaults to 1.0s for "
            "shadow/simulator and 0.1s for robot"
        ),
    )
    safety.add_argument("--hold-time", type=_positive, default=0.10)
    safety.add_argument("--command-timeout-ms", type=int, default=200)
    safety.add_argument("--command-priority", type=int, default=10)
    safety.add_argument("--countdown", type=int, default=5)

    activation = parser.add_argument_group("explicit SDK activation")
    activation.add_argument("--auto-enable", action="store_true")
    activation.add_argument("--reset-faults", action="store_true")
    activation.add_argument(
        "--confirm-real-robot",
        metavar="TOKEN",
        help=f"Required for --target robot; exact token: {REAL_ROBOT_CONFIRMATION}",
    )
    activation.add_argument(
        "--confirm-relaxed-limits",
        metavar="TOKEN",
        help=(
            "Additional acknowledgement required for --target robot "
            f"--soft-limit-policy warn; exact token: {RELAXED_LIMITS_CONFIRMATION}"
        ),
    )
    activation.add_argument("--allow-remote-simulator", action="store_true")

    output = parser.add_argument_group("logging")
    output.add_argument("--log", type=Path)
    output.add_argument("--no-log", action="store_true")
    output.add_argument("--verbose", action="store_true")
    return parser


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _apply_target_defaults(args)
    try:
        listen_host, _listen_port = parse_endpoint(args.listen)
    except SimBridgeError as exc:
        parser.error(str(exc))
    if not args.allow_remote_source and not is_loopback_host(listen_host):
        parser.error(
            "--listen must use loopback unless --allow-remote-source is explicitly selected"
        )
    if args.limit_scale > 1.0:
        parser.error("--limit-scale may not exceed 1.0")
    for name in ("torso_scale", "arm_scale", "head_scale", "wheel_scale"):
        if getattr(args, name) > 1.0:
            parser.error(f"--{name.replace('_', '-')} may not exceed 1.0")
    if args.max_sequence_gap < 1:
        parser.error("--max-sequence-gap must be at least 1")
    if args.command_timeout_ms <= 0 or args.command_priority <= 0:
        parser.error("SDK command timeout and priority must be positive")
    if args.countdown < 0:
        parser.error("--countdown must be non-negative")
    if args.return_seconds is not None and not args.return_to_zero:
        parser.error("--return-seconds requires --return-to-zero")
    if args.target == "shadow" and args.auto_enable:
        parser.error("--auto-enable is invalid in shadow mode")
    if args.target == "shadow" and args.return_to_zero:
        parser.error("--return-to-zero is invalid in shadow mode")
    if args.target == "robot":
        if args.soft_limit_policy == "enforce" and args.position_limit_mode != "reject":
            parser.error(
                "real hardware with enforced soft limits requires --position-limit-mode reject"
            )
        if (
            args.soft_limit_policy == "warn"
            and args.confirm_relaxed_limits != RELAXED_LIMITS_CONFIRMATION
        ):
            parser.error(
                "warning-only real-hardware limits require --confirm-relaxed-limits "
                f"{RELAXED_LIMITS_CONFIRMATION}"
            )
        if args.confirm_real_robot != REAL_ROBOT_CONFIRMATION:
            parser.error(
                f"real hardware requires --confirm-real-robot {REAL_ROBOT_CONFIRMATION}"
            )
        if args.countdown < 3:
            parser.error("real hardware requires --countdown of at least 3 seconds")
    if (
        args.target == "simulator"
        and not _loopback_address(args.address)
        and not args.allow_remote_simulator
    ):
        parser.error(
            "simulator mode only accepts a loopback SDK address by default; use "
            "--allow-remote-simulator for a known remote simulator"
        )
    if args.log is not None and args.no_log:
        parser.error("--log and --no-log are mutually exclusive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        run_bridge(args)
        LOGGER.info("Simulator-to-SDK bridge completed normally")
        return 0
    except (HardwareBridgeError, ReplayError, SimBridgeError, OSError) as exc:
        LOGGER.error("Bridge refused/stopped: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
