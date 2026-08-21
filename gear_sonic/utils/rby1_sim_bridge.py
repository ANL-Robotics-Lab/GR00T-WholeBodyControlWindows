"""Small, dependency-light protocol for mirroring RBY1 Isaac state to the SDK.

The learned policy remains inside Isaac Lab.  This module publishes the
realized 24-DOF articulation state after Isaac has decoded and applied the
policy action.  A separate process in ``rby1_hardware`` receives these packets
and uses the vendor SDK, whose NumPy/Python environment is intentionally kept
separate from Isaac Sim.

Packets are deliberately self-describing and strict.  In particular, every
state packet repeats the canonical Model A joint names so a changed asset order
cannot silently command the wrong hardware joint.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
import socket
import time
from typing import Any, Sequence
import uuid

import numpy as np

from gear_sonic.utils.rby1_order import (
    RBY1_MOTION_TO_ISAACLAB_DOF_SIGN,
    RBY1_MUJOCO_DOF_NAMES,
)


PROTOCOL_NAME = "gear-sonic-rby1-sim-state"
PROTOCOL_VERSION = 1
PACKET_KINDS = frozenset({"hello", "ready", "state", "stop"})
MAX_DATAGRAM_BYTES = 16_384
RBY1_SDK_JOINT_NAMES = tuple(RBY1_MUJOCO_DOF_NAMES)

# The vendor Model A URDF and motion-library model expose both wheel axes along
# -Y.  The imported Isaac USD exposes them along +Y.  The conversion is its own
# inverse, and all upper-body signs are +1.
RBY1_ISAAC_TO_SDK_SIGN = np.asarray(
    RBY1_MOTION_TO_ISAACLAB_DOF_SIGN,
    dtype=np.float64,
)


class SimBridgeError(RuntimeError):
    """A bridge packet or simulator state is unsafe or incompatible."""


@dataclass(frozen=True)
class BridgePacket:
    """Validated wire packet."""

    kind: str
    session_id: str
    sequence: int
    sent_time_ns: int
    sim_time: float | None = None
    control_hz: float | None = None
    joint_names: tuple[str, ...] = ()
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    reason: str | None = None
    normal_completion: bool = False


def parse_endpoint(value: str) -> tuple[str, int]:
    """Parse an IPv4/hostname ``HOST:PORT`` endpoint."""

    if not isinstance(value, str) or ":" not in value:
        raise SimBridgeError(f"Expected HOST:PORT, got {value!r}")
    host, port_text = value.rsplit(":", 1)
    host = host.strip()
    if not host:
        raise SimBridgeError("Bridge endpoint host cannot be empty")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise SimBridgeError(f"Bridge endpoint has an invalid port: {value!r}") from exc
    if not 1 <= port <= 65_535:
        raise SimBridgeError(f"Bridge port must be in [1, 65535], got {port}")
    return host, port


def is_loopback_host(host: str) -> bool:
    """Return whether a host is explicitly local without doing DNS lookup."""

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _finite_vector(values: Any, field: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    expected = (len(RBY1_SDK_JOINT_NAMES),)
    if vector.shape != expected:
        raise SimBridgeError(f"{field} has shape {vector.shape}, expected {expected}")
    if not np.isfinite(vector).all():
        raise SimBridgeError(f"{field} contains non-finite values")
    return vector.copy()


def _finite_number(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SimBridgeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise SimBridgeError(f"{field} must be {qualifier}, got {number}")
    return number


def encode_packet(
    kind: str,
    session_id: str,
    sequence: int,
    *,
    positions: Sequence[float] | np.ndarray | None = None,
    velocities: Sequence[float] | np.ndarray | None = None,
    control_hz: float | None = None,
    sim_time: float | None = None,
    reason: str | None = None,
    normal_completion: bool = False,
) -> bytes:
    """Encode one strict JSON datagram."""

    if kind not in PACKET_KINDS:
        raise SimBridgeError(f"Unknown bridge packet kind: {kind!r}")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise SimBridgeError("session_id must contain 1 to 128 characters")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise SimBridgeError(f"sequence must be a non-negative integer, got {sequence!r}")

    payload: dict[str, Any] = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "kind": kind,
        "session_id": session_id,
        "sequence": int(sequence),
        "sent_time_ns": time.time_ns(),
    }
    if kind in {"hello", "state"}:
        if positions is None or velocities is None or control_hz is None or sim_time is None:
            raise SimBridgeError(f"{kind} packets require state vectors, rate, and sim time")
        position_vector = _finite_vector(positions, "positions")
        velocity_vector = _finite_vector(velocities, "velocities")
        payload.update(
            {
                "joint_names": list(RBY1_SDK_JOINT_NAMES),
                "positions": position_vector.tolist(),
                "velocities": velocity_vector.tolist(),
                "control_hz": _finite_number(control_hz, "control_hz", positive=True),
                "sim_time": _finite_number(sim_time, "sim_time"),
            }
        )
    elif kind == "stop":
        payload["reason"] = str(reason or "simulator stopped")[:512]
        payload["normal_completion"] = bool(normal_completion)

    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_DATAGRAM_BYTES:
        raise SimBridgeError(
            f"Encoded bridge packet is {len(encoded)} bytes; limit is {MAX_DATAGRAM_BYTES}"
        )
    return encoded


def decode_packet(data: bytes) -> BridgePacket:
    """Decode and validate one bridge datagram."""

    if not data or len(data) > MAX_DATAGRAM_BYTES:
        raise SimBridgeError(
            f"Bridge datagram size must be in [1, {MAX_DATAGRAM_BYTES}], got {len(data)}"
        )
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimBridgeError("Bridge datagram is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise SimBridgeError("Bridge packet must be a JSON object")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise SimBridgeError(f"Unexpected bridge protocol: {payload.get('protocol')!r}")
    if payload.get("version") != PROTOCOL_VERSION:
        raise SimBridgeError(
            f"Unsupported bridge protocol version: {payload.get('version')!r}"
        )

    kind = payload.get("kind")
    if kind not in PACKET_KINDS:
        raise SimBridgeError(f"Unknown bridge packet kind: {kind!r}")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise SimBridgeError("Bridge packet has an invalid session_id")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise SimBridgeError("Bridge packet has an invalid sequence")
    sent_time_ns = payload.get("sent_time_ns")
    if not isinstance(sent_time_ns, int) or isinstance(sent_time_ns, bool) or sent_time_ns <= 0:
        raise SimBridgeError("Bridge packet has an invalid sent_time_ns")

    if kind not in {"hello", "state"}:
        reason = payload.get("reason") if kind == "stop" else None
        if reason is not None and not isinstance(reason, str):
            raise SimBridgeError("Bridge stop reason must be a string")
        normal_completion = payload.get("normal_completion", False)
        if not isinstance(normal_completion, bool):
            raise SimBridgeError("Bridge normal_completion flag must be boolean")
        return BridgePacket(
            kind,
            session_id,
            sequence,
            sent_time_ns,
            reason=reason,
            normal_completion=normal_completion,
        )

    names = payload.get("joint_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise SimBridgeError("Bridge packet joint_names must be a string list")
    joint_names = tuple(names)
    if joint_names != RBY1_SDK_JOINT_NAMES:
        raise SimBridgeError(
            "Bridge packet joint order does not exactly match RBY1 Model A: "
            f"{joint_names}"
        )
    positions = _finite_vector(payload.get("positions"), "positions")
    velocities = _finite_vector(payload.get("velocities"), "velocities")
    control_hz = _finite_number(payload.get("control_hz"), "control_hz", positive=True)
    sim_time = _finite_number(payload.get("sim_time"), "sim_time")
    if sim_time < 0.0:
        raise SimBridgeError(f"sim_time must be non-negative, got {sim_time}")
    return BridgePacket(
        kind=kind,
        session_id=session_id,
        sequence=sequence,
        sent_time_ns=sent_time_ns,
        sim_time=sim_time,
        control_hz=control_hz,
        joint_names=joint_names,
        positions=positions,
        velocities=velocities,
    )


def _as_numpy(tensor: Any) -> np.ndarray:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def extract_sdk_joint_state(robot: Any, env_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Select and recast one Isaac articulation state into SDK Model A order."""

    joint_names = tuple(str(name) for name in robot.joint_names)
    if len(joint_names) != len(set(joint_names)):
        raise SimBridgeError("Isaac articulation contains duplicate joint names")
    missing = [name for name in RBY1_SDK_JOINT_NAMES if name not in joint_names]
    if missing:
        raise SimBridgeError(f"Isaac articulation is missing RBY1 SDK joints: {missing}")
    if env_index < 0:
        raise SimBridgeError("env_index must be non-negative")

    indices = [joint_names.index(name) for name in RBY1_SDK_JOINT_NAMES]
    all_positions = _as_numpy(robot.data.joint_pos)
    all_velocities = _as_numpy(robot.data.joint_vel)
    if all_positions.ndim != 2 or all_velocities.ndim != 2:
        raise SimBridgeError(
            "Isaac joint state must have shape [num_envs, num_joints]; "
            f"got {all_positions.shape} and {all_velocities.shape}"
        )
    if env_index >= all_positions.shape[0] or env_index >= all_velocities.shape[0]:
        raise SimBridgeError(
            f"env_index {env_index} is outside Isaac state with "
            f"{all_positions.shape[0]} environments"
        )
    if max(indices) >= all_positions.shape[1] or max(indices) >= all_velocities.shape[1]:
        raise SimBridgeError("Isaac joint name list and state tensor widths disagree")

    positions = all_positions[env_index, indices] * RBY1_ISAAC_TO_SDK_SIGN
    velocities = all_velocities[env_index, indices] * RBY1_ISAAC_TO_SDK_SIGN
    return _finite_vector(positions, "Isaac positions"), _finite_vector(
        velocities, "Isaac velocities"
    )


class RBY1SimStatePublisher:
    """Paced UDP publisher with a first-pose receiver handshake."""

    def __init__(
        self,
        destination: str,
        *,
        control_hz: float,
        startup_timeout: float = 120.0,
        hello_interval: float = 0.20,
        max_pacing_lag: float = 1.0,
        rebase_on_pacing_lag: bool = False,
        allow_remote: bool = False,
    ) -> None:
        self.destination = parse_endpoint(destination)
        if not allow_remote and not is_loopback_host(self.destination[0]):
            raise SimBridgeError(
                "The simulator publisher only permits a loopback destination by default; "
                "set allow_remote=true only on an isolated trusted network"
            )
        self.control_hz = _finite_number(control_hz, "control_hz", positive=True)
        self.startup_timeout = _finite_number(
            startup_timeout, "startup_timeout", positive=True
        )
        self.hello_interval = _finite_number(
            hello_interval, "hello_interval", positive=True
        )
        self.max_pacing_lag = _finite_number(
            max_pacing_lag, "max_pacing_lag", positive=True
        )
        self.rebase_on_pacing_lag = bool(rebase_on_pacing_lag)
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self._started = False
        self._stopped = False
        self._next_deadline: float | None = None
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect(self.destination)

    def wait_for_receiver(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Hold simulation at frame zero until the SDK side reports ready."""

        if self._started:
            raise SimBridgeError("Bridge publisher handshake has already completed")
        hello = encode_packet(
            "hello",
            self.session_id,
            0,
            positions=positions,
            velocities=velocities,
            control_hz=self.control_hz,
            sim_time=0.0,
        )
        deadline = time.monotonic() + self.startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise SimBridgeError(
                    f"No SDK bridge receiver became ready at "
                    f"{self.destination[0]}:{self.destination[1]} within "
                    f"{self.startup_timeout:.1f}s"
                )
            try:
                self._socket.send(hello)
            except ConnectionResetError:
                # Windows turns an ICMP UDP port-unreachable response into
                # WSAECONNRESET. The receiver may simply still be starting.
                time.sleep(min(self.hello_interval, remaining))
                continue
            self._socket.settimeout(min(self.hello_interval, remaining))
            try:
                data = self._socket.recv(MAX_DATAGRAM_BYTES + 1)
            except (ConnectionResetError, socket.timeout):
                continue
            packet = decode_packet(data)
            if packet.kind == "stop" and packet.session_id == self.session_id:
                raise SimBridgeError(
                    f"SDK bridge receiver refused startup: {packet.reason}"
                )
            if packet.kind == "ready" and packet.session_id == self.session_id:
                self._started = True
                # The first policy/physics step may initialize CUDA kernels and
                # take substantially longer than a steady-state frame. Start
                # the real-time pacing clock when that first state is actually
                # published instead of charging startup work against the lag
                # watchdog.
                self._next_deadline = None
                return

    def _check_receiver_status(self) -> None:
        """Raise promptly when the SDK process reports a safety stop."""

        self._socket.setblocking(False)
        try:
            while True:
                try:
                    data = self._socket.recv(MAX_DATAGRAM_BYTES + 1)
                except (BlockingIOError, socket.timeout):
                    return
                packet = decode_packet(data)
                if packet.kind == "stop" and packet.session_id == self.session_id:
                    raise SimBridgeError(
                        f"SDK bridge receiver stopped: {packet.reason}"
                    )
        finally:
            self._socket.setblocking(True)

    def publish(self, positions: np.ndarray, velocities: np.ndarray) -> int:
        """Publish one realized simulator frame."""

        if not self._started or self._stopped:
            raise SimBridgeError("Bridge publisher is not in a running state")
        self.sequence += 1
        packet = encode_packet(
            "state",
            self.session_id,
            self.sequence,
            positions=positions,
            velocities=velocities,
            control_hz=self.control_hz,
            sim_time=self.sequence / self.control_hz,
        )
        try:
            self._socket.send(packet)
        except ConnectionResetError as exc:
            raise SimBridgeError("SDK bridge receiver is no longer reachable") from exc
        if self._next_deadline is None:
            self._next_deadline = time.monotonic()
        self._check_receiver_status()
        return self.sequence

    def pace(self) -> None:
        """Prevent a headless simulator from outrunning the real command clock."""

        if not self._started or self._stopped or self._next_deadline is None:
            return
        self._next_deadline += 1.0 / self.control_hz
        now = time.monotonic()
        lag = now - self._next_deadline
        if lag > self.max_pacing_lag:
            if self.rebase_on_pacing_lag:
                # GUI rendering and one-time CUDA kernels can make local Isaac
                # evaluation slower than wall time. Drop accumulated lateness;
                # the SDK receiver still independently enforces packet/state
                # watchdogs and retains strict timing for a real-robot target.
                self._next_deadline = now
                self._check_receiver_status()
                return
            raise SimBridgeError(
                f"Isaac bridge loop is {lag:.3f}s behind its real-time deadline "
                f"(limit {self.max_pacing_lag:.3f}s)"
            )
        remaining = self._next_deadline - now
        if remaining > 0.0:
            time.sleep(remaining)
        self._check_receiver_status()

    def stop(self, reason: str, *, normal_completion: bool = False) -> None:
        """Best-effort stop notification (repeated because UDP is unreliable)."""

        if self._stopped:
            return
        self._stopped = True
        if not self._started:
            return
        self.sequence += 1
        packet = encode_packet(
            "stop",
            self.session_id,
            self.sequence,
            reason=reason,
            normal_completion=normal_completion,
        )
        for _ in range(3):
            try:
                self._socket.send(packet)
            except OSError:
                break

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> RBY1SimStatePublisher:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop("simulator publisher closed")
        self.close()
