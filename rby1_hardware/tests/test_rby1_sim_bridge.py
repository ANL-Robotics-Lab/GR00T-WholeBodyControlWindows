import contextlib
import io
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import threading
import time
import unittest

import numpy as np

from gear_sonic.utils.rby1_order import RBY1_ISAACLAB_DOF_NAMES
from gear_sonic.utils.rby1_sim_bridge import (
    RBY1_ISAAC_TO_SDK_SIGN,
    RBY1_SDK_JOINT_NAMES,
    RBY1SimStatePublisher,
    SimBridgeError,
    decode_packet,
    encode_packet,
    extract_sdk_joint_state,
)
from rby1_hardware.rby1_sdk_replay import (
    DEFAULT_URDF,
    REAL_ROBOT_CONFIRMATION,
    RBY1_MODEL_A_JOINT_NAMES,
    WHEEL_INDICES,
)
from rby1_hardware.rby1_sim_to_sdk import (
    HardwareBridgeError,
    LiveTargetValidator,
    ProjectedTarget,
    TargetProjector,
    TargetRateLimiter,
    RELAXED_LIMITS_CONFIRMATION,
    _apply_target_defaults,
    _check_args,
    _receive_session_packet_with_heartbeat,
    _resolve_packet_clock,
    build_arg_parser,
    run_bridge,
)


class RBY1SimBridgeTest(unittest.TestCase):
    def test_protocol_and_sdk_runner_share_exact_24_joint_contract(self) -> None:
        self.assertEqual(RBY1_SDK_JOINT_NAMES, tuple(RBY1_MODEL_A_JOINT_NAMES))
        self.assertEqual(len(RBY1_SDK_JOINT_NAMES), 24)

    def test_state_packet_round_trip_is_strict(self) -> None:
        positions = np.linspace(-0.5, 0.5, 24)
        velocities = np.linspace(0.5, -0.5, 24)
        packet = decode_packet(
            encode_packet(
                "state",
                "session",
                7,
                positions=positions,
                velocities=velocities,
                control_hz=50.0,
                sim_time=0.14,
            )
        )
        self.assertEqual(packet.kind, "state")
        self.assertEqual(packet.sequence, 7)
        self.assertEqual(packet.joint_names, RBY1_SDK_JOINT_NAMES)
        np.testing.assert_allclose(packet.positions, positions)
        np.testing.assert_allclose(packet.velocities, velocities)

        payload = json.loads(
            encode_packet(
                "state",
                "session",
                8,
                positions=positions,
                velocities=velocities,
                control_hz=50.0,
                sim_time=0.16,
            )
        )
        payload["joint_names"][0], payload["joint_names"][1] = (
            payload["joint_names"][1],
            payload["joint_names"][0],
        )
        with self.assertRaisesRegex(SimBridgeError, "joint order"):
            decode_packet(json.dumps(payload).encode("utf-8"))

    def test_stop_packet_distinguishes_normal_completion_from_abort(self) -> None:
        normal = decode_packet(
            encode_packet(
                "stop",
                "session",
                10,
                reason="episode complete",
                normal_completion=True,
            )
        )
        aborted = decode_packet(
            encode_packet("stop", "session", 10, reason="publisher failed")
        )
        self.assertTrue(normal.normal_completion)
        self.assertFalse(aborted.normal_completion)

    def test_extract_reorders_28_joint_isaac_state_and_flips_wheels(self) -> None:
        joint_names = list(RBY1_ISAACLAB_DOF_NAMES) + [
            "backwheel",
            "backwheel2",
            "gripper_finger_r1",
            "gripper_finger_r2",
        ]
        positions = (1.0 + np.arange(len(joint_names), dtype=np.float64))[np.newaxis, :]
        velocities = (100.0 + positions).copy()
        robot = SimpleNamespace(
            joint_names=joint_names,
            data=SimpleNamespace(joint_pos=positions, joint_vel=velocities),
        )

        sdk_positions, sdk_velocities = extract_sdk_joint_state(robot)
        expected_indices = [joint_names.index(name) for name in RBY1_SDK_JOINT_NAMES]
        np.testing.assert_allclose(
            sdk_positions,
            positions[0, expected_indices] * RBY1_ISAAC_TO_SDK_SIGN,
        )
        np.testing.assert_allclose(
            sdk_velocities,
            velocities[0, expected_indices] * RBY1_ISAAC_TO_SDK_SIGN,
        )
        self.assertLess(sdk_positions[0], 0.0)
        self.assertLess(sdk_positions[1], 0.0)
        np.testing.assert_allclose(RBY1_ISAAC_TO_SDK_SIGN[2:], 1.0)

    def test_relative_projection_uses_measured_baseline_and_disables_wheels(self) -> None:
        sim_initial = np.linspace(-0.2, 0.2, 24)
        hardware_initial = np.linspace(0.3, -0.3, 24)
        projector = TargetProjector(
            sim_initial,
            hardware_initial,
            reference_mode="relative",
            torso_scale=0.10,
            arm_scale=0.25,
            head_scale=0.50,
            wheel_scale=0.10,
            enable_wheels=False,
        )
        target = projector.project(sim_initial + 0.4, np.ones(24))
        expected = hardware_initial.copy()
        expected[2:8] += 0.04
        expected[8:22] += 0.10
        expected[22:24] += 0.20
        np.testing.assert_allclose(target.positions, expected)
        np.testing.assert_allclose(target.velocities[:2], 0.0)
        np.testing.assert_allclose(target.velocities[2:8], 0.10)
        np.testing.assert_allclose(target.velocities[8:22], 0.25)
        np.testing.assert_allclose(target.velocities[22:24], 0.50)

    def test_live_validator_rejects_vendor_position_limit_violation(self) -> None:
        from rby1_hardware.rby1_sdk_replay import load_model_a_limits

        limits = load_model_a_limits(Path(DEFAULT_URDF))
        validator = LiveTargetValidator(
            limits,
            position_margin=0.01,
            limit_scale=0.50,
            enable_wheels=False,
        )
        positions = np.zeros(24)
        positions[2] = 10.0
        packet = decode_packet(
            encode_packet(
                "hello",
                "session",
                0,
                positions=np.zeros(24),
                velocities=np.zeros(24),
                control_hz=50.0,
                sim_time=0.0,
            )
        )
        with self.assertRaisesRegex(HardwareBridgeError, "hardware envelope"):
            validator.validate(
                packet,
                SimpleNamespace(positions=positions, velocities=np.zeros(24)),
            )

    def test_target_rate_limiter_bounds_live_position_command_dynamics(self) -> None:
        from rby1_hardware.rby1_sdk_replay import load_model_a_limits

        limits = load_model_a_limits(Path(DEFAULT_URDF))
        initial = ProjectedTarget(np.zeros(24), np.zeros(24))
        limiter = TargetRateLimiter(
            initial,
            limits,
            position_margin=0.01,
            limit_scale=0.50,
            enable_wheels=False,
        )
        validator = LiveTargetValidator(
            limits,
            position_margin=0.01,
            limit_scale=0.50,
            enable_wheels=False,
        )
        hello = decode_packet(
            encode_packet(
                "hello",
                "session",
                0,
                positions=np.zeros(24),
                velocities=np.zeros(24),
                control_hz=50.0,
                sim_time=0.0,
            )
        )
        validator.validate(hello, limiter.current_target)

        desired_positions = np.zeros(24)
        desired_positions[3] = 0.40
        desired_positions[10] = 0.80
        desired = ProjectedTarget(desired_positions, np.full(24, 100.0))
        previous_velocity = np.zeros(24)
        for sequence in range(1, 21):
            sim_time = sequence / 50.0
            limited = limiter.update(desired, sim_time)
            packet = decode_packet(
                encode_packet(
                    "state",
                    "session",
                    sequence,
                    positions=np.zeros(24),
                    velocities=np.zeros(24),
                    control_hz=50.0,
                    sim_time=sim_time,
                )
            )
            validator.validate(packet, limited)
            acceleration = (limited.velocities - previous_velocity) * 50.0
            self.assertLessEqual(abs(acceleration[3]), 2.5)
            self.assertLessEqual(abs(acceleration[10]), 5.0)
            previous_velocity = limited.velocities

        self.assertGreater(limiter.current_target.positions[3], 0.0)
        self.assertLess(limiter.current_target.positions[3], desired_positions[3])
        np.testing.assert_allclose(limiter.current_target.velocities[WHEEL_INDICES], 0.0)

    def test_target_rate_limiter_rejects_raw_position_limit_violation(self) -> None:
        from rby1_hardware.rby1_sdk_replay import load_model_a_limits

        limits = load_model_a_limits(Path(DEFAULT_URDF))
        limiter = TargetRateLimiter(
            ProjectedTarget(np.zeros(24), np.zeros(24)),
            limits,
            position_margin=0.01,
            limit_scale=0.50,
            enable_wheels=False,
        )
        invalid_positions = np.zeros(24)
        invalid_positions[2] = 10.0
        with self.assertRaisesRegex(HardwareBridgeError, "Projected simulator target"):
            limiter.update(
                ProjectedTarget(invalid_positions, np.zeros(24)),
                0.02,
            )

    def test_local_zero_margin_accepts_narrow_joint_target_inside_urdf_bound(self) -> None:
        from rby1_hardware.rby1_sdk_replay import load_model_a_limits

        limits = load_model_a_limits(Path(DEFAULT_URDF))
        desired_positions = np.zeros(24)
        desired_positions[9] = 0.01537  # right_arm_1 upper URDF bound is 0.01745rad
        desired = ProjectedTarget(desired_positions, np.zeros(24))

        strict_limiter = TargetRateLimiter(
            ProjectedTarget(np.zeros(24), np.zeros(24)),
            limits,
            position_margin=0.01,
            limit_scale=0.50,
            enable_wheels=False,
        )
        with self.assertRaisesRegex(HardwareBridgeError, "right_arm_1"):
            strict_limiter.update(desired, 0.02)

        local_limiter = TargetRateLimiter(
            ProjectedTarget(np.zeros(24), np.zeros(24)),
            limits,
            position_margin=0.0,
            limit_scale=0.50,
            enable_wheels=False,
        )
        local_limiter.update(desired, 0.02)

    def test_local_clamp_saturates_target_outside_narrow_vendor_bound(self) -> None:
        from rby1_hardware.rby1_sdk_replay import load_model_a_limits

        limits = load_model_a_limits(Path(DEFAULT_URDF))
        desired_positions = np.zeros(24)
        desired_positions[9] = 0.03535
        limiter = TargetRateLimiter(
            ProjectedTarget(np.zeros(24), np.zeros(24)),
            limits,
            position_margin=0.0,
            limit_scale=0.50,
            enable_wheels=False,
            clamp_position_limits=True,
        )

        for sequence in range(1, 50):
            target = limiter.update(
                ProjectedTarget(desired_positions, np.zeros(24)),
                sequence / 50.0,
            )

        self.assertLessEqual(target.positions[9], limits[9].upper)
        self.assertAlmostEqual(target.positions[9], limits[9].upper)

    def test_udp_publisher_waits_for_ready_and_marks_normal_stop(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        host, port = receiver.getsockname()
        received = []
        errors = []

        def receive_packets() -> None:
            try:
                hello_data, source = receiver.recvfrom(16_385)
                hello = decode_packet(hello_data)
                received.append(hello)
                receiver.sendto(encode_packet("ready", hello.session_id, 0), source)
                state_data, _source = receiver.recvfrom(16_385)
                received.append(decode_packet(state_data))
                stop_data, _source = receiver.recvfrom(16_385)
                received.append(decode_packet(stop_data))
            except BaseException as exc:  # test thread must report failures
                errors.append(exc)

        thread = threading.Thread(target=receive_packets, daemon=True)
        thread.start()
        publisher = RBY1SimStatePublisher(
            f"{host}:{port}",
            control_hz=50.0,
            startup_timeout=2.0,
        )
        try:
            publisher.wait_for_receiver(np.zeros(24), np.zeros(24))
            publisher.publish(np.ones(24), np.zeros(24))
            publisher.stop("episode complete", normal_completion=True)
        finally:
            publisher.close()
        thread.join(timeout=2.0)
        receiver.close()

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([packet.kind for packet in received], ["hello", "state", "stop"])
        self.assertTrue(received[-1].normal_completion)

    def test_udp_publisher_starts_pacing_clock_at_first_state(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        host, port = receiver.getsockname()
        errors = []

        def receive_packets() -> None:
            try:
                hello_data, source = receiver.recvfrom(16_385)
                hello = decode_packet(hello_data)
                receiver.sendto(encode_packet("ready", hello.session_id, 0), source)
                receiver.recvfrom(16_385)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=receive_packets, daemon=True)
        thread.start()
        publisher = RBY1SimStatePublisher(
            f"{host}:{port}",
            control_hz=50.0,
            startup_timeout=2.0,
            max_pacing_lag=0.02,
        )
        try:
            publisher.wait_for_receiver(np.zeros(24), np.zeros(24))
            # Simulate one-time CUDA/policy initialization that is longer than
            # the steady-state pacing lag limit.
            time.sleep(0.05)
            publisher.publish(np.zeros(24), np.zeros(24))
            publisher.pace()
        finally:
            publisher.close()
            receiver.close()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_publisher_can_rebase_accumulated_local_simulator_lag(self) -> None:
        publisher = RBY1SimStatePublisher(
            "127.0.0.1:9",
            control_hz=50.0,
            max_pacing_lag=0.01,
            rebase_on_pacing_lag=True,
        )
        try:
            publisher._started = True
            publisher._next_deadline = time.monotonic() - 0.10
            publisher._check_receiver_status = lambda: None
            publisher.pace()
            self.assertLess(abs(time.monotonic() - publisher._next_deadline), 0.05)
        finally:
            publisher.close()

    def test_delayed_state_wait_refreshes_finite_command_stream_lease(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        receiver_address = receiver.getsockname()
        sender_address = sender.getsockname()
        heartbeat_times = []
        errors = []

        def send_delayed_state() -> None:
            try:
                time.sleep(0.10)
                sender.sendto(
                    encode_packet(
                        "state",
                        "startup-session",
                        1,
                        positions=np.zeros(24),
                        velocities=np.zeros(24),
                        control_hz=50.0,
                        sim_time=0.02,
                    ),
                    receiver_address,
                )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=send_delayed_state, daemon=True)
        thread.start()
        try:
            packet = _receive_session_packet_with_heartbeat(
                receiver,
                source=sender_address,
                session_id="startup-session",
                timeout=0.5,
                heartbeat_interval=0.02,
                heartbeat=lambda: heartbeat_times.append(time.monotonic()),
                stop_requested=threading.Event(),
            )
        finally:
            receiver.close()
            sender.close()
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(packet.kind, "state")
        self.assertGreaterEqual(len(heartbeat_times), 3)
        self.assertLess(max(np.diff(heartbeat_times)), 0.08)

    def test_prompt_state_does_not_send_an_unnecessary_hold_heartbeat(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))
        heartbeats = []
        try:
            sender.sendto(
                encode_packet(
                    "state",
                    "prompt-session",
                    1,
                    positions=np.zeros(24),
                    velocities=np.zeros(24),
                    control_hz=50.0,
                    sim_time=0.02,
                ),
                receiver.getsockname(),
            )
            packet = _receive_session_packet_with_heartbeat(
                receiver,
                source=sender.getsockname(),
                session_id="prompt-session",
                timeout=0.5,
                heartbeat_interval=0.05,
                heartbeat=lambda: heartbeats.append(time.monotonic()),
                stop_requested=threading.Event(),
            )
        finally:
            receiver.close()
            sender.close()

        self.assertEqual(packet.kind, "state")
        self.assertEqual(heartbeats, [])

    def test_warning_policy_keeps_holding_past_packet_timeout(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender.bind(("127.0.0.1", 0))

        def send_late_state() -> None:
            time.sleep(0.10)
            sender.sendto(
                encode_packet(
                    "state",
                    "warning-session",
                    1,
                    positions=np.zeros(24),
                    velocities=np.zeros(24),
                    control_hz=50.0,
                    sim_time=0.02,
                ),
                receiver.getsockname(),
            )

        thread = threading.Thread(target=send_late_state, daemon=True)
        thread.start()
        try:
            with self.assertLogs("rby1_sim_to_sdk", level="WARNING") as logs:
                packet = _receive_session_packet_with_heartbeat(
                    receiver,
                    source=sender.getsockname(),
                    session_id="warning-session",
                    timeout=0.03,
                    heartbeat_interval=0.01,
                    heartbeat=lambda: None,
                    stop_requested=threading.Event(),
                    enforce_timeout=False,
                )
        finally:
            receiver.close()
            sender.close()
        thread.join(timeout=1.0)

        self.assertEqual(packet.kind, "state")
        self.assertTrue(any("warning-only" in message for message in logs.output))

    def test_local_simulator_timing_is_looser_than_real_robot_timing(self) -> None:
        parser = build_arg_parser()
        simulator_args = parser.parse_args(["--target", "simulator"])
        robot_args = parser.parse_args(["--target", "robot"])
        relaxed_robot_args = parser.parse_args(
            ["--target", "robot", "--soft-limit-policy", "warn"]
        )
        explicit_args = parser.parse_args(
            [
                "--target",
                "simulator",
                "--packet-timeout",
                "0.3",
                "--max-loop-lag",
                "0.4",
                "--position-margin",
                "0.002",
                "--position-limit-mode",
                "reject",
            ]
        )

        _apply_target_defaults(simulator_args)
        _apply_target_defaults(robot_args)
        _apply_target_defaults(relaxed_robot_args)
        _apply_target_defaults(explicit_args)

        self.assertEqual(simulator_args.soft_limit_policy, "warn")
        self.assertEqual(simulator_args.packet_timeout, 5.0)
        self.assertEqual(simulator_args.max_loop_lag, 1.0)
        self.assertEqual(simulator_args.position_margin, 0.0)
        self.assertEqual(simulator_args.position_limit_mode, "clamp")
        self.assertEqual(robot_args.packet_timeout, 0.20)
        self.assertEqual(robot_args.max_loop_lag, 0.10)
        self.assertEqual(robot_args.position_margin, 0.01)
        self.assertEqual(robot_args.position_limit_mode, "reject")
        self.assertEqual(robot_args.soft_limit_policy, "enforce")
        self.assertEqual(relaxed_robot_args.soft_limit_policy, "warn")
        self.assertEqual(relaxed_robot_args.packet_timeout, 5.0)
        self.assertEqual(relaxed_robot_args.max_loop_lag, 1.0)
        self.assertEqual(relaxed_robot_args.position_margin, 0.0)
        self.assertEqual(relaxed_robot_args.position_limit_mode, "clamp")
        self.assertEqual(explicit_args.packet_timeout, 0.3)
        self.assertEqual(explicit_args.max_loop_lag, 0.4)
        self.assertEqual(explicit_args.position_margin, 0.002)
        self.assertEqual(explicit_args.position_limit_mode, "reject")

    def test_warning_only_hardware_requires_separate_confirmation(self) -> None:
        parser = build_arg_parser()
        unconfirmed = parser.parse_args(
            [
                "--target",
                "robot",
                "--soft-limit-policy",
                "warn",
                "--confirm-real-robot",
                REAL_ROBOT_CONFIRMATION,
            ]
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _check_args(parser, unconfirmed)

        confirmed = parser.parse_args(
            [
                "--target",
                "robot",
                "--soft-limit-policy",
                "warn",
                "--confirm-real-robot",
                REAL_ROBOT_CONFIRMATION,
                "--confirm-relaxed-limits",
                RELAXED_LIMITS_CONFIRMATION,
            ]
        )
        _check_args(parser, confirmed)

    def test_local_clock_rebases_but_real_robot_clock_rejects_same_lag(self) -> None:
        bridge_start, deadline, lag = _resolve_packet_clock(
            bridge_start=10.0,
            sim_time=0.1,
            now=11.2,
            max_loop_lag=1.0,
            target="simulator",
            enforce=False,
        )
        self.assertAlmostEqual(bridge_start, 11.1)
        self.assertAlmostEqual(deadline, 11.2)
        self.assertEqual(lag, 0.0)

        with self.assertRaisesRegex(HardwareBridgeError, "missed simulator deadline"):
            _resolve_packet_clock(
                bridge_start=10.0,
                sim_time=0.1,
                now=11.2,
                max_loop_lag=1.0,
                target="robot",
                enforce=True,
            )

    def test_shadow_bridge_allows_only_first_state_startup_grace(self) -> None:
        port_probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port_probe.bind(("127.0.0.1", 0))
        host, port = port_probe.getsockname()
        port_probe.close()
        errors = []

        def publish_delayed_first_state() -> None:
            publisher = RBY1SimStatePublisher(
                f"{host}:{port}",
                control_hz=50.0,
                startup_timeout=2.0,
                hello_interval=0.01,
                max_pacing_lag=0.05,
            )
            try:
                publisher.wait_for_receiver(np.zeros(24), np.zeros(24))
                time.sleep(0.10)
                dynamic_positions = np.zeros(24)
                dynamic_positions[3] = 0.40
                dynamic_positions[10] = 0.80
                for _ in range(3):
                    publisher.publish(dynamic_positions, np.full(24, 100.0))
                    publisher.pace()
                publisher.stop("test episode complete", normal_completion=True)
            except BaseException as exc:
                errors.append(exc)
            finally:
                publisher.close()

        publisher_thread = threading.Thread(target=publish_delayed_first_state, daemon=True)
        publisher_thread.start()
        args = build_arg_parser().parse_args(
            [
                "--target",
                "shadow",
                "--listen",
                f"{host}:{port}",
                "--source-timeout",
                "2.0",
                "--first-packet-timeout",
                "0.5",
                "--packet-timeout",
                "0.05",
                "--max-loop-lag",
                "0.05",
                "--no-log",
            ]
        )

        run_bridge(args)
        publisher_thread.join(timeout=2.0)

        self.assertFalse(publisher_thread.is_alive())
        self.assertEqual(errors, [])

    def test_udp_publisher_propagates_receiver_safety_stop(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        host, port = receiver.getsockname()
        errors = []

        def stop_after_first_state() -> None:
            try:
                hello_data, source = receiver.recvfrom(16_385)
                hello = decode_packet(hello_data)
                receiver.sendto(encode_packet("ready", hello.session_id, 0), source)
                receiver.recvfrom(16_385)
                receiver.sendto(
                    encode_packet(
                        "stop",
                        hello.session_id,
                        0,
                        reason="tracking watchdog",
                    ),
                    source,
                )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=stop_after_first_state, daemon=True)
        thread.start()
        publisher = RBY1SimStatePublisher(
            f"{host}:{port}",
            control_hz=50.0,
            startup_timeout=2.0,
        )
        try:
            publisher.wait_for_receiver(np.zeros(24), np.zeros(24))
            with self.assertRaisesRegex(SimBridgeError, "tracking watchdog"):
                publisher.publish(np.zeros(24), np.zeros(24))
                publisher.pace()
        finally:
            publisher.close()
            receiver.close()
        thread.join(timeout=2.0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
