from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from rby1_hardware.rby1_sdk_replay import (
    DEFAULT_URDF,
    JointLimit,
    Motion,
    ReplayError,
    ReplayTrajectory,
    RBY1_MODEL_A_JOINT_NAMES,
    StateSnapshot,
    UPPER_BODY_INDICES,
    WHEEL_INDICES,
    _minimum_jerk_blend,
    _require_reset_pose_for_simulator_fit,
    _return_to_zero_pose,
    _safe_stop,
    _transition_to_first_pose,
    build_command,
    fit_trajectory_start_pose,
    load_model_a_limits,
    load_motion,
    prepare_upper_body_transition,
    prepare_trajectory,
    validate_trajectory,
)


ROOT = Path(__file__).resolve().parents[2]
GOLF_PKL = (
    ROOT
    / "data"
    / "rby1_seed_projection"
    / "robottest2"
    / "golfmotion_rby1_trainable_9fps_smoothed9.pkl"
)
BUNDLED_DANCE_PKL = (
    ROOT
    / "rby1_hardware"
    / "motions"
    / "guy_dancing_rby1_trainable_9fps_smoothed7.pkl"
)


class RBY1SDKReplayTest(unittest.TestCase):
    limits: tuple[JointLimit, ...]
    motion: Motion | None

    @classmethod
    def setUpClass(cls) -> None:
        cls.limits = load_model_a_limits(DEFAULT_URDF)
        cls.motion = load_motion(GOLF_PKL) if GOLF_PKL.is_file() else None

    def _golf_motion(self) -> Motion:
        if self.motion is None:
            self.skipTest(f"Local generated motion is absent: {GOLF_PKL}")
        assert self.motion is not None
        return self.motion

    def test_default_urdf_is_bundled_with_replay_client(self) -> None:
        self.assertEqual(
            DEFAULT_URDF,
            ROOT / "rby1_hardware" / "assets" / "model_v1.2.urdf",
        )
        self.assertTrue(DEFAULT_URDF.is_file())
        self.assertNotIn("rby1_moveit_ws", DEFAULT_URDF.parts)

    def test_vendor_limits_match_expected_model_a_shape(self) -> None:
        self.assertEqual(len(self.limits), len(RBY1_MODEL_A_JOINT_NAMES))
        self.assertAlmostEqual(self.limits[0].velocity, 15.707963268)
        torso_lower = self.limits[2].lower
        arm_upper = self.limits[19].upper
        self.assertIsNotNone(torso_lower)
        self.assertIsNotNone(arm_upper)
        assert torso_lower is not None
        assert arm_upper is not None
        self.assertAlmostEqual(torso_lower, -0.261799388)
        self.assertAlmostEqual(arm_upper, np.pi, places=6)

    def test_bundled_dance_and_documented_first_trial(self) -> None:
        motion = load_motion(BUNDLED_DANCE_PKL)
        self.assertEqual(motion.name, "guy_dancing_rby1_from_original_bvh_corrected")
        self.assertEqual(motion.positions.shape, (116, 24))
        self.assertEqual(motion.fps, 9.0)

        trajectory = prepare_trajectory(
            motion,
            control_rate=50.0,
            speed=0.10,
            reference_mode="absolute",
            start_positions=None,
            torso_scale=0.10,
            arm_scale=0.75,
            head_scale=0.10,
            enable_wheels=False,
            wheel_scale=0.25,
            scale_absolute_pose=True,
        )
        expected_scales = np.ones(24)
        expected_scales[:2] = 0.0  # wheels are disabled for the first trial
        expected_scales[2:8] = 0.10
        expected_scales[8:22] = 0.75
        expected_scales[22:24] = 0.10
        np.testing.assert_allclose(
            trajectory.positions[0],
            motion.positions[0] * expected_scales,
            atol=1.0e-7,
        )
        trajectory, _offsets = fit_trajectory_start_pose(
            trajectory,
            self.limits,
            position_margin=0.01,
        )
        self.assertEqual(
            validate_trajectory(
                trajectory,
                self.limits,
                position_margin=0.01,
                dynamic_limit_scale=0.50,
                enable_wheels=False,
            ),
            [],
        )

    def test_generated_golf_pkl_has_expected_schema(self) -> None:
        motion = self._golf_motion()
        self.assertEqual(motion.name, "golfmotion")
        self.assertEqual(motion.positions.shape, (224, 24))
        self.assertEqual(motion.fps, 9.0)

    def test_guarded_default_trajectory_passes_offline_limits(self) -> None:
        motion = self._golf_motion()
        trajectory = prepare_trajectory(
            motion,
            control_rate=50.0,
            speed=0.25,
            reference_mode="relative",
            start_positions=np.zeros(24),
            torso_scale=0.10,
            arm_scale=0.25,
            head_scale=0.25,
            enable_wheels=False,
            wheel_scale=0.25,
        )
        issues = validate_trajectory(
            trajectory,
            self.limits,
            position_margin=0.01,
            dynamic_limit_scale=0.50,
            enable_wheels=False,
        )
        self.assertEqual(issues, [])
        np.testing.assert_allclose(trajectory.velocities[:, WHEEL_INDICES], 0.0)
        self.assertGreater(trajectory.duration, motion.duration)

    def test_raw_golf_motion_is_rejected_for_direct_hardware_replay(self) -> None:
        motion = self._golf_motion()
        trajectory = prepare_trajectory(
            motion,
            control_rate=50.0,
            speed=1.0,
            reference_mode="absolute",
            start_positions=None,
            torso_scale=1.0,
            arm_scale=1.0,
            head_scale=1.0,
            enable_wheels=True,
            wheel_scale=1.0,
        )
        issues = validate_trajectory(
            trajectory,
            self.limits,
            position_margin=0.0,
            dynamic_limit_scale=1.0,
            enable_wheels=True,
        )
        affected = {(issue.joint, issue.quantity) for issue in issues}
        self.assertIn(("torso_0", "minimum position"), affected)
        self.assertIn(("left_arm_4", "maximum position"), affected)

    def test_start_pose_fit_preserves_motion_and_respects_one_sided_limits(self) -> None:
        times = np.asarray([0.0, 0.5, 1.0])
        positions = np.zeros((3, 24), dtype=np.float64)
        # Model A right_arm_1 may only roll to +1 degree and left_arm_1
        # may only roll to -1 degree. A zero-pose simulation therefore needs
        # an inward baseline shift before these relative deltas can play.
        positions[:, 9] = [0.0, 0.40, -0.20]
        positions[:, 16] = [0.0, -0.30, 0.20]
        trajectory = ReplayTrajectory(
            times=times,
            source_frames=np.asarray([0.0, 1.0, 2.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
        )

        fitted, offsets = fit_trajectory_start_pose(
            trajectory,
            self.limits,
            position_margin=0.01,
        )

        self.assertLess(offsets[9], 0.0)
        self.assertGreater(offsets[16], 0.0)
        np.testing.assert_allclose(
            fitted.positions - fitted.positions[0],
            trajectory.positions - trajectory.positions[0],
        )
        self.assertEqual(
            validate_trajectory(
                fitted,
                self.limits,
                position_margin=0.01,
                dynamic_limit_scale=0.50,
                enable_wheels=False,
            ),
            [],
        )

    def test_start_pose_fit_rejects_motion_wider_than_joint_range(self) -> None:
        positions = np.zeros((3, 24), dtype=np.float64)
        positions[:, 9] = [-4.0, 0.0, 0.5]
        trajectory = ReplayTrajectory(
            times=np.asarray([0.0, 0.5, 1.0]),
            source_frames=np.asarray([0.0, 1.0, 2.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
        )

        with self.assertRaisesRegex(ReplayError, "right_arm_1 trajectory span"):
            fit_trajectory_start_pose(
                trajectory,
                self.limits,
                position_margin=0.01,
            )

    def test_start_pose_fit_preserves_aggregate_torso_angles(self) -> None:
        positions = np.zeros((3, 24), dtype=np.float64)
        positions[:, 2] = 0.30   # torso_0 exceeds the vendor roll limit
        positions[:, 3] = -0.70  # torso_1 exceeds the vendor pitch limit
        trajectory = ReplayTrajectory(
            times=np.asarray([0.0, 0.5, 1.0]),
            source_frames=np.asarray([0.0, 1.0, 2.0]),
            positions=positions,
            velocities=np.zeros_like(positions),
            accelerations=np.zeros_like(positions),
        )

        fitted, offsets = fit_trajectory_start_pose(
            trajectory,
            self.limits,
            position_margin=0.01,
        )

        self.assertAlmostEqual(float(offsets[[2, 6]].sum()), 0.0, places=10)
        self.assertAlmostEqual(float(offsets[[3, 4, 5]].sum()), 0.0, places=10)
        np.testing.assert_allclose(
            fitted.positions[:, [2, 6]].sum(axis=1),
            trajectory.positions[:, [2, 6]].sum(axis=1),
        )
        np.testing.assert_allclose(
            fitted.positions[:, [3, 4, 5]].sum(axis=1),
            trajectory.positions[:, [3, 4, 5]].sum(axis=1),
        )
        self.assertEqual(
            validate_trajectory(
                fitted,
                self.limits,
                position_margin=0.01,
                dynamic_limit_scale=0.50,
                enable_wheels=False,
            ),
            [],
        )

    def test_sdk_command_separates_positions_from_wheel_velocities(self) -> None:
        class FakeBuilder:
            calls: list[tuple[str, tuple[object, ...]]] = []

            def __getattr__(self, name: str):
                if not name.startswith("set_"):
                    raise AttributeError(name)

                def setter(*args, **_kwargs):
                    FakeBuilder.calls.append((name, args))
                    return self

                return setter

        class FakeSDK:
            BodyComponentBasedCommandBuilder = FakeBuilder
            CommandHeaderBuilder = FakeBuilder
            ComponentBasedCommandBuilder = FakeBuilder
            JointPositionCommandBuilder = FakeBuilder
            JointVelocityCommandBuilder = FakeBuilder
            RobotCommandBuilder = FakeBuilder

        positions = np.linspace(-0.1, 0.1, 24)
        wheel_velocities = np.asarray([0.2, -0.3])

        FakeBuilder.calls.clear()
        build_command(
            FakeSDK,
            positions,
            wheel_velocities,
            self.limits,
            minimum_time=0.02,
            hold_time=0.10,
            dynamic_limit_scale=0.50,
            enable_wheels=False,
        )
        self.assertNotIn("set_mobility_command", [name for name, _args in FakeBuilder.calls])

        FakeBuilder.calls.clear()
        build_command(
            FakeSDK,
            positions,
            wheel_velocities,
            self.limits,
            minimum_time=0.02,
            hold_time=0.10,
            dynamic_limit_scale=0.50,
            enable_wheels=True,
        )
        call_names = [name for name, _args in FakeBuilder.calls]
        self.assertIn("set_mobility_command", call_names)
        velocity_values = [args[0] for name, args in FakeBuilder.calls if name == "set_velocity"]
        self.assertEqual(len(velocity_values), 1)
        np.testing.assert_allclose(velocity_values[0], wheel_velocities)

    def test_first_pose_blend_is_clamped_smooth_and_monotonic(self) -> None:
        samples = np.asarray([_minimum_jerk_blend(value) for value in np.linspace(0, 1, 101)])

        self.assertEqual(_minimum_jerk_blend(-1.0), 0.0)
        self.assertEqual(_minimum_jerk_blend(0.0), 0.0)
        self.assertEqual(_minimum_jerk_blend(1.0), 1.0)
        self.assertEqual(_minimum_jerk_blend(2.0), 1.0)
        self.assertTrue(np.all(np.diff(samples) >= 0.0))
        self.assertAlmostEqual(_minimum_jerk_blend(0.5), 0.5)

    def test_return_transition_reaches_upper_body_zero_and_stops_wheels(self) -> None:
        final_target = np.zeros(24, dtype=np.float64)
        final_target[WHEEL_INDICES] = [4.0, -2.0]
        final_target[2] = 0.10
        final_target[9] = -0.30
        final_target[16] = 0.30
        final_target[22] = 0.20

        transition = prepare_upper_body_transition(
            final_target,
            np.zeros(24, dtype=np.float64),
            self.limits,
            control_rate=50.0,
            minimum_seconds=0.05,
            dynamic_limit_scale=0.50,
        )

        np.testing.assert_allclose(
            transition.positions[0, UPPER_BODY_INDICES],
            final_target[UPPER_BODY_INDICES],
        )
        np.testing.assert_allclose(
            transition.positions[-1, UPPER_BODY_INDICES],
            0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            transition.positions[:, WHEEL_INDICES],
            np.repeat(final_target[None, WHEEL_INDICES], len(transition.times), axis=0),
        )
        np.testing.assert_allclose(transition.velocities[:, WHEEL_INDICES], 0.0)
        np.testing.assert_allclose(transition.accelerations[:, WHEEL_INDICES], 0.0)
        np.testing.assert_allclose(transition.velocities[[0, -1]], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(transition.accelerations[[0, -1]], 0.0, atol=1.0e-12)
        self.assertEqual(
            validate_trajectory(
                transition,
                self.limits,
                position_margin=0.01,
                dynamic_limit_scale=0.50,
                enable_wheels=True,
            ),
            [],
        )

    def test_return_to_zero_stream_settles_at_zero(self) -> None:
        class FakeSDK:
            class ControlManagerState:
                class State:
                    Enabled = "enabled"

        class FakeMonitor:
            def __init__(self, positions: np.ndarray) -> None:
                self.positions = positions.copy()

            def latest(self) -> StateSnapshot:
                return StateSnapshot(
                    received_at=time.monotonic(),
                    positions=self.positions.copy(),
                    velocities=np.zeros(24),
                    ready=np.ones(24, dtype=bool),
                    temperatures=np.zeros(24),
                    emo_pressed=False,
                    control_manager_state="enabled",
                )

        class FakeStream:
            def __init__(self, monitor: FakeMonitor) -> None:
                self.monitor = monitor
                self.command_count = 0

            def send_command(self, command: np.ndarray, *, timeout_ms: int) -> None:
                self.monitor.positions = command.copy()
                self.command_count += 1
                self.timeout_ms = timeout_ms

        final_target = np.zeros(24)
        final_target[WHEEL_INDICES] = [3.0, -1.0]
        final_target[2] = 0.10
        final_target[9] = -0.20
        final_target[16] = 0.20
        monitor = FakeMonitor(final_target)
        stream = FakeStream(monitor)
        fast_limits = tuple(
            JointLimit(limit.lower, limit.upper, 1000.0, 1000.0)
            for limit in self.limits
        )
        args = SimpleNamespace(
            transition_seconds=0.02,
            return_seconds=0.02,
            limit_scale=0.5,
            control_rate=100.0,
            max_loop_lag=0.10,
            state_timeout=0.20,
            max_temperature=80.0,
            enable_wheels=False,
            hold_time=0.10,
            command_timeout_ms=200,
            stationary_timeout=0.10,
            stationary_seconds=0.01,
            max_tracking_error=0.35,
            max_stationary_velocity=0.10,
        )

        def fake_build_command(
            _rby_module,
            positions,
            _wheel_velocities,
            _limits,
            **_kwargs,
        ):
            return np.asarray(positions, dtype=np.float64).copy()

        with patch(
            "rby1_hardware.rby1_sdk_replay.build_command",
            side_effect=fake_build_command,
        ):
            settled = _return_to_zero_pose(
                stream,
                FakeSDK,
                monitor,
                final_target,
                fast_limits,
                args,
                threading.Event(),
            )

        self.assertGreater(stream.command_count, 4)
        np.testing.assert_allclose(settled.positions[UPPER_BODY_INDICES], 0.0)
        np.testing.assert_allclose(settled.positions[WHEEL_INDICES], [3.0, -1.0])

    def test_first_pose_transition_uses_stream_and_settles(self) -> None:
        class FakeBuilder:
            def __getattr__(self, name: str):
                if not name.startswith("set_"):
                    raise AttributeError(name)

                def setter(*_args, **_kwargs):
                    return self

                return setter

        class FakeSDK:
            BodyComponentBasedCommandBuilder = FakeBuilder
            CommandHeaderBuilder = FakeBuilder
            ComponentBasedCommandBuilder = FakeBuilder
            JointPositionCommandBuilder = FakeBuilder
            JointVelocityCommandBuilder = FakeBuilder
            RobotCommandBuilder = FakeBuilder

            class ControlManagerState:
                class State:
                    Enabled = "enabled"

        class FakeStream:
            def __init__(self) -> None:
                self.command_count = 0

            def send_command(self, _command, *, timeout_ms: int):
                self.command_count += 1
                self.timeout_ms = timeout_ms

        class FakeMonitor:
            def latest(self) -> StateSnapshot:
                return StateSnapshot(
                    received_at=time.monotonic(),
                    positions=np.zeros(24),
                    velocities=np.zeros(24),
                    ready=np.ones(24, dtype=bool),
                    temperatures=np.zeros(24),
                    emo_pressed=False,
                    control_manager_state="enabled",
                )

        args = SimpleNamespace(
            transition_seconds=0.02,
            limit_scale=0.5,
            control_rate=100.0,
            max_loop_lag=0.10,
            state_timeout=0.20,
            max_temperature=80.0,
            enable_wheels=False,
            hold_time=0.10,
            command_timeout_ms=200,
            stationary_timeout=0.10,
            stationary_seconds=0.01,
            max_tracking_error=0.35,
            max_stationary_velocity=0.10,
        )
        stream = FakeStream()

        settled = _transition_to_first_pose(
            stream,
            FakeSDK,
            FakeMonitor(),
            np.zeros(24),
            np.zeros(24),
            self.limits,
            args,
            threading.Event(),
        )

        self.assertGreaterEqual(stream.command_count, 4)
        self.assertEqual(stream.timeout_ms, 200)
        np.testing.assert_allclose(settled.positions, 0.0)

    def test_simulator_stop_does_not_use_hanging_direct_command(self) -> None:
        class FakeBuilder:
            def __getattr__(self, name: str):
                if not name.startswith("set_"):
                    raise AttributeError(name)

                def setter(*_args, **_kwargs):
                    return self

                return setter

        class FakeSDK:
            BodyComponentBasedCommandBuilder = FakeBuilder
            CommandHeaderBuilder = FakeBuilder
            ComponentBasedCommandBuilder = FakeBuilder
            JointPositionCommandBuilder = FakeBuilder
            JointVelocityCommandBuilder = FakeBuilder
            RobotCommandBuilder = FakeBuilder

        class FakeRobot:
            direct_command_count = 0

            def get_state(self):
                return SimpleNamespace(position=np.zeros(24))

            def send_command(self, _command, _priority):
                self.direct_command_count += 1
                raise AssertionError("simulator stop must not send a direct command")

        class FakeStream:
            command_count = 0
            cancelled = False

            def send_command(self, _command, *, timeout_ms: int):
                self.command_count += 1
                self.timeout_ms = timeout_ms

            def cancel(self) -> None:
                self.cancelled = True

        robot = FakeRobot()
        stream = FakeStream()
        args = SimpleNamespace(
            target="simulator",
            limit_scale=0.5,
            enable_wheels=False,
            command_timeout_ms=200,
            command_priority=10,
        )

        _safe_stop(robot, FakeSDK, stream, None, self.limits, args)

        self.assertEqual(stream.command_count, 1)
        self.assertTrue(stream.cancelled)
        self.assertEqual(robot.direct_command_count, 0)

    def test_simulator_fit_requires_a_reset_pose(self) -> None:
        reset_pose = np.zeros(24)
        reset_pose[9] = 0.20
        _require_reset_pose_for_simulator_fit(
            reset_pose,
            target="simulator",
            fit_start_pose=True,
        )

        aborted_pose = np.zeros(24)
        aborted_pose[18] = -1.65
        with self.assertRaisesRegex(ReplayError, "left_arm_3=-1.650rad"):
            _require_reset_pose_for_simulator_fit(
                aborted_pose,
                target="simulator",
                fit_start_pose=True,
            )

        # Non-fitted simulator trajectories and explicitly confirmed hardware
        # starts may intentionally use a measured nonzero reference pose.
        _require_reset_pose_for_simulator_fit(
            aborted_pose,
            target="simulator",
            fit_start_pose=False,
        )
        _require_reset_pose_for_simulator_fit(
            aborted_pose,
            target="robot",
            fit_start_pose=True,
        )


if __name__ == "__main__":
    unittest.main()
