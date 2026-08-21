from pathlib import Path
import tempfile
import unittest

from rby1_hardware.rby1_checkpoint_eval import (
    BUNDLED_MOTION_DIR,
    CheckpointEvalError,
    build_eval_command,
    find_training_config,
    read_config_motion_path,
    resolve_checkpoint,
    resolve_motion_override,
)


ROOT = Path(__file__).resolve().parents[2]
BUNDLED_DANCE = BUNDLED_MOTION_DIR / "guy_dancing_rby1_trainable_9fps_smoothed7.pkl"


class RBY1CheckpointEvalTest(unittest.TestCase):
    def test_resolve_checkpoint_accepts_experiment_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Path(temp_dir)
            (experiment / "last.pt").touch()
            self.assertEqual(resolve_checkpoint(experiment), (experiment / "last.pt").resolve())

    def test_resolve_checkpoint_uses_highest_numbered_model_without_last(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Path(temp_dir)
            (experiment / "model_step_000900.pt").touch()
            (experiment / "model_step_003000.pt").touch()
            self.assertEqual(
                resolve_checkpoint(experiment),
                (experiment / "model_step_003000.pt").resolve(),
            )

    def test_find_training_config_checks_checkpoint_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = Path(temp_dir)
            checkpoint = experiment / "last.pt"
            config = experiment / "config.yaml"
            checkpoint.touch()
            config.touch()
            self.assertEqual(find_training_config(checkpoint), config.resolve())

    def test_read_config_motion_path_ignores_smpl_motion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text(
                "smpl_motion_file: dummy\n"
                "motion_lib_cfg:\n"
                "  motion_file: 'motions/dance.pkl' # source motion\n",
                encoding="utf-8",
            )
            self.assertEqual(read_config_motion_path(config), "motions/dance.pkl")

    def test_missing_saved_motion_falls_back_to_bundled_exact_name(self) -> None:
        self.assertTrue(BUNDLED_DANCE.is_file())
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text(
                "motion_file: tmp_rby1_video_compare/"
                "guy_dancing_rby1_trainable_9fps_smoothed7.pkl\n",
                encoding="utf-8",
            )
            override, description = resolve_motion_override(config, None)
            self.assertEqual(override, BUNDLED_DANCE.resolve())
            self.assertIn("bundled replacement", description)

    def test_old_original_bvh_name_falls_back_to_renamed_bundled_motion(self) -> None:
        self.assertTrue(BUNDLED_DANCE.is_file())
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text(
                "motion_file: missing/"
                "guy_dancing_rby1_from_original_bvh_trainable_9fps_smoothed7.pkl\n",
                encoding="utf-8",
            )
            override, description = resolve_motion_override(config, None)
            self.assertEqual(override, BUNDLED_DANCE.resolve())
            self.assertIn("renamed replacement", description)

    def test_missing_motion_without_fallback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Path(temp_dir) / "config.yaml"
            config.write_text("motion_file: missing/unknown.pkl\n", encoding="utf-8")
            with self.assertRaisesRegex(CheckpointEvalError, "--motion"):
                resolve_motion_override(config, None)

    def test_build_command_defaults_to_one_noise_free_g1_episode(self) -> None:
        checkpoint = Path("C:/experiment/last.pt")
        motion = ROOT / "rby1_hardware" / "motions" / "dance.pkl"
        command = build_eval_command(
            python_executable="python",
            checkpoint=checkpoint,
            motion_override=motion,
            headless=False,
            num_envs=1,
            encoder="g1",
            keep_observation_noise=False,
            loop=False,
            max_steps=0,
            export_onnx=False,
        )
        self.assertIn("++headless=false", command)
        self.assertIn("++num_envs=1", command)
        self.assertIn("++use_encoder=g1", command)
        self.assertIn("++run_once=true", command)
        self.assertIn(
            "++manager_env.observations.policy.enable_corruption=false", command
        )
        self.assertTrue(
            any(
                item.startswith("++manager_env.commands.motion.motion_lib_cfg.motion_file=")
                for item in command
            )
        )

    def test_onnx_export_forces_headless_single_environment(self) -> None:
        command = build_eval_command(
            python_executable="python",
            checkpoint=Path("last.pt"),
            motion_override=None,
            headless=False,
            num_envs=8,
            encoder="auto",
            keep_observation_noise=True,
            loop=False,
            max_steps=0,
            export_onnx=True,
        )
        self.assertIn("++headless=true", command)
        self.assertIn("++num_envs=1", command)
        self.assertIn("++export_onnx_only=true", command)
        self.assertNotIn("++run_once=true", command)

    def test_sdk_bridge_adds_publisher_contract_and_keeps_one_episode(self) -> None:
        command = build_eval_command(
            python_executable="python",
            checkpoint=Path("last.pt"),
            motion_override=None,
            headless=False,
            num_envs=1,
            encoder="g1",
            keep_observation_noise=False,
            loop=False,
            max_steps=0,
            export_onnx=False,
            bridge_destination="127.0.0.1:50070",
        )
        self.assertIn("++run_once=true", command)
        self.assertIn("++sim_to_sdk_bridge.enabled=true", command)
        self.assertIn(
            "++sim_to_sdk_bridge.destination=127.0.0.1:50070",
            command,
        )

    def test_sdk_bridge_rejects_looping_or_multiple_environments(self) -> None:
        common = dict(
            python_executable="python",
            checkpoint=Path("last.pt"),
            motion_override=None,
            headless=False,
            encoder="g1",
            keep_observation_noise=False,
            max_steps=0,
            export_onnx=False,
            bridge_destination="127.0.0.1:50070",
        )
        with self.assertRaisesRegex(CheckpointEvalError, "--loop"):
            build_eval_command(num_envs=1, loop=True, **common)
        with self.assertRaisesRegex(CheckpointEvalError, "--num-envs 1"):
            build_eval_command(num_envs=2, loop=False, **common)


if __name__ == "__main__":
    unittest.main()
