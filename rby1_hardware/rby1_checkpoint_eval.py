#!/usr/bin/env python3
"""Run an RBY1 training checkpoint through the repository's Isaac Lab evaluator.

By default this launcher is simulator-only.  Its optional ``--sdk-bridge`` mode
publishes the simulator's realized 24-DOF state to the separately installed SDK
receiver; policy observations and history still remain entirely inside Isaac.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "gear_sonic" / "eval_agent_trl.py"
BUNDLED_MOTION_DIR = Path(__file__).resolve().parent / "motions"
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
_MODEL_STEP_RE = re.compile(r"model_step_(\d+)\.(?:pt|pth|ckpt)$", re.IGNORECASE)
_MOTION_FILE_RE = re.compile(r"^\s*motion_file\s*:\s*(.*?)\s*$")


class CheckpointEvalError(RuntimeError):
    """A checkpoint evaluation request cannot be prepared safely."""


def _resolve_user_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


def _model_step(path: Path) -> int:
    match = _MODEL_STEP_RE.fullmatch(path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint(value: str | os.PathLike[str]) -> Path:
    """Resolve a checkpoint file, accepting an experiment directory as input."""

    requested = _resolve_user_path(value)
    if requested.is_dir():
        last_checkpoint = requested / "last.pt"
        if last_checkpoint.is_file():
            return last_checkpoint

        numbered = sorted(
            (
                path
                for path in requested.iterdir()
                if path.is_file() and _model_step(path) >= 0
            ),
            key=_model_step,
        )
        if numbered:
            return numbered[-1]
        raise CheckpointEvalError(
            f"No last.pt or model_step_*.pt checkpoint was found in: {requested}"
        )

    if not requested.is_file():
        raise CheckpointEvalError(f"Checkpoint does not exist: {requested}")
    if requested.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise CheckpointEvalError(
            f"Expected a {sorted(CHECKPOINT_SUFFIXES)} checkpoint, got: {requested}"
        )
    return requested


def find_training_config(checkpoint: Path) -> Path:
    """Match eval_agent_trl.py's companion-config lookup."""

    for candidate in (checkpoint.parent / "config.yaml", checkpoint.parent.parent / "config.yaml"):
        if candidate.is_file():
            return candidate.resolve()
    raise CheckpointEvalError(
        "The checkpoint needs its saved training config.yaml in the checkpoint "
        f"directory (or its parent): {checkpoint}"
    )


def read_config_motion_path(config_path: Path) -> str | None:
    """Read the first exact ``motion_file`` scalar without importing PyYAML."""

    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = _MOTION_FILE_RE.match(line)
        if not match:
            continue
        value = match.group(1).split(" #", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None


def _motion_match_key(name: str) -> str:
    """Normalize the one filename qualifier removed from the bundled demo."""

    return name.lower().replace("_from_original_bvh", "")


def resolve_motion_override(config_path: Path, explicit_motion: str | None) -> tuple[Path | None, str]:
    """Select an explicit motion or repair a stale saved-config path.

    A return value of ``None`` means the saved config path already resolves and
    should remain untouched.
    """

    if explicit_motion is not None:
        motion = _resolve_user_path(explicit_motion)
        if not motion.exists():
            raise CheckpointEvalError(f"Motion path does not exist: {motion}")
        return motion, "explicit --motion override"

    saved_value = read_config_motion_path(config_path)
    if not saved_value:
        raise CheckpointEvalError(
            f"No motion_file was found in {config_path}; pass --motion explicitly."
        )
    if "${" in saved_value:
        raise CheckpointEvalError(
            f"The saved motion_file is unresolved ({saved_value!r}); pass --motion explicitly."
        )

    saved_path = _resolve_user_path(saved_value, base=REPO_ROOT)
    if saved_path.exists():
        return None, f"saved config motion ({saved_value})"

    normalized_name = Path(saved_value.replace("\\", "/")).name
    bundled_candidate = (BUNDLED_MOTION_DIR / normalized_name).resolve()
    if bundled_candidate.exists():
        return bundled_candidate, (
            f"bundled replacement for missing saved path ({saved_value})"
        )

    # The copied demo was shortened from
    # ``guy_dancing_rby1_from_original_bvh_...`` to
    # ``guy_dancing_rby1_...``. Match only that known qualifier change and
    # require uniqueness so a different motion can never be selected silently.
    match_key = _motion_match_key(normalized_name)
    alias_matches = [
        candidate.resolve()
        for candidate in BUNDLED_MOTION_DIR.glob("*.pkl")
        if _motion_match_key(candidate.name) == match_key
    ]
    if len(alias_matches) == 1:
        return alias_matches[0], (
            f"bundled renamed replacement for missing saved path ({saved_value})"
        )

    raise CheckpointEvalError(
        "The motion referenced by the saved config is missing:\n"
        f"  {saved_path}\n"
        "Pass the matching PKL or motion directory with --motion."
    )


def resolve_python(value: str | None) -> str:
    """Resolve the Python executable used to launch Isaac Lab."""

    requested = value or sys.executable
    expanded = os.path.expandvars(os.path.expanduser(requested))
    has_separator = any(separator in expanded for separator in (os.sep, os.altsep) if separator)
    if has_separator or Path(expanded).is_absolute():
        path = _resolve_user_path(expanded)
        if not path.is_file():
            raise CheckpointEvalError(f"Python executable does not exist: {path}")
        return str(path)

    located = shutil.which(expanded)
    if located is None:
        raise CheckpointEvalError(f"Python executable was not found on PATH: {expanded}")
    return located


def build_eval_command(
    *,
    python_executable: str,
    checkpoint: Path,
    motion_override: Path | None,
    headless: bool,
    num_envs: int,
    encoder: str,
    keep_observation_noise: bool,
    loop: bool,
    max_steps: int,
    export_onnx: bool,
    bridge_destination: str | None = None,
    bridge_startup_timeout: float = 120.0,
    bridge_allow_remote: bool = False,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build a shell-free argv list for ``eval_agent_trl.py``."""

    if num_envs < 1:
        raise CheckpointEvalError("--num-envs must be at least 1")
    if max_steps < 0:
        raise CheckpointEvalError("--max-steps cannot be negative")
    if bridge_destination is not None:
        if export_onnx:
            raise CheckpointEvalError("--sdk-bridge cannot be combined with --export-onnx")
        if loop:
            raise CheckpointEvalError(
                "--sdk-bridge cannot be combined with --loop; hardware mirroring must stop "
                "before Isaac automatically resets the episode"
            )
        if num_envs != 1:
            raise CheckpointEvalError("--sdk-bridge requires --num-envs 1")
        if bridge_startup_timeout <= 0.0:
            raise CheckpointEvalError("--bridge-startup-timeout must be positive")
        if ":" not in bridge_destination:
            raise CheckpointEvalError("--sdk-bridge must use HOST:PORT form")

    effective_headless = headless or export_onnx
    effective_num_envs = 1 if export_onnx else num_envs
    command = [
        python_executable,
        str(EVAL_SCRIPT),
        f"+checkpoint={checkpoint.as_posix()}",
        f"++headless={'true' if effective_headless else 'false'}",
        f"++num_envs={effective_num_envs}",
    ]

    if not keep_observation_noise:
        command.extend(
            [
                "++manager_env.observations.policy.enable_corruption=false",
                "++manager_env.observations.tokenizer.enable_corruption=false",
            ]
        )
    if encoder != "auto":
        command.append(f"++use_encoder={encoder}")
    if motion_override is not None:
        command.append(
            "++manager_env.commands.motion.motion_lib_cfg.motion_file="
            f"{motion_override.as_posix()}"
        )
    if export_onnx:
        command.append("++export_onnx_only=true")
    elif not loop:
        command.append("++run_once=true")
    if max_steps:
        command.append(f"++max_render_steps={max_steps}")

    for override in extra_overrides:
        if "=" not in override:
            raise CheckpointEvalError(
                f"Hydra override must contain '=': {override!r}"
            )
        if bridge_destination is not None:
            normalized_key = override.lstrip("+").split("=", 1)[0]
            protected_keys = {
                "num_envs",
                "run_once",
                "sim_to_sdk_bridge.enabled",
                "sim_to_sdk_bridge.destination",
                "sim_to_sdk_bridge.startup_timeout",
                "sim_to_sdk_bridge.allow_remote",
            }
            if normalized_key in protected_keys:
                raise CheckpointEvalError(
                    f"--override may not replace bridge safety setting {normalized_key!r}"
                )
        command.append(override)

    if bridge_destination is not None:
        command.extend(
            [
                "++sim_to_sdk_bridge.enabled=true",
                f"++sim_to_sdk_bridge.destination={bridge_destination}",
                f"++sim_to_sdk_bridge.startup_timeout={bridge_startup_timeout}",
                "++sim_to_sdk_bridge.allow_remote="
                f"{'true' if bridge_allow_remote else 'false'}",
            ]
        )
    return command


def format_command(command: Sequence[str]) -> str:
    """Render argv in a copyable form for the current platform."""

    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def check_isaaclab_available(python_executable: str) -> None:
    probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util,sys; "
                "sys.exit(0 if importlib.util.find_spec('isaaclab') else 1)"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        suffix = f"\nInterpreter output: {detail}" if detail else ""
        raise CheckpointEvalError(
            "The selected Python interpreter cannot find Isaac Lab. Activate the "
            "training environment or pass --python PATH_TO_ENV_LAB_PYTHON."
            f"\nSelected interpreter: {python_executable}{suffix}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an RBY1 .pt checkpoint in Isaac Lab using its saved training config. "
            "It is simulator-only unless --sdk-bridge is explicitly selected."
        )
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint file, or an experiment directory containing last.pt.",
    )
    parser.add_argument(
        "--motion",
        help=(
            "Matching motion PKL/directory. If omitted, use the saved config path or an "
            "equivalent motion bundled under rby1_hardware/motions."
        ),
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="Python executable from the Isaac Lab training environment (default: current Python).",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim viewer.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of evaluation environments.")
    parser.add_argument(
        "--encoder",
        choices=("g1", "teleop", "smpl", "auto"),
        default="g1",
        help="Reference encoder to use. g1 follows the complete motion-library pose (default).",
    )
    parser.add_argument(
        "--keep-observation-noise",
        action="store_true",
        help="Keep the training-time observation corruption enabled.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep resetting and replaying; by default evaluation exits after one episode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional hard evaluation-step limit (0 means no additional limit).",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the checkpoint's ONNX models/config and exit instead of running an episode.",
    )
    parser.add_argument(
        "--sdk-bridge",
        metavar="HOST:PORT",
        help=(
            "Publish the realized 24-joint Isaac state to rby1_sim_to_sdk.py. "
            "The evaluator waits at frame zero for the receiver and then runs in real time."
        ),
    )
    parser.add_argument(
        "--bridge-startup-timeout",
        type=float,
        default=120.0,
        help="Seconds Isaac waits at frame zero for an SDK bridge receiver (default: 120).",
    )
    parser.add_argument(
        "--allow-remote-bridge",
        action="store_true",
        help="Allow a non-loopback --sdk-bridge destination on an isolated trusted network.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="HYDRA_KEY=VALUE",
        help="Append a raw Hydra override after the safe defaults; repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the command without importing or starting Isaac Lab.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not EVAL_SCRIPT.is_file():
            raise CheckpointEvalError(f"Repository evaluator is missing: {EVAL_SCRIPT}")
        checkpoint = resolve_checkpoint(args.checkpoint)
        config_path = find_training_config(checkpoint)
        motion_override, motion_description = resolve_motion_override(config_path, args.motion)
        python_executable = resolve_python(args.python_executable)
        command = build_eval_command(
            python_executable=python_executable,
            checkpoint=checkpoint,
            motion_override=motion_override,
            headless=args.headless,
            num_envs=args.num_envs,
            encoder=args.encoder,
            keep_observation_noise=args.keep_observation_noise,
            loop=args.loop,
            max_steps=args.max_steps,
            export_onnx=args.export_onnx,
            bridge_destination=args.sdk_bridge,
            bridge_startup_timeout=args.bridge_startup_timeout,
            bridge_allow_remote=args.allow_remote_bridge,
            extra_overrides=args.override,
        )

        print(f"Checkpoint: {checkpoint}", flush=True)
        print(f"Training config: {config_path}", flush=True)
        print(f"Motion: {motion_description}", flush=True)
        target_description = (
            f"Isaac Lab plus simulator-state publisher to {args.sdk_bridge}"
            if args.sdk_bridge
            else "Isaac Lab simulator only"
        )
        print(f"Target: {target_description}", flush=True)
        print(f"Command:\n{format_command(command)}", flush=True)
        if args.dry_run:
            print("Dry run complete; Isaac Lab was not started.", flush=True)
            return 0

        check_isaaclab_available(python_executable)
        print("Starting checkpoint evaluation. Press Ctrl+C to stop early.", flush=True)
        child_environment = os.environ.copy()
        child_environment.setdefault("PYTHONUNBUFFERED", "1")
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=child_environment,
                check=False,
            )
        except KeyboardInterrupt:
            print("Checkpoint evaluation interrupted.", file=sys.stderr)
            return 130
        if completed.returncode != 0:
            raise CheckpointEvalError(
                f"Isaac Lab evaluation exited with status {completed.returncode}."
            )
        return 0
    except (CheckpointEvalError, OSError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
