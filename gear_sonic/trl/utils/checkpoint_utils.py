"""Checkpoint loading helpers for lightweight policy evaluation.

Training checkpoints contain a few trainer configuration/state objects in
addition to tensors.  Importing their original classes pulls the full
Transformers/Datasets stack into the process.  On Windows, loading PyArrow
after Isaac Sim has initialized can terminate the process in native code.

Evaluation only needs the policy state dict and ``state.global_step``.  The
unpickler below replaces known training-only metadata classes with inert local
objects while leaving tensors and standard Python containers untouched.
"""

from __future__ import annotations

import pickle
import types
from pathlib import Path
from typing import Any


class _CheckpointMetadata:
    """Inert replacement for training-only objects stored in checkpoints."""


def _serialized_enum_value(value: Any) -> Any:
    """Keep an enum's serialized scalar value without importing its package."""

    return value


_METADATA_GLOBALS = frozenset(
    {
        ("trl.experimental.ppo.ppo_trainer", "OnlineTrainerState"),
        # Older checkpoints used one of these pre-TRL-0.28 locations.
        ("trl.trainer.ppo_trainer", "OnlineTrainerState"),
        ("trl.trainer.utils", "OnlineTrainerState"),
        ("trl.trainer.ppo_config", "PPOConfig"),
        ("transformers.trainer_pt_utils", "AcceleratorConfig"),
        ("accelerate.state", "PartialState"),
    }
)

_ENUM_GLOBALS = frozenset(
    {
        ("transformers.trainer_utils", "HubStrategy"),
        ("transformers.trainer_utils", "IntervalStrategy"),
        ("transformers.trainer_utils", "SaveStrategy"),
        ("transformers.trainer_utils", "SchedulerType"),
        ("transformers.training_args", "OptimizerNames"),
        ("accelerate.utils.dataclasses", "DistributedType"),
    }
)


class _EvalCheckpointUnpickler(pickle.Unpickler):
    """Unpickle tensor checkpoints without importing training-only packages."""

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in _METADATA_GLOBALS:
            return _CheckpointMetadata
        if key in _ENUM_GLOBALS:
            return _serialized_enum_value
        return super().find_class(module, name)


_EVAL_PICKLE_MODULE = types.ModuleType("gear_sonic_eval_checkpoint_pickle")
_EVAL_PICKLE_MODULE.Unpickler = _EvalCheckpointUnpickler
_EVAL_PICKLE_MODULE.Pickler = pickle.Pickler
_EVAL_PICKLE_MODULE.load = pickle.load
_EVAL_PICKLE_MODULE.loads = pickle.loads
_EVAL_PICKLE_MODULE.dump = pickle.dump
_EVAL_PICKLE_MODULE.dumps = pickle.dumps


def load_eval_checkpoint(path: str | Path, *, map_location: Any) -> dict[str, Any]:
    """Load a trusted training checkpoint without its heavyweight trainer imports.

    This retains the existing ``weights_only=False`` behavior because the saved
    metadata includes containers (notably ``deque``) that PyTorch's restricted
    weights-only unpickler cannot reconstruct.  Callers must therefore only use
    checkpoints from trusted sources, just as with the previous eval loader.
    """

    import torch

    checkpoint = torch.load(
        str(path),
        map_location=map_location,
        pickle_module=_EVAL_PICKLE_MODULE,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(checkpoint).__name__}")
    return checkpoint
