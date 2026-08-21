from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import torch

from gear_sonic.trl.utils.checkpoint_utils import load_eval_checkpoint


def test_eval_checkpoint_load_replaces_trainer_state_without_import(tmp_path: Path) -> None:
    module_name = "trl.experimental.ppo.ppo_trainer"
    previous_module = sys.modules.pop(module_name, None)
    trainer_module = ModuleType(module_name)
    trainer_state_type = type("OnlineTrainerState", (), {})
    trainer_state_type.__module__ = module_name
    trainer_module.OnlineTrainerState = trainer_state_type
    sys.modules[module_name] = trainer_module

    try:
        state = trainer_state_type()
        state.global_step = 123
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(
            {
                "policy_state_dict": {"weight": torch.tensor([1.0, 2.0])},
                "state": state,
            },
            checkpoint_path,
        )

        del sys.modules[module_name]
        checkpoint = load_eval_checkpoint(checkpoint_path, map_location="cpu")

        assert checkpoint["state"].global_step == 123
        torch.testing.assert_close(
            checkpoint["policy_state_dict"]["weight"],
            torch.tensor([1.0, 2.0]),
        )
        assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
