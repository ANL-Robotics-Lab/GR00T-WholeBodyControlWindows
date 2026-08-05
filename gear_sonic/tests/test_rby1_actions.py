from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from gear_sonic.utils.rby1_order import (
    RBY1_MUJOCO_DOF_NAMES,
    RBY1_POSITION_ACTION_SCALE,
)


ACTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "envs" / "manager_env" / "mdp" / "actions.py"
)


@pytest.fixture
def actions_module(monkeypatch):
    class FakeJointAction:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._asset = env.asset
            self._joint_names = list(env.asset.joint_names)
            self._processed_actions = torch.zeros(1, len(self._joint_names))
            self._offset = 0.0

        @property
        def processed_actions(self):
            return self._processed_actions

    isaaclab = ModuleType("isaaclab")
    envs = ModuleType("isaaclab.envs")
    mdp = ModuleType("isaaclab.envs.mdp")
    action_package = ModuleType("isaaclab.envs.mdp.actions")
    actions_cfg = ModuleType("isaaclab.envs.mdp.actions.actions_cfg")
    actions_cfg.JointActionCfg = object
    joint_actions = ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    joint_actions.JointAction = FakeJointAction
    managers = ModuleType("isaaclab.managers")
    managers.ActionTerm = object
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp,
        "isaaclab.envs.mdp.actions": action_package,
        "isaaclab.envs.mdp.actions.actions_cfg": actions_cfg,
        "isaaclab.envs.mdp.actions.joint_actions": joint_actions,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("rby1_actions_under_test", ACTIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mixed_action_dispatches_wheels_by_velocity_and_ignores_casters(actions_module):
    class FakeAsset:
        joint_names = ["left_wheel", "torso_0", "backwheel", "right_wheel", "gripper"]

        def __init__(self):
            self.data = SimpleNamespace(
                default_joint_pos=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
                default_joint_vel=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            )
            self.position_call = None
            self.velocity_call = None

        def set_joint_position_target(self, target, *, joint_ids):
            self.position_call = (target.clone(), list(joint_ids))

        def set_joint_velocity_target(self, target, *, joint_ids):
            self.velocity_call = (target.clone(), list(joint_ids))

    asset = FakeAsset()
    cfg = SimpleNamespace(
        wheel_joint_names=["right_wheel", "left_wheel"],
        passive_joint_names=["backwheel", "backwheel2"],
        wheel_velocity_limit=15.0,
        use_default_offset=True,
    )
    action = actions_module.RBY1MixedJointAction(cfg, SimpleNamespace(asset=asset))

    # Position defaults remain offsets for articulated joints, while wheel
    # entries use default velocity offsets.
    torch.testing.assert_close(
        action._offset,
        torch.tensor([[0.1, 2.0, 3.0, 0.4, 5.0]]),
    )

    action._processed_actions = torch.tensor([[20.0, 2.5, 99.0, -20.0, 5.5]])
    action.apply_actions()

    position_target, position_ids = asset.position_call
    velocity_target, velocity_ids = asset.velocity_call
    assert position_ids == [1, 4]
    torch.testing.assert_close(position_target, torch.tensor([[2.5, 5.5]]))
    assert velocity_ids == [0, 3]
    torch.testing.assert_close(velocity_target, torch.tensor([[15.0, -15.0]]))


def test_position_action_scale_covers_every_upper_body_motion_dof():
    expected = set(RBY1_MUJOCO_DOF_NAMES) - {"right_wheel", "left_wheel"}

    assert set(RBY1_POSITION_ACTION_SCALE) == expected
    assert all(scale > 0.0 for scale in RBY1_POSITION_ACTION_SCALE.values())
