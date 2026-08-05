from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch
import yaml


REWARDS_PATH = (
    Path(__file__).resolve().parents[1] / "envs" / "manager_env" / "mdp" / "rewards.py"
)
RBY1_REWARD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "manager_env"
    / "rewards"
    / "tracking"
    / "rby1_motion.yaml"
)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., 1:] *= -1.0
    return result / q.square().sum(dim=-1, keepdim=True)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    uv = torch.cross(q_xyz, vector, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


def _quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    relative = _quat_mul(q1, _quat_inv(q2))
    return 2.0 * torch.acos(relative[..., 0].abs().clamp(max=1.0))


def _get_heading_q(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., 1:3] = 0.0
    return result / torch.linalg.norm(result, dim=-1, keepdim=True)


@pytest.fixture
def rewards_module(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = object
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    utils_math = ModuleType("isaaclab.utils.math")
    utils_math.quat_apply = _quat_apply
    utils_math.quat_error_magnitude = _quat_error_magnitude
    utils_math.quat_inv = _quat_inv
    utils_math.quat_mul = _quat_mul

    commands = ModuleType("gear_sonic.envs.manager_env.mdp.commands")
    commands.ForceTrackingCommand = object
    commands.TrackingCommand = object
    commands._get_body_indexes = lambda command, names: [
        command.cfg.body_names.index(name) for name in names
    ]

    transform = ModuleType("gear_sonic.trl.utils.torch_transform")
    transform.get_heading_q = _get_heading_q

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": utils_math,
        "gear_sonic.envs.manager_env.mdp.commands": commands,
        "gear_sonic.trl.utils.torch_transform": transform,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("rby1_rewards_under_test", REWARDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _env(command, *, mapping=None):
    manager = SimpleNamespace(get_term=lambda _name: command)
    cfg = SimpleNamespace(isaaclab_to_mujoco_mapping=mapping or {})
    return SimpleNamespace(command_manager=manager, cfg=cfg, num_envs=command.anchor_pos_w.shape[0])


def test_rby1_config_prioritizes_upper_body_joints_over_base():
    config = yaml.safe_load(RBY1_REWARD_CONFIG_PATH.read_text())
    joint_term = config["tracking_upper_body_joint_pos"]
    base_term_names = [
        "tracking_planar_anchor_pos",
        "tracking_anchor_yaw",
        "tracking_base_forward_vel",
        "tracking_base_yaw_rate",
        "tracking_wheel_vel",
    ]
    base_weight_sum = sum(config[name]["weight"] for name in base_term_names)

    # Preserve a decisive upper-body signal even when every easy base reward is maximal.
    assert joint_term["weight"] >= 4.0 * base_weight_sum
    assert len(joint_term["params"]["joint_names"]) == 22
    assert len(joint_term["params"]["joint_weights"]) == 22
    assert joint_term["params"]["joint_weights"][:6] == [2.0] * 6


def test_planar_base_rewards_ignore_height_roll_and_lateral_speed(rewards_module):
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    roll_90 = torch.tensor([[2**-0.5, 2**-0.5, 0.0, 0.0]])
    command = SimpleNamespace(
        anchor_pos_w=torch.tensor([[1.0, 2.0, 100.0]]),
        robot_anchor_pos_w=torch.tensor([[1.0, 2.0, -100.0]]),
        anchor_quat_w=identity,
        robot_anchor_quat_w=roll_90,
        anchor_lin_vel_w=torch.tensor([[1.0, 9.0, 0.0]]),
        robot_anchor_lin_vel_w=torch.tensor([[1.0, -9.0, 0.0]]),
        anchor_ang_vel_w=torch.tensor([[8.0, -7.0, 0.5]]),
        robot_anchor_ang_vel_w=torch.tensor([[-8.0, 7.0, 0.5]]),
    )
    env = _env(command)

    torch.testing.assert_close(
        rewards_module.tracking_planar_anchor_pos_error(env, "motion", 0.3),
        torch.ones(1),
    )
    torch.testing.assert_close(
        rewards_module.tracking_anchor_yaw_error(env, "motion", 0.4),
        torch.ones(1),
    )
    torch.testing.assert_close(
        rewards_module.tracking_rby1_forward_velocity_error(env, "motion", 0.5),
        torch.ones(1),
    )
    torch.testing.assert_close(
        rewards_module.tracking_rby1_yaw_rate_error(env, "motion", 1.5),
        torch.ones(1),
    )


def test_wheel_velocity_reward_resolves_reference_and_robot_by_name(rewards_module):
    command = SimpleNamespace(
        anchor_pos_w=torch.zeros(1, 3),
        joint_vel=torch.tensor([[2.0, -3.0, 99.0]]),
        robot_joint_vel=torch.tensor([[-3.0, 99.0, 2.0]]),
        robot=SimpleNamespace(joint_names=["left_wheel", "torso_0", "right_wheel"]),
    )
    env = _env(
        command,
        mapping={"mujoco_dof_names": ["right_wheel", "left_wheel", "torso_0"]},
    )
    reward = rewards_module.tracking_rby1_wheel_velocity_error(
        env, "motion", std=3.0, joint_names=["right_wheel", "left_wheel"]
    )
    torch.testing.assert_close(reward, torch.ones(1))


def test_joint_position_reward_resolves_order_and_applies_weights(rewards_module):
    command = SimpleNamespace(
        anchor_pos_w=torch.zeros(1, 3),
        # Reference order: arm, torso, head.
        joint_pos=torch.tensor([[3.0, 1.0, 5.0]]),
        # Robot order: head, arm, torso. Torso and arm each differ by one radian;
        # head matches and should contribute no error.
        robot_joint_pos=torch.tensor([[5.0, 2.0, 0.0]]),
        robot=SimpleNamespace(joint_names=["head_0", "right_arm_0", "torso_0"]),
    )
    env = _env(
        command,
        mapping={"mujoco_dof_names": ["right_arm_0", "torso_0", "head_0"]},
    )

    reward = rewards_module.tracking_rby1_joint_pos_error(
        env,
        "motion",
        std=1.0,
        joint_names=["torso_0", "right_arm_0", "head_0"],
        joint_weights=[2.0, 1.0, 0.5],
    )

    expected_mean_squared_error = torch.tensor([(2.0 + 1.0) / 3.5])
    torch.testing.assert_close(reward, torch.exp(-expected_mean_squared_error))


def test_joint_position_reward_rejects_missing_mapping_joint(rewards_module):
    command = SimpleNamespace(
        anchor_pos_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 1),
        robot_joint_pos=torch.zeros(1, 1),
        robot=SimpleNamespace(joint_names=["torso_0"]),
    )
    env = _env(command, mapping={"mujoco_dof_names": ["different_joint"]})

    with pytest.raises(ValueError, match="Missing reference"):
        rewards_module.tracking_rby1_joint_pos_error(
            env, "motion", std=1.0, joint_names=["torso_0"]
        )


def test_relative_body_velocity_rewards_remove_rigid_base_motion(rewards_module):
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=["link"]),
        anchor_pos_w=torch.tensor([[0.0, 0.0, 0.0]]),
        robot_anchor_pos_w=torch.tensor([[10.0, -4.0, 0.0]]),
        anchor_quat_w=identity,
        robot_anchor_quat_w=identity,
        anchor_lin_vel_w=torch.tensor([[4.0, 5.0, 0.0]]),
        robot_anchor_lin_vel_w=torch.tensor([[-3.0, 8.0, 0.0]]),
        anchor_ang_vel_w=torch.tensor([[0.0, 0.0, 2.0]]),
        robot_anchor_ang_vel_w=torch.tensor([[0.0, 0.0, -3.0]]),
        body_pos_w=torch.tensor([[[1.0, 0.0, 0.0]]]),
        robot_body_pos_w=torch.tensor([[[11.0, -4.0, 0.0]]]),
        body_lin_vel_w=torch.tensor([[[4.0, 7.0, 0.0]]]),
        robot_body_lin_vel_w=torch.tensor([[[-3.0, 5.0, 0.0]]]),
        body_ang_vel_w=torch.tensor([[[1.0, 0.0, 2.0]]]),
        robot_body_ang_vel_w=torch.tensor([[[1.0, 0.0, -3.0]]]),
    )
    env = _env(command)

    torch.testing.assert_close(
        rewards_module.tracking_rby1_relative_body_linvel_error(
            env, "motion", std=1.0, body_names=["link"]
        ),
        torch.ones(1),
    )
    torch.testing.assert_close(
        rewards_module.tracking_rby1_relative_body_angvel_error(
            env, "motion", std=1.0, body_names=["link"]
        ),
        torch.ones(1),
    )


def test_upper_body_reward_points_support_three_weighted_points(rewards_module):
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = SimpleNamespace(
        cfg=SimpleNamespace(reward_point_body=["torso", "left_tool", "right_tool"]),
        anchor_pos_w=torch.zeros(1, 3),
        robot_anchor_pos_w=torch.zeros(1, 3),
        anchor_quat_w=identity,
        robot_anchor_quat_w=identity,
        reward_point_body_pos_w=torch.zeros(1, 3, 3),
        robot_reward_point_body_pos_w=torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
    )
    env = _env(command)
    reward = rewards_module.tracking_local_reward_points_error(
        env, "motion", std=1.0, point_weights=[1.0, 2.0, 2.0]
    )
    torch.testing.assert_close(reward, torch.exp(torch.tensor([-3.0 / 5.0])))
