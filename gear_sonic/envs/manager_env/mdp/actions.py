from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class RBY1MixedJointAction(JointAction):
    """Apply position targets to the upper body and velocity targets to the wheels.

    RBY1's existing policy/wrapper interface is a single vector in articulation
    joint order. Splitting the wheels into a second action term would change that
    ordering, so this term deliberately keeps one action vector and dispatches
    its entries by joint name when actions are applied.
    """

    cfg: RBY1MixedJointActionCfg

    def __init__(self, cfg: RBY1MixedJointActionCfg, env):
        super().__init__(cfg, env)

        missing_wheels = [name for name in cfg.wheel_joint_names if name not in self._joint_names]
        if missing_wheels:
            raise ValueError(
                f"RBY1 mixed action could not resolve wheel joints {missing_wheels}. "
                f"Resolved action joints: {self._joint_names}"
            )

        unknown_passive = [
            name for name in cfg.passive_joint_names if name not in self._joint_names
        ]
        # Clean 24-DOF RBY1 assets omit the passive casters, so their absence is
        # expected. Any passive name that is present is simply left uncommanded.
        passive_names = set(cfg.passive_joint_names) - set(unknown_passive)
        wheel_names = set(cfg.wheel_joint_names)

        self._wheel_action_ids = [
            index for index, name in enumerate(self._joint_names) if name in wheel_names
        ]
        self._position_action_ids = [
            index
            for index, name in enumerate(self._joint_names)
            if name not in wheel_names and name not in passive_names
        ]
        self._wheel_joint_ids = [
            self._asset.joint_names.index(self._joint_names[index])
            for index in self._wheel_action_ids
        ]
        self._position_joint_ids = [
            self._asset.joint_names.index(self._joint_names[index])
            for index in self._position_action_ids
        ]

        if cfg.use_default_offset:
            resolved_joint_ids = [
                self._asset.joint_names.index(name) for name in self._joint_names
            ]
            self._offset = self._asset.data.default_joint_pos[:, resolved_joint_ids].clone()
            self._offset[:, self._wheel_action_ids] = self._asset.data.default_joint_vel[
                :, self._wheel_joint_ids
            ]

    def apply_actions(self):
        """Send each processed action to the appropriate drive mode."""
        if self._position_action_ids:
            self._asset.set_joint_position_target(
                self.processed_actions[:, self._position_action_ids],
                joint_ids=self._position_joint_ids,
            )

        wheel_velocity = self.processed_actions[:, self._wheel_action_ids]
        if self.cfg.wheel_velocity_limit is not None:
            wheel_velocity = wheel_velocity.clamp(
                min=-self.cfg.wheel_velocity_limit,
                max=self.cfg.wheel_velocity_limit,
            )
        self._asset.set_joint_velocity_target(
            wheel_velocity,
            joint_ids=self._wheel_joint_ids,
        )


@configclass
class RBY1MixedJointActionCfg(JointActionCfg):
    """Configuration for :class:`RBY1MixedJointAction`."""

    class_type: type[ActionTerm] = RBY1MixedJointAction
    wheel_joint_names: tuple[str, ...] = ("right_wheel", "left_wheel")
    passive_joint_names: tuple[str, ...] = ("backwheel", "backwheel2")
    wheel_velocity_limit: float | None = 15.707963268
    use_default_offset: bool = True


# Joint ordering constants
G1_MUJOCO_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = None
