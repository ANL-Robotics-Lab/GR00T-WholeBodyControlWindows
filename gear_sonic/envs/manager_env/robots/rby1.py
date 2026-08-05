# rby1_sonic_24dof_draft.py
#
# Structural starter for adapting the GR00T/SONIC G1 robot config to RBY1.
#
# Assumptions used here:
#   1) 24-DOF policy embodiment:
#        wheels (2) + torso (6) + arms (14) + head (2)
#   2) Finger gripper joints are excluded from the first SONIC training pass.
#   3) The motion-library MJCF should expose a "collapsed" body representation:
#        25 bodies = root/base + one body per active DOF.
#      In other words, fixed decorative bodies such as EE_BODY_R / EE_BODY_L /
#      link_head_0 should not appear in the body arrays that SONIC reorders.
#
# Why this matters:
#   The current GR00T order converter assumes body tensors are either:
#      [num_dof] or [num_dof + 1] bodies.
#   For a 24-DOF RBY1 training embodiment, that means 24 or 25 bodies.
#
# IMPORTANT:
#   The mapping arrays below are updated from your Isaac Sim dynamic_control
#   articulation log for /World/envs/env_0/Robot/model. The imported USD has
#   38 bodies, not the 25 collapsed bodies assumed by the first draft.
#   For SONIC motion matching, filter the full body tensor to the 25 selected
#   bodies before applying collapsed-body mapping/reward logic.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from gear_sonic.utils.rby1_order import (
    RBY1_FULL_BODY_TO_COLLAPSED_25,
    RBY1_FULL_ISAACLAB_BODY_NAMES,
    RBY1_FULL_ISAACLAB_TO_MUJOCO_BODY,
    RBY1_ISAACLAB_BODY_NAMES_25,
    RBY1_ISAACLAB_DOF_NAMES,
    RBY1_ISAACLAB_TO_MUJOCO_BODY,
    RBY1_ISAACLAB_TO_MUJOCO_DOF,
    RBY1_MOTION_TO_ISAACLAB_DOF_SIGN,
    RBY1_MUJOCO_BODY_NAMES,
    RBY1_MUJOCO_DOF_NAMES,
    RBY1_MUJOCO_TO_ISAACLAB_BODY,
    RBY1_MUJOCO_TO_ISAACLAB_DOF,
    RBY1_POSITION_ACTION_SCALE,
)

ASSET_DIR = "gear_sonic/data/assets"

# ---------------------------------------------------------------------
# 1) ORDERING — verified from your Isaac Sim dynamic_control log
# ---------------------------------------------------------------------
# The imported USD is NOT the collapsed 25-body training model assumed in
# the first draft.  The actual PhysX articulation log shows 38 links, with
# caster/backwheel, axle, FT sensor, end-effector, and finger bodies inserted
# into the tree.  The 24-DOF policy embodiment below therefore uses:
#   - 24 selected actuated joints for policy / motion-lib DOFs
#   - a 25-body collapsed subset for SONIC motion matching
#   - an explicit full-body selector when the source tensor is the full 38-body
#     Isaac articulation tensor.
#
# IMPORTANT BODY TENSOR RULE:
#   If your tensor is full Isaac body state with 38 bodies, first select:
#       body_state_25 = body_state_full[:, RBY1_FULL_BODY_TO_COLLAPSED_25, ...]
#   Then apply the 25-body reorder arrays if your SONIC code expects them.
# ---------------------------------------------------------------------

# Backward-compatible alias for code that still expects this key/name.
# NOTE: this is now DOF names, not body names.
RBY1_ISAACLAB_JOINTS = RBY1_ISAACLAB_DOF_NAMES

# Full-joint names printed alongside the 38 links.  Several are fixed or passive
# and are intentionally not part of the 24-DOF policy embodiment.
RBY1_FULL_ISAACLAB_JOINT_NAMES = [
    None,
    "left_wheel",
    "torso_0",
    "Axil1",
    "Axil2",
    "right_wheel",
    "torso_1",
    "backwheel",
    "backwheel2",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
    "head_base",
    "left_arm_0",
    "right_arm_0",
    "head_0",
    "left_arm_1",
    "right_arm_1",
    "head_1",
    "left_arm_2",
    "right_arm_2",
    "left_arm_3",
    "right_arm_3",
    "left_arm_4",
    "right_arm_4",
    "left_arm_5",
    "right_arm_5",
    "left_arm_6",
    "right_arm_6",
    "tool_left",
    "tool_right",
    "FT_Sensor_END_left",
    "FT_Sensor_END_right",
    "FixedJoint",
    "FixedJoint0",
    "gripper_finger_r1",
    "gripper_finger_r2",
]

# ---------------------------------------------------------------------
# 2) DOF MAPPINGS — 24 selected DOFs
#
# Mapping convention follows GR00T:
#   output[i] = input[mapping[i]]
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# 3) BODY MAPPINGS — 25 selected/collapsed bodies
#
# If the tensor has full Isaac body order, use RBY1_FULL_ISAACLAB_TO_MUJOCO_BODY.
# If the tensor has already been filtered to RBY1_ISAACLAB_BODY_NAMES_25 order,
# use RBY1_ISAACLAB_TO_MUJOCO_BODY and RBY1_MUJOCO_TO_ISAACLAB_BODY below.
# ---------------------------------------------------------------------

RBY1_ISAACLAB_TO_MUJOCO_MAPPING = {
    # Backward-compatible GR00T/SONIC keys.
    "isaaclab_joints": RBY1_ISAACLAB_DOF_NAMES,
    "isaaclab_to_mujoco_dof": RBY1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": RBY1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": RBY1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": RBY1_MUJOCO_TO_ISAACLAB_BODY,

    # New explicit names/selectors for your non-collapsed USD.
    "isaaclab_dof_names": RBY1_ISAACLAB_DOF_NAMES,
    "mujoco_dof_names": RBY1_MUJOCO_DOF_NAMES,
    # PKL/clean-MJCF wheel axes are -Y; the loaded USD wheel axes are +Y.
    "motion_to_isaaclab_dof_sign": RBY1_MOTION_TO_ISAACLAB_DOF_SIGN,
    "isaaclab_body_names_25": RBY1_ISAACLAB_BODY_NAMES_25,
    "mujoco_body_names": RBY1_MUJOCO_BODY_NAMES,
    "full_isaaclab_body_names": RBY1_FULL_ISAACLAB_BODY_NAMES,
    "full_body_to_collapsed_25": RBY1_FULL_BODY_TO_COLLAPSED_25,
    "full_isaaclab_to_mujoco_body": RBY1_FULL_ISAACLAB_TO_MUJOCO_BODY,
}

# ---------------------------------------------------------------------
# 4) ACTUATION STARTING POINT
#
# Effort/velocity limits below are copied from the uploaded RBY1 URDF
# where available. Wheel effort is not specified there, so it is left as
# a clearly marked placeholder.
#
# The armature / stiffness / damping values are NOT RBY1-validated.
# They are starter placeholders patterned after the GR00T G1/H2 configs.
# Replace/tune them from RBY1 motor specs and single-env stability tests.
# ---------------------------------------------------------------------

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

ARMATURE_WHEEL = 0.010177520          # TODO: replace/tune for RBY1 wheels
ARMATURE_TORSO_HEAVY = 0.025101925    # placeholder
ARMATURE_TORSO_LIGHT = 0.010177520    # placeholder
ARMATURE_ARM_PROX = 0.010177520       # placeholder
ARMATURE_ARM_DISTAL = 0.00425         # placeholder
ARMATURE_HEAD = 0.00425               # placeholder

def _kp(armature: float) -> float:
    return armature * NATURAL_FREQ**2

def _kd(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ

STIFFNESS_WHEEL = _kp(ARMATURE_WHEEL)
DAMPING_WHEEL = _kd(ARMATURE_WHEEL)

STIFFNESS_TORSO_HEAVY = _kp(ARMATURE_TORSO_HEAVY)
DAMPING_TORSO_HEAVY = _kd(ARMATURE_TORSO_HEAVY)

STIFFNESS_TORSO_LIGHT = _kp(ARMATURE_TORSO_LIGHT)
DAMPING_TORSO_LIGHT = _kd(ARMATURE_TORSO_LIGHT)

STIFFNESS_ARM_PROX = _kp(ARMATURE_ARM_PROX)
DAMPING_ARM_PROX = _kd(ARMATURE_ARM_PROX)

STIFFNESS_ARM_DISTAL = _kp(ARMATURE_ARM_DISTAL)
DAMPING_ARM_DISTAL = _kd(ARMATURE_ARM_DISTAL)

STIFFNESS_HEAD = _kp(ARMATURE_HEAD)
DAMPING_HEAD = _kd(ARMATURE_HEAD)

WHEEL_EFFORT_LIMIT = 120.0  # TODO: replace with RBY1 wheel actuator spec
WHEEL_VELOCITY_LIMIT = 15.707963268
# Normalized action +/-1 maps to +/-10 rad/s. This covers the smoothed,
# slowed dance reference while leaving margin below the simulated drive limit.
RBY1_WHEEL_ACTION_SCALE = 10.0
# Velocity feedback gain: a 10 rad/s error can request the available 120 Nm.
# The simulator's effort limit still provides the final torque clamp.
RBY1_WHEEL_VELOCITY_DAMPING = WHEEL_EFFORT_LIMIT / RBY1_WHEEL_ACTION_SCALE

# ---------------------------------------------------------------------
# 5) ISAAC LAB ARTICULATION CONFIG
# ---------------------------------------------------------------------

RBY1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/bcarc_ziwaj0x/Downloads/TransuranicVer/GantrySystemV4/Collected_simplifiedTWPC/flatremovedRBY3.usd",
        activate_contact_sensors=True,
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Verify this in Isaac Sim. For a wheeled base, start slightly above the floor.
        pos=(0.0, 0.0, 0.02),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel", "right_wheel"],
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=WHEEL_VELOCITY_LIMIT,
            # A non-zero position gain fights continuously rotating wheel
            # targets. RBY1MixedJointAction supplies velocity targets instead.
            stiffness=0.0,
            damping=RBY1_WHEEL_VELOCITY_DAMPING,
            armature=ARMATURE_WHEEL,
        ),
        "torso_heavy": ImplicitActuatorCfg(
            joint_names_expr=["torso_0", "torso_1", "torso_2"],
            effort_limit_sim={
                "torso_0": 270.0,
                "torso_1": 270.0,
                "torso_2": 270.0,
            },
            velocity_limit_sim={
                "torso_0": 2.09439510,
                "torso_1": 2.09439510,
                "torso_2": 2.09439510,
            },
            stiffness=STIFFNESS_TORSO_HEAVY,
            damping=DAMPING_TORSO_HEAVY,
            armature=ARMATURE_TORSO_HEAVY,
        ),
        "torso_light": ImplicitActuatorCfg(
            joint_names_expr=["torso_3", "torso_4", "torso_5"],
            effort_limit_sim={
                "torso_3": 120.0,
                "torso_4": 120.0,
                "torso_5": 120.0,
            },
            velocity_limit_sim={
                "torso_3": 3.141592654,
                "torso_4": 3.141592654,
                "torso_5": 3.141592654,
            },
            stiffness=STIFFNESS_TORSO_LIGHT,
            damping=DAMPING_TORSO_LIGHT,
            armature=ARMATURE_TORSO_LIGHT,
        ),
        "arms_proximal": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_arm_0", "right_arm_1", "right_arm_2",
                "left_arm_0", "left_arm_1", "left_arm_2",
            ],
            effort_limit_sim={
                "right_arm_0": 70.0, "right_arm_1": 70.0, "right_arm_2": 70.0,
                "left_arm_0": 70.0, "left_arm_1": 70.0, "left_arm_2": 70.0,
            },
            velocity_limit_sim={
                "right_arm_0": 3.141592654, "right_arm_1": 3.141592654, "right_arm_2": 3.141592654,
                "left_arm_0": 3.141592654, "left_arm_1": 3.141592654, "left_arm_2": 3.141592654,
            },
            stiffness=STIFFNESS_ARM_PROX,
            damping=DAMPING_ARM_PROX,
            armature=ARMATURE_ARM_PROX,
        ),
        "arms_elbow": ImplicitActuatorCfg(
            joint_names_expr=["right_arm_3", "left_arm_3"],
            effort_limit_sim={
                "right_arm_3": 40.0,
                "left_arm_3": 40.0,
            },
            velocity_limit_sim={
                "right_arm_3": 3.141592654,
                "left_arm_3": 3.141592654,
            },
            stiffness=STIFFNESS_ARM_PROX,
            damping=DAMPING_ARM_PROX,
            armature=ARMATURE_ARM_PROX,
        ),
        "arms_distal": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_arm_4", "right_arm_5", "right_arm_6",
                "left_arm_4", "left_arm_5", "left_arm_6",
            ],
            effort_limit_sim={
                "right_arm_4": 10.0, "right_arm_5": 10.0, "right_arm_6": 8.0,
                "left_arm_4": 10.0, "left_arm_5": 10.0, "left_arm_6": 8.0,
            },
            velocity_limit_sim={
                "right_arm_4": 6.283185308, "right_arm_5": 6.283185308, "right_arm_6": 2.094395102,
                "left_arm_4": 6.283185308, "left_arm_5": 6.283185308, "left_arm_6": 2.094395102,
            },
            stiffness=STIFFNESS_ARM_DISTAL,
            damping=DAMPING_ARM_DISTAL,
            armature=ARMATURE_ARM_DISTAL,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_0", "head_1"],
            effort_limit_sim={
                "head_0": 1000.0,
                "head_1": 1000.0,
            },
            velocity_limit_sim={
                "head_0": 3.14,
                "head_1": 3.14,
            },
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HEAD,
        ),
    },
)

# ---------------------------------------------------------------------
# 6) ACTION SCALE
#
# Position targets use an explicit motion-sized normalization. Deriving this
# from effort / stiffness made distal arm scales as small as 0.119 rad, forcing
# the policy to emit values around 10-14 to reproduce the reference dance.
# The explicit values keep the same reference near unit action magnitude.
# ---------------------------------------------------------------------

RBY1_ACTION_SCALE = dict(RBY1_POSITION_ACTION_SCALE)

# Wheels use velocity rather than position targets, so their action scale is
# specified directly instead of being derived from effort / position stiffness.
RBY1_ACTION_SCALE["left_wheel"] = RBY1_WHEEL_ACTION_SCALE
RBY1_ACTION_SCALE["right_wheel"] = RBY1_WHEEL_ACTION_SCALE
