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
#   The mapping arrays below are derived from:
#     - IsaacLab-style breadth-first articulated traversal inferred from the
#       G1 config pattern
#     - MuJoCo depth-first active-joint order in the uploaded RBY1 XML,
#       with finger joints omitted for the 24-DOF baseline
#
#   NVIDIA recommends verifying the actual IsaacLab body_names/joint_names
#   after importing your URDF. Do that before launching long training runs.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# ---------------------------------------------------------------------
# 1) BODY ORDERING — IsaacLab traversal order, 25 bodies total
# ---------------------------------------------------------------------

RBY1_ISAACLAB_JOINTS = [
    "base",
    "wheel_r",
    "wheel_l",
    "link_torso_0",
    "link_torso_1",
    "link_torso_2",
    "link_torso_3",
    "link_torso_4",
    "link_torso_5",
    "link_right_arm_0",
    "link_left_arm_0",
    "link_head_1",
    "link_right_arm_1",
    "link_left_arm_1",
    "link_head_2",
    "link_right_arm_2",
    "link_left_arm_2",
    "link_right_arm_3",
    "link_left_arm_3",
    "link_right_arm_4",
    "link_left_arm_4",
    "link_right_arm_5",
    "link_left_arm_5",
    "link_right_arm_6",
    "link_left_arm_6",
]

# ---------------------------------------------------------------------
# 2) DOF ORDERING — 24 DOFs
#
# IsaacLab inferred order:
#   right_wheel, left_wheel, torso_0..5,
#   right_arm_0, left_arm_0, head_0,
#   right_arm_1, left_arm_1, head_1,
#   right_arm_2, left_arm_2, ...,
#   right_arm_6, left_arm_6
#
# MuJoCo active-joint order used here:
#   right_wheel, left_wheel, torso_0..5,
#   right_arm_0..6, left_arm_0..6, head_0, head_1
#
# Mapping convention follows GR00T:
#   output[i] = input[mapping[i]]
# ---------------------------------------------------------------------

RBY1_ISAACLAB_TO_MUJOCO_DOF = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 11, 14, 16, 18, 20, 22,
    9, 12, 15, 17, 19, 21, 23,
    10, 13,
]

RBY1_MUJOCO_TO_ISAACLAB_DOF = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 15, 22, 9, 16, 23, 10,
    17, 11, 18, 12, 19, 13, 20,
    14, 21,
]

# ---------------------------------------------------------------------
# 3) BODY MAPPINGS — 25 bodies
#
# Collapsed MuJoCo body order used here:
#   base,
#   wheel_r, wheel_l,
#   link_torso_0..5,
#   link_right_arm_0..6,
#   link_left_arm_0..6,
#   link_head_1, link_head_2
# ---------------------------------------------------------------------

RBY1_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 12, 15, 17, 19, 21, 23,
    10, 13, 16, 18, 20, 22, 24,
    11, 14,
]

RBY1_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 16, 23, 10, 17, 24, 11,
    18, 12, 19, 13, 20, 14, 21,
    15, 22,
]

RBY1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": RBY1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": RBY1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": RBY1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": RBY1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": RBY1_MUJOCO_TO_ISAACLAB_BODY,
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

# ---------------------------------------------------------------------
# 5) ISAAC LAB ARTICULATION CONFIG
# ---------------------------------------------------------------------

RBY1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/bcarc_ziwaj0x/Downloads/TransuranicVer/GantrySystemV4/Collected_simplifiedTWPC/flatremovedRBY2.usd",
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
            joint_names_expr=["right_wheel", "left_wheel"],
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=15.707963268,
            stiffness=STIFFNESS_WHEEL,
            damping=DAMPING_WHEEL,
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
# ---------------------------------------------------------------------

RBY1_ACTION_SCALE = {}
for actuator in RBY1_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr

    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)

    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            RBY1_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
