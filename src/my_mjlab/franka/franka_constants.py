"""Franka Emika Panda constants."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from robot_descriptions import panda_mj_description


PANDA_XML: Path = Path(panda_mj_description.MJCF_PATH)
assert PANDA_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, PANDA_XML.parent / "assets", meshdir)
    return assets

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
#
# Franka datasheet effort limits:
#   Joints 1–4: 87 Nm  (large joints)
#   Joints 5–7: 12 Nm  (small joints)
#
# Armature values taken from Menagerie mjx_panda.xml.
# PD gains follow the same inertia-based formula used by YAM and ANYmal:
#   kp = armature * ω²        (ω = 2π * natural_freq)
#   kd = 2 * ζ * armature * ω (ζ = damping ratio)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz — standard for manipulation
DAMPING_RATIO = 1.0  # critically damped; safer than 2.0 for contact-rich tasks

# Large joints (1–4): higher inertia and effort
ARMATURE_LARGE = 0.1
EFFORT_LIMIT_LARGE = 87.0

PANDA_LARGE_JOINT_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("panda_joint[1-4]",),
    stiffness=ARMATURE_LARGE * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * ARMATURE_LARGE * NATURAL_FREQ,
    effort_limit=EFFORT_LIMIT_LARGE,
    armature=ARMATURE_LARGE,
)

# Small joints (5–7): lower inertia and effort
ARMATURE_SMALL = 0.01
EFFORT_LIMIT_SMALL = 12.0

PANDA_SMALL_JOINT_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("panda_joint[5-7]",),
    stiffness=ARMATURE_SMALL * NATURAL_FREQ**2,
    damping=2.0 * DAMPING_RATIO * ARMATURE_SMALL * NATURAL_FREQ,
    effort_limit=EFFORT_LIMIT_SMALL,
    armature=ARMATURE_SMALL,
)

# Gripper: prismatic finger joints, only actuate finger_joint1;
# finger_joint2 is coupled via equality constraint in the XML (same as YAM).
ARMATURE_FINGER = 0.001
EFFORT_LIMIT_FINGER = 70.0   # N (force, not torque — prismatic joint)
NATURAL_FREQ_GRIPPER = 5 * 2.0 * 3.1415926535  # 5 Hz — slower than arm

PANDA_FINGER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("panda_finger_joint1",),
    stiffness=ARMATURE_FINGER * NATURAL_FREQ_GRIPPER**2,
    damping=2.0 * DAMPING_RATIO * ARMATURE_FINGER * NATURAL_FREQ_GRIPPER,
    effort_limit=EFFORT_LIMIT_FINGER * 0.1,  # same safety cap pattern as YAM
    armature=ARMATURE_FINGER,
)

##
# Keyframe config.
#
# Standard Franka "ready" pose — arm slightly raised and forward,
# gripper fully open.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "panda_joint1": 0.0,
        "panda_joint2": -0.785,   # -π/4
        "panda_joint3": 0.0,
        "panda_joint4": -2.356,   # -3π/4
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,    # π/2
        "panda_joint7": 0.785,    # π/4
        "panda_finger_joint1": 0.04,  # fully open (40 mm)
        "panda_finger_joint2": 0.04,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
#
# Menagerie names collision geoms as ".*_collision".
# For manipulation we only need gripper + last link collisions active
# during training (same rationale as YAM's GRIPPER_ONLY_COLLISION):
# disabling upper-arm self-collision speeds up sim significantly.
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim=3,
    friction=(0.6,),
    priority=1,
)

GRIPPER_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={
        "(panda_link[67]|panda_[lr]finger)_.*_collision": 1,
        ".*_collision": 0,
    },
    conaffinity={
        "(panda_link[67]|panda_[lr]finger)_.*_collision": 1,
        ".*_collision": 0,
    },
    condim=3,
    friction=(0.6,),
)

##
# Final config.
##

PANDA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        PANDA_LARGE_JOINT_ACTUATOR_CFG,
        PANDA_SMALL_JOINT_ACTUATOR_CFG,
        PANDA_FINGER_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_panda_robot_cfg() -> EntityCfg:
    """Get a fresh Franka Panda robot configuration instance."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(GRIPPER_ONLY_COLLISION,),
        spec_fn=get_spec,
        articulation=PANDA_ARTICULATION,
    )


# Action scale: same 0.25 * effort / stiffness formula as YAM and ANYmal.
PANDA_ACTION_SCALE: dict[str, float] = {}
for _a in PANDA_ARTICULATION.actuators:
    assert isinstance(_a, BuiltinPositionActuatorCfg)
    _e = _a.effort_limit
    _s = _a.stiffness
    assert _e is not None
    for _n in _a.target_names_expr:
        PANDA_ACTION_SCALE[_n] = 0.25 * _e / _s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_panda_robot_cfg())
    viewer.launch(robot.spec.compile())
