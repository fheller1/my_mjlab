import mujoco

from .franka.franka_constants import (
    get_panda_robot_cfg,
    PANDA_ACTION_SCALE,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg


def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_joint")
    body.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(cube_size,) * 3,
        mass=mass,
        rgba=(0.8, 0.2, 0.2, 1.0),
    )
    return spec


def franka_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_lift_cube_env_cfg()

    cfg.scene.entities = {
        "robot": get_panda_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),
    }

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = PANDA_ACTION_SCALE

    # add to hand in panda xml:
    # <site name="grasp_site" pos="0 0 0.105" size="0.005" rgba="1 0 0 0.5"/>

    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
        "grasp_site",
    )
    cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

    # TODO - this allows domain rand for fingertip friction
    # fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
    # cfg.events["fingertip_friction_slide"].params[
    #     "asset_cfg"
    # ].geom_names = fingertip_geoms
    # cfg.events["fingertip_friction_spin"].params[
    #     "asset_cfg"
    # ].geom_names = fingertip_geoms
    # cfg.events["fingertip_friction_roll"].params[
    #     "asset_cfg"
    # ].geom_names = fingertip_geoms

    # Configure collision sensor pattern.
    assert cfg.scene.sensors is not None
    for sensor in cfg.scene.sensors:
        if sensor.name == "ee_ground_collision":
            assert isinstance(sensor, ContactSensorCfg)
            sensor.primary.pattern = "hand"

    cfg.viewer.body_name = "link0"

    # Apply play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

        # Higher command resampling frequency for more dynamic play.
        assert cfg.commands is not None
        cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

    for name, entity in cfg.scene.entities.items():
        print(f"{name}: init_state = {entity.init_state}")

    return cfg
