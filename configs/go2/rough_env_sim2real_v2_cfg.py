# Go2 Rough Env Config V2 - Production Grade for Sim-to-Real Deployment
#
# Same sim2real improvements as flat V2 but WITH rough terrain:
#   - Terrain generator with stairs, slopes, bumps (curriculum learning)
#   - Height scanner (187 extra obs dims -> total 235)
#   - Larger network [512, 256, 128] needed for complex terrain
#   - Longer training needed (~8-11 hours with 16384 envs)
#
# For real robot deployment on rough terrain, you need:
#   - Go2's depth camera or LiDAR to provide height scan data
#   - Or use flat V2 policy for flat ground (simpler, no extra sensors needed)

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RoughSim2RealV2EnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough terrain - production config for real robot deployment."""

    def __post_init__(self):
        super().__post_init__()

        # === TERRAIN: KEEP rough terrain (inherited from parent) ===
        # terrain_type = "generator" with stairs, slopes, bumps
        # height_scanner enabled (187 dims)
        # terrain curriculum enabled (starts easy, gets harder)

        # === REWARDS ===
        # Positive rewards (must dominate)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Gait quality - prevents foot dragging
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.4

        # Body stability
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Smoothness
        self.rewards.dof_torques_l2.weight = -0.0005
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Foot slide penalty
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Undesired contacts
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", "base"]),
                "threshold": 1.0,
            },
        )

        # Joint deviation
        self.rewards.joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # === DOMAIN RANDOMIZATION ===
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

        self.events.base_com = EventTerm(
            func=base_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
            },
        )

        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        self.events.physics_material.params["static_friction_range"] = (0.4, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)


class UnitreeGo2RoughSim2RealV2EnvCfg_PLAY(UnitreeGo2RoughSim2RealV2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
