# Custom Go2 Flat Environment Config - Optimized for Sim-to-Real Transfer
# Based on default flat_env_cfg.py with stronger domain randomization

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatSim2RealEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 flat terrain config with aggressive domain randomization for sim-to-real."""

    def __post_init__(self):
        super().__post_init__()

        # === TERRAIN: flat plane ===
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # === REWARDS: tuned for stable, smooth gait ===
        self.rewards.flat_orientation_l2.weight = -5.0      # stronger: keep body level
        self.rewards.feet_air_time.weight = 0.25             # encourage proper gait
        self.rewards.track_lin_vel_xy_exp.weight = 1.5       # velocity tracking
        self.rewards.track_ang_vel_z_exp.weight = 0.75       # angular velocity tracking
        self.rewards.dof_torques_l2.weight = -0.0005         # stronger torque penalty (smoother)
        self.rewards.action_rate_l2.weight = -0.05           # stronger action smoothness

        # === DOMAIN RANDOMIZATION: aggressive for sim-to-real ===
        # Mass: +/- 2kg on base
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

        # COM randomization (re-enable it)
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
            },
        )

        # External force perturbations (re-enable)
        self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.base_external_force_torque.params["torque_range"] = (-5.0, 5.0)

        # Random pushes during episode
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # Friction randomization (wider range)
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)


class UnitreeGo2FlatSim2RealEnvCfg_PLAY(UnitreeGo2FlatSim2RealEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
