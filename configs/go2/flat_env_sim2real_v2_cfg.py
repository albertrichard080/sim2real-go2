# Go2 Flat Env Config V2 - Production Grade for Sim-to-Real Deployment
#
# Reward design philosophy:
#   - Tracking rewards must dominate (sum ~2.25) so robot learns to walk first
#   - Penalty rewards shape gait quality (each << tracking rewards)
#   - Total positive reward must always exceed total negative at convergence
#   - Based on: unitree_rl_lab defaults + Issue #1784 fixes + community best practices
#
# Comparison with defaults:
#   feet_air_time:       0.01 (default) -> 1.0 (ours) -- prevents foot dragging
#   flat_orientation:    0.0  (default) -> -2.5 (ours) -- keeps body level
#   action_rate:         -0.01 (default) -> -0.05 (ours) -- smoother gait
#   dof_torques:         -0.0002 (default) -> -0.0005 (ours) -- energy efficiency
#   undesired_contacts:  None (default) -> -0.5 (ours) -- no thigh ground contact
#   feet_slide:          None (default) -> -0.25 (ours) -- no foot dragging
#   joint_deviation:     None (default) -> -0.1 (ours) -- natural stance
#   friction range:      0.8/0.6 fixed (default) -> 0.4-1.4/0.3-1.0 (ours) -- floor variance
#   push_robot:          None (default) -> enabled (ours) -- perturbation robustness
#   base_com:            None (default) -> enabled (ours) -- COM uncertainty

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatSim2RealV2EnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 flat terrain - production config for real robot deployment."""

    def __post_init__(self):
        super().__post_init__()

        # === TERRAIN: flat plane ===
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # === REWARDS ===
        # --- Positive rewards (total ~2.25 - must dominate so robot learns to walk) ---
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # --- Gait quality (positive - encourage proper foot lifting) ---
        # Default: 0.01 -> Ours: 1.0 (prevents foot dragging, #1 sim2real issue)
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.4

        # --- Body stability penalties (moderate) ---
        # Default: 0.0 -> Ours: -2.5 (keep body level for real deployment)
        self.rewards.flat_orientation_l2.weight = -2.5
        # Default: -2.0 (same)
        self.rewards.lin_vel_z_l2.weight = -2.0
        # Default: -0.05 (same)
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # --- Smoothness penalties (for real hardware quality) ---
        # Default: -0.0002 -> Ours: -0.0005 (slightly more energy efficient)
        self.rewards.dof_torques_l2.weight = -0.0005
        # Default: -0.01 -> Ours: -0.05 (smoother actions on real hardware)
        self.rewards.action_rate_l2.weight = -0.05
        # Default: -2.5e-7 (same)
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # --- NEW rewards for sim2real (not in default config) ---

        # Foot slide penalty (penalizes dragging feet on ground)
        # Default: None -> Ours: -0.25
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        # Undesired contacts (penalize thigh/body hitting ground)
        # Default: None (Go2 config disables it) -> Ours: -0.5
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", "base"]),
                "threshold": 1.0,
            },
        )

        # Joint deviation from default stance
        # Default: None -> Ours: -0.1 (encourages natural posture)
        self.rewards.joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # === DOMAIN RANDOMIZATION (aggressive for sim2real) ===

        # Mass: default (-1, 3) -> ours (-2, 4) for payload variation
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 4.0)

        # COM shift: default None -> ours enabled
        self.events.base_com = EventTerm(
            func=base_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
            },
        )

        # External forces: default (0,0) -> ours (-3, 3)
        self.events.base_external_force_torque.params["force_range"] = (-3.0, 3.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        # Push robot: default None -> ours enabled
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

        # Friction: default fixed 0.8/0.6 -> ours randomized 0.4-1.4/0.3-1.0
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.4)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)


class UnitreeGo2FlatSim2RealV2EnvCfg_PLAY(UnitreeGo2FlatSim2RealV2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
