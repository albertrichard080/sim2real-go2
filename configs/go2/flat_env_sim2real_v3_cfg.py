# Go2 Flat Sim2Real V3 - Production Recipe
# ============================================================
# Fixes from real-robot deployment failure analysis:
#   1. Remove base_lin_vel from actor obs (asymmetric AC) - was causing OOD at deploy
#   2. DelayedPDActuator with 0-20ms latency randomization - matches real motor delay
#   3. Randomize actuator gains +/-20% - policy robust to Kp deployment mismatch
#   4. Heavier action_rate penalty (-0.25) - kills high-frequency vibration
#   5. Stand-still reward - robot stops trotting in place when cmd=0
#   6. Reduce joint_vel obs noise (was unrealistic +/-1.5)
#   7. Reduce velocity command range (training overshoots deployment usage)
#   8. Joint friction/armature randomization
#   9. Action smoothness penalty between consecutive steps
#   10. Initial action noise std slightly lower (0.8 vs 1.0) for stability
#
# Based on research: unitree_rl_lab, fan-ziqi/robot_lab, NVIDIA Spot, Saif Ahmad
# CAPS smoothness regularization (arxiv 2012.06644)

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


# Custom Go2 cfg with DelayedPDActuator (instead of DCMotor)
GO2_CFG_DELAYED = UNITREE_GO2_CFG.copy()
GO2_CFG_DELAYED.actuators = {
    'base_legs': DelayedPDActuatorCfg(
        joint_names_expr=['.*_hip_joint', '.*_thigh_joint', '.*_calf_joint'],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=25.0,
        damping=0.5,
        friction=0.0,
        min_delay=0,
        max_delay=4,  # 4 physics steps at 5ms = 20ms max latency
    ),
}


@configclass
class UnitreeGo2FlatSim2RealV3EnvCfg(UnitreeGo2RoughEnvCfg):
    '''Go2 flat - V3 production recipe with comprehensive sim2real fixes.'''

    def __post_init__(self):
        super().__post_init__()

        # === ROBOT: use DelayedPDActuator ===
        self.scene.robot = GO2_CFG_DELAYED.replace(prim_path='{ENV_REGEX_NS}/Robot')

        # === TERRAIN: flat plane ===
        self.scene.terrain.terrain_type = 'plane'
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # === REMOVE base_lin_vel from actor obs (asymmetric AC) ===
        # This is critical: training with lin_vel and zeroing at deploy = OOD
        self.observations.policy.base_lin_vel = None  # 48 -> 45 dim

        # === REDUCE joint_vel obs noise (was unrealistic +/-1.5) ===
        if self.observations.policy.joint_vel.noise is not None:
            self.observations.policy.joint_vel.noise = Unoise(n_min=-0.3, n_max=0.3)

        # === REWARDS ===
        # Tracking
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Gait quality
        self.rewards.feet_air_time.weight = 0.5  # reduced from 1.0; less aggressive trot
        self.rewards.feet_air_time.params['threshold'] = 0.4

        # Body stability
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05

        # Smoothness - HEAVILY increased to fight vibration
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.action_rate_l2.weight = -0.25     # was -0.05; 5x heavier
        self.rewards.dof_acc_l2.weight = -1.0e-6        # was -2.5e-7; 4x heavier

        # Foot slide
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.25,
            params={
                'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot'),
                'asset_cfg': SceneEntityCfg('robot', body_names='.*_foot'),
            },
        )

        # Undesired contacts (thighs, base)
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-1.0,  # increased from -0.5
            params={
                'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['.*_thigh', 'base']),
                'threshold': 1.0,
            },
        )

        # Joint deviation - mild penalty for staying near default pose
        self.rewards.joint_deviation = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-0.1,
            params={'asset_cfg': SceneEntityCfg('robot')},
        )

        # === DOMAIN RANDOMIZATION (aggressive for sim2real) ===

        # Mass: payload variation
        self.events.add_base_mass.params['mass_distribution_params'] = (-2.0, 4.0)

        # COM shift
        self.events.base_com = EventTerm(
            func=base_mdp.randomize_rigid_body_com,
            mode='startup',
            params={
                'asset_cfg': SceneEntityCfg('robot', body_names='base'),
                'com_range': {'x': (-0.05, 0.05), 'y': (-0.05, 0.05), 'z': (-0.02, 0.02)},
            },
        )

        # External forces
        self.events.base_external_force_torque.params['force_range'] = (-3.0, 3.0)
        self.events.base_external_force_torque.params['torque_range'] = (-3.0, 3.0)

        # Push robot
        self.events.push_robot = EventTerm(
            func=base_mdp.push_by_setting_velocity,
            mode='interval',
            interval_range_s=(10.0, 15.0),
            params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}},
        )

        # Floor friction
        self.events.physics_material.params['static_friction_range'] = (0.4, 1.4)
        self.events.physics_material.params['dynamic_friction_range'] = (0.3, 1.0)
        self.events.physics_material.params['restitution_range'] = (0.0, 0.1)

        # NEW: Randomize actuator gains +/-20% per episode
        # This makes policy robust to Kp/Kd deployment mismatch (real ~ 25-40, train 25)
        self.events.actuator_gains = EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode='reset',
            params={
                'asset_cfg': SceneEntityCfg('robot'),
                'stiffness_distribution_params': (0.8, 1.25),
                'damping_distribution_params': (0.8, 1.25),
                'operation': 'scale',
            },
        )

        # NEW: Joint friction randomization (real motors have stiction/backlash)
        self.events.joint_friction = EventTerm(
            func=base_mdp.randomize_joint_parameters,
            mode='reset',
            params={
                'asset_cfg': SceneEntityCfg('robot'),
                'friction_distribution_params': (0.8, 1.25),
                'armature_distribution_params': (0.9, 1.15),
                'operation': 'scale',
            },
        )

        # === COMMAND RANGE: smaller than typical, matches deployment use ===
        # Default is often (-1, 1) - reducing to reflect real joystick usage
        # and to encourage stand-still behavior at zero command
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # IMPORTANT: probability of zero command - encourages standing still
        self.commands.base_velocity.rel_standing_envs = 0.20  # 20% of envs get cmd=0


class UnitreeGo2FlatSim2RealV3EnvCfg_PLAY(UnitreeGo2FlatSim2RealV3EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
