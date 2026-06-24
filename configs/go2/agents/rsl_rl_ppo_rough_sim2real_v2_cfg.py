# PPO config for Go2 Rough Terrain Sim-to-Real V2
# Larger network [512, 256, 128] for complex terrain with height scan (235 obs)
#
# PRODUCTION STABILITY FIXES:
#   - noise_std_type="log": prevents RuntimeError "normal expects all elements of std >= 0.0"
#     by parameterizing std in log space (exp(log_std) is always positive)
#   - max_iterations=5000: stops before late-training numerical instability window
#   - save_interval=50: frequent checkpoints so we never lose >50 iterations of progress
#   - empirical_normalization=False: explicit, matches deployment (no normalizer)

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughSim2RealV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go2_rough_sim2real_v2"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
