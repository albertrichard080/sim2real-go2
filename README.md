# Sim-to-Real Go2 Locomotion

Training a Unitree Go2 quadruped locomotion policy in NVIDIA Isaac Lab with sim-to-real deployment.

## Project Structure

```
sim2real-go2/
├── configs/go2/                  # Isaac Lab environment configs
│   ├── flat_env_sim2real_cfg.py  # Flat terrain + aggressive domain randomization
│   ├── __init__.py               # Gym task registration
│   └── agents/
│       └── rsl_rl_ppo_sim2real_cfg.py  # PPO training config
├── deploy/
│   ├── deploy_policy.py          # Real robot deployment script
│   └── policy.onnx               # Exported ONNX policy (flat terrain)
├── checkpoints/
│   └── flat_sim2real_model_2999.pt  # Trained PyTorch checkpoint
└── README.md
```

## Training

### Prerequisites
- Isaac Sim 5.1.0
- Isaac Lab 2.1.0+ (with RSL-RL installed)

### Custom Sim-to-Real Environment Features
- Aggressive domain randomization (mass, friction, COM, external forces)
- Push perturbations during training
- Stronger flat orientation and action smoothness rewards
- Network: MLP [256, 256, 128]

### Train Flat Terrain (Sim2Real optimized)
```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-v0 \
    --num_envs 8192 --headless
```

### Train Rough Terrain
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --num_envs 16384 --max_iterations 5000 --headless
```

### Visualize Trained Policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-Play-v0 \
    --num_envs 16 \
    --checkpoint /path/to/model_2999.pt
```

## Deployment to Real Go2 EDU

### Requirements
```bash
pip install unitree_sdk2py onnxruntime numpy pygame
```

### Hardware
- Unitree Go2 EDU (low-level control required)
- USB-C to Ethernet adapter + Ethernet cable
- USB gamepad (Xbox/PS4) for velocity control

### Deploy
```bash
# Set laptop IP in Go2's subnet
sudo ifconfig <interface> 192.168.123.222 netmask 255.255.255.0

# Run deployment
python deploy/deploy_policy.py --policy deploy/policy.onnx --interface <interface>
```

### Critical Parameters (must match between training and deployment)
| Parameter | Value |
|-----------|-------|
| Action scale | 0.25 |
| PD gains (deploy) | Kp=40.0, Kd=1.0 |
| Control frequency | 50 Hz |
| Observation dims | 48 (flat) |
| Joint remapping | Isaac Lab [FL,FR,RL,RR] to SDK [FR,FL,RR,RL] |

## Training Results

### Flat Terrain (3000 iter, 8192 envs, RTX 6000 Pro)
- Training time: ~27 minutes
- Final reward: ~28+
- Fall rate: 0.22%
- Velocity error (xy): 0.216

## References
- Rudin et al. "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021)
- [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
