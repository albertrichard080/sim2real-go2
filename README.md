# E4 - Go2 RL-Based Locomotion Training and Sim-to-Real Deployment

**Course:** Cognitive Architectures for Robotics (COGAR), University of Genoa, DIBRIS<br>
**Assignment:** Team E / Assignment 13 (E4)<br>
**Student:** Richard Albert King Mechoda (8525970)<br>
**Professor:** Fulvio Mastrogiovanni<br>
**Supervisor:** Gianluca Galvagni

A reinforcement-learning locomotion policy for the Unitree Go2 quadruped, trained **entirely in
simulation** (NVIDIA Isaac Lab / Isaac Sim, PPO via RSL-RL) and transferred **zero-shot** to the
real Go2 EDU through the Unitree SDK and a ROS 2 Humble interface.

> **Full project on GitHub:** https://github.com/albertrichard080/sim2real-go2
> Because of the AulaWeb 5 MB upload limit, this archive includes the deployable ONNX policies
> (flat and rough) but **not** the large training checkpoints (`.pt`), the full training logs,
> or the LaTeX report source. The complete project, with all code, configs, checkpoints and
> source files, is in the GitHub repository above.

> **Cognitive-architecture framing (COGAR).** In the course's Sense-Reasoning/Plan-Act framework
> this is a **reactive architecture**: the learned policy is a **behaviour** (a sensory-to-motor
> mapping), and the deployment state machine with its safety reflexes is a **fixed-priority
> behaviour coordinator** that *suppresses* walking when needed (subsumption). The velocity command
> is where a deliberative Reasoning/Plan layer could later attach. See
> [`docs/02_COGNITIVE_ARCHITECTURE.md`](docs/02_COGNITIVE_ARCHITECTURE.md).

---

## Highlights
- **99.7 %** episode survival, **0.22 m/s** velocity-tracking error in simulation.
- **≈ 2×10⁹** training steps, **16 384** parallel robots, **~1 h 40 min** on one RTX Pro 6000.
- **Zero-shot** flat-terrain walking on the real Go2 (see `media/`).
- **Asymmetric actor-critic** (privileged base velocity to the critic only) - the decisive
  sim-to-real fix.
- Production deployment with a finite-state machine, safety reflexes, and dual ONNX/TorchScript
  export.

## Repository layout
```
E4_Go2_RL_SimToReal/
├── README.md                     ← you are here
├── configs/go2/                  Isaac Lab environment + PPO configs (real, V1→V2→V3)
│   ├── flat_env_sim2real_cfg.py        (V1)
│   ├── flat_env_sim2real_v2_cfg.py     (V2 - symmetric, 48-dim)
│   ├── flat_env_sim2real_v3_cfg.py     (V3 - asymmetric, 45-dim, PRODUCTION)
│   ├── rough_env_sim2real_v3_cfg.py    (rough terrain, height scan)
│   ├── __init__.py                     (Gym task registrations)
│   └── agents/                         (rsl_rl PPO configs: v2, v3, rough v3)
├── deploy/
│   ├── deploy_go2.py             Production deployment (FSM, safety, remote, ONNX)
│   ├── test_policy_advanced.py   Timing / gait / joint-limit tests
│   └── DEPLOY_README.md          Hardware setup, controls, troubleshooting
├── policies/                     (deployable ONNX; full .pt checkpoints on GitHub)
│   ├── flat_v3/   policy.onnx     (deployed flat policy, asymmetric 45-dim)
│   ├── flat_v2/   policy.onnx     (symmetric 48-dim, for the comparison)
│   └── rough_v3/  policy.onnx     (rough terrain, sim-validated)
├── report/
│   └── main.pdf                  Formal report (5-part exam structure, Sense-Plan-Act diagram)
├── slides/                       COGAR_Go2_Presentation.pptx / .pdf (PowerPoint deck)
├── docs/
│   ├── 01_TECHNICAL_REPORT.md          Full methodology (V3 ground truth)
│   ├── 02_COGNITIVE_ARCHITECTURE.md    COGAR mapping (the cognitive approach)
│   ├── 03_SIM2REAL_GAP_ANALYSIS.md     Failures, fixes, residual gap, improvements
│   └── 04_EXPERIMENT_LOG.md            V2-vs-V3 comparison + metrics
└── media/                        Demo stills + video pointers
```

## Exam deliverables
- **Report:** `report/main.pdf` - follows the exam's five-part structure, with a
  Sense-Reasoning/Plan-Act diagram.
- **Slides:** `slides/COGAR_Go2_Presentation.pptx` (editable PowerPoint) and `.pdf`.
- **Supporting docs:** `docs/` (technical report, cognitive architecture, sim-to-real gap
  analysis, experiment log).
- **Demonstration video:** https://drive.google.com/file/d/1e1jpoKKcbISyVp17HyGn1CQgQ9hJk3pM/view

## Quick start
```bash
# Train (V3 production)
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-v3 --num_envs 16384 --headless

# Validate in Isaac Sim
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-Play-v3 --num_envs 50

# Deploy on the real Go2 (laptop in the 192.168.123.x subnet, or onboard Jetson)
pip install unitree_sdk2py onnxruntime numpy pygame
python deploy/deploy_go2.py --policy policies/flat_v3/policy.onnx --interface <iface> --kp 40 --kd 1.0
```
> The config files here are copies of the live Isaac Lab task configs; to train, drop
> `configs/go2/` into
> `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/`.

## Critical parameters (must match between training and deployment)
| Parameter | Value |
|---|---|
| Action scale | 0.25 |
| Control frequency | 50 Hz (dt 0.005 × decimation 4) |
| PD gains (deploy) | Kp = 40, Kd = 1.0 |
| Observation dims | 45 (V3 flat) / 48 (V2 flat) / 232-235 (rough) |
| Joint remap (sim→sdk) | `[3,0,9,6,4,1,10,7,5,2,11,8]` |
| Default stance | hips ±0.1, front thigh 0.8, rear thigh 1.0, calf −1.5 rad |

## Safety (deployment)
State machine `IDLE → STAND_UP → READY → WALK → DAMPING → STOPPED`; sensor watchdog (100 ms),
tilt reflex (>46°), NaN/Inf guard, action/joint clipping, soft damping shutdown. **Always keep
the e-stop (SELECT / L2+B / ESC) within reach.**

## License / attribution
Academic coursework. Built on Isaac Lab (BSD-3), RSL-RL, and the Unitree SDK. Reward design adapts
community best practice (ETH `legged_gym`, Unitree `unitree_rl_lab`) with project-specific tuning.
