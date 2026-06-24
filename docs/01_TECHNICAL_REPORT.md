# Technical Report - Go2 RL Locomotion and Sim-to-Real Transfer

**Course:** Cognitive Architectures for Robotics (COGAR) - University of Genoa, DIBRIS
**Assignment:** E4 / Assignment 13 (Team E - Go2 Quadruped)
**Author:** Richard Albert King Mechoda (8525970)
**Stack:** NVIDIA Isaac Sim 5.1 · Isaac Lab · RSL-RL (PPO) · PyTorch · ONNX Runtime · Unitree SDK2 · ROS 2 Humble

> This report documents the **final (V3) production pipeline** and its real-robot deployment.
> The version history (V1 → V2 → V3) and the reasoning behind each change are in
> `04_EXPERIMENT_LOG.md` and `03_SIM2REAL_GAP_ANALYSIS.md`; the cognitive-architecture framing is
> in `02_COGNITIVE_ARCHITECTURE.md`.

---

## 1. Objective

Train a locomotion policy that makes the Unitree Go2 walk, trot, and track velocity commands on
flat terrain, **entirely in simulation**, and transfer it **zero-shot** (no real-world training)
to the physical Go2 EDU, through the Unitree SDK and a ROS 2 Humble interface.

**Headline results (simulation):** 99.7 % episode survival, 0.22 m/s velocity-tracking error,
≈ 2×10⁹ training steps in ~1 h 40 min on one RTX Pro 6000. **Hardware:** the flat policy walks zero-shot
on the real Go2 (demo videos in `media/`).

---

## 2. System architecture

### 2.1 Software / hardware
| Component | Version | Role |
|---|---|---|
| Isaac Sim | 5.1 | GPU physics simulation |
| Isaac Lab | 2.x | robot-learning framework |
| RSL-RL | 3.x | PPO implementation (ETH Zürich) |
| PyTorch | 2.5 | training |
| ONNX Runtime | 1.22 | on-robot inference |
| unitree_sdk2py | - | low-level motor control over DDS |

Training: RTX Pro 6000 Blackwell (96 GB). Deployment: Go2 EDU + onboard Jetson Orin NX.

### 2.2 Control-frequency hierarchy
| Layer | Rate | Step |
|---|---|---|
| Policy inference | 50 Hz | 20 ms |
| Physics simulation | 200 Hz | 5 ms (decimation 4) |
| Motor PD loop | 500 Hz | 2 ms (on the Unitree driver) |

---

## 3. Learning formulation

### 3.1 PPO
Proximal Policy Optimization (Schulman et al., 2017) maximises the expected discounted return
`J(πθ) = E[Σ γᵗ rₜ]` with the clipped surrogate objective
`L = E[min(rₜ(θ)Âₜ, clip(rₜ(θ),1−ε,1+ε)Âₜ)]`, `ε = 0.2`, advantages from GAE (`γ = 0.99`,
`λ = 0.95`), and a **KL-adaptive learning rate** (target KL 0.01) that self-regulates step size.

### 3.2 Networks
Actor and critic are separate MLPs `→256→256→128→` with **ELU** activations. The actor outputs
the mean of a diagonal-Gaussian over 12 joint actions (`init_noise_std` 0.8 in V3). ≈ 100 k
parameters each - small enough for 50 Hz inference on the Jetson.

### 3.3 Observation (45-dim, asymmetric)
The **actor** observes only what the real robot can measure:

| Component | Dim | Source |
|---|---|---|
| Base angular velocity | 3 | IMU gyro |
| Projected gravity | 3 | IMU quaternion |
| Velocity command (vₓ,v_y,ω_z) | 3 | operator / planner |
| Joint positions (rel. to stance) | 12 | encoders |
| Joint velocities | 12 | encoders |
| Last action | 12 | internal memory |
| **Total** | **45** | |

The **critic** additionally receives the **base linear velocity** (privileged, sim-only) - the
asymmetric actor-critic. *Projected gravity* replaces raw orientation for a singularity-free,
yaw-invariant "down" vector. Observation noise is added in training (joint-velocity noise reduced
to ±0.3 rad/s to match a real encoder).

### 3.4 Action and low-level control
The policy outputs 12 **position offsets from a nominal stance**: `q_target = q_default + 0.25·a`
(±0.25 rad ≈ ±14°). A PD loop tracks them: `τ = Kp(q_target − q) + Kd(−q̇)`, with **Kp 25 / Kd
0.5** in training and **Kp 40 / Kd 1.0** at deployment.

### 3.5 Reward (V3) - emergent gait from scalar objectives
| Sign | Term | Weight |
|---|---|---|
| + | linear-velocity tracking `exp(−‖v−v*‖²/0.25)` | 1.5 |
| + | angular-velocity tracking | 0.75 |
| + | foot air time (threshold 0.4 s) | 0.5 |
| − | flat orientation | −2.5 |
| − | vertical velocity | −2.0 |
| − | angular velocity xy | −0.05 |
| − | joint torques | −2e-4 |
| − | action rate | **−0.25** |
| − | joint accelerations | −1e-6 |
| − | feet slide | −0.25 |
| − | undesired contacts (thigh/base) | **−1.0** |
| − | joint deviation from stance | −0.1 |

No footstep pattern is ever prescribed; the trot **emerges** from optimising these terms.

### 3.6 Domain randomization (the sim-to-real bridge)
Per episode: base mass −2…+4 kg; COM ±5 cm (xy)/±2 cm (z); static friction 0.4-1.4, dynamic
0.3-1.0; restitution 0-0.1; external force/torque ±3; pushes ±0.5 m/s every 10-15 s; **actuator
gains ±20 %**; **joint friction/armature** randomised; **actuator latency 0-20 ms**
(`DelayedPDActuator`). 20 % of robots get a **zero command** (stand-still training).

### 3.7 PPO hyperparameters (V3)
`num_envs 16384 · num_steps_per_env 24 · epochs 5 · minibatches 4 · lr 1e-3 adaptive · γ 0.99 ·
λ 0.95 · clip 0.2 · entropy 0.005 · value-loss 1.0 · max-grad-norm 1.0 · desired-KL 0.01 ·
max_iterations 5000`.

---

## 4. Deployment pipeline

**Export.** Policy → TorchScript (.pt, 463 KB) and ONNX (.onnx, 452 KB).

**Joint remap.** Isaac Lab BFS order ↔ Unitree per-leg order via
`SIM_TO_REAL = [3,0,9,6,4,1,10,7,5,2,11,8]` (verified, Isaac Lab discussion #506).

**On-robot loop (50 Hz).** Read IMU (angular velocity, projected gravity) and encoders (q, q̇);
remap to sim order; build the 45-dim observation (base linear velocity not needed - it is
critic-only); run ONNX inference (~1 ms); scale and offset to joint targets; send via SDK; the
500 Hz driver PD loop tracks them. Sport mode is released first so low-level control is not
overridden.

**Executive state machine.** `IDLE → STAND_UP (2 s soft Kp ramp 5→40) → READY → WALK → DAMPING
(Kp 0/Kd 2, 1 s) → STOPPED`, driven by the wireless remote (START = walk, SELECT = e-stop),
gamepad, or keyboard.

**Safety.** Sensor watchdog (100 ms timeout → damping); tilt reflex (>46° → damping); NaN/Inf
sanitisation; action clipping ±6; joint-target clipping ±2.5 rad; soft damping shutdown.

---

## 5. Results

Simulation: **99.7 %** survival, **0.22 m/s** tracking error, ~31.2 mean reward (see
`04_EXPERIMENT_LOG.md` for the V2-vs-V3 comparison and convergence phases). Hardware: zero-shot
flat-terrain walking on the real Go2, with the asymmetric actor-critic eliminating the drift and
the smoothness penalties eliminating the chatter. A rough-terrain policy (with height scan, incl.
stair climbing) is trained and validated in simulation but not yet deployed on hardware.

---

## 6. Reproducibility
```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-v3 --num_envs 16384 --headless
# play / validate:
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Sim2Real-Play-v3 --num_envs 50
# deploy (on robot / tethered laptop in the Go2 subnet):
python deploy/deploy_go2.py --policy policies/flat_v3/policy.onnx --interface <iface> --kp 40 --kd 1.0
```
Config files in `configs/go2/`; trained policies in `policies/`; PPO settings in
`configs/go2/agents/`. Expected training time ~1 h 40 min on an RTX 6000-class GPU.

---

## 7. References
Schulman et al. 2017 (PPO); Schulman et al. 2015 (GAE); Rudin et al. 2022 (massively parallel
legged RL); Hwangbo et al. 2019 (learned actuators); Lee et al. 2020 / Miki et al. 2022
(privileged learning); Andrychowicz et al. 2019 (domain randomization); Pinto et al. 2018
(asymmetric actor-critic); NVIDIA Isaac Lab; Unitree Go2 SDK.
