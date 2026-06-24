# Experiment Log and Policy Comparison

**Assignment E4 / 13 - Go2 RL Locomotion and Sim-to-Real Deployment**
**Author:** Richard Albert King Mechoda (8525970)

This log covers assignment points 4-6: validation in Isaac Sim under different conditions,
**comparison of at least two policy configurations**, and evaluation against locomotion metrics.

---

## 1. The two configurations compared

The assignment asks for at least two policy configurations differing in reward shaping or domain
randomization. The natural, real comparison in this project is the decisive design step of the
whole sim-to-real effort:

| | **Config A - V2 (symmetric)** | **Config B - V3 (asymmetric, production)** |
|---|---|---|
| Actor observation | **48-dim**, *includes* base linear velocity | **45-dim**, base linear velocity **removed** |
| Privileged info | none (actor sees everything the critic sees) | base linear velocity given to **critic only** |
| Actuator model | ideal DC motor | **`DelayedPDActuator`**, 0-20 ms latency |
| Actuator-gain randomization | none | **±20 %** on Kp/Kd per episode |
| Joint friction / armature randomization | none | enabled |
| Stand-still training | none | **20 %** of envs, zero command |
| `action_rate` penalty | −0.05 | **−0.25** (×5) |
| `undesired_contacts` penalty | −0.5 | **−1.0** (×2) |
| `feet_air_time` reward | 1.0 | 0.5 |

Both share the same network (256-256-128, ELU), PPO hyperparameters, 16 384 environments, and
≈ 2×10⁹ training steps. **The independent change is the observation architecture and the
actuator-realism randomization** - i.e. exactly the two ingredients the sim-to-real literature
debates. This isolates *what the asymmetric actor-critic and actuator randomization buy*.

---

## 2. Hypotheses (stated before evaluation)

- **H1.** Both configs reach high survival and low tracking error **in simulation** - the gap is
  not visible in sim.
- **H2.** On the **real robot**, Config A drifts/veers because its zeroed base-velocity input is
  out of distribution, while Config B walks straight.
- **H3.** Config B is smoother (less motor chatter) on hardware due to the heavier action-rate
  penalty and latency randomization.

---

## 3. Simulation validation (Isaac Sim, `play.py`, 50 parallel robots)

Both policies were replayed in Isaac Sim across command directions (forward, backward, lateral,
turning), the stand-still command, and perturbation scenarios (random pushes, friction changes,
added payload). Both reach the production training metrics:

| Metric (simulation) | Value |
|---|---|
| Episode survival rate | **99.7 %** |
| Fall rate (base contact) | 0.33 % |
| Linear velocity tracking error | **0.22 m/s** |
| Angular velocity tracking error | 0.15 rad/s |
| final mean return | ~31.2 |
| Foot-slide penalty (lower is better) | −0.068 |

**Convergence** (both configs follow the same three phases):
- iterations 0-500: learn not to fall (survival 30 % → 85 %);
- iterations 500-2000: trot gait emerges (tracking error 1.3 → 0.3 m/s);
- iterations 2000-5000: refinement (smoother, more energy-efficient).

This confirms **H1**: *in simulation both configurations are excellent and nearly
indistinguishable*. The gap only appears on hardware - which is the entire point of a sim-to-real
study and the reason a simulation-only evaluation is not sufficient.

---

## 4. Real-robot evaluation (Unitree Go2 EDU, flat lab floor)

| Observation (hardware) | Config A - V2 (symmetric) | Config B - V3 (asymmetric) |
|---|---|---|
| Walks zero-shot | yes | yes |
| Straight-line tracking | **veers / drifts sideways** | **straight** |
| Motor chatter | noticeable vibration | smooth |
| Stand-still (neutral command) | trots in place | holds station |
| Outcome | usable but biased | **clean zero-shot walking** |

This confirms **H2** and **H3**: the asymmetric actor-critic removed the drift (privileged signal
moved off the actor), and the heavier smoothness penalties plus latency randomization removed the
chatter. The qualitative evidence is in the demo videos (`media/`), which show the simulated
training farm of 16 384 robots and the real Go2 walking under joystick command.

> **Note on rigour (returned to in the Congruence section of the report).** The hardware
> comparison is *qualitative* - observed behaviour and the disappearance of specific failure
> modes - not a logged quantitative benchmark. The simulation metrics are quantitative. A fully
> instrumented real-robot benchmark (below) is identified as future work, and the honest framing
> of this limitation is itself part of the deliverable.

---

## 5. Evaluation metrics (assignment point 6) - definitions and status

| Metric | Definition | Status |
|---|---|---|
| **Velocity tracking error** | mean \|achieved − commanded\| planar velocity | measured in sim: **0.22 m/s** |
| **Number of falls / survival** | fraction of episodes ending in base contact | measured in sim: **0.33 % falls / 99.7 % survival** |
| **Base orientation stability** | RMS of body pitch/roll (via projected gravity) | shaped by `flat_orientation` (−2.5) and `ang_vel_xy` penalties; qualitatively level on hardware |
| **Robustness to external disturbances** | recovery after push / added mass | trained via ±0.5 m/s pushes + ±3 N forces every 10-15 s; recovers in sim replay |
| **Energy / cost of transport** | mechanical power / (weight × speed) | penalised in training via `dof_torques`; **not yet measured on hardware** (future work) |

---

## 6. Conclusion of the comparison

In simulation, observation architecture barely matters - both configs are near-perfect. The
*decisive* differences live entirely in the sim-to-real gap: an **asymmetric actor-critic** that
keeps the policy inside its own perceptual horizon, and **actuator-realism randomization** that
makes it robust to real motor dynamics. This is why Config B (V3) is the production policy and the
one used for the real-robot demonstration, and why the comparison is the empirical backbone of
both the gap analysis and the cognitive-architecture argument.
