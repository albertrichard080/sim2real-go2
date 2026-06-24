# Sim-to-Real Gap Analysis

**Assignment E4 / 13 - Go2 RL Locomotion and Sim-to-Real Deployment**
**Author:** Richard Albert King Mechoda (8525970)

This document is the deliverable required by point 9 of the assignment: *analyse the sim-to-real
gap and document the main limitations, failure cases, and possible improvements.* It is written
honestly - the final policy is the result of a sequence of failures on the real Go2, each of
which exposed a specific mismatch between simulation and reality and was fixed by one targeted
change. The version progression (V1 → V2 → V3) in `configs/go2/` is the record of that process.

---

## 1. What "the gap" means here

The policy is trained **entirely in simulation** and deployed **zero-shot** - the same network
that finishes training runs unchanged on the robot, with no real-world fine-tuning. The
*sim-to-real gap* is therefore the set of differences between the Isaac Sim training world and
the physical Go2 that can make a policy that is excellent in simulation behave badly on hardware.
The five gaps that actually mattered in this project were: **(a)** reward-induced bad habits that
only hurt on real ground, **(b)** high-frequency actuator chatter, **(c)** a privileged
observation available in sim but not on the robot, **(d)** unmodelled actuator dynamics
(latency, gain error, stiction), and **(e)** a joint-ordering convention mismatch.

---

## 2. Failure cases and fixes (chronological)

### Failure 1 - Foot dragging (a reward-shaping gap)
**Symptom.** Early policies stood still or *shuffled*: the feet slid along the ground instead of
being lifted. In simulation this scored acceptably; on a real floor with real friction it could
not move.
**Root cause.** The default `feet_air_time` reward weight (0.01) gave the policy almost no
incentive to swing the legs, so optimisation found a local optimum that *drags* to avoid the cost
of swinging.
**Fix (V1→V2).** Raise `feet_air_time` to 1.0 with a 0.4 s threshold (reward feet that stay
airborne ≥ 0.4 s per step) **and** add a `feet_slide` penalty (−0.25) that directly punishes
sliding contact. Dragging stopped immediately. Later (V3) `feet_air_time` was eased to 0.5 once
the slide penalty was doing the work, to avoid an over-stepping gait.

### Failure 2 - Collapsing instead of walking (a reward-balance gap)
**Symptom.** Every episode ended in a fall.
**Root cause.** The body-orientation penalty was set too aggressively (−5.0). The policy was
punished so hard for any tilt that *collapsing* scored better than the transient tilt needed to
take a step.
**Fix (V1→V2).** Rebalance so the **positive tracking reward always dominates** the negative
shaping terms: `flat_orientation` reduced to −2.5, total positive reward (1.5 + 0.75 + air-time)
kept above the penalties. Lesson, stated in the report: in RL the *relative* magnitude of reward
terms matters more than their absolute values.

### Failure 3 - Vibration / motor chatter (an actuator-bandwidth gap)
**Symptom.** The policy walked cleanly in simulation but the real motors buzzed with
high-frequency chatter.
**Root cause.** The network's action changed too fast between consecutive 20 ms steps; the real
PD loop turned those jumps into torque spikes the rigid-body simulator never penalised.
**Fix (V2→V3).** Heavier smoothness penalties: `action_rate` ×5 (−0.05 → −0.25), `dof_acc` ×4
(−2.5e-7 → −1e-6), `dof_torques` tightened. This removed most of the vibration *before* any
actuator modelling - i.e. the cheapest fix is to ask the policy for smoother commands.

### Failure 4 - Sideways drift (a privileged-observation gap) - *the decisive one*
**Symptom.** The policy walked but consistently veered to one side on the real robot.
**Root cause.** The observation included the **base linear velocity**, which the simulator knows
exactly but the real Go2 cannot measure (no velocity sensor / VIO). At deployment we fed zeros,
so the actor was permanently **out of its training distribution** - it had learned to rely on a
signal that was now a constant lie.
**Fix (V2→V3).** Rebuild as an **asymmetric actor-critic**: the base linear velocity is given
**only to the critic** (discarded after training), and **removed from the actor** (48 → 45
dims). The actor now learns a policy that lives entirely within what the real robot can perceive.
This was, per the project notes, *the single largest improvement to real-world performance*, and
it is the conceptual centre of the cognitive analysis (epistemic state vs ground truth).

### Failure 5 - Unmodelled actuator dynamics (a dynamics gap)
**Symptom.** Residual instability and gain-sensitivity after the drift was fixed.
**Root cause.** Real motors have **latency**, **non-ideal Kp/Kd**, and **stiction/backlash** that
the ideal simulated actuator does not.
**Fix (V3).** Model them as randomisation rather than identify them exactly: switch to Isaac
Lab's **`DelayedPDActuator`** with **0-20 ms** randomised command latency, randomise **actuator
gains ±20%**, and randomise **joint friction and armature**. The policy becomes robust to
*whatever* configuration it meets on the robot - cheaper and more general than online stiffness
adaptation.

### Failure 6 - Twisted, uncoordinated motion (a convention gap)
**Symptom.** The very first deployment produced incoherent, twisted leg motion.
**Root cause.** Isaac Lab orders the 12 joints by PhysX BFS traversal (all hips, all thighs, all
calves); the Unitree SDK orders them per-leg (FR, FL, RR, RL). The action vector was being
applied to the wrong joints.
**Fix.** An explicit, verified permutation in the deploy script
(`SIM_TO_REAL = [3,0,9,6,4,1,10,7,5,2,11,8]`, and its inverse), checked against Isaac Lab
discussion #506. Stable across versions since.

---

## 3. Bridging mechanisms (why zero-shot works at all)

Beyond the per-failure fixes, three systematic mechanisms close the gap:

1. **Domain randomization.** Every episode randomises mass (−2…+4 kg), centre of mass (±5 cm),
   static/dynamic friction (0.4-1.4 / 0.3-1.0), restitution, external forces/torques (±3),
   periodic pushes (±0.5 m/s), actuator gains (±20%), joint friction/armature, and latency
   (0-20 ms). The real robot is "just one more sample" from this distribution.
2. **Observation noise matched to real sensors.** Additive uniform noise is applied to each
   observation in training. Notably, the default joint-velocity noise (±1.5 rad/s) was
   *unrealistically large* and made the policy over-filter; it was reduced to **±0.3 rad/s** to
   match a real encoder (a subtle but real fix).
3. **Stand-still training.** 20 % of simulated robots receive a zero command, so the policy
   learns to *hold station* on purpose rather than trot in place - essential for a safe handover
   from the executive layer's READY state.

---

## 4. Residual gap and limitations

- **No exteroception on the real robot.** The deployed policy is *blind* (proprioception only).
  The rough-terrain policy (with a 187-dim height scan) is trained and validated in simulation -
  including stair climbing - but **not yet deployed on hardware**, because the Go2 used has no
  configured depth/LiDAR feed into the observation; the height scan is currently zero-filled at
  deploy. Closing this is the main next step.
- **Base velocity is still unobserved, not estimated.** The asymmetric actor-critic *avoids*
  needing it rather than recovering it. A concurrent state-estimator (à la Ji et al. 2022) or an
  implicit estimator (DreamWaQ) would let the policy *use* a velocity estimate rather than work
  without one.
- **Flat ground only.** Real deployment is validated on flat lab floor. The policy walked
  zero-shot and remained stable for an extended period before requiring re-tuning; robustness
  over long horizons and varied surfaces is not yet quantified on hardware.
- **Quantitative real-robot metrics are limited.** Simulation metrics (99.7 % survival, 0.22 m/s
  tracking error) are solid; on hardware the evidence is qualitative (stable zero-shot walking,
  see demo videos) plus the disappearance of the specific failure modes above. A logged real-robot
  evaluation (cost of transport, measured tracking error, disturbance recovery) is future work.
- **No deliberative layer.** As discussed in the cognitive-architecture document, the system is
  the reactive + executive layers; a navigation/task planner on top is not implemented.

---

## 5. Possible improvements

1. Deploy the rough-terrain policy with a real depth sensor feeding the height scan (stair
   climbing on hardware).
2. Add a learned base-velocity estimator so the actor can *use* velocity instead of avoiding it.
3. Log a quantitative real-robot evaluation: measured velocity-tracking error, **cost of
   transport** vs the manufacturer's default gait, base pitch/roll stability, and push-recovery
   success rate.
4. Close the loop with a deliberative layer (e.g. Nav2) issuing the velocity commands, turning
   the skill into the motion half of a full task-and-motion-planning stack.
5. Replace fixed domain-randomization ranges with an adaptive / curriculum schedule to reduce the
   conservatism cost of very wide randomization (studied separately in the related RT2 work).
