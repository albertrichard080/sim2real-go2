# Cognitive Approach: Reading the Go2 Controller Through the COGAR Course

**Course:** Cognitive Architectures for Robotics (COGAR), University of Genoa, DIBRIS (Prof. F. Mastrogiovanni)
**Assignment:** 13 (E4) - Go2 RL-Based Locomotion and Sim-to-Real Deployment
**Author:** Richard Albert King Mechoda (8525970)

This note connects the project to the course using the course's own framework and vocabulary
(Parts 0-4). It is deliberately honest about what the project does and does not implement.

---

## 1. The course's frame: the Sense - Reasoning/Plan - Act loop

The course frames every robot architecture on one recurring loop, **Sense -> Reasoning/Plan ->
Act**, and distinguishes four families:

- **Hierarchical (Sense-Plan-Act):** all three stages, in sequence.
- **Reactive:** the Reasoning/Plan stage is removed; Sense maps directly to Act.
- **Behaviour-based:** several parallel sense-act behaviours plus an arbitration step.
- **Hybrid reactive-deliberative (a.k.a. cognitive):** a direct Sense<->Act loop with
  Reasoning/Plan sitting on top.

The deployed Go2 controller is, in these terms, a **reactive architecture**: at 50 Hz it senses
(IMU and encoders) and maps the observation directly to motor targets, with no planning and no
run-time world model. The velocity command stands in for the Reasoning/Plan stage; a planner can
later attach there, which would make the full system the *hybrid reactive-deliberative* family.

---

## 2. The policy is a behaviour (Part 4)

Part 4 defines a **behaviour** as "a mapping of sensory inputs to a pattern of motor actions used
to achieve a task," decomposed into a **perceptual schema** and a **motor schema**. The policy is
exactly this:

- **Perceptual schema** = building the 45-dim observation (projected gravity, angular velocity,
  joint states, command, last action).
- **Motor schema** = the twelve joint-position offsets, realised through the 500 Hz PD loop.

The course's System 1 / System 2 reading (Kahneman) also fits: the policy is a **Type-1,
reactive** behaviour that has been "compiled down" from a slow optimisation. That is precisely
what reinforcement learning does: deliberative credit assignment at training time is compiled into
a fast, model-free reaction at run time.

---

## 3. Behaviour coordination and subsumption (Part 4)

A single reactive behaviour is not enough on hardware; something must decide when it runs and stop
it before it breaks the robot. The deployment state machine
`IDLE -> STAND_UP -> READY -> WALK -> DAMPING -> STOPPED` does this, and it is a textbook instance
of the course's **behaviour coordination**:

- It is **competitive**, winner-take-all coordination by **fixed priority**.
- A small set of **safety reflexes** (tilt above 46 deg, a 100 ms sensor watchdog, a NaN/Inf
  guard, action and joint clipping) sit above the walking behaviour and, when triggered,
  **suppress** it - in the exact sense of Brooks' subsumption taught in the course, where
  *suppression* replaces the signal a behaviour sends to the actuators (and *inhibition* would
  block an input).
- These are also **reflexes** in the ethological sense the course gives: rapid, automatic,
  involuntary responses correlated with stimulus strength, the kind the course notes are "used for
  locomotion."

So the system realises the lower half of a behaviour-based architecture: a learned reactive skill,
coordinated and protected by a fixed-priority arbiter.

---

## 4. The course's design patterns, realised (Part 2)

The deployment code is built from the robotics design patterns the course teaches:

| Course pattern | Where it appears in this project |
|---|---|
| **Adapter** | the joint-order remap between the Isaac Lab (PhysX BFS) and Unitree (per-leg) index conventions - the course's own "adapt coordinate systems" example |
| **Computational** | the policy step: one periodic `on_update()` that collects the observation, runs one inference, and publishes the joint targets, with no blocking calls |
| **Publish-subscribe** | the Unitree DDS / ROS 2 topics push IMU, encoder and command data to the controller, which never assumes who is listening |
| **Sensor-device / interface** | the IMU and encoder readers wrap the SDK driver as pure, periodic data producers |

---

## 5. Marr's three levels and the reference architecture

The course uses **Marr's three levels** as the bridge from biology to software. The project reads
as: **Level 1 (what)** - a quadruped should walk and recover (an existence proof from ethology);
**Level 2 (processes)** - the perceptual and motor schemas of the policy behaviour plus the
fixed-priority coordinator (a finite-state machine); **Level 3 (implementation)** - ONNX inference
on the Jetson Orin NX with a PD loop on the motor driver.

Within the course's reference cognitive architecture (Darvish et al., IEEE T-RO 2021), which maps
Task Representation/Manager and the Knowledge Base to Reasoning/Plan, and the Execution Manager,
Path Planner and Controller to Act, this project implements the **Act / Controller** role: the
reusable locomotion competence on which a future Reasoning/Plan layer (a navigation or task
planner issuing velocity commands) would stand. Task-and-motion planning, raised in the
hierarchical lecture, is the natural way to close that loop.

---

## 6. Honest scope

The project realises the **reactive behaviour** and its **fixed-priority coordinator**. It does
not implement a deliberative Reasoning/Plan layer, a knowledge base, or symbolic reasoning - those
are the upper parts of the course's hybrid/cognitive family and the natural next step. Framing the
work this way keeps the engineering honest while showing that a learned locomotion controller is
not outside cognitive architecture: it is the embodied, reactive foundation that everything the
course calls "cognition" must ultimately command.
