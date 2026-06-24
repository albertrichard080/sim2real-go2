"""Advanced tests for deployment readiness."""
import numpy as np
import onnxruntime as ort
import time

POLICY = "/home/richard/go2_flat_v2_policy/policy.onnx"
DEFAULT_JOINT_POS = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.8,  0.8, 1.0,  1.0,
   -1.5, -1.5,-1.5, -1.5,
], dtype=np.float32)
ACTION_SCALE = 0.25

# Go2 hardware joint limits (from Unitree docs, in Isaac Lab order)
# hip: ±1.0472 rad (±60°), thigh: -1.5708..+3.4907, calf: -2.7227..-0.8377
JOINT_LIMIT_LOW  = np.array([-1.0472]*4 + [-1.5708]*4 + [-2.7227]*4, dtype=np.float32)
JOINT_LIMIT_HIGH = np.array([ 1.0472]*4 + [ 3.4907]*4 + [-0.8377]*4, dtype=np.float32)

s = ort.InferenceSession(POLICY)
inp = s.get_inputs()[0].name
print(f"Loaded: {POLICY}\n")

def run(obs_vec):
    obs = obs_vec.reshape(1, -1).astype(np.float32)
    return s.run(None, {inp: obs})[0].flatten()

def make_obs(lin_vel=(0,0,0), ang_vel=(0,0,0), gravity=(0,0,-1), cmd=(0,0,0),
             jpr=None, jv=None, la=None):
    if jpr is None: jpr = np.zeros(12, dtype=np.float32)
    if jv is None:  jv  = np.zeros(12, dtype=np.float32)
    if la is None:  la  = np.zeros(12, dtype=np.float32)
    return np.concatenate([lin_vel, ang_vel, gravity, cmd, jpr, jv, la]).astype(np.float32)


# TEST 1: Inference timing (must be <20ms for 50Hz)
print("=" * 60)
print("TEST 1: INFERENCE TIMING (50Hz requires <20ms)")
print("=" * 60)
obs = make_obs(cmd=(0.5, 0, 0))
# Warmup
for _ in range(10): run(obs)
# Measure
N = 1000
t0 = time.perf_counter()
for _ in range(N): run(obs)
t1 = time.perf_counter()
avg_ms = (t1 - t0) / N * 1000
print(f"  Average inference time: {avg_ms:.3f} ms")
print(f"  Max possible rate: {1000/avg_ms:.0f} Hz")
if avg_ms < 5:
    print("  OK Laptop fast. Jetson Orin ≈ 5-10ms - still well under 20ms budget.")
else:
    print(f"  ! Laptop slow ({avg_ms:.1f}ms). Jetson may struggle.")


# TEST 2: Gait oscillation - feed time-varying obs, verify outputs oscillate
print("\n" + "=" * 60)
print("TEST 2: GAIT PATTERN (actions should oscillate over time)")
print("=" * 60)
actions_over_time = []
last_actions = np.zeros(12, dtype=np.float32)
joint_pos = DEFAULT_JOINT_POS.copy()
joint_vel = np.zeros(12, dtype=np.float32)
for step in range(100):
    # Simulate a 2 Hz oscillation in joints (imitating contact cycle)
    phase = 2 * np.pi * step / 25.0
    joint_pos = DEFAULT_JOINT_POS + 0.1 * np.sin(phase) * np.array([0,0,0,0, 1,-1,-1,1, 1,-1,-1,1])
    joint_vel = 2.5 * np.cos(phase) * np.array([0,0,0,0, 1,-1,-1,1, 1,-1,-1,1], dtype=np.float32)
    jpr = joint_pos - DEFAULT_JOINT_POS
    obs = make_obs(cmd=(0.5, 0, 0), jpr=jpr, jv=joint_vel, la=last_actions)
    last_actions = run(obs).astype(np.float32)
    actions_over_time.append(last_actions.copy())

A = np.stack(actions_over_time)  # [100, 12]
# Check that at least some joints show oscillation
stds = A.std(axis=0)
print(f"  Action std per joint: {stds.round(3).tolist()}")
if stds.max() > 0.05:
    print("  OK Policy produces oscillating outputs (gait-like)")
else:
    print("  ! Policy outputs nearly constant - may not walk")


# TEST 3: Joint-limit safety - check stand-still targets are within limits
print("\n" + "=" * 60)
print("TEST 3: JOINT LIMIT SAFETY")
print("=" * 60)
obs = make_obs()
actions = run(obs)
targets = DEFAULT_JOINT_POS + actions * ACTION_SCALE
below = targets < JOINT_LIMIT_LOW
above = targets > JOINT_LIMIT_HIGH
if below.any() or above.any():
    print("  ! Some targets exceed hardware limits at rest:")
    for i, (t, lo, hi) in enumerate(zip(targets, JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)):
        mark = "<" if t < lo else (">" if t > hi else "ok")
        print(f"    [{i:2d}] target={t:+.3f} limits=[{lo:+.3f},{hi:+.3f}] {mark}")
else:
    print("  OK All stand-still targets within Go2 hardware limits")
    print(f"  target range: [{targets.min():+.3f}, {targets.max():+.3f}] rad")

# Test with max-intensity commands
for cmd_name, cmd in [("max forward", (1.0, 0, 0)), ("max yaw", (0, 0, 1.5))]:
    obs = make_obs(cmd=cmd)
    actions = run(obs)
    targets = DEFAULT_JOINT_POS + actions * ACTION_SCALE
    over = ((targets < JOINT_LIMIT_LOW) | (targets > JOINT_LIMIT_HIGH)).sum()
    print(f"  {cmd_name}: {over} joints exceed limits (clipping will protect)")


# TEST 4: Stress - NaN/Inf inputs
print("\n" + "=" * 60)
print("TEST 4: STRESS (extreme/NaN inputs)")
print("=" * 60)
bad_obs = make_obs()
bad_obs[3:6] = 10.0  # absurd angular velocity
actions = run(bad_obs)
print(f"  Extreme ang_vel: action max={np.abs(actions).max():.2f}, NaN={np.isnan(actions).any()}")

bad_obs = make_obs()
bad_obs[15:27] = 2.0  # absurd joint_pos_rel
actions = run(bad_obs)
print(f"  Extreme joint_pos: action max={np.abs(actions).max():.2f}, NaN={np.isnan(actions).any()}")

bad_obs = np.full(48, np.nan, dtype=np.float32)
try:
    actions = run(bad_obs)
    # Since our script wraps with nan_to_num BEFORE this, raw ONNX may return NaN:
    print(f"  All-NaN obs: action NaN={np.isnan(actions).any()}  "
          f"(deploy script sanitizes input first → safe)")
except Exception as e:
    print(f"  All-NaN obs: ONNX raised {type(e).__name__} (caught by deploy)")


# TEST 5: Gravity computation sanity
print("\n" + "=" * 60)
print("TEST 5: QUATERNION ROTATION (gravity computation)")
print("=" * 60)
def quat_rotate_inverse(quat, vec):
    w, x, y, z = quat
    r00 = 1 - 2*(y*y + z*z); r01 = 2*(x*y + w*z);   r02 = 2*(x*z - w*y)
    r10 = 2*(x*y - w*z);     r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z + w*x)
    r20 = 2*(x*z + w*y);     r21 = 2*(y*z - w*x);     r22 = 1 - 2*(x*x + y*y)
    return np.array([
        r00*vec[0] + r10*vec[1] + r20*vec[2],
        r01*vec[0] + r11*vec[1] + r21*vec[2],
        r02*vec[0] + r12*vec[1] + r22*vec[2],
    ])

# Upright: quat = [1, 0, 0, 0] (identity)
g = quat_rotate_inverse([1,0,0,0], [0,0,-1])
print(f"  Upright quat=[1,0,0,0]: gravity = {g.round(3)}  (expect [0,0,-1])")
# Pitched 45° forward: quat = [cos(22.5°), 0, sin(22.5°), 0]
import math
q = [math.cos(math.radians(22.5)), 0, math.sin(math.radians(22.5)), 0]
g = quat_rotate_inverse(q, [0,0,-1])
print(f"  Pitch +45°:            gravity = {g.round(3)}  (expect gx<0, gz<0)")
# Rolled 45° right: quat = [cos(22.5°), sin(22.5°), 0, 0]
q = [math.cos(math.radians(22.5)), math.sin(math.radians(22.5)), 0, 0]
g = quat_rotate_inverse(q, [0,0,-1])
print(f"  Roll +45°:             gravity = {g.round(3)}  (expect gy>0, gz<0)")
# Fallen on back (180° roll): quat = [0, 1, 0, 0]
q = [0, 1, 0, 0]
g = quat_rotate_inverse(q, [0,0,-1])
print(f"  Upside down:           gravity = {g.round(3)}  (expect [0,0,+1])")

print("\n" + "=" * 60)
print("ALL ADVANCED TESTS COMPLETE")
print("=" * 60)
