# Go2 RL Policy Deployment

Production-grade deployment script for running Isaac Lab-trained RL policies on real Unitree Go2 EDU hardware.

## Safety Features

- **State machine**: `IDLE → STAND_UP → READY → WALK → EMERGENCY_STOP`
- **Gradual stand-up**: 3s interpolation from current pose to default stance (soft PD gains ramp up)
- **Sensor watchdog**: Triggers e-stop if no data received for 100ms
- **Body-tilt safety**: Triggers e-stop if robot tilts more than 45° during walking
- **Sport mode disable**: Disabled BEFORE any low-level command
- **Emergency stop**: `Ctrl+C` / `ESC` (keyboard) / `L2+B` (gamepad)
- **Zero-torque shutdown**: Final 0.5s of zero-torque commands so robot goes limp safely

## Prerequisites

### Software (install on your laptop)
```bash
pip install unitree_sdk2py onnxruntime numpy pygame
```

### Hardware
- Unitree **Go2 EDU** (NOT Air/Pro - EDU is required for low-level control)
- **USB-C to Ethernet adapter** (laptop has no ethernet port)
- **Ethernet cable** (Cat5e/Cat6)
- Optional: USB gamepad (Xbox/PS4) for smoother analog control

## Connection Setup

1. Plug USB-C→Ethernet adapter into laptop
2. Connect Ethernet cable from adapter to Go2 (port is on robot's back under rubber cover)
3. Find your interface name:
```bash
ip addr
# Look for enx... or similar
```
4. Set static IP in Go2's subnet:
```bash
sudo ifconfig <your_interface> 192.168.123.222 netmask 255.255.255.0
```
5. Verify:
```bash
ping 192.168.123.18    # should respond
```

## Running

### Flat policy (V3 recommended - best metrics)
```bash
python deploy_go2.py \
    --policy /home/richard/go2_policies/v1_flat_sim2real/policy.onnx \
    --interface <your_interface>
```

### Rough terrain policy
```bash
python deploy_go2.py \
    --policy /home/richard/go2_policies/v2_rough_sim2real/exported/policy.onnx \
    --interface <your_interface>
```

Note: Rough terrain policy expects 187-dim height scan. Without a depth sensor mounted on the robot, this input is filled with zeros, which may cause suboptimal behavior. Use flat policy for indoor flat-ground walking.

### Optional arguments
```
--obs-dim 48              # 48=flat, 235=rough (auto-detected from ONNX if omitted)
--kp 40.0                 # policy stiffness (default tested value)
--kd 1.0                  # policy damping
--action-scale 0.25       # action scale (MUST match training)
--max-vel-xy 1.0          # max commanded linear speed (m/s)
--max-vel-yaw 1.0         # max commanded angular speed (rad/s)
--no-safety               # disable tilt safety (NOT recommended)
```

## Deploy Flow

1. **Script starts** → disables sport mode (you'll see robot go limp)
2. **Sensor wait** → waits for first IMU/motor state message
3. **Stand-up (3s)** → slowly interpolates from current pose to default stance with soft→firm PD gains
4. **READY state** → holds standing pose, prompts you to press ENTER
5. **WALK state** → RL policy active at 50 Hz, accepts velocity commands from gamepad/keyboard
6. **Stop** → any of: Ctrl+C, ESC key, L2+B gamepad, or tilt exceeds 45°

## Controls

### Gamepad
- Left stick Y: forward/backward
- Left stick X: strafe left/right
- Right stick X: turn left/right
- L2+B: emergency stop

### Keyboard (pygame window must be focused)
- W/S: forward/back
- A/D: strafe left/right
- Q/E: turn left/right
- ESC: emergency stop

## Key Parameters (must match training)

| Parameter | Value | Where it's set |
|-----------|-------|----------------|
| Action scale | 0.25 | `rough_env_cfg.py` |
| Control frequency | 50 Hz | `sim.dt=0.005, decimation=4` |
| Observation dims | 48 (flat) or 235 (rough) | `velocity_env_cfg.py` |
| Default joint pos | [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5] | `UNITREE_GO2_CFG` |
| PD gains | Kp=40, Kd=1.0 | Matches `unitree_rl_lab` recommendation |

## Joint Mapping

Isaac Lab uses URDF alphabetical order: `[FL, FR, RL, RR] x [hip, thigh, calf]`
Unitree SDK uses: `[FR, FL, RR, RL] x [hip, thigh, calf]`

The remapping `[3,4,5,0,1,2,9,10,11,6,7,8]` is applied both directions (it's symmetric). Wrong mapping is the #1 cause of deployment failure.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `No robot state received in 5s` | Network not connected | Check cable, check IP (192.168.123.222) |
| `Could not disable sport mode` | Another client has mode lock | Restart robot, check Unitree app is closed |
| Robot falls immediately | Joint mapping wrong / action scale wrong | Verify `SIM_TO_REAL` / `action-scale 0.25` |
| Robot jittery / jerky | PD gains too high | Try `--kp 30 --kd 0.8` |
| Robot too loose / can't stand | PD gains too low | Try `--kp 50 --kd 1.2` |
| Foot dragging | Policy trained without enough air-time reward | Use V2/V3 policy (has feet_slide penalty) |

## Pre-Deployment Checklist

- [ ] Go2 EDU is powered on and booted
- [ ] Ethernet cable connected
- [ ] Laptop IP set to 192.168.123.222
- [ ] `ping 192.168.123.99` succeeds
- [ ] Open floor area at least 2m x 2m
- [ ] Someone ready to catch the robot
- [ ] Emergency stop method rehearsed (Ctrl+C)
- [ ] Policy file path is correct
- [ ] Interface name is correct

## What NOT to Do

- Don't run on WiFi - latency too high for 50 Hz control
- Don't deploy rough policy without height scan data on complex terrain
- Don't change PD gains without small tests first
- Don't deploy during robot battery < 30% (motors may not have enough torque)
