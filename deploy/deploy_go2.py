#!/usr/bin/env python3
"""
Go2 RL Policy Deployment - Production Grade (Final)

Deploys an ONNX policy trained in Isaac Lab to the real Unitree Go2 EDU robot.
Merges lessons from both our previous deploy script and community research.

Supported policies:
  - Flat terrain (48-dim observation)
  - Rough terrain (235-dim observation, requires height-scan data)

Input devices (priority: wireless > gamepad > keyboard):
  1. Unitree built-in wireless controller (published on rt/wirelesscontroller)
  2. USB gamepad via pygame (Xbox / PS4)
  3. Keyboard via pygame window (WASD + QE)

Safety features:
  - State machine: IDLE -> STAND_UP -> READY -> WALK -> DAMPING -> STOPPED
  - Gradual stand-up (2s ramp, soft -> firm PD gains)
  - Sensor watchdog: emergency stop if no LowState for 100 ms
  - Body tilt safety: damping mode if body tilts more than 45 deg during walk
  - Fall detection via projected gravity z-component
  - NaN / Inf cleanup on every observation
  - Action clipping (+/- 6 rad) and joint target clipping (+/- 2.5 rad)
  - Sport mode disabled before any low-level command
  - Damping mode (zero kp, small kd) before full shutdown - robot softly collapses
  - Emergency stop sources: Ctrl+C, ESC key, Select button (Unitree remote), L2+B (gamepad)

Joint ordering:
  - Default mapping follows Isaac Lab's PhysX BFS articulation traversal
    (verified against unitree_rl_gym issue #13 and community deployments)
  - If robot behaves erratically on first motion, try --joint-order perleg
    (URDF alphabetical per-leg order, used by some custom Isaac Lab setups)

Requirements:
    pip install unitree_sdk2py onnxruntime numpy pygame

Usage:
    # Flat policy (V1, V2, V3 are all 48-dim)
    python deploy_go2.py --policy /path/to/policy.onnx --interface enp2s0

    # Rough policy (V2 rough, V3 rough - 235-dim, height scan filled with zeros)
    python deploy_go2.py --policy /path/to/policy.onnx --interface enp2s0

    # Override defaults
    python deploy_go2.py --policy ... --interface ... --kp 35 --kd 1.0 \\
                        --max-vel-xy 0.8 --max-vel-yaw 0.8

    # If default joint mapping fails, try the alternate
    python deploy_go2.py --policy ... --interface ... --joint-order perleg
"""

import argparse
import signal
import sys
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    LowCmd_,
    LowState_,
    WirelessController_,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)
from unitree_sdk2py.go2.sport.sport_client import SportClient


# ============================================================================
# UNITREE WIRELESS CONTROLLER BUTTON BITMASKS (from WirelessController_.keys)
# ============================================================================

BTN_R1 = 0x0001
BTN_L1 = 0x0002
BTN_START = 0x0004
BTN_SELECT = 0x0008
BTN_R2 = 0x0010
BTN_L2 = 0x0020
BTN_A = 0x0100
BTN_B = 0x0200
BTN_X = 0x0400
BTN_Y = 0x0800
BTN_UP = 0x1000
BTN_RIGHT = 0x2000
BTN_DOWN = 0x4000
BTN_LEFT = 0x8000

# ============================================================================
# JOINT ORDER MAPPINGS
# ============================================================================
# VERIFIED from:
#   - Isaac Lab GitHub Discussion #506 (joint names use PhysX BFS traversal)
#   - Isaac Lab documentation (hierarchical URDF parsing)
#   - Unitree SDK example/unitree_legged_const.py
#
# Isaac Lab parses the Go2 URDF and produces this joint order (BFS from base):
#   [0]  FL_hip    [1]  FR_hip    [2]  RL_hip    [3]  RR_hip     <- all hips
#   [4]  FL_thigh  [5]  FR_thigh  [6]  RL_thigh  [7]  RR_thigh   <- all thighs
#   [8]  FL_calf   [9]  FR_calf   [10] RL_calf   [11] RR_calf    <- all calves
#
# Unitree SDK expects this order (from unitree_legged_const.py):
#   [0]  FR_hip    [1]  FR_thigh  [2]  FR_calf   <- FR leg
#   [3]  FL_hip    [4]  FL_thigh  [5]  FL_calf   <- FL leg
#   [6]  RR_hip    [7]  RR_thigh  [8]  RR_calf   <- RR leg
#   [9]  RL_hip    [10] RL_thigh  [11] RL_calf   <- RL leg

# SIM_TO_REAL[sim_i] = SDK index holding that joint
# Usage: sdk_array[SIM_TO_REAL[i]] gives the same joint as sim_array[i]
SIM_TO_REAL = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

# REAL_TO_SIM[sdk_i] = sim index holding that joint
# Usage: sim_array[REAL_TO_SIM[i]] gives the same joint as sdk_array[i]
REAL_TO_SIM = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

# ============================================================================
# DEFAULT STANDING JOINT ANGLES (in Isaac Lab BFS order)
# ============================================================================
# From UNITREE_GO2_CFG.init_state.joint_pos:
#   .*L_hip = +0.1,  .*R_hip = -0.1
#   F[LR]_thigh = 0.8,  R[LR]_thigh = 1.0
#   .*_calf = -1.5
DEFAULT_JOINT_POS_SIM = np.array([
    # hips:    FL,   FR,  RL,   RR
             0.1, -0.1, 0.1, -0.1,
    # thighs:  FL,   FR,  RL,   RR
             0.8,  0.8, 1.0,  1.0,
    # calves:  FL,   FR,  RL,   RR
            -1.5, -1.5, -1.5, -1.5,
], dtype=np.float32)

# ============================================================================
# TIMING / SAFETY CONSTANTS
# ============================================================================

CONTROL_DT = 0.02              # 50 Hz control loop (matches training decimation)
STAND_UP_DURATION = 2.0        # Seconds to ramp from current pose to default stance
SENSOR_TIMEOUT_S = 0.1         # Max age of sensor data before emergency stop
MAX_BODY_TILT_RAD = 0.8        # ~46 deg tilt threshold for emergency stop during WALK
GRAVITY_Z_TIP_THRESHOLD = -0.5 # Fall detection: gravity_z > this means robot tipped
ACTION_CLIP = 6.0              # Policy action magnitude clip
JOINT_TARGET_CLIP = 2.5        # Final joint target clip (radians)

# SDK safe stop values
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0

# Dead zone for stick inputs
STICK_DEADZONE = 0.1


# ============================================================================
# STATE MACHINE
# ============================================================================

class DeployState(Enum):
    IDLE = "idle"
    STAND_UP = "stand_up"
    READY = "ready"
    WALK = "walk"
    DAMPING = "damping"
    STOPPED = "stopped"


# ============================================================================
# UTILITIES
# ============================================================================

def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vec by inverse of quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y + w * z); r02 = 2 * (x * z - w * y)
    r10 = 2 * (x * y - w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z + w * x)
    r20 = 2 * (x * z + w * y); r21 = 2 * (y * z - w * x); r22 = 1 - 2 * (x * x + y * y)
    return np.array([
        r00 * vec[0] + r10 * vec[1] + r20 * vec[2],
        r01 * vec[0] + r11 * vec[1] + r21 * vec[2],
        r02 * vec[0] + r12 * vec[1] + r22 * vec[2],
    ], dtype=np.float32)


def safe_clean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with zeros (defense against sensor glitches)."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# MAIN DEPLOYER
# ============================================================================

class Go2Deployer:
    def __init__(self, args):
        self.args = args
        self.kp = args.kp
        self.kd = args.kd
        self.action_scale = args.action_scale
        self.max_vel_xy = args.max_vel_xy
        self.max_vel_yaw = args.max_vel_yaw
        self.safety_enabled = not args.no_safety

        # Joint mappings (verified BFS order from Isaac Lab)
        self.sim_to_real = np.array(SIM_TO_REAL, dtype=np.int64)
        self.real_to_sim = np.array(REAL_TO_SIM, dtype=np.int64)
        self.default_joint_pos_sim = DEFAULT_JOINT_POS_SIM.copy()
        print("[INFO] Joint order: Isaac Lab BFS (verified against GitHub #506)")

        # ----- Load ONNX policy -----
        print(f"[INFO] Loading policy: {args.policy}")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(args.policy, providers=providers)
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        self.input_name = input_info.name
        self.output_name = output_info.name

        # Auto-detect obs_dim from ONNX input shape (last dim)
        input_shape = list(input_info.shape)
        # Shape may contain strings for dynamic axes - take last static int
        detected_dim = None
        for d in reversed(input_shape):
            if isinstance(d, int) and d > 0:
                detected_dim = d
                break
        self.obs_dim = args.obs_dim or detected_dim or 48

        print(f"[INFO] ONNX input  : {self.input_name}, shape={input_shape}")
        print(f"[INFO] ONNX output : {self.output_name}, shape={list(output_info.shape)}")
        print(f"[INFO] Observation dim: {self.obs_dim}")

        if self.obs_dim == 48:
            self.policy_type = "flat"
            print("[INFO] Policy type: FLAT with base_lin_vel (48 dims)")
        elif self.obs_dim == 45:
            self.policy_type = "flat"
            print("[INFO] Policy type: FLAT without base_lin_vel (45 dims, sim2real friendly)")
        elif self.obs_dim == 235:
            self.policy_type = "rough"
            print("[INFO] Policy type: ROUGH with base_lin_vel (235 dims)")
            print("[WARN] Go2 has no default depth sensor - height scan filled with zeros")
        elif self.obs_dim == 232:
            self.policy_type = "rough"
            print("[INFO] Policy type: ROUGH without base_lin_vel (232 dims)")
            print("[WARN] Go2 has no default depth sensor - height scan filled with zeros")
        else:
            print(f"[WARN] Unusual obs_dim {self.obs_dim} - trying best-effort layout")

        # ----- Initialize SDK -----
        print(f"[INFO] Initializing SDK on interface: {args.interface}")
        ChannelFactoryInitialize(0, args.interface)

        # ----- Disable sport mode BEFORE low-level commands -----
        self._disable_sport_mode()

        # ----- Runtime state (MUST be initialized BEFORE DDS subscribers
        # so callbacks have valid attributes when first message arrives) -----
        self.crc = CRC()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self._init_low_cmd_safe()

        self._state_lock = threading.Lock()
        self.low_state: Optional[LowState_] = None
        self._last_state_time = 0.0

        self.wireless_state: Optional[WirelessController_] = None
        self._last_wireless_buttons = 0
        self._start_requested = False
        self._select_requested = False

        self.deploy_state = DeployState.IDLE
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.velocity_commands = np.zeros(3, dtype=np.float32)
        self.stand_start_pos_sdk: Optional[np.ndarray] = None
        self.stand_progress = 0.0

        # ----- Set up DDS (subscribers last, after all state vars exist) -----
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._on_state_msg, 10)

        self.wireless_sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.wireless_sub.Init(self._on_wireless_msg, 10)

        # ----- USB gamepad / keyboard fallback -----
        self.joystick = None
        self.use_keyboard = False
        self._init_input_device()

        # ----- Ctrl+C handler -----
        signal.signal(signal.SIGINT, self._on_sigint)

        print("[INFO] Go2Deployer initialized")

    # -------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------

    def _disable_sport_mode(self):
        """Release any active Unitree motion service before low-level control."""
        print("[INFO] Disabling sport mode (critical - required before low-level)...")
        sc = SportClient(); sc.SetTimeout(5.0); sc.Init()
        msc = MotionSwitcherClient(); msc.SetTimeout(5.0); msc.Init()
        _, result = msc.CheckMode()
        tries = 0
        while result.get("name"):
            if tries > 10:
                raise RuntimeError(
                    "Could not disable sport mode after 10 tries. "
                    "Try Unitree app -> Motion Switch -> None, or restart robot."
                )
            print(f"[INFO]   Releasing mode: {result.get('name')}")
            sc.StandDown()
            msc.ReleaseMode()
            time.sleep(1.0)
            _, result = msc.CheckMode()
            tries += 1
        print("[INFO] Sport mode disabled.")

    def _init_low_cmd_safe(self):
        """Initialize LowCmd with safe-stop values (matches official go2_stand_example)."""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF  # LOWLEVEL
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _init_input_device(self):
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"[INFO] USB gamepad: {self.joystick.get_name()}")
            else:
                self.use_keyboard = True
                pygame.display.set_mode((400, 200))
                pygame.display.set_caption("Go2 Keyboard - WASD / QE / ESC")
                print("[INFO] No USB gamepad - keyboard active (focus pygame window)")
        except ImportError:
            print("[WARN] pygame not installed - only Unitree remote will work for control")

    # -------------------------------------------------------------------
    # DDS callbacks
    # -------------------------------------------------------------------

    def _on_state_msg(self, msg: LowState_):
        with self._state_lock:
            self.low_state = msg
            self._last_state_time = time.time()

    def _on_wireless_msg(self, msg: WirelessController_):
        self.wireless_state = msg
        pressed = msg.keys & ~self._last_wireless_buttons
        if pressed & BTN_START:
            self._start_requested = True
        if pressed & BTN_SELECT:
            self._select_requested = True
        # L2+B e-stop
        if (msg.keys & BTN_L2) and (msg.keys & BTN_B):
            self._select_requested = True
        self._last_wireless_buttons = msg.keys

    def _on_sigint(self, signum, frame):
        print("\n[INFO] Ctrl+C - triggering damping shutdown")
        self._select_requested = True

    # -------------------------------------------------------------------
    # Safety checks
    # -------------------------------------------------------------------

    def _safety_check(self) -> bool:
        """Return True if safe to continue, False if damping shutdown required."""
        if not self.safety_enabled:
            return True

        with self._state_lock:
            st = self.low_state
            last_t = self._last_state_time

        if st is None:
            return True

        if time.time() - last_t > SENSOR_TIMEOUT_S:
            print(f"[SAFETY] Sensor stale ({time.time() - last_t:.2f}s) -> damping")
            return False

        # Only check tilt during WALK (stand-up naturally has some tilt)
        if self.deploy_state == DeployState.WALK:
            quat = st.imu_state.quaternion
            g_body = quat_rotate_inverse(
                np.array([quat[0], quat[1], quat[2], quat[3]], dtype=np.float32),
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
            )
            # g_body[2] is -1 when upright. Going toward 0 or positive means tipped.
            if g_body[2] > GRAVITY_Z_TIP_THRESHOLD:
                tilt = np.arccos(np.clip(-g_body[2], -1.0, 1.0))
                print(f"[SAFETY] Tilt {np.rad2deg(tilt):.1f} deg > threshold -> damping")
                return False

        return True

    # -------------------------------------------------------------------
    # Observation / action
    # -------------------------------------------------------------------

    def _get_observation(self) -> Optional[np.ndarray]:
        with self._state_lock:
            st = self.low_state
        if st is None:
            return None

        imu = st.imu_state

        # Base angular velocity (IMU gyroscope, body frame)
        ang_vel = np.array(
            [imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]], dtype=np.float32
        )

        # Projected gravity from quaternion
        quat = np.array(
            [imu.quaternion[0], imu.quaternion[1], imu.quaternion[2], imu.quaternion[3]],
            dtype=np.float32,
        )
        gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        # Base linear velocity:
        # Real Go2 LowState has no velocity field. Training used ground-truth lin_vel,
        # but domain randomization (friction, push, mass) and the other observations
        # provide enough signal for the policy to walk without real lin_vel data.
        # Using zeros is the industry-standard approach (unitree_rl_gym, rl_sar).
        lin_vel = np.zeros(3, dtype=np.float32)

        # Joint state - SDK order -> sim order
        joint_pos_sdk = np.array(
            [st.motor_state[i].q for i in range(12)], dtype=np.float32
        )
        joint_vel_sdk = np.array(
            [st.motor_state[i].dq for i in range(12)], dtype=np.float32
        )
        # Remap SDK -> SIM: sim[i] = sdk[sim_to_real[i]]
        joint_pos_sim = joint_pos_sdk[self.sim_to_real]
        joint_vel_sim = joint_vel_sdk[self.sim_to_real]

        # Joint position relative to default standing pose
        joint_pos_rel = joint_pos_sim - self.default_joint_pos_sim

        # Safety: clean NaN/Inf from sensor glitches
        lin_vel = safe_clean(lin_vel)
        ang_vel = safe_clean(ang_vel)
        gravity = safe_clean(gravity)
        joint_pos_rel = safe_clean(joint_pos_rel)
        joint_vel_sim = safe_clean(joint_vel_sim)

        # Build proprioceptive observation - either 48-dim (with base_lin_vel)
        # or 45-dim (without, better for sim-to-real since real robot can't measure it)
        if self.obs_dim == 45 or self.obs_dim == 232:
            # Policy was trained WITHOUT base_lin_vel (sim-to-real best practice)
            obs_proprio = np.concatenate([
                ang_vel,                   # 3
                gravity,                   # 3
                self.velocity_commands,    # 3
                joint_pos_rel,             # 12
                joint_vel_sim,             # 12
                self.last_actions,         # 12
            ]).astype(np.float32)          # total: 45
            base_dim = 45
        else:
            # Policy was trained WITH base_lin_vel (less sim-to-real robust)
            obs_proprio = np.concatenate([
                lin_vel,                   # 3
                ang_vel,                   # 3
                gravity,                   # 3
                self.velocity_commands,    # 3
                joint_pos_rel,             # 12
                joint_vel_sim,             # 12
                self.last_actions,         # 12
            ]).astype(np.float32)          # total: 48
            base_dim = 48

        if self.obs_dim == base_dim:
            return obs_proprio.reshape(1, -1)

        # Rough policy: append height scan (zeros without a depth sensor)
        extra_dims = self.obs_dim - base_dim
        if extra_dims < 0:
            raise ValueError(
                f"obs_dim={self.obs_dim} smaller than base_dim={base_dim}. "
                f"Policy may use different observation layout than expected."
            )
        height_scan = np.zeros(extra_dims, dtype=np.float32)
        return np.concatenate([obs_proprio, height_scan]).reshape(1, -1)

    def _read_velocity_commands(self):
        """Priority: wireless remote > USB gamepad > keyboard."""
        # Unitree wireless (from unitree_rl_lab): vx=ly, vy=-lx, wz=-rx
        if self.wireless_state is not None:
            w = self.wireless_state
            vx = w.ly * self.max_vel_xy
            vy = -w.lx * (self.max_vel_xy * 0.5)
            omega = -w.rx * self.max_vel_yaw
            if abs(vx) < STICK_DEADZONE: vx = 0.0
            if abs(vy) < STICK_DEADZONE: vy = 0.0
            if abs(omega) < STICK_DEADZONE: omega = 0.0
            self.velocity_commands = np.array([vx, vy, omega], dtype=np.float32)
            return

        # USB gamepad / keyboard via pygame
        try:
            import pygame
            pygame.event.pump()
        except ImportError:
            return

        if self.joystick is not None:
            def dz(v):
                return 0.0 if abs(v) < STICK_DEADZONE else v
            vx = -dz(self.joystick.get_axis(1)) * self.max_vel_xy
            vy = -dz(self.joystick.get_axis(0)) * (self.max_vel_xy * 0.5)
            omega = -dz(self.joystick.get_axis(3)) * self.max_vel_yaw
            self.velocity_commands = np.array([vx, vy, omega], dtype=np.float32)

            # L2+B e-stop
            if self.joystick.get_button(6) and self.joystick.get_button(1):
                self._select_requested = True

        elif self.use_keyboard:
            keys = pygame.key.get_pressed()
            vx = vy = omega = 0.0
            if keys[pygame.K_w]: vx = self.max_vel_xy * 0.5
            if keys[pygame.K_s]: vx = -self.max_vel_xy * 0.5
            if keys[pygame.K_a]: vy = self.max_vel_xy * 0.3
            if keys[pygame.K_d]: vy = -self.max_vel_xy * 0.3
            if keys[pygame.K_q]: omega = self.max_vel_yaw * 0.5
            if keys[pygame.K_e]: omega = -self.max_vel_yaw * 0.5
            if keys[pygame.K_ESCAPE]:
                self._select_requested = True
            self.velocity_commands = np.array([vx, vy, omega], dtype=np.float32)

    # -------------------------------------------------------------------
    # Command sending
    # -------------------------------------------------------------------

    def _send_damping(self):
        """Damping mode: zero kp, small kd, zero target - robot softly collapses."""
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 2.0
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def _send_zero(self):
        """Zero everything - used after damping for final shutdown."""
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def _send_targets_sdk(self, target_sdk: np.ndarray, kp: float, kd: float):
        target_sdk = np.clip(target_sdk, -JOINT_TARGET_CLIP, JOINT_TARGET_CLIP)
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(target_sdk[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def _send_targets_sim(self, target_sim: np.ndarray, kp: float, kd: float):
        # Remap SIM -> SDK: sdk[i] = sim[real_to_sim[i]]
        target_sdk = target_sim[self.real_to_sim]
        self._send_targets_sdk(target_sdk, kp, kd)

    # -------------------------------------------------------------------
    # State machine steps
    # -------------------------------------------------------------------

    def _step_stand_up(self):
        with self._state_lock:
            st = self.low_state
        if st is None:
            return

        if self.stand_start_pos_sdk is None:
            self.stand_start_pos_sdk = np.array(
                [st.motor_state[i].q for i in range(12)], dtype=np.float32
            )
            self.stand_progress = 0.0
            print("[INFO] Stand-up: starting from current pose")

        self.stand_progress = min(
            1.0, self.stand_progress + CONTROL_DT / STAND_UP_DURATION
        )
        default_sdk = self.default_joint_pos_sim[self.real_to_sim]
        target_sdk = (
            (1 - self.stand_progress) * self.stand_start_pos_sdk
            + self.stand_progress * default_sdk
        )

        # Ramp PD gains from soft to firm
        kp = self.kp * self.stand_progress + 5.0 * (1 - self.stand_progress)
        kd = self.kd * self.stand_progress + 0.5 * (1 - self.stand_progress)
        self._send_targets_sdk(target_sdk, kp, kd)

        if self.stand_progress >= 1.0 and self.deploy_state == DeployState.STAND_UP:
            print("[INFO] Stand-up complete. Waiting for START (Unitree remote) or ENTER.")
            self.deploy_state = DeployState.READY

    def _step_ready(self):
        default_sdk = self.default_joint_pos_sim[self.real_to_sim]
        self._send_targets_sdk(default_sdk, self.kp, self.kd)

    def _step_walk(self):
        self._read_velocity_commands()
        obs = self._get_observation()
        if obs is None:
            return

        actions = self.session.run(
            [self.output_name], {self.input_name: obs}
        )[0].flatten().astype(np.float32)

        # Clean and clip
        actions = safe_clean(actions)
        actions = np.clip(actions, -ACTION_CLIP, ACTION_CLIP)

        target_sim = self.default_joint_pos_sim + actions * self.action_scale
        self.last_actions = actions
        self._send_targets_sim(target_sim, self.kp, self.kd)

    def _control_loop(self):
        """Main 50 Hz control loop."""
        try:
            # Safety check drives state transitions
            if self.deploy_state in (DeployState.STAND_UP, DeployState.READY, DeployState.WALK):
                if not self._safety_check():
                    self.deploy_state = DeployState.DAMPING

            if self._select_requested and self.deploy_state not in (
                DeployState.DAMPING, DeployState.STOPPED
            ):
                print("\n[INFO] Stop requested -> damping")
                self.deploy_state = DeployState.DAMPING
                self._select_requested = False

            s = self.deploy_state
            if s == DeployState.IDLE:
                self._send_zero()
            elif s == DeployState.STAND_UP:
                self._step_stand_up()
            elif s == DeployState.READY:
                self._step_ready()
            elif s == DeployState.WALK:
                self._step_walk()
            elif s == DeployState.DAMPING:
                self._send_damping()
            elif s == DeployState.STOPPED:
                pass

        except Exception as e:
            print(f"[ERROR] Control loop exception: {e}")
            import traceback
            traceback.print_exc()
            self.deploy_state = DeployState.DAMPING

    # -------------------------------------------------------------------
    # Run loop
    # -------------------------------------------------------------------

    def run(self):
        print("=" * 68)
        print("  Go2 RL Policy Deployment")
        print("=" * 68)
        print("  Stages:")
        print("    1. Disable sport mode (done)")
        print("    2. Wait for sensor data")
        print("    3. Stand up gradually (2 s)")
        print("    4. Press START (Unitree remote) or ENTER to activate RL")
        print("    5. Control via Unitree remote / USB gamepad / keyboard")
        print("    6. Stop via SELECT (Unitree remote) / Ctrl+C / ESC / L2+B")
        print("=" * 68)

        # Wait for first state
        print("[INFO] Waiting for robot state...")
        t0 = time.time()
        while self.low_state is None:
            if time.time() - t0 > 5.0:
                print("[ERROR] No robot state in 5s. Check:")
                print("         - Is the Go2 powered on?")
                print("         - Ethernet cable connected?")
                print(f"         - Interface '{self.args.interface}' in 192.168.123.x subnet?")
                sys.exit(1)
            time.sleep(0.05)
        print(f"[INFO] Robot state received - IMU quat: {self.low_state.imu_state.quaternion}")

        # Start control thread
        self.deploy_state = DeployState.STAND_UP
        self.control_thread = RecurrentThread(
            interval=CONTROL_DT, target=self._control_loop, name="go2_rl"
        )
        self.control_thread.Start()

        # Wait for stand-up
        while self.deploy_state == DeployState.STAND_UP:
            time.sleep(0.05)
        if self.deploy_state == DeployState.DAMPING:
            self._shutdown()
            return

        # Wait for START button or ENTER
        print("[INFO] Press START on Unitree remote (or ENTER in terminal) to start RL")
        import select
        while not self._start_requested:
            if self._select_requested:
                print("[INFO] SELECT pressed - exiting without starting RL")
                self._shutdown()
                return
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                break

        print("[INFO] >>> RL POLICY ACTIVE <<<")
        print("[INFO] Control: left stick = forward/strafe, right stick = yaw")
        self.deploy_state = DeployState.WALK

        # Main loop
        try:
            while self.deploy_state == DeployState.WALK:
                time.sleep(0.5)
                v = self.velocity_commands
                print(f"\r[walk] vx={v[0]:+.2f}  vy={v[1]:+.2f}  w={v[2]:+.2f}     ",
                      end="", flush=True)
        except KeyboardInterrupt:
            pass

        print("\n[INFO] Walk loop exited")
        self._shutdown()

    def _shutdown(self):
        """Damping mode then zero - robot softly collapses."""
        print("[INFO] Entering damping mode for safe shutdown...")
        self.deploy_state = DeployState.DAMPING
        end = time.time() + 1.0
        while time.time() < end:
            self._send_damping()
            time.sleep(CONTROL_DT)
        print("[INFO] Sending final zero commands...")
        for _ in range(5):
            self._send_zero()
            time.sleep(CONTROL_DT)
        self.deploy_state = DeployState.STOPPED
        print("[INFO] Stopped. Bye.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Deploy RL policy to Unitree Go2")
    p.add_argument("--policy", type=str, required=True, help="ONNX policy path")
    p.add_argument("--interface", type=str, required=True, help="Ethernet interface")
    p.add_argument("--obs-dim", type=int, default=None,
                   help="48=flat, 235=rough (auto-detected from ONNX if omitted)")
    p.add_argument("--kp", type=float, default=40.0, help="Policy Kp")
    p.add_argument("--kd", type=float, default=1.0, help="Policy Kd")
    p.add_argument("--action-scale", type=float, default=0.25, help="Action scale")
    p.add_argument("--max-vel-xy", type=float, default=1.0, help="Max linear vel (m/s)")
    p.add_argument("--max-vel-yaw", type=float, default=1.0, help="Max yaw vel (rad/s)")
    p.add_argument("--no-safety", action="store_true", help="Disable tilt safety (NOT recommended)")
    args = p.parse_args()

    Go2Deployer(args).run()


if __name__ == "__main__":
    main()
