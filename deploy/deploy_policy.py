"""
Go2 RL Policy Deployment Script
Deploys an ONNX policy trained in Isaac Lab to the real Unitree Go2 EDU robot.

Usage:
    python deploy_policy.py --policy policy.onnx --interface eth0

Requirements:
    pip install unitree_sdk2py onnxruntime numpy pygame
"""

import argparse
import time
import sys
import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

# ============================================================
# CRITICAL: Joint Mapping Between Isaac Lab and Unitree SDK
# ============================================================
# Isaac Lab order (URDF alphabetical):
#   0: FL_hip,  1: FL_thigh,  2: FL_calf
#   3: FR_hip,  4: FR_thigh,  5: FR_calf
#   6: RL_hip,  7: RL_thigh,  8: RL_calf
#   9: RR_hip, 10: RR_thigh, 11: RR_calf
#
# Unitree SDK order:
#   0: FR_hip,  1: FR_thigh,  2: FR_calf
#   3: FL_hip,  4: FL_thigh,  5: FL_calf
#   6: RR_hip,  7: RR_thigh,  8: RR_calf
#   9: RL_hip, 10: RL_thigh, 11: RL_calf

# Maps Isaac Lab joint index -> SDK joint index
SIM_TO_REAL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Maps SDK joint index -> Isaac Lab joint index
REAL_TO_SIM = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Default standing joint angles (from UNITREE_GO2_CFG, in Isaac Lab order)
DEFAULT_JOINT_POS_SIM = np.array([
    0.1,  0.8, -1.5,   # FL: hip, thigh, calf
   -0.1,  0.8, -1.5,   # FR: hip, thigh, calf
    0.1,  1.0, -1.5,   # RL: hip, thigh, calf
   -0.1,  1.0, -1.5,   # RR: hip, thigh, calf
], dtype=np.float32)

# PD gains for real robot deployment
# Training simulation used DCMotor model with Kp=25, Kd=0.5
# unitree_rl_lab recommends Kp=40, Kd=1.0 for deployment
# Official stand example uses Kp=60, Kd=5 (too stiff for RL)
# Start with training-matched values, increase if robot feels loose
KP = 40.0
KD = 1.0

# Special SDK stop values (from unitree_legged_const.py)
PosStopF = 2.146e9
VelStopF = 16000.0

# Action scale (from rough_env_cfg.py)
ACTION_SCALE = 0.25

# Control frequency (must match training: dt=0.005 * decimation=4 = 50Hz)
CONTROL_DT = 0.02  # 50 Hz


class Go2Deployer:
    def __init__(self, policy_path, interface):
        # Load ONNX policy
        self.session = ort.InferenceSession(policy_path)
        print(f"[INFO] Loaded ONNX policy: {policy_path}")
        print(f"[INFO] Input: {self.session.get_inputs()[0].name}, shape: {self.session.get_inputs()[0].shape}")
        print(f"[INFO] Output: {self.session.get_outputs()[0].name}, shape: {self.session.get_outputs()[0].shape}")

        # Initialize SDK
        ChannelFactoryInitialize(0, interface)

        # Disable sport mode (CRITICAL — must be done before low-level control)
        print("[INFO] Disabling sport mode...")
        sc = SportClient()
        sc.SetTimeout(5.0)
        sc.Init()
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result['name']:
            sc.StandDown()
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)
        print("[INFO] Sport mode disabled.")

        # Publishers and subscribers
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self.state_callback, 10)

        # CRC calculator
        self.crc = CRC()

        # State
        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.velocity_commands = np.zeros(3, dtype=np.float32)  # vx, vy, omega_z

        # Control state machine
        self.mode = "idle"  # idle -> stand -> walk
        self.stand_progress = 0.0

        # Initialize low_cmd
        self._init_low_cmd()

        # Try importing gamepad
        self.joystick = None
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"[INFO] Gamepad found: {self.joystick.get_name()}")
            else:
                print("[INFO] No gamepad found. Use keyboard: WASD=move, QE=turn, SPACE=start, ESC=stop")
        except ImportError:
            print("[INFO] pygame not installed. Use keyboard: WASD=move, QE=turn, SPACE=start, ESC=stop")

    def _init_low_cmd(self):
        """Initialize low command with correct headers (matches official example)."""
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = PosStopF   # safe stop position
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF   # safe stop velocity
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def state_callback(self, msg: LowState_):
        """Receive robot state from SDK."""
        self.low_state = msg

    def get_observation(self):
        """
        Build 48-dim observation vector matching Isaac Lab training:
        [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
         velocity_commands(3), joint_pos_rel(12), joint_vel(12), last_actions(12)]
        """
        if self.low_state is None:
            return None

        state = self.low_state
        imu = state.imu_state

        # Base angular velocity from gyroscope (in body frame)
        ang_vel = np.array([imu.gyroscope[0], imu.gyroscope[1], imu.gyroscope[2]], dtype=np.float32)

        # Projected gravity from quaternion
        quat = np.array([imu.quaternion[0], imu.quaternion[1],
                         imu.quaternion[2], imu.quaternion[3]], dtype=np.float32)  # w,x,y,z
        gravity = self._quat_rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32))

        # Base linear velocity estimate (from accelerometer integration or SDK estimate)
        # Note: Real robot doesn't have ground-truth lin_vel. We use velocity estimate.
        # For basic walking, using [0,0,0] or IMU-derived estimate works.
        lin_vel = np.array([state.velocity[0], state.velocity[1], state.velocity[2]], dtype=np.float32) \
            if hasattr(state, 'velocity') else np.zeros(3, dtype=np.float32)

        # Joint positions and velocities (SDK order -> Isaac Lab order)
        joint_pos_sdk = np.array([state.motor_state[i].q for i in range(12)], dtype=np.float32)
        joint_vel_sdk = np.array([state.motor_state[i].dq for i in range(12)], dtype=np.float32)

        # Remap SDK order to Isaac Lab order
        joint_pos_sim = joint_pos_sdk[REAL_TO_SIM]
        joint_vel_sim = joint_vel_sdk[REAL_TO_SIM]

        # Joint positions relative to default stance
        joint_pos_rel = joint_pos_sim - DEFAULT_JOINT_POS_SIM

        # Construct observation (48 dims, same order as training)
        obs = np.concatenate([
            lin_vel,                    # 3: base linear velocity
            ang_vel,                    # 3: base angular velocity
            gravity,                    # 3: projected gravity
            self.velocity_commands,     # 3: velocity commands [vx, vy, omega_z]
            joint_pos_rel,              # 12: joint positions relative to default
            joint_vel_sim,              # 12: joint velocities
            self.last_actions,          # 12: last actions
        ], dtype=np.float32)

        return obs.reshape(1, -1)

    def _quat_rotate_inverse(self, quat, vec):
        """Rotate vector by inverse of quaternion. quat=[w,x,y,z]"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Rotation matrix from quaternion
        r00 = 1 - 2*(y*y + z*z); r01 = 2*(x*y + w*z);   r02 = 2*(x*z - w*y)
        r10 = 2*(x*y - w*z);     r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z + w*x)
        r20 = 2*(x*z + w*y);     r21 = 2*(y*z - w*x);     r22 = 1 - 2*(x*x + y*y)
        # Inverse rotation = transpose
        out = np.array([
            r00*vec[0] + r10*vec[1] + r20*vec[2],
            r01*vec[0] + r11*vec[1] + r21*vec[2],
            r02*vec[0] + r12*vec[1] + r22*vec[2],
        ], dtype=np.float32)
        return out

    def read_gamepad(self):
        """Read gamepad inputs and update velocity commands."""
        if self.joystick is not None:
            import pygame
            pygame.event.pump()
            # Left stick: forward/lateral, Right stick: yaw
            vx = -self.joystick.get_axis(1) * 1.0    # forward (inverted Y axis)
            vy = -self.joystick.get_axis(0) * 0.5     # lateral
            omega = -self.joystick.get_axis(3) * 1.0   # yaw rate
            self.velocity_commands = np.array([vx, vy, omega], dtype=np.float32)

            # Dead zone
            for i in range(3):
                if abs(self.velocity_commands[i]) < 0.1:
                    self.velocity_commands[i] = 0.0

    def send_joint_commands(self, target_positions_sim):
        """
        Send joint position commands to robot.
        target_positions_sim: 12 joint targets in Isaac Lab order
        """
        # Remap Isaac Lab order -> SDK order
        target_positions_sdk = target_positions_sim[SIM_TO_REAL]

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(target_positions_sdk[i])
            self.low_cmd.motor_cmd[i].kp = KP
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = KD
            self.low_cmd.motor_cmd[i].tau = 0.0

        # CRC is REQUIRED
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

    def stand_up(self):
        """Gradually move to default standing pose over 2 seconds."""
        if self.low_state is None:
            return False

        self.stand_progress += CONTROL_DT / 2.0  # 2 second ramp
        self.stand_progress = min(self.stand_progress, 1.0)

        # Get current joint positions (SDK order)
        current_pos_sdk = np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)

        # Default positions in SDK order
        default_pos_sdk = DEFAULT_JOINT_POS_SIM[SIM_TO_REAL]

        # Interpolate
        target = current_pos_sdk * (1 - self.stand_progress) + default_pos_sdk * self.stand_progress

        # Send directly in SDK order (no remapping needed here)
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(target[i])
            self.low_cmd.motor_cmd[i].kp = KP
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = KD
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.cmd_pub.Write(self.low_cmd)

        return self.stand_progress >= 1.0

    def control_step(self):
        """Main control loop called at 50 Hz."""
        if self.low_state is None:
            return

        if self.mode == "stand":
            done = self.stand_up()
            if done:
                print("[INFO] Standing complete. Press START/SPACE to activate RL policy.")
                self.mode = "ready"

        elif self.mode == "walk":
            # Read gamepad
            self.read_gamepad()

            # Get observation
            obs = self.get_observation()
            if obs is None:
                return

            # Run ONNX inference
            actions = self.session.run(None, {"obs": obs})[0].flatten()

            # Scale actions and add to default joint positions
            target_positions = DEFAULT_JOINT_POS_SIM + actions * ACTION_SCALE

            # Save for next observation
            self.last_actions = actions.astype(np.float32)

            # Send to robot
            self.send_joint_commands(target_positions)

    def run(self):
        """Main entry point."""
        print("=" * 60)
        print("  Go2 RL Policy Deployment")
        print("=" * 60)
        print("Controls:")
        print("  1. Robot will stand up first")
        print("  2. Then RL policy takes over")
        print("  3. Use gamepad left stick to control direction")
        print("  4. Ctrl+C to stop")
        print("=" * 60)

        # Wait for first state message
        print("[INFO] Waiting for robot state...")
        while self.low_state is None:
            time.sleep(0.1)
        print("[INFO] Robot state received!")

        # Phase 1: Stand up
        print("[INFO] Standing up...")
        self.mode = "stand"

        # Start control thread at 50 Hz
        control_thread = RecurrentThread(interval=CONTROL_DT, target=self.control_step, name="rl_control")
        control_thread.Start()

        # Wait for stand to complete
        while self.mode == "stand":
            time.sleep(0.1)

        # Phase 2: Activate RL policy
        input("[INFO] Press ENTER to activate RL policy (or Ctrl+C to stop)...")
        print("[INFO] RL policy ACTIVE! Use gamepad to control.")
        self.mode = "walk"

        # Keep running
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            self.mode = "idle"
            time.sleep(0.5)
            print("[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(description="Deploy RL policy to Go2")
    parser.add_argument("--policy", type=str, required=True, help="Path to ONNX policy file")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface connected to Go2")
    args = parser.parse_args()

    deployer = Go2Deployer(args.policy, args.interface)
    deployer.run()


if __name__ == "__main__":
    main()
