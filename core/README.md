# Master Robotics Architecture Guide: Humanoid, ANYmal, AMR, & Cobot

An end-to-end, professor-level guide covering first principles, kinematic formulations, MDP terms, headless Docker video visualization, ROS2 integration, and training logs for **Humanoid Bipedal Motion Tracking**, **ANYmal Quadruped Locomotion**, **Autonomous Mobile Robots (AMR)**, and **Cobot 6-DOF Manipulator Arms**.

---

## Table of Contents
1. [First Principles & Kinematic Foundations](#1-first-principles--kinematic-foundations)
2. [Headless Docker Camera & Universal Video Recording](#2-headless-docker-camera--universal-video-recording)
3. [ROS2 Integration Infrastructure](#3-ros2-integration-infrastructure)
4. [Section A: Humanoid Bipedal Motion Imitation](#4-section-a-humanoid-bipedal-motion-imitation)
5. [Section B: ANYmal Quadruped Locomotion Architecture](#5-section-b-anymal-quadruped-locomotion-architecture)
6. [Section C: Autonomous Mobile Robot (AMR / TurtleBot3) Navigation](#6-section-c-autonomous-mobile-robot-amr--turtlebot3-navigation)
7. [Section D: Cobot 6-DOF Manipulator Arm Target Reaching](#7-section-d-cobot-6-dof-manipulator-arm-target-reaching)
8. [Comprehensive Step-by-Step Usage Guide](#8-comprehensive-step-by-step-usage-guide)
9. [Log Analysis & TensorBoard Metrics](#9-log-analysis--tensorboard-metrics)
10. [Troubleshooting & Failure Modes](#10-troubleshooting--failure-modes)

---

## 1. First Principles & Kinematic Foundations

### Comparison of Robot Kinematic Paradigms

```
   HUMANOID (Bipedal)       ANYMAL-C (Quadrupedal)    AMR (Differential Drive)   COBOT (6-DOF Arm)
   ------------------       ──────────────────────    ───────────────────────   ─────────────────
     O  (Head/Torso)             ┌─────────┐ (Base)        ┌───────────────┐     [LINK_6 / FLANGE]
    /|\                         / ┌───────┐ \              │   LIDAR/IMU   │          │  (EE Target)
   / | \                       /  │  COM  │  \             │ ┌───────────┐ │          ◯  (Wrist 1,2,3)
    / \                       /   └───────┘   \            │ │  BASE LINK│ │          │  (Elbow)
   /   \                     /                 \           └─┴─[W_L]─[W_R]─┴─┘          ◯  (Shoulder)
  FOOT FOOT               LF_FOOT RF_FOOT LH_FOOT RH_FOOT                                ───[BASE]───

   • High Center of Mass     • Low Center of Mass     • Fixed Planar SE(2)      • Fixed Base Kinematic Chain
   • 2 Points of Contact     • 4 Points of Contact    • 2 Continuous Wheels     • 6 Revolute Joints
   • Statically Unstable     • Support Polygon        • Non-Holonomic Twist     • Forward/Inverse Kinematics
   • Inverted Pendulum       • Trot / Pace Gaits      • Differential Drive      • End-Effector 3D Reaching
   • PD Torque Control       • ActuatorNet Control    • Speed (v, w) Control    • Joint Position Control
```

#### 1. Differential Drive Kinematics (AMR / TurtleBot3)
$$w_L = \frac{v - \omega \frac{b}{2}}{r}, \qquad w_R = \frac{v + \omega \frac{b}{2}}{r}$$
Where $r = 0.033 \, \text{m}$ (wheel radius) and $b = 0.160 \, \text{m}$ (wheel base separation).

#### 2. Serial Chain Manipulator Kinematics (Cobot 6-DOF Arm)
The Cobot arm operates as a serial open kinematic chain of 6 revolute joints $(\theta_1, \theta_2, \theta_3, \theta_4, \theta_5, \theta_6)$. The end-effector pose $\mathbf{T}_{ee} \in SE(3)$ is computed via forward kinematics:

$$\mathbf{T}_{ee}(\boldsymbol{\theta}) = \mathbf{A}_1(\theta_1) \mathbf{A}_2(\theta_2) \mathbf{A}_3(\theta_3) \mathbf{A}_4(\theta_4) \mathbf{A}_5(\theta_5) \mathbf{A}_6(\theta_6)$$

The policy controls joint target positions to minimize end-effector 3D Euclidean distance error to target position $\mathbf{p}_{\text{target}}$:

$$r_{\text{cobot}} = 2.0 \cdot \exp\left(-\frac{\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{target}}\|^2}{\sigma^2}\right) + 5.0 \cdot \mathbb{I}(\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{target}}\| < 0.05)$$

---

## 2. Headless Docker Camera & Universal Video Recording

### 1. Centralized Video Recorder Location
The universal video recording utility is located at:
`/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/utils/video_recorder.py`

### 2. Configurable Modes (`mode='video'`, `'ros2'`, `'both'`)
You can configure the video recorder mode when wrapping your environment:

```python
from core.utils.video_recorder import PeriodicVideoRecorderWrapper

# Default Mode: 'video' (Direct MP4 video file recording to disk)
env = PeriodicVideoRecorderWrapper(
    env,
    mode="video",               # Default mode: saves MP4 clips to /workspace/data/videos/
    video_folder="/workspace/data/videos",
    record_interval_s=3600.0,   # Record 1 clip every 1 hour
    video_length_s=60.0,         # 1 minute clip duration
)
```

---

## 3. ROS2 Integration Infrastructure

Bridge subscriptions (`custom_bridge.yaml`) map ROS2 network inputs into `core/source/`:
- `/cmd_vel` $\to$ AMR & ANYmal speed twist commands.
- `/cobot/target_pose` $\to$ Cobot end-effector 3D goal.
- `/cobot/joint_trajectory` $\to$ Cobot joint trajectory overrides.
- `/humanoid/motion_target` $\to$ Humanoid walking direction target.

---

## 4. Section A: Humanoid Bipedal Motion Imitation
- **Task**: `Isaac-Humanoid-Imitation-v0`
- **Source**: `core/source/humanoid/`

---

## 5. Section B: ANYmal Quadruped Locomotion Architecture
- **Task**: `Isaac-Anymal-C-v0`
- **Source**: `core/source/anymal/`

---

## 6. Section C: Autonomous Mobile Robot (AMR / TurtleBot3) Navigation

### AMR Architecture (`core/source/amr/`)
- **Task Name**: `Isaac-AMR-Navigation-v0`
- **Descriptions**: TurtleBot3 Burger URDF/Xacro, SDF, and DAE mesh geometry in `core/source/amr/descriptions/turtlebot3/`.
- **MDP Terms**: `DifferentialDriveAction` ([v, w] -> wheel speeds), base velocity & 2D goal displacement observations, target proximity & reach bonus rewards.

---

## 7. Section D: Cobot 6-DOF Manipulator Arm Target Reaching

### Cobot Architecture (`core/source/cobot/`)
- **Task Name**: `Isaac-Cobot-Reaching-v0`
- **Descriptions**: URDF Xacros (`cobot.xacro`, `cobot_arm.xacro`, `cobot_ros2_control.xacro`, `gripper.xacro`) and DAE meshes (`rdr/`, `ur5/`) in `core/source/cobot/descriptions/`.
- **MDP Terms**: `CobotArmActionCfg` (Joint position targets for `joint_1` to `joint_6`), end-effector 3D position error observations (`link_6`), exponential reach proximity & goal reach rewards.

---

## 8. Comprehensive Step-by-Step Usage Guide

### Phase 1: Launching RL Policy Training for Any Robot Head

```bash
# 1. Humanoid Motion Imitation Training
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-Imitation-v0 --headless

# 2. ANYmal Quadruped Locomotion Training
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Anymal-C-v0 --headless

# 3. AMR Mobile Robot Navigation Training
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-AMR-Navigation-v0 --headless

# 4. Cobot Manipulator Arm Target Reaching Training
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cobot-Reaching-v0 --headless
```

### Phase 2: Launching ROS2 State Publishers (/tf Tree)
```bash
# Humanoid
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=humanoid

# ANYmal
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=anymal

# AMR Mobile Robot
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=amr

# Cobot Arm
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=cobot
```

### Phase 3: Sending ROS2 Control Commands

#### A. Teleoperate AMR or ANYmal (/cmd_vel)
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.2}}"
```

#### B. Command Cobot 3D Target Pose (/cobot/target_pose)
```bash
ros2 topic pub /cobot/target_pose geometry_msgs/msg/PoseStamped "{pose: {position: {x: 0.4, y: 0.2, z: 0.5}}}"
```

---

## 9. Log Analysis & TensorBoard Metrics

### Console Log Sample (Cobot Reaching Session)

```text
------------------------------------------------------------------------------------
 Learning iteration 150 / 1000

               Mean action noise std: 0.52
               Mean reward / step: 4.1200
               Mean episode length: 120.00
               Mean episode reward: 494.40

              - Rewards/target_proximity: 1.9120
              - Rewards/reach_bonus: 2.2500
              - Rewards/action_l2: -0.0080
              - Loss/surrogate: 0.0062
              - Loss/value: 0.0810

 Computation time: 0.62s (FPS: 148200)
------------------------------------------------------------------------------------
```

---

## 10. Troubleshooting & Failure Modes

1. **Cobot Arm Vibrating Wildly**:
   - *Cause*: High stiffness/damping ratio in PD actuator configuration.
   - *Fix*: Verify stiffness `800.0` and damping `40.0` in `core/source/cobot/tasks/cobot_env_cfg.py`.

2. **AMR Spinning in Circles**:
   - *Cause*: Differential wheel base separation mismatch.
   - *Fix*: Verify wheel radius `0.033` and wheel base `0.160` in `core/source/amr/mdp/actions.py`.
