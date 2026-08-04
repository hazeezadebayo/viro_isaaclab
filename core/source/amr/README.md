# Autonomous Mobile Robot (AMR / TurtleBot3) Navigation Architecture

## 1. Overview & Educational Purpose

The **AMR Mobile Robot Head** models a two-wheeled differential drive mobile robot (TurtleBot3 Burger). Mobile robots operate on planar $SE(2)$ non-holonomic manifolds $(x, y, \theta)$.

The primary educational goal is to train neural policies to master **non-holonomic planar movement**, goal-directed navigation, and vision-based **traversability** (staying on a drivable path using a camera).

The head is split into three learning problems:

1. **Locomotion** — a low-level policy maps a target twist $(v, \omega)$ into wheel angular velocities.
2. **Navigation** — a high-level policy maps proprioceptive + goal observations into a twist command consumed by the frozen, pre-trained locomotion policy.
3. **Traversability** — a standalone, camera-based policy maps a down-sampled *occupancy mask* of a white figure-8 path (plus the goal in body frame) into differential-drive twists.

---

## 2. Simulation Environment Setup

Registered task identifiers (see `core/source/register_tasks.py`):

| Task ID | Description |
| :--- | :--- |
| `Isaac-AMR-Locomotion-v0` | Low-level velocity tracking over mixed flat/rough terrain with a terrain curriculum |
| `Isaac-AMR-Navigation-v0` | Hierarchical navigation behind obstacle cones (uses the frozen locomotion policy) |
| `Isaac-AMR-Traversability-v0` | Camera-based goal reaching on a white figure-8 path over a black ground plane |

- **Physics Engine**: Nvidia PhysX (Fabric GPU acceleration)
- **Simulation Time-step ($\Delta t_{\text{sim}}$)**: $1 / 100 \, \text{s} = 10 \, \text{ms}$
- **Locomotion Policy Decimation ($K$)**: 10 ($\Delta t_{\text{policy}} = 0.1 \, \text{s}$, 10 Hz control loop)
- **Navigation Policy Decimation ($K$)**: 40 ($\Delta t_{\text{policy}} = 0.4 \, \text{s}$, 2.5 Hz control loop; low-level locomotion runs at its own 10 Hz inside)
- **Traversability Policy Decimation ($K$)**: 10 ($\Delta t_{\text{policy}} = 0.1 \, \text{s}$; the camera renders every policy step)
- **Episode Horizon**: 10.0 s (locomotion & traversability), 8.0 s (navigation, matches command resampling)

> **Traversability must be trained/played with `--enable_cameras`** so the offscreen RTX camera pipeline is active.

---

## 3. Robot Kinematics & Joint Breakdown (2 Actuated Wheel Joints)

```
                            ┌────────────────┐
                            │  TURTLEBOT3    │
                            │   CHASSIS      │
                            └───────┬────────┘
                   ┌────────────────┴────────────────┐
          Left Wheel Joint                  Right Wheel Joint
          (wheel_left_joint)                (wheel_right_joint)
                   │                                 │
            [ LEFT TIRE ]                     [ RIGHT TIRE ]
```

### Kinematic Specifications

- **Wheel Radius ($r$)**: $0.033 \, \text{m}$ ($33 \, \text{mm}$)
- **Wheel Separation Base ($b$)**: $0.160 \, \text{m}$ ($160 \, \text{mm}$)
- **Non-Holonomic Constraint**: No instantaneous lateral sliding velocity along local $y$-axis ($\dot{y}_{\text{body}} = 0$).

---

## 4. Mathematical Action Space

### 4a. Differential-Drive Action ($\mathcal{A} \in [-1, 1]^2$)

The policy outputs unscaled twist actions $\mathbf{a}_t = [a_v, a_\omega] \in [-1, 1]^2$. The `DifferentialDriveAction` term maps this twist vector into wheel angular velocity targets $(w_L, w_R)$:

$$v = a_v \cdot v_{\text{max}} = a_v \cdot 0.22 \, \text{m/s}$$
$$\omega = a_\omega \cdot \omega_{\text{max}} = a_\omega \cdot 2.84 \, \text{rad/s}$$
$$w_L = \frac{v - \omega \frac{b}{2}}{r}, \qquad w_R = \frac{v + \omega \frac{b}{2}}{r}$$

Wheel velocity targets are clamped to the physical wheel limit $w_{\text{max}} = 11.0 \, \text{rad/s}$ (matches the URDF joint limit and TurtleBot3 Burger's rated specs).

The action term clips the commanded twist to the linear/angular velocity limits. Wheel targets are applied as **joint velocity targets** through `IdealPDActuatorCfg` (stiffness 0, damping 10). This action is used by **locomotion and traversability** directly.

### 4b. Navigation Action ($\mathcal{A} \in [-1, 1]^2$)

The high-level navigation policy produces the same twist command $[a_v, a_\omega]$, but it is fed through `PreTrainedPolicyAction`, which runs the **frozen low-level locomotion policy** (TorchScript, `low_level_decimation = 4`) at 25 Hz and applies the resulting wheel velocity targets to the robot.

---

## 5. Mathematical Observation Space

### 5a. Locomotion Policy ($\mathcal{S} \in \mathbb{R}^{3 + 2 + 2}$)

$$\mathbf{s}_t = \begin{bmatrix} v_x & \omega_z & \theta_{\text{roll}} & \theta_{\text{pitch}} & v_{\text{cmd}} & \omega_{\text{cmd}} & a_{v, t-1} & a_{\omega, t-1} \end{bmatrix}$$

1. **Base Velocities ($v_x, \omega_z$)**: Current forward speed and yaw turning rate.
2. **Projected Gravity ($\theta_{\text{roll}}, \theta_{\text{pitch}}$)**: Roll/pitch alignment of the base with the world gravity vector (used for rough-terrain robustness).
3. **Velocity Command ($v_{\text{cmd}}, \omega_{\text{cmd}}$)**: Target twist sampled by the `UniformVelocityCommandCfg` command term.
4. **Last Action ($a_{v, t-1}, a_{\omega, t-1}$)**: Previous policy output for smoothness conditioning.

### 5b. Navigation Policy ($\mathcal{S} \in \mathbb{R}^{2 + 3 + 2 + 3}$)

$$\mathbf{s}_t = \begin{bmatrix} v_x & \omega_z & g_x & g_y & \theta_g & c_x & c_y & c_z \end{bmatrix}$$

1. **Base Velocities ($v_x, \omega_z$)**: Current forward speed and yaw turning rate.
2. **Goal Pose ($g_x, g_y, \theta_g$)**: Goal position/heading sampled behind the obstacle by `ObstacleBlockedPoseCommand`.
3. **Cone Position ($c_x, c_y, c_z$)**: World position of the obstacle cone the robot must navigate behind.

### 5c. Traversability Policy ($\mathcal{S} \in \mathbb{R}^{3 + 3 + 192 + 2}$)

$$\mathbf{s}_t = \begin{bmatrix} v_x & \omega_z & \mathbf{g}_b & \mathbf{m}_{16 \times 12} & a_{v, t-1} & a_{\omega, t-1} \end{bmatrix}$$

1. **Base Velocities ($v_x, \omega_z$)**: Current forward speed and yaw turning rate.
2. **Goal in Body Frame ($\mathbf{g}_b \in \mathbb{R}^3$)**: The on-path goal rotated into the robot's base frame (`goal_in_base`), so the policy knows which direction to drive.
3. **Occupancy Mask ($\mathbf{m} \in [0,1]^{192}$)**: The RGB camera frame (64x48, forward-mounted, ~15° pitch) is **grayscaled, thresholded** (white path = 1, black ground = 0) and **area-pooled** down to a 16x12 grid (`camera_occupancy_mask`). Each cell encodes the fraction of white path visible in that image region. This hand-written mask is the entire "perception" of the task — no CNN required.
4. **Last Action ($a_{v, t-1}, a_{\omega, t-1}$)**: Previous policy output for smoothness conditioning.

The live camera stream can be inspected with:

```bash
python3 /workspace/core/ros2_ws/image_listener.py --topic /amr/camera/rgb
```

---

## 6. Reward Function Formulation

### 6a. Locomotion ($\mathcal{R}$)

$$\mathcal{R}_t = 1.5 \exp\left(-\frac{(v_{\text{cmd}} - v_x)^2}{2 \cdot 0.3^2}\right) + 1.0 \exp\left(-\frac{(\omega_{\text{cmd}} - \omega_z)^2}{2 \cdot 0.5^2}\right) - 0.01 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 - 0.5 v_z^2 - 0.002 (\omega_x^2 + \omega_y^2)$$

- **Forward/Turn Tracking Rewards**: Exponential rewards on matching commanded linear/angular velocity.
- **Action Smoothness Penalty**: $-0.01$ per action-rate unit prevents aggressive wheel motor acceleration.
- **Regularization Penalties**: Discourage unwanted vertical bouncing ($-0.5 v_z^2$) and pitching/rolling rotation ($-0.002 (\omega_x^2 + \omega_y^2)$).

### 6b. Navigation ($\mathcal{R}$)

$$\mathcal{R}_t = 5.0 \tanh\left(1 - \frac{d_{\text{goal}}}{2.0}\right) + 1.0 \tanh\left(1 - \frac{d_{\text{goal}}}{0.2}\right) - 5.0 \, \mathbb{I}(d_{\text{cone}} < 0.6) + 0.5 \, \text{progress} - 200 \, \mathbb{I}(\text{terminated})$$

- **Goal Proximity Rewards**: $\tanh$ shaping over distance to the goal pose, coarse and fine-grained.
- **Cone Proximity Penalty**: Hard penalty inside the collision radius (0.6 m).
- **Goal Progress**: Dense reward proportional to how much of the velocity is aimed at the goal.
- **Termination Penalty**: Sparse $-200$ for episodes that end early without reaching the goal.

### 6c. Traversability ($\mathcal{R}$)

$$\mathcal{R}_t = 1.5 \, \underbrace{\Delta d_{\text{goal}}}_{\text{goal progress}} + 10 \, \mathbb{I}(d_{\text{goal}} < 0.15) + \underbrace{\exp\left(-\frac{d_{\text{path}}^2}{2 \cdot 0.12^2}\right)}_{\text{on-path}} - \underbrace{\max(0, d_{\text{path}} - 0.16)}_{\text{off-path penalty}} - 0.01 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$$

- **Goal Progress ($\Delta d_{\text{goal}}$)**: Reduction in distance to the on-path goal between steps (dense shaping).
- **Goal Reached**: Sparse $+10$ when within $0.15$ m of the goal; the goal then resamples ahead.
- **On-Path Reward**: Gaussian bump that is 1.0 on the centerline and decays with distance off it.
- **Off-Path Penalty**: Linear penalty once the robot leaves the path strip (width $0.32$ m, half = $0.16$ m).
- **Action Smoothness**: $-0.01$ action-rate penalty for smooth driving.

> **Teaching note**: the rewards use ground-truth distance to the path centerline, while the policy
> *only* sees the camera mask. This makes the target easy to inspect: a well-trained policy keeps
> the mask mostly full (white) while steering toward the goal.

---

## 7. What the AMR Learns to Do

1. **Phase 1 (Locomotion)**: Track commanded linear/angular velocities over flat and rough terrain, rejecting pushes and terrain noise.
2. **Phase 2 (Navigation)**: Combine goal-approach and obstacle-cone awareness to plan a trajectory that ends behind the obstacle facing the goal direction.
3. **Phase 3 (Traversability)**: Read the white-path occupancy mask, keep the path under the robot, and drive along the figure-8 toward an on-path goal.

---

## 8. How to Teach New Behaviors

### Example 1: LiDAR-Based Obstacle Avoidance
1. Enable `RayCaster` 2D LiDAR sensor in `NavigationSceneCfg`.
2. Add 36-beam distance array to `ObservationsCfg`.
3. Add collision penalty term: $-10.0 \cdot \mathbb{I}(d_{\text{obstacle}} < 0.20 \, \text{m})$.

### Example 2: ConvNet End-to-End Vision
1. Replace the `path_mask` term with `mdp.image` (raw RGB) and enlarge the camera to 128x128.
2. Change the actor/critic to a small CNN encoder in `agents/rsl_rl_ppo_cfg.py`.
3. Compare convergence against the hand-written mask version — a great lesson on feature engineering.

### Example 3: ROS2 Nav2 Integration
1. Launch ROS2 State Publisher: `ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=amr`.
2. Publish goal pose from Nav2 map server on topic `/cmd_vel`.

---

## 9. Repository Layout

```
core/source/amr/
├── descriptions/
│   ├── amr.py                          # AMR_BURGER_CFG (ArticulationCfg)
│   └── turtlebot3/model.urdf           # Self-contained URDF (mass/inertia/collision)
├── mdp/
│   ├── actions.py                      # DifferentialDriveAction (twist -> wheel vel)
│   ├── locomotion/                     # commands, curriculums, myTerrainCfg
│   ├── navigation/                     # pre_trained_policy_action, policy_path,
│   │                                   # commands, rewards, observations, events, curriculums
│   └── traversability/                 # myPathTerrainCfg (figure-8 path), commands
│                                       # (path goals), observations (occupancy mask),
│                                       # rewards (on-path / goal progress), events
├── tasks/
│   ├── amr_locomotion_env_cfg.py
│   ├── amr_navigation_env_cfg.py
│   └── amr_traversability_env_cfg.py
├── agents/rsl_rl_ppo_cfg.py            # 3 PPO runner configs (locomotion / navigation / traversability)
└── README.md
```
