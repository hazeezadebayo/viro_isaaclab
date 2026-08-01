# Autonomous Mobile Robot (AMR / TurtleBot3) Navigation Architecture

## 1. Overview & Educational Purpose

The **AMR Mobile Robot Head** models a two-wheeled differential drive mobile robot (TurtleBot3 Burger). Mobile robots operate on planar $SE(2)$ non-holonomic manifolds $(x, y, \theta)$.

The primary educational goal is to train a neural policy to master **non-holonomic planar movement**, obstacle-free trajectory generation, and ROS2 Nav2 / `/cmd_vel` velocity control integration.

---

## 2. Simulation Environment Setup

- **Task Identifier**: `Isaac-AMR-Navigation-v0`
- **Physics Engine**: Nvidia PhysX (Fabric GPU acceleration)
- **Simulation Time-step ($\Delta t_{\text{sim}}$)**: $1 / 120 \, \text{s} \approx 8.33 \, \text{ms}$
- **Policy Control Decimation ($K$)**: 2 ($\Delta t_{\text{policy}} = 1 / 60 \, \text{s} \approx 16.67 \, \text{ms}$, 60 Hz control loop)
- **Episode Horizon**: 10.0 seconds (600 policy steps)

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

## 4. Mathematical Action Space ($\mathcal{A} \in [-1, 1]^2$)

The policy outputs unscaled twist actions $\mathbf{a}_t = [a_v, a_\omega] \in [-1, 1]^2$. The `DifferentialDriveAction` term maps this twist vector into wheel angular velocity targets $(w_L, w_R)$:

$$v = a_v \cdot v_{\text{max}} = a_v \cdot 0.5 \, \text{m/s}$$
$$\omega = a_\omega \cdot \omega_{\text{max}} = a_\omega \cdot 1.0 \, \text{rad/s}$$
$$w_L = \frac{v - \omega \frac{b}{2}}{r}, \qquad w_R = \frac{v + \omega \frac{b}{2}}{r}$$

---

## 5. Mathematical Observation Space ($\mathcal{S} \in \mathbb{R}^7$)

$$\mathbf{s}_t = \begin{bmatrix} v_x & \omega_z & \Delta x_{\text{goal}} & \Delta y_{\text{goal}} & \theta_{\text{goal}} & a_{v, t-1} & a_{\omega, t-1} \end{bmatrix}$$

1. **Base Velocities ($v_x, \omega_z$)**: Current forward speed and yaw turning rate.
2. **Goal Displacement Vector ($\Delta x_{\text{goal}}, \Delta y_{\text{goal}}$)**: $2\text{D}$ relative vector to 2D target position $(x_{\text{target}}, y_{\text{target}})$.
3. **Heading Angle Alignment ($\theta_{\text{goal}}$)**: Angle error between robot forward vector and target vector.

---

## 6. Reward Function Formulation ($\mathcal{R}$)

$$\mathcal{R}_t = 2.0 \, \exp\left(-\frac{d_{\text{goal}}^2}{0.25}\right) + 5.0 \, \mathbb{I}(d_{\text{goal}} < 0.05) - 0.01 \|\mathbf{a}\|^2$$

- **Target Proximity Reward**: Continuous exponential reward scoring proximity to target location $(1.5, 0.0)$.
- **Goal Reach Bonus**: $+5.0$ sparse bonus when robot arrives within $0.05 \, \text{m}$ of target.
- **Action Smoothness Penalty**: $-0.01 \|\mathbf{a}\|^2$ prevents aggressive wheel motor acceleration.

---

## 7. What the AMR Learns to Do

1. **Phase 1**: Learns differential turning dynamics ($w_L \neq w_R$).
2. **Phase 2**: Aligns heading vector $\theta_{\text{goal}}$ toward target coordinates.
3. **Phase 3**: Drives straight toward target and decelerates smoothly upon arrival.

---

## 8. How to Teach New Behaviors

### Example 1: LiDAR-Based Obstacle Avoidance
1. Enable `RayCaster` 2D LiDAR sensor in `AmrSceneCfg`.
2. Add 36-beam distance array to `ObservationsCfg`.
3. Add collision penalty term: $-10.0 \cdot \mathbb{I}(d_{\text{obstacle}} < 0.20 \, \text{m})$.

### Example 2: ROS2 Nav2 Integration
1. Launch ROS2 State Publisher: `ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=amr`.
2. Publish goal pose from Nav2 map server on topic `/cmd_vel`.
