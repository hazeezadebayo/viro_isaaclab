# Humanoid Bipedal Motion Tracking & Imitation Architecture

## 1. Overview & Educational Purpose

The **Humanoid Biped Head** simulates a 17-DOF bipedal robot executing **Deep Reinforcement Learning Motion Imitation**. Bipedal locomotion is inherently statically unstable due to its high Center of Mass ($\text{CoM}$) and narrow 2-point foot contact support polygon. 

The primary goal of this environment is to train a neural network policy $\pi_\theta(a|s)$ to track dynamic human reference motion capture data $q_{\text{ref}}(t)$ extracted from real video recordings or biomechanical models.

---

## 2. Simulation Environment Setup

- **Task Identifier**: `Isaac-Humanoid-Imitation-v0`
- **Physics Engine**: Nvidia PhysX (Fabric GPU acceleration)
- **Simulation Time-step ($\Delta t_{\text{sim}}$)**: $1 / 120 \, \text{s} \approx 8.33 \, \text{ms}$
- **Policy Control Decimation ($K$)**: 2 ($\Delta t_{\text{policy}} = 1 / 60 \, \text{s} \approx 16.67 \, \text{ms}$, 60 Hz control loop)
- **Episode Horizon**: 10.0 seconds (600 policy steps)

---

## 3. Robot Kinematics & Joint Breakdown (17 Actuated DOF)

```
                       [ HEAD / TORSO ]
                              │
                     ┌────────┴────────┐
               Left Shoulder     Right Shoulder
                     │                 │
                Left Elbow        Right Elbow
                     │                 │
             [ LEFT HAND ]     [ RIGHT HAND ]
                              │
                       [ PELVIS / WAIST ]
                     ┌────────┴────────┐
                 Left Hip          Right Hip
                     │                 │
                 Left Knee         Right Knee
                     │                 │
                 Left Ankle        Right Ankle
                     │                 │
             [ LEFT FOOT ]     [ RIGHT FOOT ]
```

### Joint Registry Table

| Joint Group | Joint Name | Range of Motion | Control Type | Stiffness ($K_p$) | Damping ($K_d$) |
|-------------|------------|-----------------|--------------|------------------|-----------------|
| **Torso / Waist** | `pelvis`, `left_waist`, `right_waist` | $[-0.5, 0.5] \, \text{rad}$ | PD Effort | 800.0 | 40.0 |
| **Left Leg** | `left_thigh_0`, `left_thigh_1`, `left_thigh_2`, `left_shin`, `left_foot` | $[-1.5, 1.5] \, \text{rad}$ | PD Effort | 800.0 | 40.0 |
| **Right Leg** | `right_thigh_0`, `right_thigh_1`, `right_thigh_2`, `right_shin`, `right_foot` | $[-1.5, 1.5] \, \text{rad}$ | PD Effort | 800.0 | 40.0 |
| **Left Arm** | `left_upper_arm`, `left_lower_arm` | $[-1.2, 1.2] \, \text{rad}$ | PD Effort | 400.0 | 20.0 |
| **Right Arm** | `right_upper_arm`, `right_lower_arm` | $[-1.2, 1.2] \, \text{rad}$ | PD Effort | 400.0 | 20.0 |

---

## 4. Mathematical Observation Space ($\mathcal{S} \in \mathbb{R}^{78}$)

The observation vector fed into the PPO actor policy at time step $t$ consists of:

$$\mathbf{s}_t = \begin{bmatrix} \hat{\mathbf{q}}_t & \dot{\mathbf{q}}_t & \mathbf{q}_{\text{ref}}(t) - \mathbf{q}_t & \dot{\mathbf{q}}_{\text{ref}}(t) - \dot{\mathbf{q}}_t & \mathbf{v}_{\text{base}} & \boldsymbol{\omega}_{\text{base}} & \mathbf{g}_{\text{proj}} & \mathbf{a}_{t-1} \end{bmatrix}$$

1. **Normalized Joint Positions ($\hat{\mathbf{q}}_t \in \mathbb{R}^{17}$)**: Joint angles scaled relative to lower/upper physical bounds $[-1, 1]$.
2. **Normalized Joint Velocities ($\dot{\mathbf{q}}_t \in \mathbb{R}^{17}$)**: Joint angular velocities scaled by $0.1$.
3. **Reference Joint Error ($\mathbf{q}_{\text{ref}}(t) - \mathbf{q}_t \in \mathbb{R}^{17}$)**: Displacement from video-retargeted reference motion.
4. **Base Body Velocities ($\mathbf{v}_{\text{base}} \in \mathbb{R}^3, \boldsymbol{\omega}_{\text{base}} \in \mathbb{R}^3$)**: Linear and angular velocity of the torso.
5. **Projected Gravity Vector ($\mathbf{g}_{\text{proj}} \in \mathbb{R}^3$)**: $\mathbf{R}_{\text{base}}^T \cdot [0, 0, -1]^T$ indicating vertical posture tilt.

---

## 5. Mathematical Action Space ($\mathcal{A} \in [-1, 1]^{17}$)

The policy outputs unscaled actions $\mathbf{a}_t \in [-1, 1]^{17}$. The `PDTrackingAction` class maps these actions directly into target joint motor torques $\boldsymbol{\tau}_t$:

$$\mathbf{q}_{\text{target}} = \mathbf{q}_{\text{ref}}(t) + \mathbf{a}_t \cdot \mathbf{s}_{\text{scale}}$$
$$\boldsymbol{\tau}_t = \mathbf{K}_p \left( \mathbf{q}_{\text{target}} - \mathbf{q}_t \right) - \mathbf{K}_d \dot{\mathbf{q}}_t$$

Where $\mathbf{s}_{\text{scale}} = 0.50$, $\mathbf{K}_p = 800.0$, and $\mathbf{K}_d = 40.0$.

---

## 6. Reward Function Formulation ($\mathcal{R}$)

The total reward per step is a weighted sum of tracking, posture, and energy terms:

$$\mathcal{R}_t = w_{\text{pose}} r_{\text{pose}} + w_{\text{vel}} r_{\text{vel}} + w_{\text{alive}} r_{\text{alive}} + w_{\text{upright}} r_{\text{upright}} - w_{\text{torque}} \|\boldsymbol{\tau}\|^2 - w_{\text{action}} \|\mathbf{a}\|^2$$

| Reward Term | Math Expression | Weight ($w$) | Educational Purpose |
|-------------|-----------------|--------------|---------------------|
| **Pose Tracking ($r_{\text{pose}}$)** | $\exp\left(-\frac{\|\mathbf{q} - \mathbf{q}_{\text{ref}}\|^2}{0.25}\right)$ | $+2.0$ | Encourages accurate imitation of reference gait |
| **Velocity Tracking ($r_{\text{vel}}$)** | $\exp\left(-\frac{\|\dot{\mathbf{q}} - \dot{\mathbf{q}}_{\text{ref}}\|^2}{1.0}\right)$ | $+1.0$ | Ensures velocity phase alignment |
| **Survival ($r_{\text{alive}}$)** | $\mathbb{I}(\text{height} > 0.8 \, \text{m})$ | $+2.0$ | Prevents the robot from falling down |
| **Upright Posture ($r_{\text{upright}}$)** | $(\mathbf{g}_{\text{proj}} \cdot [0,0,-1]) > 0.90$ | $+0.5$ | Keeps torso upright |
| **Torque Penalty** | $-\|\boldsymbol{\tau}\|^2$ | $-0.0001$ | Reduces motor overheating & energy waste |
| **Action Smoothness** | $-\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ | $-0.005$ | Eliminates high-frequency joint jitter |

---

## 7. What the Humanoid Learns to Do

1. **Phase 1 (0–100 Iterations)**: Exploration. The humanoid frequently falls over as it learns to maintain balance and avoid gravity collapse.
2. **Phase 2 (100–400 Iterations)**: Stance & Gait Discovery. The policy learns to transfer weight between left and right foot support polygons.
3. **Phase 3 (400+ Iterations)**: Synchronized Motion Tracking. The biped smoothly mimics the walking velocity and joint trajectories from the motion capture reference JSON.

---

## 8. How to Teach New Behaviors

### Example 1: Teaching Running / Jumping
1. Replace reference motion file `core/data/motion_capture/human_walk_retargeted.json` with a running/jumping JSON file generated via `third_party/retarget_tools/run_retarget.sh`.

2. Increase target velocity reward weight $w_{\text{vel}}$ from $1.0$ to $3.0$.

### Example 2: Teaching Obstacle Avoidance
1. Add LiDAR or depth camera point-cloud observations into `ObservationsCfg`.
2. Add collision penalty term for foot/shin contact with obstacles.
