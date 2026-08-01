# ANYmal Quadruped Locomotion Architecture

## 1. Overview & Educational Purpose

The **ANYmal-C Quadruped Head** models a 12-DOF commercial quadrupedal robot (ANYbotics ANYmal-C). Quadrupeds feature a low Center of Mass ($\text{CoM}$) and a 4-point foot contact support polygon, making them statically stable at rest and dynamically versatile over rough terrain.

The educational goal of this environment is to teach **ActuatorNet Neural Control**, quadrupedal trot/pace gait synchronization, and robust blind locomotion over uneven ground planes.

---

## 2. Simulation Environment Setup

- **Task Identifier**: `Isaac-Anymal-C-v0`
- **Physics Engine**: Nvidia PhysX (Fabric GPU acceleration)
- **Simulation Time-step ($\Delta t_{\text{sim}}$)**: $1 / 120 \, \text{s} \approx 8.33 \, \text{ms}$
- **Policy Control Decimation ($K$)**: 2 ($\Delta t_{\text{policy}} = 1 / 60 \, \text{s} \approx 16.67 \, \text{ms}$, 60 Hz control loop)
- **Episode Horizon**: 20.0 seconds (1200 policy steps)

---

## 3. Robot Kinematics & Joint Breakdown (12 Actuated DOF)

```
                            ┌────────────────┐
                            │  ANYmal-C BASE │
                            └───────┬────────┘
             ┌──────────────────────┼──────────────────────┐
      Front Left (LF)        Front Right (RF)       Hind Left (LH)         Hind Right (RH)
     ┌───────┴───────┐      ┌───────┴───────┐      ┌───────┴───────┐      ┌───────┴───────┐
     │  LF_HAA (Hip) │      │  RF_HAA (Hip) │      │  LH_HAA (Hip) │      │  RH_HAA (Hip) │
     │  LF_HFE (Thigh)      │  RF_HFE (Thigh)      │  LH_HFE (Thigh)      │  RH_HFE (Thigh)
     │  LF_KFE (Knee)│      │  RF_KFE (Knee)│      │  LH_KFE (Knee)│      │  RH_KFE (Knee)│
     └───────┬───────┘      └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
          [FOOT]                 [FOOT]                 [FOOT]                 [FOOT]
```

### Joint Registry Table (4 Legs $\times$ 3 Joints/Leg)

| Leg Identifier | Joint Name | Nominal Target ($q_0$) | Range of Motion | Actuator Model |
|----------------|------------|------------------------|-----------------|----------------|
| **Left Front** | `LF_HAA`, `LF_HFE`, `LF_KFE` | $[0.0, 0.4, -0.73] \, \text{rad}$ | $[-0.8, 0.8] \times [-1.5, 1.5] \times [-2.6, -0.2]$ | ActuatorNet MLP |
| **Right Front** | `RF_HAA`, `RF_HFE`, `RF_KFE` | $[0.0, 0.4, -0.73] \, \text{rad}$ | $[-0.8, 0.8] \times [-1.5, 1.5] \times [-2.6, -0.2]$ | ActuatorNet MLP |
| **Left Hind** | `LH_HAA`, `LH_HFE`, `LH_KFE` | $[0.0, -0.4, 0.73] \, \text{rad}$ | $[-0.8, 0.8] \times [-1.5, 1.5] \times [0.2, 2.6]$ | ActuatorNet MLP |
| **Right Hind** | `RH_HAA`, `RH_HFE`, `RH_KFE` | $[0.0, -0.4, 0.73] \, \text{rad}$ | $[-0.8, 0.8] \times [-1.5, 1.5] \times [0.2, 2.6]$ | ActuatorNet MLP |

---

## 4. Mathematical Observation Space ($\mathcal{S} \in \mathbb{R}^{48}$)

$$\mathbf{s}_t = \begin{bmatrix} \mathbf{v}_{\text{base}} & \boldsymbol{\omega}_{\text{base}} & \mathbf{g}_{\text{proj}} & \mathbf{q}_t - \mathbf{q}_0 & \dot{\mathbf{q}}_t & \mathbf{a}_{t-1} \end{bmatrix}$$

1. **Base Velocities ($\mathbf{v}_{\text{base}} \in \mathbb{R}^3, \boldsymbol{\omega}_{\text{base}} \in \mathbb{R}^3$)**: Measured linear and angular velocity in body frame.
2. **Projected Gravity ($\mathbf{g}_{\text{proj}} \in \mathbb{R}^3$)**: Orientation relative to gravity vector.
3. **Joint Position Offsets ($\mathbf{q}_t - \mathbf{q}_0 \in \mathbb{R}^{12}$)**: Displacement from default stance.
4. **Joint Velocities ($\dot{\mathbf{q}}_t \in \mathbb{R}^{12}$)**: Joint angular rates scaled by $0.1$.
5. **Last Action Vector ($\mathbf{a}_{t-1} \in \mathbb{R}^{12}$)**: Previous policy command.

---

## 5. Mathematical Action Space ($\mathcal{A} \in [-1, 1]^{12}$)

The policy outputs joint target offsets $\mathbf{a}_t \in [-1, 1]^{12}$. Target joint position is:

$$\mathbf{q}_{\text{target}} = \mathbf{q}_0 + 0.25 \cdot \mathbf{a}_t$$

ActuatorNet neural network converts $(\mathbf{q}_{\text{target}} - \mathbf{q}_t, \dot{\mathbf{q}}_t)$ into physical SEA (Series Elastic Actuator) motor torques.

---

## 6. Reward Function Formulation ($\mathcal{R}$)

$$\mathcal{R}_t = 1.5 \, r_{\text{progress}} + 2.0 \, r_{\text{alive}} + 0.3 \, r_{\text{upright}} + 0.8 \, r_{\text{target}} - 0.01 \|\mathbf{a}\|^2 - 0.005 \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 - 0.005 \, P_{\text{power}}$$

- **Progress Reward ($r_{\text{progress}}$)**: Linear displacement toward target location $(1000.0, 0, 0)$.
- **Upright Posture Bonus ($r_{\text{upright}}$)**: Encourages base orientation cosine similarity $> 0.90$.
- **Power Consumption Penalty ($P_{\text{power}}$)**: $\sum_j |\tau_j \cdot \dot{q}_j|$, penalizes mechanical energy loss.

---

## 7. What ANYmal Learns to Do

1. **Exploration**: ANYmal balances on four legs and learns to withstand external push perturbations.
2. **Gait Emergence**: Discovers synchronized diagonal leg pairing (Trotting: LF+RH swing while RF+LH stance).
3. **Terrain Navigation**: Maintains constant forward velocity over ground planes without tripping.

---

## 8. How to Teach New Behaviors

### Example 1: Dynamic ROS2 Command Velocity Tracking ($v_x, v_y, \omega_z$)
1. Add command observation $\mathbf{c}_{\text{cmd}} = [v_{x,\text{cmd}}, v_{y,\text{cmd}}, \omega_{z,\text{cmd}}]$ to `ObservationsCfg`.
2. Add tracking reward terms:
   $$r_{\text{lin\_vel}} = \exp\left(-\frac{\|v_{xy} - v_{xy,\text{cmd}}\|^2}{0.25}\right), \qquad r_{\text{ang\_vel}} = \exp\left(-\frac{(\omega_z - \omega_{z,\text{cmd}})^2}{0.25}\right)$$

### Example 2: Stair & Rough Terrain Climbing
1. Change terrain importer in `AnymalSceneCfg` from `"plane"` to `"rough"` or `"pyramid_stairs"`.
2. Add foot clearance reward for lifting feet above step height during swing phase.
