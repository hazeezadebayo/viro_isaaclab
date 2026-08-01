# Cobot 6-DOF Manipulator Arm Target Reaching Architecture

## 1. Overview & Educational Purpose

The **Cobot Manipulator Head** models a 6-DOF industrial collaborative manipulator arm (Universal Robots UR5). Cobots are open-chain serial kinematic mechanisms mounted to a fixed base.

The primary educational objective is to train a neural network policy to master **Inverse Kinematics (IK)**, 3D end-effector Cartesian target reaching, orientation alignment, and collision-free manipulation.

---

## 2. Simulation Environment Setup

- **Task Identifier**: `Isaac-Cobot-Reaching-v0`
- **Physics Engine**: Nvidia PhysX (Fabric GPU acceleration)
- **Simulation Time-step ($\Delta t_{\text{sim}}$)**: $1 / 120 \, \text{s} \approx 8.33 \, \text{ms}$
- **Policy Control Decimation ($K$)**: 2 ($\Delta t_{\text{policy}} = 1 / 60 \, \text{s} \approx 16.67 \, \text{ms}$, 60 Hz control loop)
- **Episode Horizon**: 10.0 seconds (600 policy steps)

---

## 3. Robot Kinematics & Joint Breakdown (6 Actuated Joint DOF)

```
                            [ BASE LINK ] (world fixed)
                                  │
                             joint_1 (Shoulder Pan)
                                  │
                             joint_2 (Shoulder Lift)
                                  │
                             joint_3 (Elbow)
                                  │
                             joint_4 (Wrist 1)
                                  │
                             joint_5 (Wrist 2)
                                  │
                             joint_6 (Wrist 3 / Flange)
                                  │
                        [ END-EFFECTOR / GRIPPER ]
```

### Joint Specifications Table

| Joint Name | Kinematic Function | Range of Motion | Stiffness ($K_p$) | Damping ($K_d$) |
|------------|--------------------|-----------------|------------------|-----------------|
| `joint_1` | Shoulder Pan | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |
| `joint_2` | Shoulder Lift | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |
| `joint_3` | Elbow | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |
| `joint_4` | Wrist 1 | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |
| `joint_5` | Wrist 2 | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |
| `joint_6` | Wrist 3 (Flange) | $[-2\pi, 2\pi] \, \text{rad}$ | 800.0 | 40.0 |

---

## 4. Mathematical Observation Space ($\mathcal{S} \in \mathbb{R}^{21}$)

$$\mathbf{s}_t = \begin{bmatrix} \hat{\mathbf{q}}_t & \dot{\mathbf{q}}_t & \mathbf{p}_{\text{target}} - \mathbf{p}_{\text{ee}} & \mathbf{a}_{t-1} \end{bmatrix}$$

1. **Normalized Joint Angles ($\hat{\mathbf{q}}_t \in \mathbb{R}^6$)**: Current joint angles normalized to $[-1, 1]$.
2. **Joint Velocities ($\dot{\mathbf{q}}_t \in \mathbb{R}^6$)**: Joint angular velocity rates scaled by $0.1$.
3. **End-Effector Target Displacement ($\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{ee}} \in \mathbb{R}^3$)**: 3D spatial vector from flange `link_6` to goal object position $(0.4, 0.0, 0.4)$.
4. **Last Action Vector ($\mathbf{a}_{t-1} \in \mathbb{R}^6$)**: Previous joint target commands.

---

## 5. Mathematical Action Space ($\mathcal{A} \in [-1, 1]^6$)

The policy outputs target joint angle offsets $\mathbf{a}_t \in [-1, 1]^6$. Joint target positions are calculated as:

$$\mathbf{q}_{\text{target}} = \mathbf{q}_{\text{default}} + 0.50 \cdot \mathbf{a}_t$$

The PD motor controller drives actuators to target angles with stiffness $K_p = 800.0$ and damping $K_d = 40.0$.

---

## 6. Reward Function Formulation ($\mathcal{R}$)

$$\mathcal{R}_t = 2.0 \, \exp\left(-\frac{\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{target}}\|^2}{0.04}\right) + 5.0 \, \mathbb{I}(\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{target}}\| < 0.05) - 0.01 \|\mathbf{a}\|^2$$

- **Target Proximity Reward**: Continuous exponential reward scoring 3D distance to target.
- **Reach Bonus**: $+5.0$ sparse bonus when end-effector reaches within $0.05 \, \text{m}$ of target.
- **Action Penalty**: $-0.01 \|\mathbf{a}\|^2$ penalizes high joint acceleration and energy consumption.

---

## 7. What the Cobot Learns to Do

1. **Phase 1**: Explores joint space and learns forward kinematics mapping.
2. **Phase 2**: Coordinates shoulder (`joint_1`, `joint_2`) and elbow (`joint_3`) to bring end-effector into target workspace volume.
3. **Phase 3**: Fine-tunes wrist joints (`joint_4`, `joint_5`, `joint_6`) to precision-align end-effector onto target.

---

## 8. How to Teach New Behaviors

### Example 1: Pick-and-Place Manipulation
1. Add Robotiq 2F-85 gripper articulation to `CobotSceneCfg`.
2. Add object primitive (cube/cylinder) and goal bin.
3. Add gripper grasp reward term: $r_{\text{grasp}} = \mathbb{I}(\text{finger contact}) \cdot \mathbb{I}(\text{object lifted})$.

### Example 2: Peg-in-Hole Assembly / Force Control
1. Add force-torque sensor at `link_6` flange.
2. Add contact force minimization penalty to prevent jamming during peg insertion.

---

## 9. Vision-Language-Action (VLA) Framework ($\pi_0$, $\pi_{0.5}$, SmolVLA, ACT)

The Cobot head includes a dedicated VLA sub-module in `core/source/cobot/vla/` for training and inferencing Vision-Language-Action models:

- **$\pi_0$ & $\pi_{0.5}$ (Physical Intelligence)**: Continuous Flow-Matching ODE policy over vision-language embeddings.
- **SmolVLA**: Lightweight VLM backbone for fast real-time edge execution.
- **ACT**: Action Chunking with Transformers CVAE policy.

### Quickstart Execution
```bash
# 1. Collect Trajectory Dataset
python /workspace/core/source/cobot/vla/dataset_collector.py --episodes 50

# 2. Train VLA Model (pi0, pi0.5, smolvla, act)
python /workspace/core/source/cobot/vla/train_vla.py --model pi0 --epochs 10

# 3. Closed-Loop Inference
python /workspace/core/source/cobot/vla/inference_vla.py --model pi0 --prompt "reach red object"
```

For the complete theoretical and mathematical VLA guide, read **[core/source/cobot/vla/README.md](file:///home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/source/cobot/vla/README.md)**.

