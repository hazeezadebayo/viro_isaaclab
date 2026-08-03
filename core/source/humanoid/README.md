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

---

## 9. new section

### 1. Input Dimension — What Do We Feed Precisely?

The observation vector is **78-dimensional** and is constructed in `humanoid_imitation_env_cfg.py` (`ObservationsCfg.PolicyCfg`):

| Term | Function | Dimension | Scale / Notes |
|------|----------|-----------|---------------|
| `base_height` | `mdp.base_pos_z` | 1 | Base link Z height |
| `base_lin_vel` | `mdp.base_lin_vel` | 3 | Base linear velocity (world frame) |
| `base_ang_vel` | `mdp.base_ang_vel` | 3 | Base angular velocity | **scaled by 0.25** |
| `joint_pos_norm` | `mdp.joint_pos_limit_normalized` | 17 | Joint positions normalized to `[-1, 1]` |
| `joint_vel_rel` | `mdp.joint_vel_rel` | 17 | Joint velocities | **scaled by 0.1** |
| `joint_pos_err` | `mdp.joint_pos_ref_error` | 17 | `q_robot - q_ref` (tracking error) |
| `joint_vel_err` | `mdp.joint_vel_ref_error` | 17 | `dq_robot - dq_ref` (velocity tracking error) | **scaled by 0.1** |
| `phase` | `mdp.motion_phase` | 2 | `[sin(phase), cos(phase)]` normalized reference phase |
| `actions` | `mdp.last_action` | 17 | Previous policy action |
| **Total** | | **78** | |

**Precisely:** The policy receives `[1 + 3 + 3 + 17 + 17 + 17 + 17 + 2 + 17] = 78` floats per environment per step. There are **no cameras, no LiDAR, no depth sensors** — this is purely **proprioceptive** (state-based) RL.

---

### 2. Observation Space — What Does the Humanoid "See"?

The humanoid has **no visual or range sensors**. It does not "see" the world. Instead it senses its **own internal body state**:

#### Proprioceptive Sensors Used

| Sensor | Source | Description |
|--------|--------|-------------|
| **Base height** | `asset.data.root_pos_w[:, 2]` | Z-position of pelvis |
| **Base linear velocity** | `asset.data.root_lin_vel_b` | 3-DoF velocity of torso in body frame |
| **Base angular velocity** | `asset.data.root_ang_vel_b` | 3-DoF angular velocity of torso |
| **Joint positions** | `asset.data.joint_pos` | 17 joint angles |
| **Joint velocities** | `asset.data.joint_vel` | 17 joint angular velocities |
| **Projected gravity** | `asset.data.projected_gravity_b` | Used in `base_up_proj`, `base_angle_to_target` |
| **Reference motion** | `motion_loader.get_current_frame()` | External mocap reference trajectory |

#### How Reference Motion Is Recorded

The reference motion is loaded from `core/data/motion_capture/human_walk_retargeted.json` by `ReferenceMotionLoader` (`mdp/motion_loader.py`). This JSON contains:
- `joint_positions`: `[N_frames, 17]` retargeted joint angles
- `joint_velocities`: `[N_frames, 17]` finite-differenced velocities
- `fps`: 60.0 Hz

At each simulation step, the loader linearly interpolates between frames based on `env_times` to produce `q_ref` and `dq_ref` for all 4096 environments simultaneously.

#### Observation Recording
The observations are **not recorded to disk** during training. They are computed on-the-fly each step via the Isaac Lab `ObservationManager` and fed directly to the policy network. If you want to log them, you would add `env.extras["observations"] = obs` in a custom callback.

---

### 3. Environment — Goal, Events, and Terrain

#### Environment Class
`HumanoidImitationEnv` inherits from `ManagerBasedRLEnv` and is registered as `Isaac-Humanoid-Imitation-v0`.

#### Precise Goal
Track a **looping reference walking motion** (`human_walk_retargeted.json`) as accurately as possible while remaining upright and energetically efficient. This is **not** a locomotion-to-target task — it is **pure motion imitation**.

#### Terrain
- **Flat plane** (`terrain_type="plane"`)
- Physics material: static friction = 1.0, dynamic friction = 1.0, restitution = 0.0

#### Events (Domain Randomization & Reset)

| Event | Mode | Purpose |
|-------|------|---------|
| `reset_base` | `reset` | Randomize root pose (`pose_range={}` defaults → actually uses Isaac Lab defaults) and zero velocity |
| `reset_robot_joints` | `reset` | Randomize joint positions by `±0.2 rad`, velocities by `±0.1 rad/s` around default |
| *(No push, no friction randomization in imitation env)* | | |

**Note:** The basic `humanoid_env_cfg.py` (non-imitation) adds `randomize_rigid_body_material` and `push_by_setting_velocity` events, but the **imitation variant does not** — it uses cleaner resets to stabilize learning.

#### Episode Horizon
- `episode_length_s = 16.0` seconds
- `decimation = 2` → policy runs at **60 Hz** (sim runs at 120 Hz)
- Motion duration: **~3.12 seconds** (loops every 187 frames at 60 FPS)

---

### 4. Rewards — Scoring and Science

The total reward per environment per step is:

```
R_total = 1.5 * r_pose_tracking + 0.5 * r_vel_tracking 
        + 1.0 * r_alive + 0.2 * r_upright 
        - 0.01 * r_action_rate - 0.002 * r_power
```

#### Reward Terms with Exact Formulas

| Term | Weight | Formula | Purpose |
|------|--------|---------|---------|
| `pose_tracking` | **+1.5** | `exp(-5.0 * Σ(q_robot - q_ref)²)` | Exponential pose matching |
| `vel_tracking` | **+0.5** | `exp(-0.1 * Σ(dq_robot - dq_ref)²)` | Velocity phase alignment |
| `alive` | **+1.0** | `1.0 if root_height > 0.7m else 0.0` | Survival bonus |
| `upright` | **+0.2** | `1.0 if projected_gravity_z > 0.90 else 0.0` | Torso uprightness |
| `action_rate` | **-0.01** | `Σ(a_t - a_{t-1})²` | Smoothness penalty |
| `power` | **-0.002** | `Σ \|τ * dq\|` (weighted by gear ratio) | Energy efficiency |

#### The Science Behind the Rewards

1. **Exponential tracking rewards** (`exp(-k * error²)`): This is a **kernel density** style reward that is smooth, bounded `[0, 1]`, and has a well-behaved gradient near the optimum. The `tracking_k` parameters (5.0 for pose, 0.1 for velocity) were tuned so that typical human walking deviations yield rewards in the `[0.3, 0.9]` range, leaving headroom for improvement.

2. **Survival bonus** (`is_alive`): In legged RL, the agent must first learn to not fall. A binary `height > 0.7m` check is simpler and more stable than a continuous penalty on height.

3. **Upright bonus**: Uses `base_up_proj` (projection of body up-vector onto world up). Threshold 0.90 means the torso can tilt up to ~25° before losing the bonus.

4. **Action smoothness**: Prevents high-frequency jitter that would waste energy and look unnatural.

5. **Power consumption**: Computed as `τ · ω` (torque times joint velocity), scaled by gear ratios. This approximates electrical power and discourages aggressive torque commands.

---

### 5. Forward Pass, Backpropagation, Loss Estimation, and Model Architecture

#### This codebase does NOT contain the training loop.

Isaac Lab uses **RSL-RL** (`isaaclab_rl`) as the training backend. The humanoid package only provides:
- The environment configuration
- The MDP terms (observations, actions, rewards, terminations, events)

#### Model Architecture (`agents/rsl_rl_ppo_cfg.py`)

```python
policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,          # Initial std dev for action distribution
    actor_hidden_dims=[400, 200, 100],   # 3-layer MLP
    critic_hidden_dims=[400, 200, 100],  # 3-layer MLP
    activation="elu",                     # ELU activation
)
algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=2.0,
    use_clipped_value_loss=True,
    clip_param=0.2,               # PPO epsilon
    entropy_coef=0.0,             # No entropy bonus (deterministic imitation)
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=5.0e-4,
    schedule="adaptive",          # KL-adaptive LR
    gamma=0.99,
    lam=0.95,                     # GAE lambda
    desired_kl=0.01,
    max_grad_norm=1.0,
)
```

#### Where the Training Code Lives

The forward pass, loss computation, and backpropagation happen inside **RSL-RL** (installed separately as `isaaclab_rl`), specifically in:
- `isaaclab_rl/rsl_rl/runners/on_policy_runner.py` — collects rollouts, computes GAE, calls `learn()`
- `isaaclab_rl/rsl_rl/algorithms/ppo.py` — PPO loss, backprop, optimizer step

You would launch training via:
```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-Imitation-v0 --headless
```

---

### 6. Output/Action Space — What Is Expected of the Robot?

#### Action Space
- **Dimension:** 17 (one per actuated joint)
- **Range:** `[-1, 1]` (normalized, unbounded in practice but clipped by the action term)
- **Type:** Residual position offsets added to the reference motion pose

#### Action Application — `PDTrackingAction`

The raw policy action `a ∈ [-1, 1]^17` is processed by `PDTrackingAction.process_actions()`:

```python
# 1. Get reference joint positions from motion loader
q_ref, _ = env.motion_loader.get_current_frame()

# 2. Compute target pose: policy output is a RESIDUAL offset
q_target = q_ref[:, joint_ids] + (a_raw * action_scale)   # action_scale = 0.25

# 3. Read current state
q_current = asset.data.joint_pos[:, joint_ids]
dq_current = asset.data.joint_vel[:, joint_ids]

# 4. Compute PD torques
tau = Kp * (q_target - q_current) - Kd * dq_current       # Kp=100, Kd=10

# 5. Clip torques
tau = clamp(tau, -200.0, 200.0)

# 6. Apply as joint effort targets
asset.set_joint_effort_target(tau, joint_ids)
```

#### What the Robot Must Do
The humanoid must produce joint torques that cause its 17 joints to track the reference walking motion. The policy is learning **when and how much to deviate** from the reference pose to maintain balance, not learning the pose itself.

---

### 7. Where Does MPC Come In? Where Does WBC Come In?

#### MPC (Model Predictive Control)
**MPC is NOT present in this codebase.** This is a pure RL approach. There is no trajectory optimizer, no contact scheduling, and no predictive model of future states. The policy is reactive: it maps current state → current action.

#### WBC (Whole-Body Control)
**WBC is NOT present as a separate module.** However, the `PDTrackingAction` term is a **simplified form of operational-space joint control**:

| Concept | Implementation Here | Full WBC Would Add |
|---------|---------------------|-------------------|
| Joint-level PD | ✅ `Kp*(q_target - q) - Kd*dq` | Task-space PD (Cartesian) |
| Priority stacking | ❌ None | Contact, base, arm priorities |
| Contact forces | ❌ Not optimized | QP-based contact force optimization |
| Feasibility | ❌ No constraints | Inequality constraints (friction cone, torque limits) |

#### Where They Would Fit in a Production Pipeline

In a production humanoid system (like Boston Dynamics Atlas or Unitree), the hierarchy is:

```
High-Level Planner (MPC)
    ↓ desired base trajectory + footstep plan
Whole-Body Controller (WBC)
    ↓ joint torques with contact force optimization
Low-Level Joint PD (hardware)
    ↓ motor commands
```

In this Isaac Lab codebase, **the RL policy replaces both the MPC and WBC layers**. The policy learns an implicit model of the dynamics and outputs joint efforts directly. This is common in simulation-to-real transfer research because:
- MPC requires an accurate dynamics model (hard to get for soft contacts)
- WBC requires solving a QP at ~1 kHz (computationally expensive)
- RL can learn a compressed policy that runs in a single forward pass (~1 ms on GPU)

---

### Summary Diagram

```
Reference Motion JSON
        ↓
  MotionLoader (interpolates q_ref, dq_ref)
        ↓
  ObservationManager (78-D vector: state + error + phase)
        ↓
  RSL-RL PPO Actor [400,200,100] MLP → 17-D action ∈ [-1,1]
        ↓
  PDTrackingAction (q_target = q_ref + a*scale)
        ↓
  PD Controller: τ = Kp(q_target - q) - Kd*dq
        ↓
  PhysX Simulation (120 Hz)
        ↓
  Reward Computation → PPO Critic + GAE → Backprop
```

The humanoid package is a **self-contained, state-only, imitation learning environment** with no hierarchical control, no MPC, and no WBC — the neural policy is the entire controller.

---

## 10. VLA, MPC + WBC Hybridization, and Real-Robot Deployment

### 1. How Would a VLA Be Used Here?

A **Vision-Language-Action** model would replace the current policy architecture. Today the humanoid project has:

```
78-D proprioceptive state → MLP [400,200,100] → 17-D residual joint offsets
```

A VLA would change this to something like:

```
RGB images + language instruction + proprioceptive state 
    → frozen vision encoder (e.g., ViT) + LLM tokenizer 
    → VLA backbone (e.g., π0, OpenVLA, or custom) 
    → 17-D residual joint offsets
```

#### Where It Would Plug In

| Component | Current Implementation | VLA-Augmented Version |
|-----------|----------------------|----------------------|
| **Policy** | `RslRlPpoActorCriticCfg` MLP | VLA model that conditions on vision + language |
| **Observation** | 78-D state vector | Multi-modal: RGB tokens + state tokens + language tokens |
| **Action** | 17-D joint residuals | Same 17-D residuals (or higher-level commands) |
| **Reference motion** | Fixed JSON loop | Could be conditioned by language ("walk", "run", "wave") |

#### What Datamentors Specifically Wants

The job description mentions **"Vision-Language-Action (VLA) models that give our robots natural-language reasoning and autonomy."** This means:

- A field operator could say **"go to the patient room and pick up the tray"** instead of writing a navigation script
- The VLA would break this into: visual perception (where is the tray?) → motion planning (walk there) → manipulation (grasp tray)
- The current humanoid project would be the **locomotion backbone** that the VLA calls as a skill

#### Code Evidence for Integration Point

In `humanoid_imitation_env_cfg.py`, the observations are computed by `ObservationManager`:

```python
class PolicyCfg(ObsGroup):
    base_height = ObsTerm(func=mdp.base_pos_z)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    # ...
    actions = ObsTerm(func=mdp.last_action)
```

A VLA integration would add:

```python
# New observation terms
rgb_image = ObsTerm(func=mdp.camera_rgb, params={"sensor_cfg": SceneEntityCfg("camera")})
language_embedding = ObsTerm(func=mdp.language_embedding, params={"prompt": "walk forward"})
```

The VLA model would then run **outside** Isaac Lab's `ObservationManager` — it would take the RGB image and language token, produce a latent, and that latent would either:
1. Replace part of the 78-D observation vector fed to the existing MLP, or
2. Replace the MLP entirely, with the VLA directly outputting actions

---

### 2. Can the Humanoid Training Be Combined with MPC + WBC?

**Yes.** This is the standard hybrid architecture in modern legged robotics. The humanoid project provides the **high-level policy**, and MPC + WBC provide the **mid-level stabilizer** and **low-level executor**.

#### The Three-Layer Stack

```
┌───────────────────────────────────────────────────┐
│  LAYER 3: High-Level Policy (RL)                  │
│  "I want to walk forward at 0.5 m/s"              │
│  Output: Reference motion / desired CoM vel       │
│  ← This is what the humanoid project trains        │
└───────────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────┐
│  LAYER 2: MPC (Model Predictive Control)          │
│  "Over the next 2 seconds, optimize..."           │
│  - CoM trajectory tracking                         │
│  - Footstep placement / timing                     │
│  - Angular momentum management                     │
│  Output: Desired base pose, foot swing targets     │
└───────────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────────┐
│  LAYER 1: WBC (Whole-Body Control)                │
│  "Find joint torques that satisfy all tasks"       │
│  - QP optimization with priorities                 │
│  - Contact force optimization                      │
│  - Torque limit enforcement                        │
│  Output: τ = joint efforts [17-D]                  │
└───────────────────────────────────────────────────┘
                        ↓
                ┌───────────────┐
                │  Real Hardware │
                └───────────────┘
```

#### What Each Layer Controls

| Layer | Controls | Optimizes | Constraints |
|-------|----------|-----------|-------------|
| **RL Policy** | High-level behavior | Reward function (imitation, energy) | None — pure trial-and-error |
| **MPC** | CoM trajectory, footstep schedule | Future cost over horizon (2-4 steps ahead) | Dynamics model, ground contact |
| **WBC** | Joint torques | Multi-objective QP | Friction cone, torque limits, unilateral contacts |

#### What WBC Specifically Optimizes

The WBC solves a **hierarchical QP** (Quadratic Program) at ~1 kHz:

```
MINIMIZE:  || J₁ᵀ × τ + J₁_ff ||²    ← Priority 1: Base pose / CoM
           + || J₂ᵀ × τ + J₂_ff ||²  ← Priority 2: Foot swing tracking
           + || J₃ᵀ × τ + J₃_ff ||²  ← Priority 3: Joint regularization

SUBJECT TO:
    Λ(q) × q̈ + g = J_cᵀ × f + τ     ← Rigid body dynamics
    f_iz ≥ 0                           ← Unilateral contacts (feet can push, not pull)
    || f_ixy || ≤ μ × f_iz            ← Friction cone
    τ_min ≤ τ ≤ τ_max                 ← Joint torque limits
    q_min ≤ q ≤ q_max                 ← Joint position limits
```

Where:
- `J₁`, `J₂`, `J₃` = Jacobians for different tasks
- `f` = contact forces at feet
- `Λ(q)` = centroidal mass matrix
- `g` = gravity + Coriolis

**The WBC output is joint torques τ that simultaneously satisfy ALL these constraints while tracking ALL these tasks.**

#### How the Humanoid Policy Fits In

The current `PDTrackingAction` is a **crude approximation of Layer 1 (WBC)** — it only does joint-space PD with no contact optimization.

If you hybridize with MPC + WBC:

```python
# Instead of PDTrackingAction directly applying torques:
class HybridAction(ActionTerm):
    def process_actions(self, actions):
        # 1. RL policy generates reference motion / behavior
        q_ref = self.policy(obs)  # Could be the trained MLP or VLA
        
        # 2. MPC optimizes CoM trajectory and footstep timing
        com_traj, foot_swing_targets = self.mpc.solve(q_ref, dt=0.001)
        
        # 3. WBC computes joint torques that track both
        #    - the MPC's base/CoM task
        #    - the RL policy's joint reference
        tau = self.wbc.solve(
            base_task=com_traj,
            foot_tasks=foot_swing_targets,
            joint_ref=q_ref,
        )
        
        self._processed_actions = tau
```

**This is exactly what Boston Dynamics Atlas, Unitree, and ANYmal do in research papers.** The RL policy learns *what* to do; MPC plans *how* to do it safely; WBC computes *exactly how* to apply torques.

---

### 3. Hardware Interface, ROS 2, and Latency

#### Yes, You Need a Hardware Interface

Every robot joint has a **physical communication bus**:

| Robot | Joint Interface | Protocol |
|-------|----------------|----------|
| Unitree G1/H1 | Built-in motor driver board | Ethernet / RS485 / CAN |
| Boston Dynamics Spot | Custom PCB | Ethernet + ROS 2 topics |
| ANYmal | Custom actuator board | EtherCAT |
| Datamentors Ardia | Likely custom | Whatever they designed |

The hardware interface is the **software layer that translates "apply 50 Nm to knee joint"** into **the specific bytes/registers** the motor driver expects.

```python
# Pseudocode for a real hardware interface
class RealRobotInterface:
    def set_joint_effort(self, joint_name, torque):
        # 1. Map joint name to CAN/EtherCAT ID
        actuator_id = self.joint_name_to_id[joint_name]  # e.g., "left_knee" → CAN ID 0x05
        
        # 2. Encode torque into protocol bytes
        message = self.protocol.encode_torque(actuator_id, torque)
        
        # 3. Send to hardware bus
        self.can_bus.send(message)
        
        # 4. Read back feedback
        feedback = self.can_bus.receive()
        actual_position = feedback.position
        actual_velocity = feedback.velocity
        actual_torque = feedback.torque
```

#### Why ROS 2 (and Not Just Direct Connection)

You **can** connect directly to joint middleware without ROS 2. Many research labs do this. But Datamentors requires ROS 2 for several specific reasons:

##### 1. It's the Industry Standard for Multi-Sensor Fusion

A humanoid robot doesn't just have joints. It has:
- Cameras (RGB, depth)
- LiDAR
- IMU
- Force/torque sensors at feet
- Battery management
- Emergency stop circuits

ROS 2 provides:
- **Standardized message types**: `sensor_msgs/Image`, `sensor_msgs/Imu`, `geometry_msgs/Twist`
- **Time synchronization**: All sensors get a common timestamp via `ros_clock`
- **Transport layer**: DDS (Data Distribution Service) handles multicast, discovery, and reliability

If you connect directly to joints without ROS 2, you'd need to write custom code for **every sensor** and **every robot variant** in the fleet. ROS 2 gives you a common bus.

##### 2. Real-Time Guarantees

ROS 2 has a **Real-Time (RT) execution profile**:

```bash
# Run with PREEMPT_RT kernel for hard real-time
sudo apt install ros-${ROS_DISTRO}-ros2run
ros2 run --real-time ...
```

This ensures:
- The control loop is not preempted by the OS
- Memory allocation happens in pre-allocated pools (no malloc in the control path)
- Threads have fixed priorities

Without ROS 2, you'd need to write your own real-time Linux configuration, thread pinning, and memory pools — which is exactly what ROS 2 already solved.

##### 3. Latency Is NOT the Problem You Think

You asked: *"why can't we connect the middleware of the joints themselves directly?"*

You **can**. For a single joint at 1 kHz, direct CAN/EtherCAT access has **lower latency** than ROS 2 (~0.1ms vs ~0.5-1ms).

But Datamentors is building a **fleet of heterogeneous robots** (humanoids, quadrupeds, mobile bases). The latency cost of ROS 2 is worth paying because:

| Concern | Direct Connection | ROS 2 |
|---------|------------------|-------|
| Single-joint latency | ~0.1 ms | ~0.5 ms |
| Multi-sensor fusion | Custom code | Built-in |
| Fleet management | Custom | `ros2 launch`, `ros2 lifecycle` |
| Debugging/visualization | Custom | `rviz2`, `rqt`, `foxglove` |
| Safety/E-stop | Custom | ROS 2 `lifecycle` + `system_metrics` |

##### 4. The Actual Latency Problem

The real latency hazard is not ROS 2 itself — it's the **full stack**:

```
Camera frame capture
    ↓ ~2-5 ms (exposure + USB transfer)
ROS 2 topic publish/deserialize
    ↓ ~0.5-1 ms
Policy inference (VLA or MLP)
    ↓ ~1-5 ms (Jetson Orin ~2ms, desktop GPU ~0.5ms)
ROS 2 topic publish/deserialize
    ↓ ~0.5-1 ms
CAN/EtherCAT message to joint
    ↓ ~0.5-1 ms (bus arbitration + packet time)
Joint torque loop (1 kHz)
    ↓ ~1 ms
Total end-to-end latency: 5-15 ms
```

For a 1 kHz humanoid (1 ms control period), **5-15 ms total latency is catastrophic**. This is why:
- **Inference runs on the robot's embedded GPU** (Jetson Orin, not a workstation)
- **ROS 2 shared memory transport** avoids copy overhead for intra-process communication
- **The PD/WBC layer runs on a separate real-time core** while the VLA inference runs on the GPU core
- **Prediction horizons in MPC** compensate for known latency

#### What the Hardware Interface Actually Looks Like

Based on the Datamentors tech stack and the humanoid codebase, the deployment pipeline would be:

```python
# On the robot's embedded computer (Jetson Orin / x86 real-time)

# Thread 1: Real-time control (1 kHz, pinned to CPU core 0)
def control_loop():
    while True:
        # 1. Read joint states from EtherCAT/CAN
        q, dq, tau_actual = hardware_interface.read_feedback()
        
        # 2. Compute PD / WBC torques (could be from RL policy, MPC, or hybrid)
        tau_command = controller.compute(q, dq, obs)
        
        # 3. Send torque commands
        hardware_interface.send_torque(tau_command)
        
        # 4. Log for safety monitoring
        safety_monitor.check(tau_command, tau_actual, q, dq)
        
        sleep(0.001)  # 1 kHz

# Thread 2: Policy inference (runs as fast as possible, GPU)
def inference_loop():
    while True:
        # 1. Collect observations (proprioceptive + vision)
        obs = observation_collector.get_latest()
        
        # 2. Run VLA or MLP policy
        action = policy_model.forward(obs)
        
        # 3. Write to shared memory buffer (lock-free ring buffer)
        shared_buffer.write(action)
        
        # No sleep — inference runs at maximum throughput

# Thread 3: Non-real-time (telemetry, fleet comms, logging)
def telemetry_loop():
    ros2_publisher.publish(joint_states, diagnostics, battery)
```

#### Why Datamentors Needs ROS 2 Specifically

From the job description:
- **"Expert ROS 2 and real-time control"** — they already have a ROS 2-based fleet
- **"agnostic orchestration platform"** — ROS 2 is the middleware that makes heterogeneous robots (humanoid, quadruped, mobile) interoperable
- **"fleet of 600 units per year"** — you cannot manage 600 robots with custom per-joint drivers. You need a standardized stack.
- **"defence and civil protection"** sectors — these require **certifiable, auditable, deterministic** software, which ROS 2's type-safe interfaces and real-time profile provide

#### Summary

| Question | Answer |
|----------|--------|
| **VLA usage** | Replace/augment the 78-D state policy with vision+language conditioned actions. The current project is the locomotion backbone the VLA would invoke as a skill. |
| **RL + MPC + WBC** | Yes. RL = high-level policy ("what to do"). MPC = mid-level planner ("CoM trajectory over next 2s"). WBC = low-level executor ("joint torques satisfying dynamics + contacts"). The humanoid PD layer is a simplified WBC. |
| **Direct joint connection vs ROS 2** | You *can* connect directly, but for a fleet of heterogeneous robots with cameras, IMUs, and safety requirements, ROS 2 provides the only maintainable, real-time, multi-sensor, multi-robot abstraction layer. The latency cost (~0.5-1ms) is acceptable compared to the engineering cost of replacing it. |
