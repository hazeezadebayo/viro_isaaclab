# Core Robotics Architecture Guide: Humanoid, ANYmal, AMR, & Cobot

An end-to-end reference guide covering first principles, kinematic formulations, MDP terms, USD trajectory export, ROS2 integration, and in-container execution for **Humanoid Bipedal Motion Tracking**, **ANYmal Quadruped Locomotion**, **Autonomous Mobile Robots (AMR)**, and **Cobot 6-DOF Manipulator Arms**.

---

## 1. Kinematic Foundations & Paradigms

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

---

## 2. In-Container Step-by-Step Execution Guide

Once you launch the Docker environment on your host machine:
```powershell
.\launcher.ps1 build
.\launcher.ps1 up -Head humanoid -Headless
```

You enter the container shell using:
```bash
docker exec -it isaac-sim bash
```

Once inside the container (`root@docker-desktop:/workspace#`), execute the following commands based on your workflow:

---

### Step 1: Train an RL Policy Inside Container

Run Isaac Lab's official training runner (`/isaac-sim/python.sh`) inside the container. Here is the full command with all available input parameters configured:

```bash
# Option A: Automated USD & MP4 Video Export (Docker Desktop / WSL2 without Vulkan drivers)
USD_EXPORT=1 USD_INTERVAL=1800 USD_LENGTH=10 \
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-Imitation-v0 \
    --headless \
    --num_envs 16 \
    --max_iterations 1000 \
    --seed 42 \
    --experiment_name humanoid_imitation \
    --logger tensorboard \
    --device cuda:0


# Option B: Native NVIDIA Vulkan Video Recording (Native Linux / Ubuntu with Vulkan drivers)
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-Imitation-v0 \
    --headless \
    --num_envs 16 \
    --max_iterations 1000 \
    --seed 42 \
    --video \
    --video_length 3600 \
    --video_interval 108000 \
    --enable_cameras \
    --experiment_name humanoid_imitation \
    --logger tensorboard \
    --device cuda:0
```

#### Training Commands for Other Robot Heads:
```bash
# ANYmal-C Quadruped
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Anymal-C-v0 --headless --num_envs 16 --max_iterations 1000

# AMR TurtleBot3 Navigation
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-AMR-Navigation-v0 --headless --num_envs 16 --max_iterations 1000

# Cobot UR5e Manipulator Reaching
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cylinder-Cobot-v0 --headless --num_envs 16 --max_iterations 7000
```

#### Complete In-Container `train.py` Parameter Reference:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--task` | `str` | *Required* | Gym task ID (`Isaac-Humanoid-Imitation-v0`, `Isaac-Anymal-C-v0`, `Isaac-AMR-Navigation-v0`, `Isaac-Lift-Cylinder-Cobot-v0`). |
| `--headless` | `flag` | `False` | Disables GUI viewport for fast headless simulation inside Docker. |
| `--num_envs` | `int` | `4096` (16 CLI) | Number of parallel simulation environments running on GPU CUDA tensors. |
| `--max_iterations` | `int` | *From Agent Cfg* | Maximum RL training iterations (e.g. `1000`). |
| `--seed` | `int` | `42` | Random number generator seed for reproducible training runs. |
| `--video` | `flag` | `False` | Enables periodic video clip recording during training. |
| `--video_length` | `int` | `3600` | Duration of each recorded video clip in **simulation steps** (3600 steps = 1 min @ 60Hz). |
| `--video_interval` | `int` | `108000` | Sim steps between video recording starts (108000 steps = 30 mins @ 60Hz). |
| `--enable_cameras` | `flag` | `False` | Enables offscreen camera rendering pipelines for video capture. |
| `--checkpoint` | `str` | `None` | Path to PyTorch `.pt` model file to resume training from a prior checkpoint. |
| `--experiment_name` | `str` | *From Agent Cfg* | Override directory name under `core/logs/rsl_rl/` where logs are saved. |
| `--logger` | `str` | `"tensorboard"` | Logging framework (`tensorboard`, `wandb`, `neptune`). |
| `--device` | `str` | `"cuda:0"` | Target PyTorch device (`cuda:0` or `cpu`). |

> **Checkpoints & Logs Output**: Trained PyTorch policy weights (`model_*.pt`) and TensorBoard metrics automatically save to `/workspace/core/logs/rsl_rl/<experiment>/<run>/` (persisted on your host at `core/logs/rsl_rl/`).

---

### Step 2: Evaluate (Play) a Trained Checkpoint Inside Container

To evaluate a policy checkpoint (`.pt` file) saved during training:

```bash
/isaac-sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-Imitation-v0 \
    --checkpoint ./core/logs/rsl_rl/humanoid/<run>/model_1000.pt
```

---

### Step 3: Standalone USD-to-MP4 Video Conversion

To convert any generated `.usda` scene file directly into an `.mp4` video without Isaac Lab dependencies:

```bash
/isaac-sim/python.sh core/utils/usd_to_mp4.py core/logs/usd/trajectory_t1.usda
```

> **Output**: The output MP4 file is written to `core/logs/usd/trajectory_t1.mp4`. Both `.usda` and `.mp4` are accessible in host directory `core/logs/usd/`.

---

### Step 4: Video Motion Retargeting (Humanoid Imitation Target)

Generate 3D motion-capture targets (`human_walk_retargeted.json`) from an RGB walking video:

```bash
# Retarget RGB video to humanoid motion JSON
bash third_party/retarget_tools/run_retarget.sh \
    third_party/retarget_tools/bodypose3d/media/cam0_test.mp4 \
    core/data/motion_capture
```

> **Output**: Writes JSON target data to `core/data/motion_capture/human_walk_retargeted.json`.

---

### Step 5: Cobot Vision-Language-Action (VLA) Fine-Tuning

Fine-tune pre-trained VLA architectures ($\pi_0$, SmolVLA, ACT) on demonstration datasets:

```bash
# 1-Click Master VLA Pipeline (Collect -> Fine-Tune -> Inference)
bash third_party/vla_tools/run_vla_pipeline.sh pi0

# Manual VLA Fine-Tuning Execution
python3 core/source/cobot/vla/train_vla.py \
    --model pi0 \
    --pretrained_hub lerobot/pi0_ur5 \
    --dataset core/data/vla/cobot_vla_dataset.h5 \
    --output_dir core/logs/vla
```

---

## 3. Robot Task Configurations & Parameters

Defaults configured in `core/source/<head>/tasks/*_env_cfg.py` and `agents/rsl_rl_ppo_cfg.py`:

| Head | Registered Gym Task ID | `sim.dt` | `decimation` | `step_dt` | `scene.num_envs` | Output Log Path |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Humanoid** | `Isaac-Humanoid-Imitation-v0` | $1/120\,\text{s}$ | 2 | $1/60\,\text{s}$ | 4096 (16 CLI) | `core/logs/rsl_rl/humanoid/` |
| **ANYmal** | `Isaac-Anymal-C-v0` | $1/200\,\text{s}$ | 4 | $1/50\,\text{s}$ | 4096 (16 CLI) | `core/logs/rsl_rl/anymal/` |
| **AMR** | `Isaac-AMR-Navigation-v0` | $1/100\,\text{s}$ | 4 | $1/25\,\text{s}$ | 4096 (16 CLI) | `core/logs/rsl_rl/amr/` |
| **Cobot** | `Isaac-Lift-Cylinder-Cobot-v0` | $0.01\,\text{s}$ | 2 | $0.02\,\text{s}$ | 4096 (16 CLI) | `core/logs/rsl_rl/cobot/` |

---

## 4. TensorBoard Live Metrics

TensorBoard monitors all training runs automatically.

- **Host Web Access**: [http://localhost:6006](http://localhost:6006)
- **Container Path**: `/workspace/core/logs/rsl_rl/`

---

## 5. ROS2 Integration Infrastructure

Bridge subscriptions (`custom_bridge.yaml`) map ROS2 network inputs into `core/source/`:
- `/cmd_vel` $\to$ AMR & ANYmal speed twist commands.
- `/cobot/target_pose` $\to$ Cobot end-effector 3D goal.
- `/humanoid/motion_target` $\to$ Humanoid walking direction target.

```bash
# Launch ROS2 Robot State Publisher inside container
ros2 launch viro_ros2_ws robot_state_publisher.launch.py robot_type:=humanoid
```
