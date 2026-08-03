# ANYmal-C Position-Based Locomotion & Local Navigation

This directory contains the migrated configurations and MDP logic for the ANYmal-C quadruped robot, supporting **position-based locomotion** (walking to target coordinate frames) and **local navigation** (obstacle avoidance using a hierarchical policy architecture).

---

## 1. Overview of Registered Environments (8 Tasks)

There are 8 distinct environments registered for ANYmal-C. They are structured as a combination of two core tasks (Locomotion vs. Navigation), two terrain types (Rough vs. Flat), and two execution modes (Train vs. Play).

| Gym Environment ID | Task Type | Terrain | Mode / Purpose |
| :--- | :--- | :--- | :--- |
| **`Isaac-Anymal-C-v0`** | Locomotion | Rough | **Train** locomotion to walk to coordinates on rough terrain |
| **`Isaac-Anymal-C-Play-v0`** | Locomotion | Rough | **Visualize/Evaluate** locomotion on rough terrain |
| **`Isaac-Anymal-C-Flat-v0`** | Locomotion | Flat | **Train** locomotion to walk to coordinates on flat ground |
| **`Isaac-Anymal-C-Flat-Play-v0`** | Locomotion | Flat | **Visualize/Evaluate** locomotion on flat ground |
| **`Isaac-Anymal-C-Navigation-v0`** | Navigation | Rough | **Train** navigation + obstacle avoidance on rough terrain |
| **`Isaac-Anymal-C-Navigation-Play-v0`** | Navigation | Rough | **Visualize/Evaluate** navigation + obstacle avoidance on rough terrain |
| **`Isaac-Anymal-C-Navigation-Flat-v0`** | Navigation | Flat | **Train** navigation + obstacle avoidance on flat ground |
| **`Isaac-Anymal-C-Navigation-Flat-Play-v0`** | Navigation | Flat | **Visualize/Evaluate** navigation + obstacle avoidance on flat ground |

---

## 2. Core Architecture Dimensions

### A. Core Task
* **Locomotion** (`tasks/anymal_locomotion_*_env_cfg.py`):
  * The network directly maps base inputs, joint states, and target coordinates to motor target commands.
* **Navigation** (`tasks/anymal_navigation_*_env_cfg.py`):
  * Hierarchical setup. A high-level navigation policy receives base states and relative obstacle (red cone) positions, and outputs velocity targets.
  * These velocity targets are fed directly into a pre-trained low-level locomotion policy, which translates them into physical joint commands.
  * Obstacles are spawned dynamically via `RigidObjectCfg` inside the task configuration files.

### B. Terrain Type
* **Rough Terrain**:
  * Procedurally generated slopes, steps, and obstacles.
  * Active height scanner (`RayCasterCfg`) mounted on the robot base scans elevation around feet and passes height data to the policy.
* **Flat Terrain**:
  * Flat ground plane plane.
  * Height scanner is disabled (`self.scene.height_scanner = None`) to simplify observations and accelerate training.

### C. Execution Mode
* **Train Mode (`-v0`)**:
  * Configured for large-scale GPU simulation (spawns 4,096 environments).
  * Enables action/observation noises, randomized physical properties (mass, com, body friction), and random pushes to train highly robust policies.
* **Play Mode (`-Play-v0`)**:
  * Configured for interactive visualization (spawns 1 to 50 environments).
  * Disables domain randomizations, noises, and pushes for clean evaluation of checkpoint models.

---

## 3. Policy & Checkpoint Management

### Centralized Checkpoint Directory (`core/logs`)
Checkpoints are saved and loaded from the centralized `core/logs/rsl_rl/` directory.

```text
core/logs/rsl_rl/
├── anymal_c_rough/
│   ├── pretrained/
│   │   └── policy.pt    # Pretrained locomotion fallback (rough)
│   └── <run_folder>/    # Datetime folders from training
└── anymal_c_flat/
    ├── pretrained/
    │   └── policy.pt    # Pretrained locomotion fallback (flat)
    └── <run_folder>/    # Datetime folders from training
```

### Dynamic Checkpoint Resolver
The navigation environment configurations implement a dynamic path resolver:
1. It scans `core/logs/rsl_rl/anymal_c_rough` (or `anymal_c_flat`) for active training runs.
2. It automatically identifies the latest date-time run folder and selects the highest iteration checkpoint (e.g., `model_2000.pt`).
3. If no training runs are found, it falls back to the `pretrained/policy.pt` model weights.

---

## 4. Train & Play Execution Commands

Run execution commands using the root-level launcher `launcher.ps1`:

### Training Locomotion
```powershell
.\launcher.ps1 train -Head anymal -Task Isaac-Anymal-C-v0 -NumEnvs 4096
```

### Training Navigation
```powershell
.\launcher.ps1 train -Head anymal -Task Isaac-Anymal-C-Navigation-v0 -NumEnvs 2048
```

### Visualizing/Playing a Task Checkpoint
```powershell
.\launcher.ps1 play -Head anymal -Task Isaac-Anymal-C-Play-v0 -Checkpoint ./core/logs/rsl_rl/anymal_c_rough/pretrained/policy.pt
```
