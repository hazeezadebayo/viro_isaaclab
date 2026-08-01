# Isaac RL Studio - Build & Usage Guide

This guide covers two things:

1. **Part 1 - Docker Build** — how the container images are built and how caching works.
2. **Part 2 - Usage** — a complete, step-by-step walkthrough: from opening a terminal, to
   visualizing the simulation in your browser, running headless training, loading your own
   URDF/USD robot, jogging joints, and tuning rewards.

If you are brand new to this project, start at **Part 2, Section 2.1**.

---

# Part 1 - Docker Build

This project uses a **layered Docker build strategy** for optimal caching:

```
┌─────────────────────────────────────┐
│  Layer 7: Primitives (app code)     │  <-- Changes most often
├─────────────────────────────────────┤
│  Layer 6: Entrypoint                │
├─────────────────────────────────────┤
│  Layer 5: ROS2 environment          │
├─────────────────────────────────────┤
│  Layer 4: ROS2 packages             │
├─────────────────────────────────────┤
│  Layer 3: ROS2 repo setup           │
├─────────────────────────────────────┤
│  Layer 2: Isaac Lab verification    │  <-- Cached as long as base image doesn't change
├─────────────────────────────────────┤
│  Layer 1: Base system utilities     │
├─────────────────────────────────────┤
│  Layer 0: nvcr.io/nvidia/isaac-lab  │  <-- 7GB cached by Docker
└─────────────────────────────────────┘
```

## Build Options

### Option 1: Full build (recommended)
```powershell
.\launcher.ps1 build
# or
docker compose -f docker/docker-compose.yml build isaac-sim
docker compose -f docker/docker-compose.yml build isaac-viz
```

This builds both:
- `isaac-rl-studio:2.1.0-humble` — the main Isaac Lab + ROS2 image
- `isaac-rl-studio-viz:2.1.0-humble` — thin additive layer on top with Xvfb +
  x11vnc + noVNC for browser visualization (`http://localhost:6080/vnc.html`)

### Option 2: Build visualization layer on top of the main image
```bash
docker compose -f docker/docker-compose.yml build isaac-viz
# or standalone:
docker build -f docker/Dockerfile.viz -t isaac-rl-studio-viz:2.1.0-humble \
    --build-arg VIZ_BASE=isaac-rl-studio:2.1.0-humble ..
```

## Cache Behavior

**What gets cached:**
- The 7GB `nvcr.io/nvidia/isaac-lab:2.1.0` base image is cached by Docker automatically
- Each Dockerfile layer is cached independently
- Changing `primitives/` only rebuilds Layer 7
- Changing ROS2 configs only rebuilds Layers 3-5
- Changing the base image invalidates Layers 1-7
- Both images share the same `docker/entrypoint.sh`; changing it rebuilds only
  the entrypoint + primitives COPY layers (the apt/ROS2 layers reuse cache)

**What invalidates cache:**
- Changing `FROM` image or tag
- Changing files copied in a layer
- Changing `RUN` command content

## Verification

```bash
docker images | Select-String "isaac"
```

Should show the base image and any intermediate layers.

---

# Part 2 - Usage Guide

## 2.1 Start here: from a blank terminal to a running simulation

Open **PowerShell** on your Windows machine.

```powershell
# Go to the project folder (adjust if you cloned it somewhere else)
cd C:\Users\azeez.adebayo\env_ideahub\ideahub\isaaclab

# Make sure Docker Desktop is running (look for the whale icon in the system tray)
# then build the images (first build is slow; later builds reuse the cache)
.\launcher.ps1 build
```

When the build finishes you have two images ready:
- `isaac-rl-studio` (the "sim" container — training / headless work)
- `isaac-rl-studio-viz` (the "viz" container — view the sim in your browser)

You never run Isaac Sim directly on your Windows machine. Everything runs **inside
the container**; the Windows machine is just a remote control + display.

## 2.2 How the project is organized

```
ideahub/isaaclab/
├── docker/               # Build files (Dockerfile.core, Dockerfile.viz, compose, entrypoint)
├── primitives/           # Core framework code (base env/agent classes, ROS2 bridge)
│   ├── core/
│   └── ros2/
├── heads/                # THE SCENARIOS. One folder per robot/environment ("a head")
│   ├── template/         # Unitree Go1 quadruped + terrain
│   └── humanoid/         # Isaac-Humanoid-v0 walking task
│       ├── config/       # env_cfg.py, robot_cfg.py, scene_cfg.py, reward_cfg.py, agent_cfg.py
│       ├── description/  # URDFs / USDs / meshes for the robot
│       ├── rl/           # train.yaml / play.yaml (reference configs)
│       └── ros2/         # ROS2 topic bridge settings for this head
├── workspace/            # Persisted logs + data (mounted into the container)
├── launcher.ps1          # The one command-line tool you need
└── .env                  # Default settings (image, ports, head name, ...)
```

Important mental model:
- **`heads/<name>` is read-only inside the container.** To change a robot, a config or a
  reward, edit the files on your **Windows host** under `heads/<name>/...` and the container
  sees the change immediately. Restart the container to make sure a fresh environment uses it.
- **`workspace/` is shared and persistent.** Logs, checkpoints, and videos written by the
  container land in your `workspace/logs` and `workspace/data` folders and survive restarts.

Two heads ship with the project:
- `template` → Unitree Go1 quadruped on a generated terrain (the default).
- `humanoid` → the classic `Isaac-Humanoid-v0` walking task.

## 2.3 HOW TO VISUALIZE the simulation (browser / noVNC + recordings)

No X server, no VNC client, no extra software needed — just a browser.

> **Platform note (Windows / Docker Desktop / WSL2):** Isaac Sim's 3D viewport **cannot
> render** on Docker Desktop. NVIDIA does not support Isaac Sim on WSL2, and Docker Desktop
> forwards only CUDA, not the Vulkan/OpenGL driver the Isaac renderer needs. So on Windows the
> noVNC page shows the virtual **desktop**, not the robot. On a native-Linux host with the
> NVIDIA driver, the viewport does render in the browser.

```powershell
# 1. Start the visualization container (this replaces the plain sim container)
.\launcher.ps1 up -Head humanoid -Viz
```

`up -Head humanoid -Viz` maps the head to the `Isaac-Humanoid-v0` task and the container
**auto-starts it as headless training**. The browser views are:

```
TensorBoard (auto-started): http://localhost:6006
noVNC desktop:              http://localhost:6080/vnc.html
```

Follow launch progress while you wait:

```powershell
docker exec -it isaac-rl-studio-viz tail -f /var/log/isaac-viz/task.log
```

### 2.3.1 Periodic video clips (camera recording)

The trainer's built-in `--video` flags record a short MP4 every N steps — no live GUI needed.
Enable via PowerShell before `up`:

```powershell
$env:VIZ_VIDEO = "1"            # 0/1
$env:VIZ_VIDEO_INTERVAL = "2000"  # clip every 2000 steps
$env:VIZ_VIDEO_LENGTH  = "2000"   # ~60-90 s of footage per clip
.\launcher.ps1 up -Head humanoid -Viz
```

Clips land in `workspace/logs/rsl_rl/humanoid/<run>/videos/train/`. Video recording requires
the Isaac RTX renderer, which needs **Vulkan ray tracing** — available on native Linux with an
NVIDIA driver, but **not** on Docker Desktop/WSL2 (even a software Vulkan driver such as
llvmpipe has no ray tracing, so this cannot be fixed by installing packages). On Docker
Desktop, do **not** set `VIZ_VIDEO=1`: the trainer hangs at renderer init. Use the behavior
report in 2.3.2 instead.

Manual launch (also how you run heads without a mapped task):

```bash
# 2. Open a shell inside the viz container
docker exec -it isaac-rl-studio-viz bash
```

```bash
# 3. Load the environment (ROS, Isaac, head paths) — do this in every new shell
source /usr/local/bin/isaac-ros-entrypoint.sh

# 4. Launch the humanoid environment headless, recording a clip every 2000 steps
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 --headless --num_envs 4 \
    --video --video_interval 2000 --video_length 2000
```

> Why `--num_envs 4`? For recording, keep it small so each step is fast; training the
> headless policy is independent of what you record.

**Change head / port:**
- Different robot: `.\launcher.ps1 up -Head template -Viz` (then run the template's
  environment, see Sections 2.6-2.8 for how).
- Different ports: edit `.env` → `VIZ_PORT`, `VNC_PORT`, `TENSORBOARD_PORT` (or set
  `$env:VIZ_PORT="7000"` etc. in PowerShell before `up`).

> **Networking note:** `isaac-viz` uses a bridge network with published ports
> (`6080` noVNC, `5900` VNC, `6006` TensorBoard), so `localhost` works on any host
> (including Docker Desktop on Windows). Only `isaac-sim` uses `NETWORK_MODE=host`
> (required for ROS2 DDS discovery).
>
> **If the browser says "Failed to connect to server":** open a terminal and check
> `docker exec isaac-rl-studio-viz sh -c "ps | grep -E 'Xvfb|x11vnc|websockify'"`.
> All three must be running. If they are not, restart the container:
> `docker compose -f docker/docker-compose.yml restart isaac-viz`. The entrypoint
> automatically clears stale Xvfb display locks before restarting the display.

### 2.3.2 Behavior report (headless, no renderer required)

If no renderer is available (e.g. Docker Desktop), generate plot-based visual evidence of
behavior instead. This uses no cameras and no GPU rendering:

```bash
# inside the container
source /usr/local/bin/isaac-ros-entrypoint.sh
python /workspace/primitives/core/eval_report.py --task Isaac-Humanoid-v0 --num_envs 4 --steps 600
# with a trained policy:
python /workspace/primitives/core/eval_report.py --task Isaac-Humanoid-v0 \
    --checkpoint /workspace/logs/rsl_rl/humanoid/<run>/model_500.pt
```

Report: `workspace/logs/eval/<timestamp>/report.html` (CoM position/height, joints, reward,
top-down trajectory) — open it directly in your browser.

## 2.4 HOW NOT TO VISUALIZE (headless training)

Headless = no window rendered at all. It is the fastest mode (GPU does pure compute) and is
what you use for real training runs.

```powershell
# Start the plain simulation container (not the viz one)
.\launcher.ps1 up -Head humanoid -Headless
#   (-Headless and -NoGui are the same thing; omit them to keep the container in GUI mode)
```

```bash
docker exec -it isaac-rl-studio bash
source /usr/local/bin/isaac-ros-entrypoint.sh

# Big headless training run
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 500
```

That trains 4096 humanoids in parallel on your GPU. Watch progress with TensorBoard
(optional; the container usually ships it):

```bash
python -m tensorboard.main --logdir /workspace/logs/rsl_rl/humanoid --port 6006
```
then open `http://localhost:6006` in your browser (host networking forwards it).

**Stop/cleanup:**
```powershell
.\launcher.ps1 kill        # stop containers
.\launcher.ps1 clean       # remove containers + volumes (WARNING: wipes workspace/logs+data)
.\launcher.ps1 logs isaac-viz   # follow viz container logs
.\launcher.ps1 logs isaac-sim   # follow sim container logs
```

## 2.5 Launching a bare scene (no training) and the ROS2 bridge

The primitives include a small launch script that starts Isaac Sim with a head and the
ROS2 bridge, but does **not** start training:

```bash
# inside a running container
source /usr/local/bin/isaac-ros-entrypoint.sh
python /workspace/primitives/ros2/launch/isaac_sim.launch.py --head humanoid
# add --no-gui for headless, or run it in the viz container to see it in the browser
```

This is useful to confirm a scene loads, and to test ROS2 topics while the sim is running.
In a second terminal inside the container you can then inspect ROS2:

```bash
source /usr/local/bin/isaac-ros-entrypoint.sh
ros2 topic list
ros2 topic echo /joint_states
```

## 2.6 Loading a URDF or USD file

> A **URDF** is the ROS XML text description of a robot. Isaac Sim does not load URDFs
> directly — it needs a **USD** (Universal Scene Description) file. So the job is:
> put your files in the project, **convert URDF → USD once**, then point a head at the USD.

### 2.6.1 Where to put the files

Copy your robot package (`.urdf` + any meshes) into a head's description folder on the
Windows host:

```
heads\my_robot\description\robot.urdf
heads\my_robot\description\meshes\...
```

Because `heads/` is mounted into the container, the files are instantly visible inside.
Put **converted USD output** under `workspace/data` so it persists and is writable:

```
workspace\data\robots\my_robot\robot.usd
```

### 2.6.2 Convert a URDF to USD

Open a shell in the sim container, then:

```bash
source /usr/local/bin/isaac-ros-entrypoint.sh

# Isaac Sim ships a standalone URDF importer example. Adjust the path if your
# version names it differently (run: ls /isaac-sim/standalone_examples/api/ | grep urdf)
/isaac-sim/python.sh standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py \
    --urdf /workspace/data/robots/my_robot/robot.urdf \
    --usd-path /workspace/data/robots/my_robot \
    --merge-mesh
```

Alternative — the same thing using the Isaac Lab converter API (most reliable on Isaac Lab 2.1):

```bash
python - <<'PY'
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

converter = UrdfConverter(
    UrdfConverterCfg(
        asset_path="/workspace/data/robots/my_robot/robot.urdf",
        usd_dir="/workspace/data/robots/my_robot",
        usd_file_name="robot.usd",
        force_usd_conversion=True,
        make_instanceable=True,          # recommended for RL (fast multi-env cloning)
    )
)
print("USD written to:", converter.usd_path)
simulation_app.close()
PY
```

Result: `workspace/data/robots/my_robot/robot.usd`.

### 2.6.3 View / inspect a USD in the browser

In the **viz** container:

```bash
source /usr/local/bin/isaac-ros-entrypoint.sh
python - <<'PY'
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import open_stage
ok, stage = open_stage("/workspace/data/robots/my_robot/robot.usd")
print("Stage open:", ok, "| prims:", len(stage.GetPrims()))

while simulation_app.is_running():
    simulation_app.update()
simulation_app.close()
PY
```

Then look at your browser tab (localhost:6080). You can orbit the camera with the middle
mouse button. Press `Ctrl+C` in the terminal to close.

### 2.6.4 Wire the USD into a head so you can train on it

The robot a head uses is defined in `heads/<name>/config/robot_cfg.py`. Open it on your
host and change the `usd_path` to your converted file:

```python
ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/data/robots/my_robot/robot.usd",   # <-- your file
        ...
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), joint_pos={".*": 0.0}),
    actuators={"body": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0)},
)
```

Notes:
- `init_state.pos` is where the robot spawns — set `z` just above the ground.
- Start with `ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=..., damping=...)`
  for a new robot; refine per-joint stiffness later.
- If the robot falls apart or explodes, it usually means bad collision geometry in the
  URDF, missing inertia, or self-collision — re-convert with `--collision-from-visuals`
  and check the actuator stiffness.
- After editing on the host: `docker compose -f docker/docker-compose.yml restart isaac-viz`
  (or `isaac-sim`) so the next env creation uses the new robot.

## 2.7 Running training (the full recipe)

Training uses Isaac Lab's built-in RSL-RL trainer against a **registered gym task**.
A head becomes trainable when its `register.py` calls `gym.register(...)` — the humanoid
head already does this (`Isaac-Humanoid-v0`). The template head has a commented example.

```bash
# 1. (in a container shell)
source /usr/local/bin/isaac-ros-entrypoint.sh

# 2. Train headless (fast, for real runs)
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-v0 --headless --num_envs 4096 --max_iterations 500

# 3. Or train with visualization (watch in browser, small num_envs)
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 16 --max_iterations 200
```

> The entrypoint puts Isaac Sim's interpreter on `PATH`, so `python` resolves to
> `/isaac-sim/kit/python/bin/python3`. The trainer scripts live under
> `/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/` — there is no
> `isaaclab_rl.rsl_rl.train` module to run with `-m`.
> For a custom head, `gym.spec("Isaac-Humanoid-v0")` must load after Kit boots; the viz
> container auto-launch imports the head's `register.py` first (see Section 2.3).

**Where the results go** (they persist in your `workspace/` folder):

```
workspace/logs/rsl_rl/humanoid/<timestamp>/            # training run
workspace/logs/rsl_rl/humanoid/<timestamp>/model_500.pt   # checkpoints
workspace/logs/rsl_rl/humanoid/<timestamp>/events.out.tfevents.*   # TensorBoard
```

**Resume a run** (add these flags):
```bash
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 \
    --resume \
    --load_run <timestamp> \
    --checkpoint model_250.pt
```

**Key knobs:**
| Flag | Meaning |
|---|---|
| `--num_envs N` | Parallel environments (4-16 for viz, 1000-8192 for headless) |
| `--max_iterations N` | Total PPO iterations (more = longer/better) |
| `--headless` | No rendering |
| `--resume --load_run <ts> --checkpoint <file>` | Continue a run |
| `--seed N` | Reproducibility |

Hyperparameters live in `heads/humanoid/config/agent_cfg.py` (learning rate, hidden dims,
entropy, clip, gamma, ...). The `rl/train.yaml` files are reference configs — the real
training reads `agent_cfg.py`.

## 2.8 Playing back a trained policy

```bash
source /usr/local/bin/isaac-ros-entrypoint.sh

# Auto-load the latest checkpoint of the run
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Humanoid-v0 --num_envs 16

# Load a specific run/checkpoint
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-v0 \
    --load_run <timestamp> \
    --checkpoint model_500.pt \
    --num_envs 16
```

Run this inside the **viz** container (no `--headless`) and watch the learned walking in
your browser. Run it with `--headless` in the sim container if you just want to evaluate.

## 2.9 Jogging the joints (manual joint control test)

This is the fastest way to sanity-check that a robot's joints respond, which direction is
positive, and whether your actuators are stiff enough. Run this in the **viz** container:

```bash
source /usr/local/bin/isaac-ros-entrypoint.sh
python - <<'PY'
import sys, math
import torch

sys.path.insert(0, "/workspace/head")

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaaclab.envs import ManagerBasedRLEnv
from config.env_cfg import EnvCfg        # the head mounted at /workspace/head

cfg = EnvCfg()
cfg.scene.num_envs = 1                    # one robot is enough to jog
env = ManagerBasedRLEnv(cfg)
env.reset()

robot = env.scene.articulations["robot"]
print("Joint names:", robot.joint_names)
print("Joint limits:\n", robot.data.joint_pos_limits[0])

n = robot.num_joints
for step in range(600):
    phase = step * 0.05
    targets = torch.zeros((1, n), device=env.device)
    for j in range(n):
        targets[0, j] = 0.5 * math.sin(phase + j)   # sweep every joint sinusoidally
    robot.set_joint_position_targets(targets)
    env.step(torch.zeros((1, env.num_actions), device=env.device))
    if step % 30 == 0:
        print(f"step {step}: pos={robot.data.joint_pos[0, :4].tolist()}")

env.close()
simulation_app.close()
PY
```

Watch the robot's joints wave back and forth in the browser tab. Each joint name is
printed, so you can change the loop to move one specific joint:

```python
idx = list(robot.joint_names).index("L_thigh_joint")   # replace with a real name
targets = torch.zeros((1, n), device=env.device)
targets[0, idx] = 0.8 * math.sin(phase)
robot.set_joint_position_targets(targets)
```

If a joint barely moves, increase its stiffness in `robot_cfg.py`. If it slams to its
limit, reduce the amplitude or check the joint limits printed above.

## 2.10 Tuning rewards

Rewards tell the agent *what* to do. They are defined as a list of terms, each with a
function, a **weight**, and optional parameters. Positive weight = the agent is rewarded;
negative weight = it is penalized.

For the humanoid head, open `heads/humanoid/config/env_cfg.py` and find `HumanoidRewardsCfg`:

```python
@configclass
class HumanoidRewardsCfg:
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    move_to_target = RewTerm(func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)})
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)          # negative = penalty
    energy = RewTerm(func=mdp.power_consumption, weight=-0.005, params={...})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits_penalty_ratio, weight=-0.25, params={...})
```

**Rules of thumb:**
- The total reward = `sum(term_weight * term_value)`. A term's value is typically in
  `[0, 1]` for bonuses or unbounded for penalties, so weight alone is not the full story.
- Start every new task with: a **progress/goal** bonus (tells it where to go), an
  **alive/upright** bonus (tells it to survive), and small **penalties** for wasting energy
  and hitting joint limits.
- Make penalties small and negative; big penalties make agents stop trying.
- If the robot collapses immediately, increase the alive/upright weights. If it walks in
  circles, raise `move_to_target`/`progress`. If it jitters, raise the `action_l2` penalty.

**Writing a custom reward term** — use the template head's `heads/template/config/reward_cfg.py`
as a pattern. A term function has the signature `func(env, **params)` and returns a tensor
with one value per environment:

```python
def upright_bonus(env, asset_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    upright = torch.prod(projected_gravity[:, :2] < threshold, dim=1)
    return upright.float()
```

Then register it in the head's reward config and give it a weight:

```python
custom_upright = RewTerm(func=upright_bonus, weight=0.2, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.93})
```

**Debugging rewards:** after training, watch the per-term reward curves in TensorBoard
(they appear as `rewards/progress`, `rewards/action_l2`, etc.). If a term is huge and
drowning out the others, lower its weight. If a term is always near zero, it is never
being triggered — check its parameters.

## 2.11 Quick reference (cheat sheet)

```powershell
# Windows host (PowerShell), from the project folder
.\launcher.ps1 build                                    # build both images
.\launcher.ps1 up -Head humanoid -Viz                   # viz container: headless training + TensorBoard
.\launcher.ps1 up -Head humanoid -Headless              # plain headless sim container
.\launcher.ps1 up -Head template                        # default GUI sim container
.\launcher.ps1 logs [isaac-sim|isaac-viz]               # follow logs
.\launcher.ps1 kill                                     # stop containers
.\launcher.ps1 clean                                    # remove containers+volumes
```

```bash
# Inside a container
source /usr/local/bin/isaac-ros-entrypoint.sh                                            # always first
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 --headless --num_envs 4096                                  # train
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Humanoid-v0 --headless --num_envs 16 \
    --video --video_interval 2000 --video_length 2000                                    # train + record clips
python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Humanoid-v0 --num_envs 16                                               # play
python -m tensorboard.main --logdir /workspace/logs --port 6006                          # metrics
python /workspace/primitives/core/eval_report.py --task Isaac-Humanoid-v0 --steps 600    # behavior report (no renderer)
ros2 topic list                                             # ROS2 sanity check
```

**Browser URLs:** TensorBoard = `http://localhost:6006` · noVNC desktop = `http://localhost:6080/vnc.html`
(the Isaac viewport only renders on native-Linux hosts) · eval report = `workspace/logs/eval/<timestamp>/report.html`
