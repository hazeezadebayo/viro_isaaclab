# Isaac RL Studio - Build & Usage Guide

This guide covers:
1. **Docker Build** — how the container image is built and how caching works.
2. **Headless Simulation Execution** — running training, loading URDFs, viewing video capture, and ROS2 integration.

---

## 1. Docker Build

This project uses a **layered Docker build strategy** for optimal caching:

```
┌─────────────────────────────────────┐
│  Layer 5: Core App & Systems        │  <-- Changes most often
├─────────────────────────────────────┤
│  Layer 4: Entrypoint Script         │
├─────────────────────────────────────┤
│  Layer 3: ROS2 Environment & Bridge │
├─────────────────────────────────────┤
│  Layer 2: Isaac Lab verification    │  <-- Cached as long as base image doesn't change
├─────────────────────────────────────┤
│  Layer 1: Base system utilities     │
├─────────────────────────────────────┤
│  Layer 0: nvcr.io/nvidia/isaac-lab  │  <-- 7GB cached by Docker
└─────────────────────────────────────┘
```

### Build Command
```powershell
.\launcher.ps1 build
# or
docker compose -f docker/docker-compose.yml build isaac-sim
```

This builds `isaac-rl-studio:2.1.0-humble` — the main Isaac Lab + ROS2 image.

---

## 2. Usage Guide

### 2.1 Starting the Headless Simulation Container
```powershell
# Select target head: humanoid | anymal | amr | cobot
.\launcher.ps1 up -Head humanoid -Headless
```

### 2.2 Headless Video Capture
Headless runs capture video using IsaacLab's **built-in `--video` recording** (recommended):

- Pass `--video --video_length <steps> --video_interval <steps> --enable_cameras` to `train.py`/`play.py`.
- `--video_length` — clip length in **simulation steps** (humanoid step_dt = 1/60 s → 3600 steps = 1 minute).
- `--video_interval` — simulation steps between clip **starts** (humanoid → 108000 steps = 30 minutes).
- A 1-minute clip is recorded immediately at simulation start, then a new clip every interval.
- Clips are saved to `<cwd>/logs/rsl_rl/<experiment>/<run>/videos/{train,play}/`.

Run from the host so output persists to `core/logs/`:
```powershell
.\launcher.ps1 train -Head humanoid                    # 1-min clip at start, then every 30 min
.\launcher.ps1 train -Head humanoid -VideoLengthMin 2 -VideoIntervalMin 15
```

The optional `PeriodicVideoRecorderWrapper` (`core/utils/video_recorder.py`) provides MP4 clips
(moviepy/ffmpeg) plus live ROS2 streaming for scripted/custom runners:
- MP4 clips: `/workspace/core/logs/videos/sim_clip_*.mp4`
- Live ROS2 stream topic: `/camera/rgb/image_raw`
- View live stream: `python3 core/ros2_ws/image_listener.py`

> **Known limitation (Docker Desktop):** `--video` frames are rendered through the NVIDIA Vulkan
> driver, which Docker Desktop's WSL2 GPU passthrough does **not** expose (only CUDA/compute libs
> are passed; `vkCreateInstance` fails with `ERROR_INCOMPATIBLE_DRIVER`). No MP4 clips are produced
> under Docker Desktop. Use **TensorBoard** for training visualization instead. To obtain real policy
> videos, run the stack in a native WSL2 distro with nvidia-container-toolkit configured for the
> `graphics` capability.

### 2.3 TensorBoard (reproducible compose service)
A `tensorboard` service is declared in `docker-compose.yml` and starts automatically with
`docker compose up -d` (or standalone). It serves `core/logs/rsl_rl` at http://localhost:6006:
```powershell
docker compose -f docker/docker-compose.yml up -d tensorboard
# open http://localhost:6006
```
The `launcher.ps1 up`/`train`/`play` commands start only the `isaac-sim` service; start TensorBoard
explicitly when needed.
