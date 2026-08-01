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

### 2.2 In-Scene Camera & Video Captures
Visual feedback is provided via **In-Scene `TiledCamera` sensors** and **`PeriodicVideoRecorderWrapper`** (`core/utils/video_recorder.py`).

- Saved MP4 clips: `/workspace/data/videos/`
- Live ROS2 Stream Topic: `/camera/rgb/image_raw`

View live simulation video stream:
```bash
python3 core/ros2_ws/image_listener.py
```
