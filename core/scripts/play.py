#!/usr/bin/env python3
"""Play (evaluate) a trained policy inside the container using Isaac Lab's canonical RSL-RL runner.

This is a thin wrapper around ``/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py``.
It must be executed inside the container (``/workspace/core`` is mounted and the tasks in
``core/source/register_tasks.py`` are auto-registered on startup via ``sitecustomize.py``).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _find_rsl_rl_dir() -> Path:
    candidates = [
        Path("/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl"),
        Path("/workspace/IsaacLab/scripts/reinforcement_learning/rsl_rl"),
    ]
    for candidate in candidates:
        if (candidate / "play.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate Isaac Lab RSL-RL scripts. Expected /workspace/isaaclab or /workspace/IsaacLab."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a trained policy inside the container.")
    parser.add_argument("--task", type=str, required=True, help="Registered gym task ID.")
    parser.add_argument("--num_envs", type=int, help="Number of parallel environments.")
    parser.add_argument("--checkpoint", type=str, help="Trained policy checkpoint .pt file.")
    parser.add_argument("--experiment_name", type=str, help="Log directory name under core/logs/rsl_rl/.")
    parser.add_argument("--logger", type=str, choices=["tensorboard", "wandb", "neptune"], help="Logging backend.")
    parser.add_argument("--device", type=str, help="PyTorch device (cuda:0 or cpu).")
    parser.add_argument("--real-time", dest="real_time", action="store_true", help="Run in real time.")
    parser.add_argument("--video", action="store_true", help="Record periodic MP4 clips during play.")
    parser.add_argument("--video_length", type=int, help="Video clip length in simulation steps.")
    parser.add_argument("--video_interval", type=int, help="Sim steps between video clip starts.")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable offscreen cameras for video capture.")
    return parser


def main() -> None:
    args, extra = _build_parser().parse_known_args()

    script = _find_rsl_rl_dir() / "play.py"
    cmd = [sys.executable, str(script), "--task", args.task]
    for flag in ("num_envs", "checkpoint", "experiment_name", "logger", "device", "video_length", "video_interval"):
        value = getattr(args, flag)
        if value is not None:
            cmd += [f"--{flag}", str(value)]
    if args.real_time:
        cmd.append("--real-time")
    if args.video:
        cmd.append("--video")
    if args.enable_cameras:
        cmd.append("--enable_cameras")
    cmd += extra

    print(f"[INFO] Playing task '{args.task}' via {script}")
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
