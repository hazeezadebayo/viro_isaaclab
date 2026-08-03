#!/usr/bin/env python3
"""Visualize the AMR traversability camera -> occupancy-mask pipeline.

Runs the ``Isaac-AMR-Traversability-v0`` environment headlessly (with ``--enable_cameras``),
drives the robot with a scripted action and dumps PNG files to ``core/logs/mask_debug`` showing:

* the raw RGB camera frame,
* the full-resolution thresholded binary (white path vs black ground),
* the low-resolution 16x12 occupancy mask that the policy actually receives.

Also prints per-frame pixel statistics so the threshold can be tuned.

Usage (inside the container):

    python scripts/debug_mask.py --num_envs 4 --steps 5
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug the AMR traversability camera mask.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--steps", type=int, default=5, help="Number of simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# tiled cameras require the offscreen renderer
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from core.source.amr.mdp.traversability.observations import camera_occupancy_mask  # noqa: E402

OUT_DIR = Path("/workspace/core/logs/mask_debug")
MASK_HEIGHT, MASK_WIDTH = 16, 12
THRESHOLD = 0.5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make("Isaac-AMR-Traversability-v0", num_envs=args_cli.num_envs)
    env.reset()
    unwrapped = env.unwrapped
    sensor_cfg = SceneEntityCfg("tiled_camera")

    for step in range(args_cli.steps):
        # scripted drive: straight ahead with a gentle steering oscillation
        t = float(step)
        actions = torch.zeros((args_cli.num_envs, 2), device=env.unwrapped.device)
        actions[:, 0] = 1.0
        actions[:, 1] = 0.4 * np.sin(t * 0.6)
        env.step(actions)

        sensor = unwrapped.scene.sensors["tiled_camera"]
        rgb = sensor.data.output["rgb"].cpu().numpy()  # (N, H, W, 3) uint8

        mask_obs = camera_occupancy_mask(
            unwrapped, sensor_cfg, mask_height=MASK_HEIGHT, mask_width=MASK_WIDTH, threshold=THRESHOLD
        )
        mask_grid = mask_obs.cpu().reshape(-1, MASK_HEIGHT, MASK_WIDTH).numpy()

        for i in range(args_cli.num_envs):
            frame = rgb[i]
            gray = frame.mean(axis=-1)
            binary = (gray > THRESHOLD * 255.0).astype(np.uint8) * 255

            cv2.imwrite(str(OUT_DIR / f"step{step:02d}_env{i}_rgb.png"), frame[:, :, ::-1])
            cv2.imwrite(
                str(OUT_DIR / f"step{step:02d}_env{i}_binary.png"),
                cv2.resize(binary, (binary.shape[1] * 8, binary.shape[0] * 8), interpolation=cv2.INTER_NEAREST),
            )
            cv2.imwrite(
                str(OUT_DIR / f"step{step:02d}_env{i}_mask.png"),
                cv2.resize(
                    (mask_grid[i] * 255).astype(np.uint8),
                    (MASK_WIDTH * 16, MASK_HEIGHT * 16),
                    interpolation=cv2.INTER_NEAREST,
                ),
            )

        print(f"[step {step}] rgb shape={rgb.shape} dtype={rgb.dtype}")
        print(f"  rgb min={rgb.min()} mean={rgb.mean():.2f} max={rgb.max()}")
        print(f"  gray white-fraction (>threshold): {(gray > THRESHOLD * 255.0).mean():.3f}")
        print(f"  mask obs min={mask_obs.min():.3f} mean={mask_obs.mean():.3f} max={mask_obs.max():.3f}")

    print(f"[INFO] Mask debug images written to {OUT_DIR}")
    simulation_app.close()


if __name__ == "__main__":
    main()
