# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset Demonstrations Collector for Training VLA Models (pi0, pi0.5, SmolVLA, ACT)."""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import h5py
import torch

from isaaclab.envs import ManagerBasedRLEnv
from core.source.cobot.vla.cobot_env_cfg import CobotEnvCfg_VLA


def _capture_rgb(env: ManagerBasedRLEnv) -> np.ndarray:
    """Reads the current RGB frame [H, W, 3] uint8 from the in-scene tiled camera."""
    cam = env.scene.sensors["tiled_camera"]
    rgb = cam.data.output["rgb"]  # [num_envs, H, W, 3] uint8
    return rgb[0].cpu().numpy().astype(np.uint8)


def _heuristic_action(step: int, total_steps: int, grasp_start: float = 0.4) -> np.ndarray:
    """Scripted reach-and-grasp demo policy in the env action space (22-D: 6 arm + 16 hand).

    The arm is eased from its default posture into a forward reach posture while the
    Allegro hand closes after ``grasp_start`` fraction of the episode. This is a
    deterministic bootstrap policy for demonstration collection and can be replaced by
    teleoperation or a learned policy.
    """
    phase = step / max(1, total_steps - 1)
    progress = (1.0 - np.cos(np.pi * min(max(phase, 0.0), 1.0))) * 0.5  # eased 0 -> 1

    arm = np.zeros(6)
    arm[0] = 0.15 * progress  # shoulder_pan: slight +y turn toward the object
    arm[1] = -1.2 * progress  # shoulder_lift: bend forward
    arm[2] = 1.2 * progress   # elbow: extend arm
    arm[3] = -0.4 * progress  # wrist_1: align palm downward

    hand = np.zeros(16)
    if phase >= grasp_start:
        grasp_progress = (phase - grasp_start) / max(1e-6, 1.0 - grasp_start)
        hand[:] = min(grasp_progress * 2.0, 1.0)

    return np.clip(np.concatenate([arm, hand]), -1.0, 1.0)


def collect_vla_demonstrations(
    output_h5_path: str = "/workspace/core/data/vla/cobot_vla_dataset.h5",
    num_episodes: int = 50,
    max_steps_per_episode: int = 200,
    language_prompt: str = "reach and touch the target red object",
):
    """Collects multi-modal trajectory demonstrations (RGB camera frames, joint states, actions, text prompt).

    Saved HDF5 Dataset Schema:
      - /episode_<idx>/image: RGB camera frames [T, H, W, 3] (uint8)
      - /episode_<idx>/joint_pos: Joint position angles [T, 22] (float32)
      - /episode_<idx>/joint_vel: Joint angular rates [T, 22] (float32)
      - /episode_<idx>/actions: Applied joint action targets [T, 22] (float32)
      - /episode_<idx>.attrs["language_prompt"]: Natural language instruction string

    The action space matches the Cobot task environment: 6 arm joints + 16 Allegro hand joints.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
    print(f"[VLA Collector] Initializing IsaacLab Cobot Environment for VLA data collection...")

    env_cfg = CobotEnvCfg_VLA()
    env_cfg.scene.num_envs = 1  # Single environment for clean demonstration trajectory recording
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with h5py.File(output_h5_path, "w") as h5_file:
        print(f"[VLA Collector] Saving HDF5 VLA dataset to: {output_h5_path}")

        total_episodes_saved = 0

        for ep in range(num_episodes):
            obs, _ = env.reset()
            # Warm up the scene so the camera and sensor buffers are populated before capture.
            env.step(torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32))

            ep_group = h5_file.create_group(f"episode_{ep:04d}")

            frames_list = []
            joint_pos_list = []
            joint_vel_list = []
            actions_list = []

            print(f"[VLA Collector] Recording Episode #{ep + 1}/{num_episodes} Prompt: '{language_prompt}'...")

            for step in range(max_steps_per_episode):
                # 1. Extract RGB camera image frame from the in-scene camera sensor
                rgb_frame = _capture_rgb(env)

                # 2. Extract robot joint states (6 arm + 16 hand = 22 joints)
                robot_asset = env.scene["robot"]
                q_pos = robot_asset.data.joint_pos[0].cpu().numpy()
                q_vel = robot_asset.data.joint_vel[0].cpu().numpy()

                # 3. Compute heuristic / scripted expert action toward target
                action_np = _heuristic_action(step, max_steps_per_episode)
                action_tensor = torch.from_numpy(action_np).unsqueeze(0).to(env.device).float()

                # Record step data
                frames_list.append(rgb_frame)
                joint_pos_list.append(q_pos)
                joint_vel_list.append(q_vel)
                actions_list.append(action_np)

                # Step simulation physics
                obs, rew, term, trunc, _ = env.step(action_tensor)
                if term or trunc:
                    break

            # Save trajectory arrays to HDF5 group
            ep_group.create_dataset("image", data=np.array(frames_list, dtype=np.uint8), compression="gzip")
            ep_group.create_dataset("joint_pos", data=np.array(joint_pos_list, dtype=np.float32))
            ep_group.create_dataset("joint_vel", data=np.array(joint_vel_list, dtype=np.float32))
            ep_group.create_dataset("actions", data=np.array(actions_list, dtype=np.float32))
            ep_group.attrs["language_prompt"] = language_prompt

            total_episodes_saved += 1

        env.close()
        print(f"[VLA Collector] SUCCESS! Recorded {total_episodes_saved} episodes to '{output_h5_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect VLA Demonstrations Dataset for Cobot.")
    parser.add_argument("--output", type=str, default="/workspace/core/data/vla/cobot_vla_dataset.h5")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    collect_vla_demonstrations(output_h5_path=args.output, num_episodes=args.episodes)
