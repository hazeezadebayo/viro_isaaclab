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
from core.source.cobot.tasks.cobot_env_cfg import CobotEnvCfg


def collect_vla_demonstrations(
    output_h5_path: str = "/workspace/data/cobot_vla_dataset.h5",
    num_episodes: int = 50,
    max_steps_per_episode: int = 200,
    language_prompt: str = "reach and touch the target red object",
):
    """Collects multi-modal trajectory demonstrations (RGB camera frames, joint states, actions, text prompt).

    Saved HDF5 Dataset Schema:
      - /episode_idx/image: RGB camera frames [T, H, W, 3] (uint8)
      - /episode_idx/joint_pos: Joint position angles [T, 6] (float32)
      - /episode_idx/joint_vel: Joint angular rates [T, 6] (float32)
      - /episode_idx/actions: Applied joint action targets [T, 6] (float32)
      - /episode_idx/language_prompt: Natural language instruction string
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
    print(f"[VLA Collector] Initializing IsaacLab Cobot Environment for VLA data collection...")

    env_cfg = CobotEnvCfg()
    env_cfg.scene.num_envs = 1  # Single environment for clean demonstration trajectory recording
    env = ManagerBasedRLEnv(cfg=env_cfg)

    h5_file = h5py.File(output_h5_path, "w")
    print(f"[VLA Collector] Saving HDF5 VLA dataset to: {output_h5_path}")

    total_episodes_saved = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_group = h5_file.create_group(f"episode_{ep:04d}")

        frames_list = []
        joint_pos_list = []
        joint_vel_list = []
        actions_list = []

        print(f"[VLA Collector] Recording Episode #{ep + 1}/{num_episodes} Prompt: '{language_prompt}'...")

        for step in range(max_steps_per_episode):
            # 1. Extract RGB camera image frame from simulation camera
            camera_sensor = env.scene.sensors["tiled_camera"]
            rgb_frame = camera_sensor.data.output["rgb"][0].cpu().numpy()  # [H, W, 3] uint8

            # 2. Extract robot joint states
            robot_asset = env.scene["robot"]
            q_pos = robot_asset.data.joint_pos[0].cpu().numpy()  # [6]
            q_vel = robot_asset.data.joint_vel[0].cpu().numpy()  # [6]

            # 3. Compute heuristic / expert action toward target
            ee_body_idx, _ = robot_asset.find_bodies("link_6")
            ee_pos = robot_asset.data.body_pos_w[0, ee_body_idx[0]].cpu().numpy()
            target_pos = np.array([0.4, 0.0, 0.4])
            error = target_pos - ee_pos

            # Simple IK proportional controller action
            action_np = np.clip(error * 2.0, -1.0, 1.0)
            if action_np.shape[0] < 6:
                action_np = np.pad(action_np, (0, 6 - action_np.shape[0]))

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

    h5_file.close()
    env.close()
    print(f"[VLA Collector] SUCCESS! Recorded {total_episodes_saved} episodes to '{output_h5_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect VLA Demonstrations Dataset for Cobot.")
    parser.add_argument("--output", type=str, default="/workspace/data/cobot_vla_dataset.h5")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    collect_vla_demonstrations(output_h5_path=args.output, num_episodes=args.episodes)
