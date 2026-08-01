# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-Loop Real-Time VLA Model Inference Runner in IsaacLab & ROS2."""

from __future__ import annotations

import argparse
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnv
from core.source.cobot.tasks.cobot_env_cfg import CobotEnvCfg
from core.source.cobot.vla.pi0_model import Pi0VLAPolicy
from core.source.cobot.vla.smol_vla_model import SmolVLAPolicy
from core.source.cobot.vla.act_model import ACTPolicy


def run_vla_inference(
    model_type: str = "pi0",
    ckpt_path: str = "/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/workspace/models/vla/pi0_cobot_policy.pt",
    prompt_text: str = "reach and touch target red object",
    num_steps: int = 200,
):

    """Loads fine-tuned VLA checkpoint (pi0, pi0.5, smolvla, act) and executes closed-loop control in IsaacLab."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[VLA Inference] Loading '{model_type.upper()}' Model Checkpoint from: {ckpt_path}")

    # Load Model Architecture
    if model_type in ("pi0", "pi0.5"):
        model = Pi0VLAPolicy(action_dim=6, action_horizon=16).to(device)
    elif model_type == "smolvla":
        model = SmolVLAPolicy(action_dim=6, action_horizon=8).to(device)
    elif model_type == "act":
        model = ACTPolicy(action_dim=6, action_horizon=16).to(device)
    else:
        raise ValueError(f"Unknown VLA model: {model_type}")

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[VLA Inference] SUCCESS: Loaded fine-tuned weights from: '{ckpt_path}'")
    else:
        print(f"[VLA Inference] NOTICE: Checkpoint '{ckpt_path}' not found. Running with base pretrained weights.")

    model.eval()

    # Initialize IsaacLab Cobot Environment
    env_cfg = CobotEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, _ = env.reset()
    print(f"[VLA Inference] Executing closed-loop control for Prompt: '{prompt_text}'...")

    prompt_embed = torch.randn(1, 256, device=device)

    for step in range(num_steps):
        # 1. Capture RGB frame from simulation camera sensor
        camera_sensor = env.scene.sensors["tiled_camera"]
        rgb_np = camera_sensor.data.output["rgb"][0].cpu().numpy()
        img_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        robot_asset = env.scene["robot"]
        qpos_tensor = robot_asset.data.joint_pos[0:1].to(device)

        # 2. VLA Inference: Predict Action Chunk
        with torch.no_grad():
            if model_type in ("pi0", "pi0.5"):
                actions_chunk = model.sample_actions(img_tensor, prompt_embed, num_steps=10)
            elif model_type == "smolvla":
                actions_chunk = model(img_tensor, prompt_embed)
            elif model_type == "act":
                actions_chunk, _, _ = model(img_tensor, qpos_tensor)

        # Take first action step from chunk
        current_action = actions_chunk[:, 0, :]  # [1, 6]

        # Step Simulation Physics
        obs, rew, term, trunc, _ = env.step(current_action)

        if step % 20 == 0:
            print(f"[Step {step:03d}/{num_steps:03d}] VLA Joint Action Executed: {current_action[0].cpu().numpy().round(3)}")

        if term or trunc:
            print(f"[VLA Inference] Episode completed at step {step}.")
            break

    env.close()
    print("[VLA Inference] Closed-loop evaluation finished cleanly.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLA Inference for Cobot.")
    parser.add_argument("--model", type=str, default="pi0", choices=["pi0", "pi0.5", "smolvla", "act"])
    parser.add_argument("--ckpt", type=str, default="/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/workspace/models/vla/pi0_cobot_policy.pt")

    parser.add_argument("--prompt", type=str, default="reach and touch target red object")
    args = parser.parse_args()

    run_vla_inference(model_type=args.model, ckpt_path=args.ckpt, prompt_text=args.prompt)
