# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the AMR traversability task (pseudo-HIL)."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from .ros_scene import get_ground_truth, path_tangent_at


def reset_base_on_path(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_noise: float = 0.02,
    yaw_noise: float = 0.1,
):
    """Reset the robot onto a random point of the white figure-8 path.

    The robot spawns on a random on-path cell center of the ROS occupancy grid, facing
    along the local path tangent with small noise, with zero linear and angular velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_state = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    gt = get_ground_truth(env.device)
    centers = gt["centers"]  # (M, 2)

    idx = torch.randint(0, centers.shape[0], (n,), device=env.device)
    pos_local = centers[idx]  # (n, 2)

    heading = path_tangent_at(
        gt["grid"], pos_local, gt["resolution"], gt["origin_x"], gt["origin_y"], gt["centers"]
    )
    heading = heading + (torch.rand(n, device=env.device) - 0.5) * 2.0 * yaw_noise

    env_origins = env.scene.env_origins[env_ids]
    root_state[:, 0] = env_origins[:, 0] + pos_local[:, 0] + (torch.rand(n, device=env.device) - 0.5) * 2.0 * position_noise
    root_state[:, 1] = env_origins[:, 1] + pos_local[:, 1] + (torch.rand(n, device=env.device) - 0.5) * 2.0 * position_noise
    root_state[:, 2] = env_origins[:, 2]

    # quaternion from yaw about z (w, x, y, z)
    root_state[:, 3] = torch.cos(heading / 2.0)
    root_state[:, 4] = 0.0
    root_state[:, 5] = 0.0
    root_state[:, 6] = torch.sin(heading / 2.0)

    root_state[:, 7:] = 0.0

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
