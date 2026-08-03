# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the AMR traversability task.

Rewards encourage staying on the white path (camera mask is the observation; these rewards use
the ground-truth centerline distance), making dense progress toward the goal and reaching it.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .myPathTerrainCfg import PATH_WIDTH, path_centerline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def path_centerline_distance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ground-truth distance (m) from the robot to the nearest path centerline point."""
    asset = env.scene[asset_cfg.name]
    robot_local = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    centerline = path_centerline(env.device)  # (M, 2)
    dist = torch.norm(robot_local.unsqueeze(1) - centerline.unsqueeze(0), dim=2)
    return dist.min(dim=1).values


def on_path_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth reward for being on the path: ``exp(-0.5 * (d / std)^2)``.

    Is 1.0 when the robot is exactly on the centerline and decays with distance off it.
    """
    dist = path_centerline_distance(env, asset_cfg)
    return torch.exp(-0.5 * (dist / std) ** 2)


def off_path_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5 * PATH_WIDTH,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty that grows linearly once the robot leaves the path strip."""
    dist = path_centerline_distance(env, asset_cfg)
    return torch.clamp(dist - threshold, min=0.0)


def path_goal_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reward equal to the reduction in distance to the goal since the last step."""
    asset = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)

    dist_to_goal = torch.norm(asset.data.root_pos_w[:, :2] - cmd_term.pos_command_w[:, :2], dim=1)
    progress = cmd_term.dist_to_goal_prev - dist_to_goal
    cmd_term.dist_to_goal_prev = dist_to_goal

    return progress


def goal_reached_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary reward for getting within ``threshold`` meters of the goal."""
    asset = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    dist_to_goal = torch.norm(asset.data.root_pos_w[:, :2] - cmd_term.pos_command_w[:, :2], dim=1)
    return (dist_to_goal < threshold).float()
