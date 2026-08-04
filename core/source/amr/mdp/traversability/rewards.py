# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the AMR traversability task (pseudo-HIL).

All ground truth is read from the ROS2 occupancy grid (the synthetic world node's map),
so rewards are always consistent with the camera observation. Rewards encourage staying
on the white path (reward vs. penalty based on grid-cell distance/occupancy), making
dense progress toward the goal and reaching it.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .ros_scene import distance_to_path, get_ground_truth

#: Half-width of the white path strip (m). Wider than the robot footprint.
PATH_WIDTH = 0.32


def _robot_gt(env: ManagerBasedRLEnv) -> dict:
    """Cache and return the ROS grid ground-truth + robot-local positions."""
    gt = get_ground_truth(env.device)
    robot = env.scene["robot"]
    robot_local = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return gt, robot_local


def path_centerline_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Distances (m) from the robot to the nearest on-path grid cell center."""
    gt, robot_local = _robot_gt(env)
    return distance_to_path(
        gt["grid"], robot_local, gt["resolution"], gt["origin_x"], gt["origin_y"], gt["centers"]
    )


def on_path_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth reward for being on the path: ``exp(-0.5 * (d / std)^2)``.

    Is 1.0 when the robot is exactly on the centerline and decays with distance off it.
    """
    del asset_cfg
    dist = path_centerline_distance(env, SceneEntityCfg("robot"))
    return torch.exp(-0.5 * (dist / std) ** 2)


def off_path_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5 * PATH_WIDTH,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty that grows linearly once the robot leaves the path strip."""
    del asset_cfg
    dist = path_centerline_distance(env, SceneEntityCfg("robot"))
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