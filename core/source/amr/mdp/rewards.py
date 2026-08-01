# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for AMR mobile robot navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_target_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (5.0, 0.0, 0.0),
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse bonus reward when AMR reaches within threshold distance of navigation goal."""
    asset: Articulation = env.scene[asset_cfg.name]
    dist = torch.linalg.norm(torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w.torch[:, :3], dim=-1)
    return (dist < threshold).float()


def position_tracking_fine_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (5.0, 0.0, 0.0),
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous exponential reward for proximity to navigation target position."""
    asset: Articulation = env.scene[asset_cfg.name]
    dist = torch.linalg.norm(torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w.torch[:, :3], dim=-1)
    return torch.exp(-torch.square(dist / std))
