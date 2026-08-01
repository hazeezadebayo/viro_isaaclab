# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for Cobot manipulator arm reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_proximity_reward(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.4, 0.0, 0.4),
    std: float = 0.2,
    ee_body_name: str = "link_6",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous exponential reward scoring end-effector proximity to target."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_idx, _ = asset.find_bodies(ee_body_name)
    ee_pos_w = asset.data.body_pos_w[:, ee_body_idx[0]]
    dist = torch.linalg.norm(torch.tensor(target_pos, device=env.device) - ee_pos_w, dim=-1)

    return torch.exp(-torch.square(dist / std))


def end_effector_reach_bonus(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float] = (0.4, 0.0, 0.4),
    threshold: float = 0.05,
    ee_body_name: str = "link_6",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse bonus reward when end-effector reaches within threshold distance of target."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_idx, _ = asset.find_bodies(ee_body_name)
    ee_pos_w = asset.data.body_pos_w[:, ee_body_idx[0]]
    dist = torch.linalg.norm(torch.tensor(target_pos, device=env.device) - ee_pos_w, dim=-1)

    return (dist < threshold).float()
