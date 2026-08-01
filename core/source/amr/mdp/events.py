# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation events and resets for AMR mobile robot environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_amr_position_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset AMR root pose within random uniform range."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    root_state = asset.data.default_root_state[env_ids].clone()
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range.get("x", (0.0, 0.0)))
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range.get("y", (0.0, 0.0)))
    root_state[:, 0] += dx
    root_state[:, 1] += dy

    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
