# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation events and resets for Cobot manipulator arm environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_cobot_joints_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float] = (-0.2, 0.2),
    velocity_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset cobot arm joints with random uniform offsets from default state."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    joint_pos += torch.empty_like(joint_pos).uniform_(*position_range)
    joint_vel += torch.empty_like(joint_vel).uniform_(*velocity_range)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
