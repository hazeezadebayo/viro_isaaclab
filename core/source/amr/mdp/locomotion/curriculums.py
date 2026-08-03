# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the AMR velocity-tracking locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg, RewardTermCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: "ManagerBasedRLEnv", env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the AMR drove when commanded to move at a desired velocity.

    The terrain difficulty increases when the robot covers more than half of the terrain size and
    decreases when it travels less than half of the distance implied by the commanded velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def increase_reward_weight_over_time(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_name: str,
    increase: float,
    episodes_per_increase: int = 1,
    max_increases: int = 100000,
) -> torch.Tensor:
    """Increase the weight of a reward term after every ``episodes_per_increase`` episodes.

    The weight mutation is applied once per episode boundary through the reward manager API.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    num_increases = num_episodes // episodes_per_increase

    if num_increases > max_increases:
        return torch.tensor(0.0, device=env.device)

    if env.common_step_counter % env.max_episode_length != 0:
        return torch.tensor(0.0, device=env.device)

    if (num_episodes + 1) % episodes_per_increase == 0:
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight += increase
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)

    return torch.tensor(0.0, device=env.device)
