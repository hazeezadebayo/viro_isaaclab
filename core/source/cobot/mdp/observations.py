# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Cobot manipulator arm environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def end_effector_target_error(
    env: ManagerBasedEnv,
    target_pos: tuple[float, float, float] = (0.4, 0.0, 0.4),
    ee_body_name: str = "link_6",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """3D position error between cobot end-effector flange link_6 and target object position."""
    asset: Articulation = env.scene[asset_cfg.name]
    ee_body_idx, _ = asset.find_bodies(ee_body_name)
    ee_pos_w = asset.data.body_pos_w[:, ee_body_idx[0]]
    target_pos_w = torch.tensor(target_pos, device=env.device)

    return target_pos_w - ee_pos_w
