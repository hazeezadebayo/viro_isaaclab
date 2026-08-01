# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for AMR mobile robot environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base height above ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2:3]


def target_position_error_b(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float] = (5.0, 0.0, 0.0), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Position error vector to navigation goal target expressed in robot base body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_vec_w = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w.torch[:, :3]
    target_vec_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w.torch, target_vec_w)
    return target_vec_b[:, :2]  # Return 2D (x, y) displacement in base frame
