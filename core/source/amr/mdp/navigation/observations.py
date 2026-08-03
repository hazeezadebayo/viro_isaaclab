# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the AMR local navigation task."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse


def cone_position_b(env, robot_cfg: SceneEntityCfg, cone_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation: the position of the cone relative to the robot's base frame."""
    robot = env.scene[robot_cfg.name]
    cone = env.scene[cone_cfg.name]

    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    cone_pos = cone.data.root_pos_w

    vec_w = cone_pos - robot_pos
    vec_b = quat_rotate_inverse(robot_quat, vec_w)

    return vec_b
