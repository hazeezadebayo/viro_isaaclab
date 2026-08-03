# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the AMR traversability task.

The camera feed is converted into a low-resolution occupancy mask: the RGB frame is
thresholded to white-path / black-ground and area-pooled down to a small grid. This gives the
policy a compact, interpretable view of *where the path is* without a CNN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse


def camera_occupancy_mask(
    env,
    sensor_cfg: SceneEntityCfg,
    mask_height: int = 16,
    mask_width: int = 12,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Thresholded, area-pooled white-path occupancy mask from the RGB camera.

    The raw RGB frame is converted to grayscale, binarized against ``threshold`` and
    average-pooled to ``(mask_height, mask_width)``. Each mask cell therefore encodes the
    fraction of white path visible in that image region. Returns shape ``(num_envs,
    mask_height * mask_width)``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    rgb = sensor.data.output["rgb"]  # (num_envs, H, W, 3) uint8

    gray = rgb.float().mean(dim=-1) / 255.0  # (num_envs, H, W)
    binary = (gray > threshold).float()  # 1 -> white path, 0 -> black ground

    mask = F.interpolate(
        binary.unsqueeze(1),
        size=(mask_height, mask_width),
        mode="area",
    ).squeeze(1)  # (num_envs, mask_height, mask_width)

    return mask.flatten(1)


def goal_in_base(env, command_name: str) -> torch.Tensor:
    """Goal position in the robot's base frame (3-D world offset rotated into base).

    The command term places goals on the path in world coordinates; this observation gives the
    policy the goal relative to itself so it can steer toward it.
    """
    cmd_term = env.command_manager.get_term(command_name)
    goal_w = cmd_term.pos_command_w

    robot = env.scene["robot"]
    vec_w = goal_w - robot.data.root_pos_w
    return quat_rotate_inverse(robot.data.root_quat_w, vec_w)
