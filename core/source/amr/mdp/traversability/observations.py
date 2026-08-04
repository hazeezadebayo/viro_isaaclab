# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the AMR traversability task (pseudo-HIL over ROS2).

The camera feed comes from the ROS2 synthetic world node (``/amr/camera/rgb``) which
renders a forward-ahead view of the grid from the robot's current pose. The raw RGB
frame is converted into a low-resolution occupancy mask: thresholded to white-path /
black-ground and area-pooled down to a small grid. This gives the policy a compact,
interpretable view of *where the path is* without a CNN.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from isaaclab.utils.math import quat_rotate_inverse

from .ros_scene import get_ros_scene


def ros_camera_mask(
    env,
    mask_height: int = 16,
    mask_width: int = 12,
    threshold: float = 0.5,
    timeout: float = 5.0,
) -> torch.Tensor:
    """Thresholded, area-pooled white-path occupancy mask from the ROS2 camera.

    Publishes the robot pose to ROS2 topic (/amr/robot_pose), then reads the latest
    frame from ROS2 topic (/amr/camera/rgb) and binarizes / area-pools it.
    Returns shape ``(num_envs, mask_height * mask_width)``.

    Raises:
        RuntimeError: if no camera frame arrives within ``timeout`` seconds.
    """
    scene = get_ros_scene()
    robot = env.scene["robot"]

    xyz = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]
    yaw = torch.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
    scene.publish_pose(float(xyz[0].cpu()), float(xyz[1].cpu()), float(yaw.cpu()))

    deadline = time.time() + timeout
    rgb = None
    while time.time() < deadline:
        rgb = scene.camera_frame(env.device)
        if rgb is not None:
            break
        time.sleep(0.05)

    if rgb is None:
        raise RuntimeError(
            f"No camera frame received on ROS2 topic /amr/camera/rgb within {timeout}s. "
            "Ensure ROS2 daemon and synthetic world node are running."
        )

    gray = rgb.float().mean(dim=-1) / 255.0  # (H, W)
    binary = (gray > threshold).float()
    mask = F.interpolate(
        binary[None, None],
        size=(mask_height, mask_width),
        mode="area",
    ).squeeze(1)  # (1, mask_height, mask_width)

    return mask.flatten(1).expand(env.num_envs, -1)


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
