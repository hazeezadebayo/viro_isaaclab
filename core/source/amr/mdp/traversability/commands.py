# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the AMR traversability task (pseudo-HIL).

Generates a goal pose that lies on the white figure-8 path (sampled from the ROS2
occupancy grid), at least ``min_goal_distance`` meters away from the robot's current
position. The robot must use the camera mask to stay on the path while driving toward
the goal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand

from .ros_scene import get_ground_truth

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PathGoalCommand(UniformPose2dCommand):
    """Goal generator that samples goal positions on the white figure-8 path.

    The goal is a world-frame 2-D position sampled uniformly from the on-path cells of
    the ROS occupancy grid. The previous distance to the goal is tracked so reward terms
    can compute dense progress.
    """

    cfg: PathGoalCommandCfg

    def __init__(self, cfg: PathGoalCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self.robot = env.scene[cfg.asset_name]
        # distance (m) to the current goal from the previous step; read/written by the
        # goal-progress reward term
        self.dist_to_goal_prev = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor):
        gt = get_ground_truth(self.device)
        centers = gt["centers"]  # (M, 2) on-path cell centers in local coords
        robot_pos = self.robot.data.root_pos_w[env_ids]
        env_origins = self._env.scene.env_origins[env_ids]

        robot_local = robot_pos[:, :2] - env_origins[:, :2]  # (n, 2)
        dist_to_path = torch.norm(robot_local.unsqueeze(1) - centers.unsqueeze(0), dim=2)  # (n, M)

        # candidate goal cells must be far enough from the current position
        valid = dist_to_path > self.cfg.min_goal_distance
        if not valid.any():
            valid = torch.ones_like(valid, dtype=torch.bool)
        else:
            # rows with no valid point fall back to the whole path
            valid = valid | (valid.sum(dim=1) == 0).unsqueeze(1)

        # randomly pick one valid index per environment (uniform over the valid subset)
        n = len(env_ids)
        counts = valid.sum(dim=1)  # (n,)
        rank = (torch.rand(n, device=self.device) * counts).long().clamp(max=counts - 1)
        cum = torch.cumsum(valid.float(), dim=1)
        choices = torch.argmax((cum > rank.unsqueeze(1)).float(), dim=1)  # (n,)

        goal_local = centers[choices]  # (n, 2)

        self.pos_command_w[env_ids, 0] = env_origins[:, 0] + goal_local[:, 0]
        self.pos_command_w[env_ids, 1] = env_origins[:, 1] + goal_local[:, 1]
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        # point the heading command along the straight line to the goal
        delta = goal_local - robot_local
        self.heading_command_w[env_ids] = torch.atan2(delta[:, 1], delta[:, 0])

        # record the initial distance for dense progress rewards
        rows = torch.arange(n, device=self.device)
        self.dist_to_goal_prev[env_ids] = dist_to_path[rows, choices]


@dataclass
class PathGoalCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the on-path goal generator."""

    class_type: type = PathGoalCommand

    asset_name: str = "robot"
    simple_heading: bool = False

    #: Minimum straight-line distance (m) between the robot and a newly sampled goal.
    min_goal_distance: float = 1.0

    #: Ranges are not used for sampling (goals always lie on the path), but are required
    #: by the base configuration. They bound the path extent.
    ranges: UniformPose2dCommandCfg.Ranges = field(
        default_factory=lambda: UniformPose2dCommandCfg.Ranges(
            pos_x=(-2.4, 2.4),
            pos_y=(-1.2, 1.2),
            heading=(-math.pi, math.pi),
        )
    )
