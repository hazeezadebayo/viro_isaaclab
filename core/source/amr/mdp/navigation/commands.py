# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for the AMR local navigation task.

Generates a goal pose relative to an obstacle (a cone), so that the robot must navigate around
the obstacle to reach the goal. An angular offset makes the goal easier or harder to reach.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPose2dCommandCfg
from isaaclab.envs.mdp.commands.pose_2d_command import UniformPose2dCommand
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ObstacleBlockedPoseCommand(UniformPose2dCommand):
    """Generates a goal pose relative to an obstacle.

    Can apply an angular offset to make the goal easier to reach (not directly behind the obstacle).
    """

    cfg: ObstacleBlockedPoseCommandCfg

    def __init__(self, cfg: ObstacleBlockedPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_cfg.name]
        self.cone = env.scene[cfg.obstacle_cfg.name]

    def _resample_command(self, env_ids: torch.Tensor):
        # 1. Get Positions
        robot_pos = self.robot.data.root_pos_w[env_ids]
        cone_pos = self.cone.data.root_pos_w[env_ids]

        # 2. Calculate Base Direction Vector (Robot -> Cone)
        vec_robot_to_cone = cone_pos[:, :2] - robot_pos[:, :2]
        dist_robot_to_cone = torch.norm(vec_robot_to_cone, dim=1, keepdim=True)
        # Normalize
        direction = vec_robot_to_cone / (dist_robot_to_cone + 1e-6)

        # 3. Apply Angular Offset (The Curriculum Part)
        angle_range = self.cfg.goal_pose_angle_range
        angle_offset = (torch.rand(len(env_ids), device=self.device) * (angle_range[1] - angle_range[0])) + angle_range[0]

        # Rotation Matrix for 2D vectors
        cos_a = torch.cos(angle_offset).unsqueeze(1)
        sin_a = torch.sin(angle_offset).unsqueeze(1)

        # Rotated direction vector
        rot_dir_x = direction[:, 0:1] * cos_a - direction[:, 1:2] * sin_a
        rot_dir_y = direction[:, 0:1] * sin_a + direction[:, 1:2] * cos_a
        rotated_direction = torch.cat([rot_dir_x, rot_dir_y], dim=1)

        # 4. Sample Goal Position
        min_d, max_d = self.cfg.goal_distance_behind_obstacle
        rand_dist = torch.rand(len(env_ids), device=self.device) * (max_d - min_d) + min_d

        # Goal = ConePos + RotatedDirection * Distance
        goal_pos = cone_pos[:, :2] + rotated_direction * rand_dist.unsqueeze(1)

        # 5. Set Command Buffers
        self.pos_command_w[env_ids, :2] = goal_pos
        self.heading_command_w[env_ids] = (torch.rand(len(env_ids), device=self.device) * 2 * torch.pi) - torch.pi


@dataclass
class ObstacleBlockedPoseCommandCfg(UniformPose2dCommandCfg):
    class_type: type = ObstacleBlockedPoseCommand

    asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot"))
    obstacle_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("cone"))

    goal_distance_behind_obstacle: tuple[float, float] = (1.0, 2.0)

    # Angle offset in radians. Start wide (easy) and tighten with the curriculum.
    goal_pose_angle_range: tuple[float, float] = (-1.507, 1.507)
