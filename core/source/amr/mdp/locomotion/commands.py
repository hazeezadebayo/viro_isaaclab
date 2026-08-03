# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command configurations for the AMR velocity-tracking locomotion task.

The AMR receives 2-DoF planar velocity commands ``[v_x, omega_z]`` expressed in its base
frame, mirroring the ROS2 ``/cmd_vel`` convention used by Nav2.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class AmrVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command configuration for the AMR differential-drive robot."""

    asset_name: str = "robot"
    heading_command: bool = False
    heading_control_stiffness: float = 0.5
    rel_standing_envs: float = 0.02
    rel_heading_envs: float = 0.0
    command_shape: tuple[int, int] = (2,)
    ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.22),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-2.84, 2.84),
    )
