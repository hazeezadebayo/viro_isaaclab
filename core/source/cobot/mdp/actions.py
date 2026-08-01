# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for Cobot 6-DOF Manipulator Arm."""

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.utils import configclass


@configclass
class CobotArmActionCfg(isaac_mdp.JointPositionActionCfg):
    """Action specifications for Cobot joint position targets."""

    asset_name: str = "robot"
    joint_names: list[str] = ["joint_.*"]
    scale: float = 0.5
    use_default_offset: bool = True
