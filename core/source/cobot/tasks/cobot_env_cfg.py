# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VLA-facing Cobot environment configurations.

This module provides the canonical ``CobotEnvCfg`` alias used by the VLA data-collection
and inference scripts, as well as ``CobotEnvCfg_VLA`` -- a variant that augments the RL
task environment with an in-scene RGB camera sensor for visual observations.

The RL task environments themselves live in ``joint_pos2_cobot_env_cfg`` and
``joint_pos3_cobot_env_cfg``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from core.source.cobot.tasks.joint_pos2_cobot_env_cfg import CobotUR5eCylinderLiftEnvCfg

# Canonical Cobot env config alias (used by VLA collector/inference and documentation)
CobotEnvCfg = CobotUR5eCylinderLiftEnvCfg


@configclass
class CobotUR5eCylinderLiftEnvCfg_VLA(CobotUR5eCylinderLiftEnvCfg):
    """Cobot lift environment with an in-scene RGB camera for VLA observation collection."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # In-scene camera sensor for visual observations (used only by VLA collection/inference).
        self.scene.tiled_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link/front_cam",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 0.85),
                rot=(0.95372, 0.0, 0.30071, 0.0),
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            width=640,
            height=480,
        )


# Alias matching the name referenced by the VLA scripts
CobotEnvCfg_VLA = CobotUR5eCylinderLiftEnvCfg_VLA
