# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain configuration for the AMR locomotion task.

Sub-terrains are chosen to be traversable by a differential-drive robot (0.105 m wheelbase):
flat patches, gentle height-field noise, gentle pyramid slopes and low boxes.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

VELOCITY_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(14.0, 14.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.35), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.4, grid_height_range=(0.02, 0.06), platform_width=2.0
        ),
    },
)
