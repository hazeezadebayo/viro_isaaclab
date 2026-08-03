# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Figure-8 traversability path definition for the AMR camera task.

The task world is a flat, black ground plane overlaid with a white figure-8 path built from
visual-only cuboid strips. The robot must use a camera (rendered as an occupancy mask) to stay
on the white path and reach a goal that also lies on the path.

This module defines:

* the ground-truth figure-8 centerline (used by commands, events and rewards),
* the visual white path strips that are spawned into each environment,
* helper tensors for GPU-side distance computation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg

import isaaclab.sim as sim_utils

# -----------------------------------------------------------------------------
# Path geometry
# -----------------------------------------------------------------------------

#: Half-width of the white path strip (m). Wider than the robot footprint.
PATH_WIDTH = 0.32

#: Vertical position of the path strips. They float slightly above the ground plane
#: so there is no z-fighting with the black terrain.
_PATH_STRIP_THICKNESS = 0.02
_PATH_STRIP_Z = 0.005 + _PATH_STRIP_THICKNESS / 2.0

#: Scale of the Bernoulli lemniscate (figure-8). The path spans roughly
#: ``[-2.4, 2.4] x [-1.2, 1.2]`` meters around each environment origin.
_LEMNISCATE_SCALE = 2.4

#: Number of dense samples along the centerline (for distance computation).
_NUM_CENTERLINE_POINTS = 800

#: Number of straight cuboid strips used to approximate the curved path visually.
_NUM_PATH_STRIPS = 48

#: Environment-local XY coordinates of the ground-truth path centerline.
_PATH_CENTERLINE_NP: np.ndarray = None  # type: ignore
#: Heading (rad) of the path tangent at each centerline sample.
_PATH_TANGENT_NP: np.ndarray = None  # type: ignore
#: Same centerline as a CPU torch tensor, moved to the device by callers.
_PATH_CENTERLINE_TORCH: torch.Tensor = None  # type: ignore


def _build_centerline() -> tuple[np.ndarray, np.ndarray]:
    """Compute the figure-8 centerline and its tangent headings."""
    t = np.linspace(0.0, 2.0 * math.pi, _NUM_CENTERLINE_POINTS, endpoint=False)
    denom = 1.0 + np.sin(t) ** 2
    x = _LEMNISCATE_SCALE * np.cos(t) / denom
    y = _LEMNISCATE_SCALE * np.sin(t) * np.cos(t) / denom
    centerline = np.stack([x, y], axis=-1)  # (M, 2)

    # Tangent heading of each segment (wrap-around to close the loop).
    tangent = np.empty(_NUM_CENTERLINE_POINTS)
    for i in range(_NUM_CENTERLINE_POINTS):
        p0 = centerline[i]
        p1 = centerline[(i + 1) % _NUM_CENTERLINE_POINTS]
        tangent[i] = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    return centerline, tangent


_PATH_CENTERLINE_NP, _PATH_TANGENT_NP = _build_centerline()
_PATH_CENTERLINE_TORCH = torch.as_tensor(_PATH_CENTERLINE_NP, dtype=torch.float32)


def path_centerline(device: str) -> torch.Tensor:
    """Return the path centerline as a (M, 2) tensor on ``device``."""
    return _PATH_CENTERLINE_TORCH.to(device)


def build_path_assets() -> list[AssetBaseCfg]:
    """Create the visual-only white path strips for one environment.

    The strips are straight cuboids chained along the centerline. They have no collision
    shape, so the robot drives over them freely; they exist purely for the camera.
    """
    assets: list[AssetBaseCfg] = []

    # Anchor the strips along the dense centerline at regular arc-length steps. Each strip
    # reaches halfway into both of its neighbors so the white path is visually continuous
    # even where the figure-8 curves tightly.
    points = _PATH_CENTERLINE_NP
    seg_edges = np.linspace(0, len(points), _NUM_PATH_STRIPS + 1, dtype=int)
    chords = []
    for i in range(_NUM_PATH_STRIPS):
        p0, p1 = points[seg_edges[i]], points[seg_edges[i + 1] % len(points)]
        chords.append(float(np.linalg.norm(p1 - p0)))
    chords = np.array(chords)

    for i in range(_NUM_PATH_STRIPS):
        p0, p1 = points[seg_edges[i]], points[seg_edges[i + 1] % len(points)]
        mid = (p0 + p1) / 2.0
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        # extend into both neighbors so adjacent strips always overlap
        prev_len = chords[(i - 1) % _NUM_PATH_STRIPS]
        next_len = chords[(i + 1) % _NUM_PATH_STRIPS]
        length = 0.5 * prev_len + chords[i] + 0.5 * next_len + 0.02

        assets.append(
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/path_seg_{i:02d}",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(float(mid[0]), float(mid[1]), _PATH_STRIP_Z),
                    rot=(float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0))),
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(length, PATH_WIDTH, _PATH_STRIP_THICKNESS),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
                ),
            )
        )

    return assets
