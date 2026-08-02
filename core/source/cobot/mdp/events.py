from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_three_objects(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    min_dist: float,
):
    """
    Vectorized reset of 3 cylindrical objects.

    Objects:
        object1
        object2
        object3

    Constraints:
        center distance >= min_dist
    """

    device = env.device
    num_envs = len(env_ids)

    objects = [
        env.scene["object1"],
        env.scene["object2"],
        env.scene["object3"],
    ]

    x_min, x_max = pose_range["x"]
    y_min, y_max = pose_range["y"]

    # ------------------------------------------------------------------
    # Sample XY positions for all environments and all objects.
    #
    # Shape:
    #   positions_xy = [num_envs, 3, 2]
    # ------------------------------------------------------------------

    positions_xy = torch.empty(
        (num_envs, 3, 2),
        device=device,
    )

    valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    while not torch.all(valid):

        invalid_idx = (~valid).nonzero(as_tuple=False).squeeze(-1)

        n_invalid = len(invalid_idx)

        positions_xy[invalid_idx, :, 0] = torch.rand(
            (n_invalid, 3),
            device=device,
        ) * (x_max - x_min) + x_min

        positions_xy[invalid_idx, :, 1] = torch.rand(
            (n_invalid, 3),
            device=device,
        ) * (y_max - y_min) + y_min

        p1 = positions_xy[:, 0]
        p2 = positions_xy[:, 1]
        p3 = positions_xy[:, 2]

        d12 = torch.linalg.norm(p1 - p2, dim=1)
        d13 = torch.linalg.norm(p1 - p3, dim=1)
        d23 = torch.linalg.norm(p2 - p3, dim=1)

        valid = (
            (d12 >= min_dist)
            & (d13 >= min_dist)
            & (d23 >= min_dist)
        )

    # ------------------------------------------------------------------
    # Apply positions
    # ------------------------------------------------------------------

    for obj_idx, obj in enumerate(objects):

        root_state = obj.data.default_root_state[env_ids].clone()

        positions = root_state[:, 0:3]

        positions[:, 0] += positions_xy[:, obj_idx, 0]
        positions[:, 1] += positions_xy[:, obj_idx, 1]

        positions += env.scene.env_origins[env_ids]

        orientations = root_state[:, 3:7]

        pose = torch.cat(
            [positions, orientations],
            dim=-1,
        )

        velocities = torch.zeros(
            (num_envs, 6),
            device=device,
        )

        obj.write_root_pose_to_sim(
            pose,
            env_ids=env_ids,
        )

        obj.write_root_velocity_to_sim(
            velocities,
            env_ids=env_ids,
        )