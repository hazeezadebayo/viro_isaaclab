# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import ContactSensor
#from isaaclab.sensors import FrameTransformerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b

def contact_active(sensor: ContactSensor, threshold: float):

    force = (
        sensor.data.force_matrix_w
        .norm(dim=-1)
        .squeeze(-1)
        .squeeze(-1)
    )

    return (force > threshold).float()

def fingertip_contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:

    contact_index: ContactSensor = env.scene["contact_index"]
    contact_middle: ContactSensor = env.scene["contact_middle"]
    contact_ring: ContactSensor = env.scene["contact_ring"]
    contact_thumb: ContactSensor = env.scene["contact_thumb"]

    thumb  = contact_active(contact_thumb, threshold)
    index  = contact_active(contact_index, threshold)
    middle = contact_active(contact_middle, threshold)
    ring   = contact_active(contact_ring, threshold)

    #print(index_sensor.data.force_matrix_w.shape)
    #print(index_sensor.data.force_matrix_w)
    #print("net =", index_sensor.data.net_forces_w.norm(dim=-1))
    #print("filtered =", index_sensor.data.force_matrix_w.norm(dim=-1))

    return torch.stack(
        [thumb, index, middle, ring],
        dim=-1,
    )

def fingertip_cube_vectors(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]

    cube_pos = object.data.root_pos_w

    index_tip = env.scene["index_tip_frame"].data.target_pos_w[..., 0, :]
    middle_tip = env.scene["middle_tip_frame"].data.target_pos_w[..., 0, :]
    ring_tip = env.scene["ring_tip_frame"].data.target_pos_w[..., 0, :]
    thumb_tip = env.scene["thumb_tip_frame"].data.target_pos_w[..., 0, :]

    index_vec = cube_pos - index_tip
    middle_vec = cube_pos - middle_tip
    ring_vec = cube_pos - ring_tip
    thumb_vec = cube_pos - thumb_tip

    return torch.cat(
        [
            index_vec,
            middle_vec,
            ring_vec,
            thumb_vec,
        ],
        dim=-1,
    )
