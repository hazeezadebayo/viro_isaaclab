# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def lift_while_grasping(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    lifting_height: float,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    
    condition = object.data.root_pos_w[:, 2] > minimal_height
    
    height_reward = torch.clamp(
        (object.data.root_pos_w[:, 2] - minimal_height) / lifting_height, #
        min=0.0,
        max=1.0
    )

    contact_index: ContactSensor = env.scene["contact_index"]
    contact_middle: ContactSensor = env.scene["contact_middle"]
    contact_ring: ContactSensor = env.scene["contact_ring"]
    contact_thumb: ContactSensor = env.scene["contact_thumb"]

    thumb  = contact_active(contact_thumb, threshold)
    index  = contact_active(contact_index, threshold)
    middle = contact_active(contact_middle, threshold)
    ring   = contact_active(contact_ring, threshold)
    
    return condition.float() * height_reward * (thumb + index + middle + ring)

def cube_is_moved(
    env: ManagerBasedRLEnv,
    lift_threshold: float,
    vel_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward (penalty) for cube moving in x or y direction from it's previous position"""
    object: RigidObject = env.scene[object_cfg.name]

    vel_xy = object.data.root_lin_vel_w[:, :2]
    reward = torch.clamp(vel_xy - vel_threshold, min=0.0)
    height = object.data.root_pos_w[:, 2]

    reward = torch.norm(vel_xy, dim=1) * (height < lift_threshold).float()

    return reward

def contact_active(sensor: ContactSensor, threshold: float):

    force = (
        sensor.data.force_matrix_w
        .norm(dim=-1)
        .squeeze(-1)
        .squeeze(-1)
    )

    return (force > threshold).float()

def grasp_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
) -> torch.Tensor:

    contact_index: ContactSensor = env.scene["contact_index"]
    contact_middle: ContactSensor = env.scene["contact_middle"]
    contact_ring: ContactSensor = env.scene["contact_ring"]
    contact_thumb: ContactSensor = env.scene["contact_thumb"]

    thumb  = contact_active(contact_thumb, threshold)
    index  = contact_active(contact_index, threshold)
    middle = contact_active(contact_middle, threshold)
    ring   = contact_active(contact_ring, threshold)

    #num_fingers = index + middle + ring

    #thumb touching cube
    #AND
    #at least two other fingers touching cube
    #grasp = (thumb > 0.5) & (num_fingers >= 2)
    #return grasp.float()
    return thumb + index + middle + ring

def palm_contact_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
) -> torch.Tensor:

    contact_palm: ContactSensor = env.scene["contact_palm"]

    palm  = contact_active(contact_palm, threshold)

    return palm

def thumb_opposition_reward2(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]

    cube_pos = object.data.root_pos_w

    thumb_tip = env.scene["thumb_tip_frame"].data.target_pos_w[..., 0, :]
    index_tip = env.scene["index_tip_frame"].data.target_pos_w[..., 0, :]
    middle_tip = env.scene["middle_tip_frame"].data.target_pos_w[..., 0, :]
    ring_tip = env.scene["ring_tip_frame"].data.target_pos_w[..., 0, :]

    d_thumb = torch.norm(cube_pos - thumb_tip, dim=1)
    d_index = torch.norm(cube_pos - index_tip, dim=1)
    d_middle = torch.norm(cube_pos - middle_tip, dim=1)    
    d_ring = torch.norm(cube_pos - ring_tip, dim=1) 

    r_index = 1 - torch.tanh(d_index/std)
    r_middle = 1 - torch.tanh(d_middle/std)
    r_ring = 1 - torch.tanh(d_ring/std)   
    thumb_reward = 1 - torch.tanh(d_thumb / std)
    
    finger_reward = (r_index + r_middle + r_ring) / 3

    return 0.6 * thumb_reward + 0.4 * finger_reward

def thumb_opposition_reward_ee(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos = object.data.root_pos_w
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    thumb_tip = env.scene["thumb_tip_frame"].data.target_pos_w[..., 0, :]

    index_tip = env.scene["index_tip_frame"].data.target_pos_w[..., 0, :]
    middle_tip = env.scene["middle_tip_frame"].data.target_pos_w[..., 0, :]
    ring_tip = env.scene["ring_tip_frame"].data.target_pos_w[..., 0, :]

    d_thumb = torch.norm(cube_pos - thumb_tip, dim=1)

    d_fingers = (
        torch.norm(cube_pos - index_tip, dim=1)
        + torch.norm(cube_pos - middle_tip, dim=1)
        + torch.norm(cube_pos - ring_tip, dim=1)
    ) / 3.0

    thumb_reward = 1 - torch.tanh(d_thumb / std)
    finger_reward = 1 - torch.tanh(d_fingers / std)

    return 0.5 * (thumb_reward + finger_reward) * (1 - torch.tanh(object_ee_distance / std))

def finger_closure_reward(
    env,
    std: float,
    asset_cfg=SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):

    robot = env.scene[asset_cfg.name]

    finger_ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, finger_ids]

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return torch.mean(torch.abs(q), dim=1) * (1 - torch.tanh(object_ee_distance / std))

def finger_palm_closure_reward(
    env,
    threshold: float,
    asset_cfg=SceneEntityCfg("robot"),
):

    robot = env.scene[asset_cfg.name]

    finger_ids = asset_cfg.joint_ids

    q = robot.data.joint_pos[:, finger_ids]

    contact_palm: ContactSensor = env.scene["contact_palm"]

    palm  = contact_active(contact_palm, threshold)

    return torch.mean(torch.abs(q), dim=1) * palm
