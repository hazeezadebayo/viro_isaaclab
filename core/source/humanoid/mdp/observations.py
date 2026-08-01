# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


def base_yaw_roll(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Yaw and roll of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

    return torch.cat((yaw.unsqueeze(-1), roll.unsqueeze(-1)), dim=-1)


def base_up_proj(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b.torch

    return base_up_vec[:, 2].unsqueeze(-1)


def base_heading_proj(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Projection of the base forward vector onto the world forward vector."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w.torch[:, :3]
    to_target_pos = torch.cat((to_target_pos[:, :2], torch.zeros_like(to_target_pos[:, 2:3])), dim=-1)
    to_target_dir = math_utils.normalize(to_target_pos)
    # compute base forward vector
    heading_vec = math_utils.quat_apply(asset.data.root_quat_w.torch, asset.data.FORWARD_VEC_B.torch)
    # compute dot product between heading and target direction
    heading_proj = torch.bmm(heading_vec.view(env.num_envs, 1, 3), to_target_dir.view(env.num_envs, 3, 1))

    return heading_proj.view(env.num_envs, 1)


def base_angle_to_target(
    env: ManagerBasedEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Angle between the base forward vector and the vector to the target."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute desired heading direction
    to_target_pos = torch.tensor(target_pos, device=env.device) - asset.data.root_pos_w.torch[:, :3]
    walk_target_angle = torch.atan2(to_target_pos[:, 1], to_target_pos[:, 0])
    # compute base forward vector
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    # normalize angle to target to [-pi, pi]
    angle_to_target = walk_target_angle - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    return angle_to_target.unsqueeze(-1)


def joint_pos_ref_error(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Observation term for joint position reference tracking error (q_robot - q_ref)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "motion_loader") and env.motion_loader is not None:
        q_ref, _ = env.motion_loader.get_current_frame()
    else:
        q_ref = asset.data.default_joint_pos.torch

    q_robot = asset.data.joint_pos.torch
    if q_ref.shape != q_robot.shape:
        q_ref = q_ref[:, : q_robot.shape[-1]]

    return q_robot - q_ref


def joint_vel_ref_error(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Observation term for joint velocity reference tracking error (dq_robot - dq_ref)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(env, "motion_loader") and env.motion_loader is not None:
        _, dq_ref = env.motion_loader.get_current_frame()
    else:
        dq_ref = asset.data.default_joint_vel.torch

    dq_robot = asset.data.joint_vel.torch
    if dq_ref.shape != dq_robot.shape:
        dq_ref = dq_ref[:, : dq_robot.shape[-1]]

    return dq_robot - dq_ref


def motion_phase(env: ManagerBasedEnv) -> torch.Tensor:
    """Observation term for normalized reference motion phase (sin(phase), cos(phase))."""
    if hasattr(env, "motion_loader") and env.motion_loader is not None and env.motion_loader.duration_s > 0.0:
        phase = (env.motion_loader.env_times / env.motion_loader.duration_s) * 2.0 * torch.pi
        sin_phase = torch.sin(phase).unsqueeze(-1)
        cos_phase = torch.cos(phase).unsqueeze(-1)
        return torch.cat((sin_phase, cos_phase), dim=-1)
    else:
        return torch.zeros(env.num_envs, 2, device=env.device)

