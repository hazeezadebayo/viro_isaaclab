# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    return (up_proj > threshold).float()


def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        self.potentials[env_ids] = -torch.linalg.norm(to_target_pos, ord=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]
        survived = self._env.termination_manager.time_outs[env_ids]
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = survived.float().mean().item()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.linalg.norm(to_target_pos, ord=2, dim=-1) / env.step_dt
        return self.potentials - self.prev_potentials


class joint_pos_limits_penalty_ratio(ManagerTermBase):
    """Penalty for violating joint position limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        gear_ratio: dict[str, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos,
            asset.data.soft_joint_pos_limits[..., 0],
            asset.data.soft_joint_pos_limits[..., 1],
        )
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled
        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.sum(
            torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1
        )


def joint_position_tracking_reward(
    env: ManagerBasedRLEnv,
    tracking_k: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Imitation learning reward for tracking reference motion joint positions.

    Computes: r_pos = exp( - tracking_k * sum_j (q_robot,j - q_ref,j)^2 )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q_robot = asset.data.joint_pos
    n_joints = q_robot.shape[-1]

    if hasattr(env, "motion_loader") and env.motion_loader is not None:
        q_ref, _ = env.motion_loader.get_current_frame()
    else:
        q_ref = asset.data.default_joint_pos

    if q_ref.shape[-1] > n_joints:
        q_ref = q_ref[:, :n_joints]
    elif q_ref.shape[-1] < n_joints:
        pad = torch.zeros(q_ref.shape[0], n_joints - q_ref.shape[-1], device=q_ref.device)
        q_ref = torch.cat([q_ref, pad], dim=-1)

    pos_diff = torch.sum(torch.square(q_robot - q_ref), dim=-1)
    return torch.exp(-tracking_k * pos_diff)


def joint_velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    tracking_k: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Imitation learning reward for tracking reference motion joint velocities.

    Computes: r_vel = exp( - tracking_k * sum_j (dq_robot,j - dq_ref,j)^2 )
    """
    asset: Articulation = env.scene[asset_cfg.name]
    dq_robot = asset.data.joint_vel
    n_joints = dq_robot.shape[-1]

    if hasattr(env, "motion_loader") and env.motion_loader is not None:
        _, dq_ref = env.motion_loader.get_current_frame()
    else:
        dq_ref = asset.data.default_joint_vel

    if dq_ref.shape[-1] > n_joints:
        dq_ref = dq_ref[:, :n_joints]
    elif dq_ref.shape[-1] < n_joints:
        pad = torch.zeros(dq_ref.shape[0], n_joints - dq_ref.shape[-1], device=dq_ref.device)
        dq_ref = torch.cat([dq_ref, pad], dim=-1)

    vel_diff = torch.sum(torch.square(dq_robot - dq_ref), dim=-1)
    return torch.exp(-tracking_k * vel_diff)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalty for large changes in action commands between consecutive simulation steps."""
    if env.action_manager.prev_action is None:
        return torch.zeros(env.num_envs, device=env.device)
    action_diff = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(action_diff), dim=-1)
