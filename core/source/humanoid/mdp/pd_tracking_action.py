# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom PD Tracking Action term for humanoid motion imitation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


class PDTrackingAction(ActionTerm):
    """Action term that computes effort using a PD controller against a reference pose.

    The policy outputs residual position offsets. The target joint position is computed as:
        q_target = q_ref + actions * action_scale

    The torque applied to the joint motors is given by:
        tau = Kp * (q_target - q_robot) - Kd * dq_robot
    """

    cfg: "PDTrackingActionCfg"

    def __init__(self, cfg: "PDTrackingActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._asset: Articulation = self._env.scene[self.cfg.asset_name]

        # Resolve joints governed by this action term
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Buffers for raw actions, target positions, and computed torques
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_pos = torch.zeros_like(self._raw_actions)

        # Parse Kp gain
        if isinstance(cfg.kp, (float, int)):
            self._kp = torch.full((self.num_envs, self.action_dim), float(cfg.kp), device=self.device)
        elif isinstance(cfg.kp, dict):
            self._kp = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.kp, self._joint_names)
            self._kp[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported type for kp: {type(cfg.kp)}")

        # Parse Kd gain
        if isinstance(cfg.kd, (float, int)):
            self._kd = torch.full((self.num_envs, self.action_dim), float(cfg.kd), device=self.device)
        elif isinstance(cfg.kd, dict):
            self._kd = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.kd, self._joint_names)
            self._kd[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported type for kd: {type(cfg.kd)}")

        # Parse action scale
        if isinstance(cfg.action_scale, (float, int)):
            self._scale = float(cfg.action_scale)
        elif isinstance(cfg.action_scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.action_scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported type for action_scale: {type(cfg.action_scale)}")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # 1. Fetch q_ref from environment motion loader if attached, otherwise use default joint pos
        if hasattr(self._env, "motion_loader") and self._env.motion_loader is not None:
            q_ref, _ = self._env.motion_loader.get_current_frame()
        else:
            q_ref = self._asset.data.default_joint_pos

        # Align q_ref columns to the full joint count before slicing to joint_ids
        n_full = self._asset.num_joints
        if q_ref.shape[-1] > n_full:
            q_ref = q_ref[:, :n_full]
        elif q_ref.shape[-1] < n_full:
            pad = torch.zeros(q_ref.shape[0], n_full - q_ref.shape[-1], device=q_ref.device)
            q_ref = torch.cat([q_ref, pad], dim=-1)

        # Slice to only the joints this action term controls
        if not isinstance(self._joint_ids, slice):
            q_ref = q_ref[:, self._joint_ids]

        # 2. Compute target position with residual policy offset
        self._target_pos = q_ref + (self._raw_actions * self._scale)

        # 3. Read current joint state
        current_q = self._asset.data.joint_pos[:, self._joint_ids]
        current_dq = self._asset.data.joint_vel[:, self._joint_ids]

        # 4. Calculate PD Torque: tau = Kp * (q_target - q) - Kd * dq
        tau = self._kp * (self._target_pos - current_q) - self._kd * current_dq

        # 5. Optional effort clipping
        if self.cfg.clip_effort is not None:
            tau = torch.clamp(tau, min=-self.cfg.clip_effort, max=self.cfg.clip_effort)

        self._processed_actions = tau

    def apply_actions(self):
        """Applies computed torques directly to articulation joints."""
        self._asset.set_joint_effort_target(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class PDTrackingActionCfg(ActionTermCfg):
    """Configuration for custom PD tracking action term."""

    # class_type must reference PDTrackingAction AFTER it is defined above.
    class_type: type = PDTrackingAction

    asset_name: str = "robot"
    """Name of the articulation asset in the scene. Defaults to 'robot'."""

    joint_names: list[str] = [".*"]
    """Joint names or regex patterns to target with this action term."""

    kp: float | dict[str, float] = 100.0
    """Proportional gain (stiffness) for PD controller."""

    kd: float | dict[str, float] = 10.0
    """Derivative gain (damping) for PD controller."""

    action_scale: float | dict[str, float] = 0.25
    """Scaling factor applied to the policy residual action offset."""

    clip_effort: float | None = None
    """Optional maximum effort threshold for clipping computed torques."""
