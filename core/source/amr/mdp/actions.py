# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for Autonomous Mobile Robot (AMR / Differential Drive)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Configuration for differential drive velocity action term (linear v, angular w -> wheel velocities)."""

    class_type: type = None

    asset_name: str = "robot"
    left_wheel_name: str = "wheel_left_joint"
    right_wheel_name: str = "wheel_right_joint"
    wheel_radius: float = 0.033
    wheel_base: float = 0.160
    max_wheel_vel: float = 15.0


class DifferentialDriveAction(ActionTerm):
    """Translates [v_x, omega_z] twist commands into differential wheel velocity targets."""

    cfg: DifferentialDriveActionCfg

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._asset: Articulation = self._env.scene[self.cfg.asset_name]
        self._wheel_ids, _ = self._asset.find_joints([self.cfg.left_wheel_name, self.cfg.right_wheel_name])

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 2, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2  # [linear velocity v, angular velocity w]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        v = self._raw_actions[:, 0]
        w = self._raw_actions[:, 1]

        r = self.cfg.wheel_radius
        b = self.cfg.wheel_base

        # Inverse Kinematics for Differential Drive:
        # v_left  = (v - w * b / 2) / r
        # v_right = (v + w * b / 2) / r
        w_left = (v - w * (b / 2.0)) / r
        w_right = (v + w * (b / 2.0)) / r

        wheel_vels = torch.stack([w_left, w_right], dim=-1)
        wheel_vels = torch.clamp(wheel_vels, min=-self.cfg.max_wheel_vel, max=self.cfg.max_wheel_vel)

        self._processed_actions = wheel_vels

    def apply_actions(self):
        self._asset.set_joint_velocity_target_index(target=self._processed_actions, joint_ids=self._wheel_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


DifferentialDriveActionCfg.class_type = DifferentialDriveAction
