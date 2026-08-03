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
    max_wheel_vel: float = 11.0
    max_lin_vel: float = 0.22
    max_ang_vel: float = 2.84
    bounding_strategy: str = "clip"
    no_reverse: bool = False


class DifferentialDriveAction(ActionTerm):
    """Translates normalized ``[v_x, omega_z]`` twist commands into differential wheel velocity targets.

    The raw actions are normalized to ``[-1, 1]``. They are scaled by ``max_lin_vel`` (m/s) and
    ``max_ang_vel`` (rad/s) before the differential-drive inverse kinematics map them to wheel
    angular velocity targets (rad/s)::

        v = a_v * max_lin_vel
        w = a_w * max_ang_vel
        w_left  = (v - w * wheel_base / 2) / wheel_radius
        w_right = (v + w * wheel_base / 2) / wheel_radius

    A positive wheel target rotates the wheel forward (about -y_body), producing forward motion
    along the base +x axis. When ``no_reverse`` is enabled the forward velocity is clamped to
    non-negative values, which is useful for curriculum stages or car-like driving.
    """

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

        # normalize the raw actions to the robot's twist limits
        v = self._raw_actions[:, 0] * self.cfg.max_lin_vel
        w = self._raw_actions[:, 1] * self.cfg.max_ang_vel

        if self.cfg.no_reverse:
            v = torch.clamp(v, min=0.0)

        r = self.cfg.wheel_radius
        b = self.cfg.wheel_base

        # Inverse Kinematics for Differential Drive:
        # v_left  = (v - w * b / 2) / r
        # v_right = (v + w * b / 2) / r
        w_left = (v - w * (b / 2.0)) / r
        w_right = (v + w * (b / 2.0)) / r

        wheel_vels = torch.stack([w_left, w_right], dim=-1)

        if self.cfg.bounding_strategy == "clip":
            wheel_vels = torch.clamp(wheel_vels, min=-self.cfg.max_wheel_vel, max=self.cfg.max_wheel_vel)
        elif self.cfg.bounding_strategy == "tanh":
            wheel_vels = torch.tanh(wheel_vels / self.cfg.max_wheel_vel) * self.cfg.max_wheel_vel
        else:
            raise ValueError(f"Unsupported bounding strategy: {self.cfg.bounding_strategy}")

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
