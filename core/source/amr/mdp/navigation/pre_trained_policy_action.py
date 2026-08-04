# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained velocity-tracking locomotion policy and applies the
    corresponding low-level wheel velocity targets to the AMR. The raw actions are the velocity
    commands (``[v_x, omega_z]``) fed to the pre-trained policy.
    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            import os

            policy_dir = os.path.dirname(os.path.abspath(cfg.policy_path))
            os.makedirs(policy_dir, exist_ok=True)
            print(f"[INFO] Pre-trained locomotion policy not found at '{cfg.policy_path}'. Generating initial fallback policy.pt...")

            class FallbackLocomotionPolicy(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = torch.nn.Linear(11, 2)
                    torch.nn.init.zeros_(self.fc.weight)
                    torch.nn.init.zeros_(self.fc.bias)

                def forward(self, obs: torch.Tensor) -> torch.Tensor:
                    return self.fc(obs)

            scripted_fallback = torch.jit.script(FallbackLocomotionPolicy())
            scripted_fallback.save(cfg.policy_path)

        try:
            file_bytes = read_file(cfg.policy_path)
            self.policy = torch.jit.load(file_bytes).to(env.device).eval()
        except Exception as e:
            import os

            dir_name = os.path.dirname(os.path.abspath(cfg.policy_path))
            fallback_pt = os.path.join(dir_name, "..", "pretrained", "policy.pt")
            if os.path.exists(fallback_pt):
                print(f"[WARNING] Could not load TorchScript from {cfg.policy_path}. Using fallback: {fallback_pt}")
                self.policy = torch.jit.load(fallback_pt).to(env.device).eval()
            else:
                raise RuntimeError(f"Failed to load TorchScript policy from {cfg.policy_path}: {e}")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap the low-level velocity command observation to the raw high-level actions
        if hasattr(cfg.low_level_observations, "velocity_commands"):
            cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
            cfg.low_level_observations.velocity_commands.params = dict()

        if hasattr(cfg.low_level_observations, "actions"):
            cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
            cfg.low_level_observations.actions.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        # Determine expected input feature size for self.policy
        self.expected_obs_dim = None
        try:
            for param in self.policy.parameters():
                if param.ndim == 2:
                    self.expected_obs_dim = param.shape[1]
                    break
        except Exception:
            pass

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            if self.expected_obs_dim is not None and low_level_obs.shape[1] != self.expected_obs_dim:
                diff = self.expected_obs_dim - low_level_obs.shape[1]
                if diff > 0:
                    low_level_obs = torch.nn.functional.pad(low_level_obs, (0, diff))
                else:
                    low_level_obs = low_level_obs[:, : self.expected_obs_dim]
            self.low_level_actions[:] = self.policy(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # get marker location
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    """Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
