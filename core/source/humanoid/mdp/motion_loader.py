# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference motion dataset loader and trajectory manager for imitation learning."""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch


class ReferenceMotionLoader:
    """Manages loading and interpolating reference motion trajectories for batched environments.

    Loads retargeted motion capture data (JSON/NPY format containing joint_positions and joint_velocities)
    and provides batched reference joint target queries for imitation reward computation and action terms.
    """

    def __init__(
        self,
        motion_file: str,
        num_envs: int,
        device: str | torch.device = "cuda:0",
        loop: bool = True,
    ):
        self.motion_file = motion_file
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.loop = loop

        filename = os.path.basename(self.motion_file)
        if not os.path.isabs(self.motion_file):
            # Single canonical location: core/data/motion_capture. The container mount
            # (/workspace/core/data/motion_capture) and the host-side relative path both
            # resolve to the SAME directory. workspace/data is intentionally NOT consulted.
            candidate_paths = [
                os.path.join("/workspace", "core", "data", "motion_capture", filename),
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "motion_capture", filename)
                ),
            ]
            for path in candidate_paths:
                if os.path.exists(path):
                    self.motion_file = path
                    break
            else:
                self.motion_file = candidate_paths[0]

        self._load_motion()

        # Phase tracking buffer across environments
        self.env_times = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _generate_synthetic_motion(self, filepath: str) -> dict:
        """Generates a default synthetic humanoid gait motion trajectory if no motion capture file exists."""
        num_frames = 120
        fps = 60.0
        joint_names = [
            "pelvis", "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
            "torso", "left_shoulder_pitch", "left_shoulder_roll", "left_elbow",
            "right_shoulder_pitch", "right_shoulder_roll", "right_elbow"
        ]
        t = torch.linspace(0, 2 * torch.pi, num_frames)
        pos = torch.zeros((num_frames, len(joint_names)))
        pos[:, 3] = 0.2 * torch.sin(t)          # left_hip_pitch
        pos[:, 4] = 0.4 * torch.sin(t + 0.5)    # left_knee
        pos[:, 8] = -0.2 * torch.sin(t)         # right_hip_pitch
        pos[:, 9] = -0.4 * torch.sin(t + 0.5)   # right_knee
        dt = 1.0 / fps
        vel = torch.zeros_like(pos)
        vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)

        data = {
            "metadata": {"fps": fps, "num_frames": num_frames, "duration_s": num_frames / fps},
            "joint_names": joint_names,
            "joint_positions": pos.tolist(),
            "joint_velocities": vel.tolist(),
        }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        return data

    def _load_motion(self) -> None:
        """Loads motion json data into PyTorch tensors."""
        if not os.path.exists(self.motion_file):
            data = self._generate_synthetic_motion(self.motion_file)
        else:
            with open(self.motion_file, "r") as f:
                data = json.load(f)

        self.fps = float(data["metadata"].get("fps", 60.0))
        self.joint_names = data["joint_names"]
        self.num_joints = len(self.joint_names)

        positions = torch.tensor(data["joint_positions"], dtype=torch.float32, device=self.device)
        velocities = torch.tensor(data["joint_velocities"], dtype=torch.float32, device=self.device)

        self.num_frames = positions.shape[0]
        self.duration_s = (self.num_frames - 1) / self.fps if self.num_frames > 1 else 0.0

        self.joint_pos_trajectory = positions  # Shape: [N_frames, N_joints]
        self.joint_vel_trajectory = velocities  # Shape: [N_frames, N_joints]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Resets motion time phase for given environments (random initial phase or frame 0)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Initialize to uniform random frame times for trajectory diversity
        if self.duration_s > 0.0:
            self.env_times[env_ids] = torch.rand(len(env_ids), device=self.device) * self.duration_s
        else:
            self.env_times[env_ids] = 0.0

    def step(self, dt: float) -> None:
        """Advances motion time by simulation step dt."""
        self.env_times += dt
        if self.loop and self.duration_s > 0.0:
            self.env_times = torch.fmod(self.env_times, self.duration_s)

    def get_current_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets current reference joint position and velocity targets for all environments.

        Returns:
            q_ref: Reference joint positions tensor of shape [num_envs, num_joints]
            dq_ref: Reference joint velocities tensor of shape [num_envs, num_joints]
        """
        if self.duration_s <= 0.0:
            q_ref = self.joint_pos_trajectory[0].unsqueeze(0).repeat(self.num_envs, 1)
            dq_ref = self.joint_vel_trajectory[0].unsqueeze(0).repeat(self.num_envs, 1)
            return q_ref, dq_ref

        # Compute continuous frame indices for linear interpolation
        frame_idx_float = self.env_times * self.fps
        idx0 = torch.floor(frame_idx_float).long() % self.num_frames
        idx1 = (idx0 + 1) % self.num_frames
        blend = (frame_idx_float - torch.floor(frame_idx_float)).unsqueeze(-1)

        q0 = self.joint_pos_trajectory[idx0]
        q1 = self.joint_pos_trajectory[idx1]
        q_ref = (1.0 - blend) * q0 + blend * q1

        dq0 = self.joint_vel_trajectory[idx0]
        dq1 = self.joint_vel_trajectory[idx1]
        dq_ref = (1.0 - blend) * dq0 + blend * dq1

        return q_ref, dq_ref
