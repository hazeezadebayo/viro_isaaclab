# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for AMR mobile robot environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate episode when maximum steps buffer is reached."""
    return env.episode_length_buf >= env.max_episode_length
