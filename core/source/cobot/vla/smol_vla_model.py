# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SmolVLA: Lightweight Vision-Language-Action Policy Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmolVLAPolicy(nn.Module):
    """SmolVLA Policy combining lightweight Vision-Language backbone with Action MLP head."""

    def __init__(self, action_dim: int = 6, action_horizon: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Lightweight CNN Feature Extractor
        self.vision_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Text Prompt Projection
        self.text_net = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

        # Multimodal Action Fusion Head
        self.action_head = nn.Sequential(
            nn.Linear(64 * 4 * 4 + hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_horizon * action_dim),
        )

    def forward(self, image: torch.Tensor, prompt_embed: torch.Tensor) -> torch.Tensor:
        """- image: [B, 3, H, W]
        - prompt_embed: [B, 256]
        Returns: predicted action chunk [B, action_horizon, action_dim]
        """
        B = image.shape[0]
        v_feat = self.vision_net(image).flatten(1)  # [B, 1024]
        t_feat = self.text_net(prompt_embed)        # [B, hidden_dim]

        fused = torch.cat([v_feat, t_feat], dim=-1)
        action_flat = self.action_head(fused)
        actions = action_flat.view(B, self.action_horizon, self.action_dim)
        return actions
