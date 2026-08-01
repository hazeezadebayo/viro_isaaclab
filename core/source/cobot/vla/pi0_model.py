# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical Intelligence Pi0 / Pi0.5 Flow-Matching Vision-Language-Action (VLA) Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLanguageEncoder(nn.Module):
    """Vision-Language Encoder processing RGB image frames and text prompt embeddings."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # ConvNet Vision Backbone processing camera RGB image [B, 3, 224, 224] -> [B, 256, embed_dim]
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.vision_proj = nn.Linear(128 * 8 * 8, embed_dim)

        # Text Prompt Embedding Layer
        self.prompt_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor, prompt_embed: torch.Tensor) -> torch.Tensor:
        # image: [B, 3, H, W]
        vis_features = self.vision_backbone(image).flatten(1)
        vis_tokens = self.vision_proj(vis_features)  # [B, embed_dim]

        text_tokens = self.prompt_proj(prompt_embed)  # [B, embed_dim]

        # Concatenate Vision and Language Multimodal Tokens
        multimodal_context = vis_tokens + text_tokens  # [B, embed_dim]
        return multimodal_context


class FlowMatchingActionDecoder(nn.Module):
    """Flow Matching Continuous Normalizing Flow Action Decoder for Pi0 / Pi0.5."""

    def __init__(self, action_dim: int = 6, action_horizon: int = 16, embed_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim * action_horizon + 1, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim * action_horizon),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Predicts vector field v_t(x_t, t | context) for flow matching integration.
        - x_t: [B, action_horizon * action_dim]
        - t: [B, 1] continuous flow time in [0, 1]
        - context: [B, embed_dim] multimodal vision-language features
        """
        inputs = torch.cat([x_t, t, context], dim=-1)
        v_pred = self.net(inputs)
        return v_pred


class Pi0VLAPolicy(nn.Module):
    """Physical Intelligence Pi0 / Pi0.5 Vision-Language-Action Policy."""

    def __init__(self, action_dim: int = 6, action_horizon: int = 16, embed_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.encoder = VisionLanguageEncoder(embed_dim=embed_dim)
        self.decoder = FlowMatchingActionDecoder(action_dim=action_dim, action_horizon=action_horizon, embed_dim=embed_dim)

    def forward(self, image: torch.Tensor, prompt_embed: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        context = self.encoder(image, prompt_embed)
        v_pred = self.decoder(x_t, t, context)
        return v_pred

    @torch.no_grad()
    def sample_actions(
        self, image: torch.Tensor, prompt_embed: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Closed-loop Flow Matching ODE integration to sample action trajectory chunk."""
        B = image.shape[0]
        context = self.encoder(image, prompt_embed)

        # Start from Gaussian noise x_0 ~ N(0, I)
        x_t = torch.randn(B, self.action_horizon * self.action_dim, device=image.device)
        dt = 1.0 / num_steps

        # Euler ODE Integration from t=0 to t=1
        for step in range(num_steps):
            t = torch.full((B, 1), step * dt, device=image.device)
            v_pred = self.decoder(x_t, t, context)
            x_t = x_t + v_pred * dt

        # Reshape to [B, action_horizon, action_dim]
        actions_chunk = x_t.view(B, self.action_horizon, self.action_dim)
        return actions_chunk
