# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action Chunking with Transformers (ACT) CVAE VLA Policy Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTPolicy(nn.Module):
    """Action Chunking with Transformers (ACT) CVAE Encoder-Decoder Policy Model."""

    def __init__(
        self, action_dim: int = 6, action_horizon: int = 16, latent_dim: int = 32, hidden_dim: int = 512
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_dim = latent_dim

        # ResNet / ConvNet Vision Backbone
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.image_proj = nn.Linear(64 * 7 * 7, hidden_dim)

        # CVAE Encoder: Encodes (action_sequence, image_features, joint_pos) -> latent z (mean, logvar)
        self.cvae_encoder = nn.Sequential(
            nn.Linear(action_horizon * action_dim + hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu and logvar
        )

        # Transformer Decoder: Predicts action chunk given (image_features, joint_pos, z)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.latent_proj = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

        # Learnable Action Positional Embeddings
        self.query_embed = nn.Embedding(action_horizon, hidden_dim)

    def encode_cvae(
        self, actions: torch.Tensor, image_feat: torch.Tensor, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act_flat = actions.flatten(1)
        inputs = torch.cat([act_flat, image_feat, qpos], dim=-1)
        stats = self.cvae_encoder(inputs)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        image: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """- image: [B, 3, H, W]
        - qpos: [B, action_dim]
        - actions: Optional [B, action_horizon, action_dim] for CVAE training
        Returns: (pred_actions_chunk, mu, logvar)
        """
        B = image.shape[0]
        img_feat = self.image_proj(self.image_encoder(image).flatten(1))

        if actions is not None:
            # Training Mode with CVAE Encoder
            mu, logvar = self.encode_cvae(actions, img_feat, qpos)
            z = self.reparameterize(mu, logvar)
        else:
            # Inference Mode: Sample z ~ N(0, I)
            mu, logvar = None, None
            z = torch.zeros(B, self.latent_dim, device=image.device)

        z_qpos = torch.cat([z, qpos], dim=-1)
        memory = self.latent_proj(z_qpos).unsqueeze(1) + img_feat.unsqueeze(1)  # [B, 1, hidden_dim]

        # Target queries [B, action_horizon, hidden_dim]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Transformer decoding
        decoded = self.transformer_decoder(tgt=queries, memory=memory)
        pred_actions = self.action_out(decoded)  # [B, action_horizon, action_dim]

        return pred_actions, mu, logvar
