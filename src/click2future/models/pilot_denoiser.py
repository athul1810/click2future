"""Compact pixel-space video denoiser for the proposed E1-E4 pilot.

This is a forward/backward-tested architecture prototype, not a trained video generator.
All modes use identical backbone dimensions. Numeric action fields are broadcast over
the image; spatial action fields are localized by the same selected-object mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DenoiserOutput:
    predicted_noise: Tensor
    positions: Tensor | None = None  # [B,T_future,M,2], frame-0 canonical object order
    collision_logits: Tensor | None = None  # [B,T_future,M,M], onset intervals, same order


def _time_embedding(timesteps: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / (half - 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _positions_from_heatmaps(logits: Tensor) -> Tensor:
    """Return normalized (x,y) moments for heatmaps shaped [B,N,T,H,W]."""
    height, width = logits.shape[-2:]
    probabilities = logits.float().flatten(-2).softmax(-1)
    x = torch.linspace(0, 1, width, device=logits.device) if width > 1 else logits.new_tensor([0.5])
    y = (
        torch.linspace(0, 1, height, device=logits.device)
        if height > 1
        else logits.new_tensor([0.5])
    )
    grid_y, grid_x = torch.meshgrid(y.float(), x.float(), indexing="ij")
    horizontal = (probabilities * grid_x.flatten()).sum(-1)
    vertical = (probabilities * grid_y.flatten()).sum(-1)
    return torch.stack((horizontal, vertical), dim=-1).permute(0, 2, 1, 3)


class _SpaceTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embedding_dimension: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.spatial = nn.Conv3d(in_channels, out_channels, (1, 3, 3), padding=(0, 1, 1))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(embedding_dimension, out_channels * 2)
        self.temporal = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, (3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, (1, 3, 3), padding=(0, 1, 1)),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, features: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.spatial(F.silu(self.norm1(features)))
        scale, bias = self.film(F.silu(embedding)).chunk(2, dim=-1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None, None])
        hidden = hidden + bias[:, :, None, None, None]
        return self.shortcut(features) + self.temporal(F.silu(hidden))


class PilotDenoiser(nn.Module):
    """Predict noise for `[B,3,T_future,H,W]` using frame-0 pixels/mask and action.

    `first_frame`: `[B,3,H,W]`, normalized consistently with future frames.
    `selected_mask`: `[B,1,H,W]`; `force_xy`: normalized `[B,2]`.
    `timesteps`: one diffusion timestep per clip, `[B]`.
    Auxiliary outputs are training predictions, not additional conditioning inputs.
    When using training.auxiliary, head slots follow frame-0 mask-centre x/y order;
    the label adapter handles simulator IDs without passing masks/labels into this model.
    """

    def __init__(
        self,
        base_channels: int = 32,
        conditioning: str = "spatial",
        auxiliary: bool = False,
        max_objects: int = 4,
    ):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8, at least 8")
        if conditioning not in {"first_frame", "numeric", "spatial"}:
            raise ValueError("conditioning must be first_frame, numeric, or spatial")
        if max_objects < 2:
            raise ValueError("max_objects must be at least 2")
        self.conditioning = conditioning
        self.max_objects = max_objects
        embedding_dimension = base_channels * 4
        self.time_projection = nn.Sequential(
            nn.Linear(128, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.action_projection = nn.Sequential(
            nn.Linear(2, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.stem = nn.Conv3d(9, base_channels, (1, 3, 3), padding=(0, 1, 1))
        self.encoder1 = _SpaceTimeBlock(base_channels, base_channels, embedding_dimension)
        self.down1 = nn.Conv3d(
            base_channels, base_channels * 2, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.encoder2 = _SpaceTimeBlock(base_channels * 2, base_channels * 2, embedding_dimension)
        self.down2 = nn.Conv3d(
            base_channels * 2,
            base_channels * 4,
            (1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.bottleneck = _SpaceTimeBlock(base_channels * 4, base_channels * 4, embedding_dimension)
        self.attention_norm = nn.LayerNorm(base_channels * 4)
        self.temporal_attention = nn.MultiheadAttention(base_channels * 4, 4, batch_first=True)
        self.decoder2 = _SpaceTimeBlock(base_channels * 6, base_channels * 2, embedding_dimension)
        self.decoder1 = _SpaceTimeBlock(base_channels * 3, base_channels, embedding_dimension)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv3d(base_channels, 3, (1, 3, 3), padding=(0, 1, 1))
        self.position_head = nn.Conv3d(base_channels * 4, max_objects, 1) if auxiliary else None
        self.collision_head = (
            nn.Linear(base_channels * 4, max_objects * max_objects) if auxiliary else None
        )

    def forward(
        self,
        noisy_future: Tensor,
        timesteps: Tensor,
        first_frame: Tensor,
        selected_mask: Tensor,
        force_xy: Tensor,
    ) -> DenoiserOutput:
        if noisy_future.ndim != 5 or noisy_future.shape[1] != 3:
            raise ValueError("noisy_future must have shape [B,3,T,H,W]")
        batch, _, frames, height, width = noisy_future.shape
        if batch < 1 or frames < 1 or min(height, width) < 4 or height % 4 or width % 4:
            raise ValueError(
                "Nonempty batch, at least one future frame, and positive spatial dimensions "
                "divisible by 4 needed"
            )
        expected = (
            (first_frame, (batch, 3, height, width), "first_frame"),
            (selected_mask, (batch, 1, height, width), "selected_mask"),
            (force_xy, (batch, 2), "force_xy"),
            (timesteps, (batch,), "timesteps"),
        )
        for value, shape, name in expected:
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

        mask = selected_mask.to(dtype=noisy_future.dtype)
        action = force_xy.to(dtype=noisy_future.dtype)
        if self.conditioning == "first_frame":
            mask = torch.zeros_like(mask)
            action = torch.zeros_like(action)
        fields = action[:, :, None, None].expand(-1, -1, height, width)
        if self.conditioning == "spatial":
            fields = fields * mask
        conditioning = torch.cat((first_frame, mask, fields), dim=1)
        conditioning = conditioning[:, :, None].expand(-1, -1, frames, -1, -1)
        embedding = self.time_projection(_time_embedding(timesteps, 128))
        embedding = embedding + self.action_projection(action)

        first = self.encoder1(self.stem(torch.cat((noisy_future, conditioning), dim=1)), embedding)
        second = self.encoder2(self.down1(first), embedding)
        hidden = self.bottleneck(self.down2(second), embedding)
        _, channels, _, low_height, low_width = hidden.shape
        tokens = hidden.permute(0, 3, 4, 2, 1).reshape(-1, frames, channels)
        positions = torch.arange(frames, device=tokens.device)
        tokens_with_time = tokens + _time_embedding(positions, channels)[None].to(tokens.dtype)
        normalized = self.attention_norm(tokens_with_time)
        attended, _ = self.temporal_attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended
        hidden = tokens.reshape(batch, low_height, low_width, frames, channels).permute(
            0, 4, 3, 1, 2
        )

        predicted_positions = collision_logits = None
        if self.position_head is not None:
            pooled = hidden.mean(dim=(-2, -1)).transpose(1, 2)
            predicted_positions = _positions_from_heatmaps(self.position_head(hidden))
            collision_logits = self.collision_head(pooled).reshape(
                batch, frames, self.max_objects, self.max_objects
            )
            collision_logits = (collision_logits + collision_logits.transpose(-1, -2)) * 0.5

        up = F.interpolate(hidden, size=second.shape[2:], mode="nearest")
        up = self.decoder2(torch.cat((up, second), dim=1), embedding)
        up = F.interpolate(up, size=first.shape[2:], mode="nearest")
        up = self.decoder1(torch.cat((up, first), dim=1), embedding)
        predicted_noise = self.output(F.silu(self.output_norm(up)))
        return DenoiserOutput(predicted_noise, predicted_positions, collision_logits)
