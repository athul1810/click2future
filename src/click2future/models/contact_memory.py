"""Temporal residual denoising around a frozen sampled-world model (E8)."""

import torch
from torch import nn

from .object_motion import embedding
from .world_motion import WORLD_BOUNDS, WorldTrajectoryDiffusion

MEMORY_CONTRACT = "frozen-world-backbone-reset-versus-carried-gru-residual-1"


class ContactMemoryDiffusion(nn.Module):
    """Carry or reset identical GRU weights; neither mode enforces contact.

    Memory is causal within the residual. The frozen denoiser still attends to
    the entire noisy future, so this is not an end-to-end causal simulator.
    Every call starts from zero state: worlds, actions and seeds cannot leak
    memory into one another. Initial residuals are exactly zero.
    """

    def __init__(self, *, temporal_memory=True, memory_width=64, **backbone_config):
        super().__init__()
        if (
            not isinstance(temporal_memory, bool)
            or isinstance(memory_width, bool)
            or not isinstance(memory_width, int)
            or not 8 <= memory_width <= 256
            or backbone_config.get("world_condition") is not True
            or backbone_config.get("relational") is not True
        ):
            raise ValueError("Use a conditioned relational backbone and bounded memory width")
        self.backbone = WorldTrajectoryDiffusion(**backbone_config).requires_grad_(False).eval()
        self.temporal_memory, self.memory_width = temporal_memory, memory_width
        self.future_frames, self.sigma_data = self.backbone.future_frames, self.backbone.sigma_data
        self.world_condition = self.relational = True
        self.input = nn.Sequential(nn.Linear(51, memory_width), nn.SiLU())
        self.memory = nn.GRU(memory_width, memory_width, batch_first=True)
        self.output = nn.Linear(memory_width, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def residual(self, motion, sigma, features, force, world):
        """Predict a learned correction; no future labels or collision solver."""
        batch, frames = motion.shape[:2]
        velocity = torch.diff(motion, dim=1, prepend=torch.zeros_like(motion[:, :1]))
        centers = features[:, None, :, :2] + 0.5 * motion[..., :2]
        relative = centers.flip(2) - centers
        half_sizes = 0.5 * (features[:, :, 2:4] + features[:, :, 2:4].flip(1))
        gaps = relative.abs() - half_sizes[:, None]
        bounds = world.new_tensor(WORLD_BOUNDS)
        normalized = 2 * (world - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
        local_force = features[..., -1:] * force[:, None] / 0.5
        context = torch.cat((features, local_force, normalized, normalized.flip(1)), -1)
        noise = embedding(sigma.log() / 4, 16)
        inputs = torch.cat(
            (
                motion,
                velocity,
                motion.flip(2),
                relative,
                gaps,
                context[:, None].expand(-1, frames, -1, -1),
                noise[:, None, None].expand(-1, frames, 2, -1),
            ),
            -1,
        )
        tokens = self.input(inputs).transpose(1, 2).reshape(batch * 2, frames, -1)
        if self.temporal_memory:
            encoded, _ = self.memory(tokens)
        else:
            # Each frame gets a fresh zero state, with the same active parameters.
            encoded, _ = self.memory(tokens.reshape(batch * 2 * frames, 1, -1))
            encoded = encoded.reshape(batch * 2, frames, -1)
        correction = self.output(encoded).reshape(batch, 2, frames, 4).transpose(1, 2)
        scale = sigma[:, None, None, None]
        return scale * self.sigma_data / (scale.square() + self.sigma_data**2).sqrt() * correction

    def forward(self, noisy, sigma, features, force, world):
        with torch.no_grad():
            base = self.backbone(noisy, sigma, features, force, world)
        return base + self.residual(base, sigma, features, force, world)

    @torch.inference_mode()
    def sample(self, features, force, *, seeds, **kwargs):
        return WorldTrajectoryDiffusion.sample(self, features, force, seeds=seeds, **kwargs)
