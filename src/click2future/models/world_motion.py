"""Sampled physical worlds and relational denoising, without a dynamics engine."""

import torch
from torch import nn

from .object_motion import TrajectoryDiffusion, embedding

WORLD_CONTRACT = "shared-independent-uniform-mass-friction-restitution-1"
WORLD_BOUNDS = ((0.5, 2.0), (0.1, 0.8), (0.0, 0.4))


def sample_world(seed, device="cpu"):
    """The same seed gives the same world across forces and selected masks.

    These are independent prior draws, never the hidden properties of an input
    scene. CPU generation keeps the latent draw independent of compute device.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("Use a nonnegative seed below 2**63")
    generator = torch.Generator().manual_seed(seed ^ 0x574F524C44)
    uniform = torch.rand((1, 2, 3), generator=generator)
    bounds = torch.tensor(WORLD_BOUNDS)
    return (bounds[:, 0] + uniform * (bounds[:, 1] - bounds[:, 0])).to(device)


class BoundWorld(nn.Module):
    """Bind a batch latent without mutable state on the shared denoiser."""

    def __init__(self, model, world):
        super().__init__()
        self.model, self.world = model, world
        self.future_frames, self.sigma_data = model.future_frames, model.sigma_data
        self.training = model.training

    def forward(self, noisy, sigma, features, force):
        return self.model(noisy, sigma, features, force, self.world)


class WorldTrajectoryDiffusion(TrajectoryDiffusion):
    """E7: inherited EDM with per-world context and learned pair messages.

    World labels supervise training. Sampling draws the declared prior and never
    reads a scene specification. Messages see noise-scaled relative states, not
    oracle contacts. Neither messages nor the renderer enforce valid physics.
    """

    def __init__(self, *, world_condition=True, relational=True, **kwargs):
        if not isinstance(world_condition, bool) or not isinstance(relational, bool):
            raise ValueError("world_condition must be boolean")
        super().__init__(**kwargs)
        self.world_condition, self.relational = world_condition, relational
        self.world_context = nn.Sequential(
            nn.Linear(3, self.width), nn.SiLU(), nn.Linear(self.width, self.width)
        )
        self.messages = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * self.width + 8, self.width),
                nn.SiLU(),
                nn.Linear(self.width, self.width),
            )
            for _ in self.blocks
        )
        for adapter in [self.world_context, *self.messages]:
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    def forward(self, noisy, sigma, features, force, world):
        batch = len(noisy)
        bounds = world.new_tensor(WORLD_BOUNDS)
        if (
            noisy.shape != (batch, self.future_frames, 2, 4)
            or sigma.shape != (batch,)
            or features.shape != (batch, 2, 11)
            or force.shape != (batch, 2)
            or world.shape != (batch, 2, 3)
            or any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != noisy.device
                for value in (noisy, sigma, features, force, world)
            )
            or (sigma <= 0).any()
            or (world < bounds[:, 0]).any()
            or (world > bounds[:, 1]).any()
            or not ((features[..., -1] == 0) | (features[..., -1] == 1)).all()
            or not (features[..., -1].sum(1) == 1).all()
        ):
            raise ValueError("Provide finite observable inputs and in-prior physical latents")
        scale, sd = sigma[:, None, None, None], self.sigma_data
        denominator = scale.square() + sd * sd
        scaled = noisy / denominator.sqrt()
        local_force = features[..., -1:] * force[:, None] / 0.5
        context = self.condition(torch.cat((features, local_force), -1))
        if self.world_condition:
            # Inverse mass reflects acceleration; no equation is integrated here.
            normalized = 2 * (world - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
            normalized = torch.stack(
                ((world[..., 0].reciprocal() - 1.25) / 0.75, *normalized.unbind(-1)[1:]), -1
            )
            context = context + self.world_context(normalized)
        time = embedding(torch.arange(self.future_frames, device=noisy.device), self.width)
        noise = self.noise(embedding(sigma.log() / 4, self.width))
        tokens = self.motion(scaled) + context[:, None]
        tokens = tokens + time[None, :, None] + noise[:, None, None]
        relative = scaled[..., :2].flip(2) - scaled[..., :2]
        velocity = torch.diff(relative, dim=1, prepend=relative[:, :1])
        initial = (features[:, :, :2].flip(1) - features[:, :, :2])[:, None]
        initial = initial.expand(-1, self.future_frames, -1, -1)
        centers = initial + 0.5 * relative
        half_sizes = 0.5 * (features[:, :, 2:4] + features[:, :, 2:4].flip(1))
        gaps = centers.abs() - half_sizes[:, None]
        edges = torch.cat((relative, velocity, initial, gaps), -1)
        for block, message in zip(self.blocks, self.messages, strict=True):
            tokens = block(tokens.flatten(1, 2)).unflatten(1, (self.future_frames, 2))
            sender = tokens.flip(2) if self.relational else tokens
            edge_input = edges if self.relational else torch.zeros_like(edges)
            tokens = tokens + message(torch.cat((tokens, sender, edge_input), -1))
        residual = self.output(self.norm(tokens)).reshape_as(noisy)
        return sd * sd / denominator * noisy + scale * sd / denominator.sqrt() * residual

    @torch.inference_mode()
    def sample(self, features, force, *, seeds, **kwargs):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Use distinct seeds")
        samples = []
        for seed in seeds:
            bound = BoundWorld(self, sample_world(seed, features.device))
            samples.append(
                TrajectoryDiffusion.sample(bound, features, force, seeds=[seed], **kwargs)
            )
        return torch.cat(samples)
