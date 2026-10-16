"""Compact conditional video VAE baseline with a learned conditional Gaussian prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class CVAEOutput:
    reconstruction: Tensor
    posterior_mean: Tensor
    posterior_logvar: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor


def _statistics(value):
    mean, logvar = value.float().chunk(2, dim=-1)
    return mean, logvar.clamp(-10, 5)


class ConditionalVideoVAE(nn.Module):
    """Inputs are first RGB [-1,1], binary selected mask, and scaled numeric force.

    A global clip latent models variation across whole futures. Only the posterior
    sees future RGB. Inference samples the learned prior and never invokes it.
    This is a pixel baseline; it does not settle the diffusion representation choice.
    """

    def __init__(self, *, future_frames=31, base_channels=16, latent_dim=64):
        super().__init__()
        if (
            any(
                isinstance(x, bool) or not isinstance(x, int) or x < 1
                for x in (future_frames, base_channels, latent_dim)
            )
            or base_channels % 8
        ):
            raise ValueError("Use positive dimensions and a base width divisible by eight")
        self.future_frames, self.latent_dim = future_frames, latent_dim
        self.base_channels = base_channels
        c = base_channels
        condition_layers, posterior_layers = [], []
        for incoming, outgoing in ((6, c), (c, 2 * c), (2 * c, 4 * c)):
            condition_layers += [
                nn.Conv2d(incoming, outgoing, 4, stride=2, padding=1),
                nn.GroupNorm(8, outgoing),
                nn.SiLU(),
            ]
        for incoming, outgoing in ((3, c), (c, 2 * c), (2 * c, 4 * c)):
            posterior_layers += [
                nn.Conv3d(incoming, outgoing, 3, stride=(1, 2, 2), padding=1),
                nn.GroupNorm(8, outgoing),
                nn.SiLU(),
            ]
        self.condition_encoder = nn.Sequential(*condition_layers)
        self.future_encoder = nn.Sequential(*posterior_layers)
        self.prior = nn.Sequential(
            nn.Linear(16 * c, 4 * c), nn.SiLU(), nn.Linear(4 * c, 2 * latent_dim)
        )
        self.posterior = nn.Sequential(
            nn.Linear(80 * c, 8 * c), nn.SiLU(), nn.Linear(8 * c, 2 * latent_dim)
        )
        self.latent_projection = nn.Linear(latent_dim, 4 * c)
        self.decoder_stem = nn.Sequential(
            nn.Conv3d(8 * c + 2, 4 * c, 3, padding=1), nn.GroupNorm(8, 4 * c), nn.SiLU()
        )
        self.decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(incoming, outgoing, (1, 3, 3), padding=(0, 1, 1)),
                    nn.GroupNorm(8, outgoing),
                    nn.SiLU(),
                )
                for incoming, outgoing in ((4 * c, 2 * c), (2 * c, c), (c, c))
            ]
        )
        self.output = nn.Conv3d(c, 3, 3, padding=1)

    def configuration(self):
        return {
            "future_frames": self.future_frames,
            "base_channels": self.base_channels,
            "latent_dim": self.latent_dim,
        }

    def _condition(self, first_frame, selected_mask, force_xy):
        if (
            first_frame.ndim != 4
            or first_frame.shape[1] != 3
            or len(first_frame) < 1
            or min(first_frame.shape[-2:]) < 16
            or any(x % 8 for x in first_frame.shape[-2:])
        ):
            raise ValueError("First frames must be [B,3,H,W] with H,W >= 16 divisible by eight")
        b, _, h, w = first_frame.shape
        for tensor in (first_frame, selected_mask, force_xy):
            if tensor.device != first_frame.device or not torch.isfinite(tensor).all():
                raise ValueError("Condition tensors must be finite on the same device")
        if not first_frame.is_floating_point() or ((first_frame < -1) | (first_frame > 1)).any():
            raise ValueError("First RGB must be floating in [-1,1]")
        if (
            selected_mask.shape != (b, 1, h, w)
            or not ((selected_mask == 0) | (selected_mask == 1)).all()
            or not selected_mask.bool().flatten(1).any(1).all()
        ):
            raise ValueError("Selected masks must be nonempty binary [B,1,H,W]")
        if force_xy.shape != (b, 2) or not force_xy.is_floating_point():
            raise ValueError("Numeric force must be floating [B,2]")
        fields = force_xy[:, :, None, None].expand(-1, -1, h, w)
        features = self.condition_encoder(
            torch.cat((first_frame, selected_mask.float(), fields), 1)
        )
        summary = F.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
        return features, summary

    def _decode(self, features, latent):
        b, _, h, w = features.shape
        frames = self.future_frames
        condition = features[:, :, None].expand(-1, -1, frames, -1, -1)
        z = self.latent_projection(latent.to(features.dtype))[:, :, None, None, None]
        z = z.expand(-1, -1, frames, h, w)
        time = torch.arange(1, frames + 1, device=features.device, dtype=features.dtype) / frames
        time_fields = torch.stack((time, time.square()))[None, :, :, None, None]
        time_fields = time_fields.expand(b, -1, -1, h, w)
        hidden = self.decoder_stem(torch.cat((condition, z, time_fields), dim=1))
        for block in self.decoder_blocks:
            hidden = F.interpolate(hidden, scale_factor=(1, 2, 2), mode="nearest")
            hidden = block(hidden)
        return self.output(hidden).tanh()

    def forward(self, first_frame, selected_mask, force_xy, future, *, generator=None):
        features, condition = self._condition(first_frame, selected_mask, force_xy)
        if (
            future.shape != (len(first_frame), 3, self.future_frames, *first_frame.shape[-2:])
            or not future.is_floating_point()
            or future.device != first_frame.device
            or not torch.isfinite(future).all()
            or ((future < -1) | (future > 1)).any()
        ):
            raise ValueError("Future RGB must match [B,3,F,H,W] and lie in [-1,1]")
        future_features = F.adaptive_avg_pool3d(self.future_encoder(future), (4, 2, 2)).flatten(1)
        mean, logvar = _statistics(self.posterior(torch.cat((condition, future_features), dim=1)))
        prior_mean, prior_logvar = _statistics(self.prior(condition))
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        latent = mean + (0.5 * logvar).exp() * noise
        return CVAEOutput(self._decode(features, latent), mean, logvar, prior_mean, prior_logvar)

    @torch.inference_mode()
    def sample(self, first_frame, selected_mask, force_xy, *, seeds: list[int]):
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
            )
        ):
            raise ValueError("Provide distinct nonnegative integer sample seeds")
        if self.training:
            raise ValueError("Call model.eval() before sampling")
        features, condition = self._condition(first_frame, selected_mask, force_xy)
        mean, logvar = _statistics(self.prior(condition))
        samples = []
        for seed in seeds:
            generator = torch.Generator(device=first_frame.device).manual_seed(seed)
            noise = torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
            )
            samples.append(self._decode(features, mean + (0.5 * logvar).exp() * noise))
        return torch.stack(samples)  # [K,B,3,F,H,W]
