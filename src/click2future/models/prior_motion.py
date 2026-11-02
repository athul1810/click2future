"""E11: test a deterministic decoder after conditioning on a sampled world."""

import torch

from .world_motion import WorldTrajectoryDiffusion, sample_world

PRIOR_CONTRACT = "world-prior-only-versus-conditional-edm-1"


class PriorTrajectoryModel(WorldTrajectoryDiffusion):
    def __init__(self, *, world_prior_only, **kwargs):
        if not isinstance(world_prior_only, bool):
            raise ValueError("world_prior_only must be boolean")
        super().__init__(**kwargs)
        self.world_prior_only = world_prior_only

    def forward(self, noisy, sigma, features, force, world):
        if self.world_prior_only:
            if (
                noisy.shape != (len(features), self.future_frames, 2, 4)
                or sigma.shape != (len(features),)
                or not noisy.is_floating_point()
                or not sigma.is_floating_point()
                or not torch.isfinite(noisy).all()
                or not torch.isfinite(sigma).all()
                or (sigma <= 0).any()
                or noisy.device != features.device
                or sigma.device != features.device
            ):
                raise ValueError("Provide finite, aligned pose and positive noise tensors")
            # Same backbone; no sampled pose/noise enters this arm's decoder.
            # Its motion-input weights consequently have zero gradient.
            noisy, sigma = torch.zeros_like(noisy), torch.ones_like(sigma)
        return super().forward(noisy, sigma, features, force, world)

    @torch.inference_mode()
    def decode_world(self, features, force, world):
        """One neural evaluation, with an explicit world and no dynamics solver."""
        return self(
            features.new_zeros((len(features), self.future_frames, 2, 4)),
            features.new_ones((len(features),)),
            features,
            force,
            world,
        )

    @torch.inference_mode()
    def sample(self, features, force, *, seeds, steps=16, sigma_min=0.002, sigma_max=20.0):
        if not self.world_prior_only:
            return super().sample(
                features,
                force,
                seeds=seeds,
                steps=steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        if (
            features.shape != (1, 2, 11)
            or force.shape != (1, 2)
            or isinstance(steps, bool)
            or steps != 1
            or not 0 < sigma_min < sigma_max
            or not seeds
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Use one condition, distinct seeds and exactly one direct step")
        was_training = self.training
        self.eval()
        try:
            values = [
                self.decode_world(features, force, sample_world(seed, features.device))[0]
                for seed in seeds
            ]
            result = torch.stack(values)
            if not torch.isfinite(result).all():
                raise ValueError("Direct decoder returned nonfinite poses")
            return result
        finally:
            self.train(was_training)
