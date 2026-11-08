"""A learned force response whose zero-action displacement is exactly zero."""

import torch

from .prior_motion import PriorTrajectoryModel

REST_ANCHOR_CONTRACT = "shared-neural-force-minus-zero-response-1"


class RestAnchoredPriorModel(PriorTrajectoryModel):
    backbone_evaluations = 2

    def forward(self, noisy, sigma, features, force, world):
        # Train and sample the same two neural evaluations. This represents
        # action response relative to rest, rather than correcting a video.
        response = super().forward(noisy, sigma, features, force, world)
        rest = super().forward(noisy, sigma, features, torch.zeros_like(force), world)
        return response - rest
