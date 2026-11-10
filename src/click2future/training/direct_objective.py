"""Remove nuisance Monte Carlo weights from the deterministic E11 objective."""

import math
from dataclasses import replace

import torch

from .contact_geometry import contact_objective

DIRECT_OBJECTIVE = "expected-clipped-lognormal-direct-loss-1"


def expected_coefficients(config):
    """Exact coefficients of the existing loss, including both clipped tails.

    The direct decoder ignores sigma and corruption. Its point, temporal,
    acceleration and stationary errors therefore have the same expectation
    after replacing the independent EDM weight by its mean. Geometry terms
    instead receive the probability of the original low-noise activation.
    """
    config.validate()
    if not config.world_prior_only or config.risk_penetration_weight != 0:
        raise ValueError("Expected loss requires a direct decoder and disabled nonlinear tail risk")
    mean, std = config.log_sigma_mean, config.log_sigma_std
    low, high = math.log(config.sigma_min), math.log(config.sigma_max)

    def cdf(value):
        return 0.5 * math.erfc(-value / math.sqrt(2))

    inverse_square = (
        cdf((low - mean) / std) / config.sigma_min**2
        + math.exp(-2 * mean + 2 * std**2)
        * (cdf((high - mean + 2 * std**2) / std) - cdf((low - mean + 2 * std**2) / std))
        + cdf((mean - high) / std) / config.sigma_max**2
    )
    probability = (
        1.0
        if config.physics_sigma_max == config.sigma_max
        else cdf((math.log(config.physics_sigma_max) - mean) / std)
    )
    return {
        "edm_weight": config.sigma_data**-2 + inverse_square,
        "physics_probability": probability,
        "equivalent_sigma": inverse_square**-0.5,
    }


def expected_direct_objective(model, clean, features, force, geometry, config):
    """The expected historical objective, without changing the sampled model.

    This is a training-only reduction of random gradient variance. It does not
    add a simulator, motion projection, rest override or new quality threshold.
    In particular, removing that variance does not guarantee better videos.
    """
    core = getattr(model, "model", model)
    if not getattr(core, "world_prior_only", False) or any(
        isinstance(layer, torch.nn.Dropout) and layer.training and layer.p != 0
        for layer in core.modules()
    ):
        raise ValueError("Expected loss needs a deterministic direct decoder")
    coefficients = expected_coefficients(config)
    sigma = clean.new_full((len(clean),), coefficients["equivalent_sigma"])
    probability = coefficients["physics_probability"]
    # Reuse the frozen loss arithmetic. The equivalent sigma gives exactly the
    # expected EDM weight; every geometry term gets its former activation rate.
    effective = replace(
        config,
        physics_sigma_max=max(config.physics_sigma_max, coefficients["equivalent_sigma"]),
        penetration_weight=config.penetration_weight * probability,
        contact_weight=config.contact_weight * probability,
        boundary_weight=config.boundary_weight * probability,
        dynamics_residual_weight=config.dynamics_residual_weight * probability,
        collision_impulse_weight=config.collision_impulse_weight * probability,
    )
    loss, parts = contact_objective(
        model, clean, features, force, sigma, torch.zeros_like(clean), geometry, effective
    )
    return loss, {
        **parts,
        "expected_edm_weight": clean.new_tensor(coefficients["edm_weight"]),
        "physics_activation_probability": clean.new_tensor(probability),
    }
