"""Training-only upper-tail penetration risk; never edits sampled trajectories."""

import math

import torch


def upper_tail(values, fraction, dim=-1):
    """Mean of the worst ceil(fraction*n) entries, an empirical CVaR-style loss."""
    if (
        isinstance(fraction, bool)
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
        or not values.is_floating_point()
        or not values.shape[dim]
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Tail risk needs finite floating values and a fraction in (0,1]")
    count = math.ceil(values.shape[dim] * fraction)
    return values.topk(count, dim=dim, sorted=False).values.mean(dim)


def penetration_risk(depth, eligible, *, margin_pixels=1.0, fraction=0.25):
    """Focus on the worst times and examples, using the unchanged native margin.

    Geometry has normalized image units. Squared units and the one-pixel margin
    match the existing loss. Ineligible high-noise rows contribute exactly zero.
    The selected tail uses an explicit ceiling, not a finite-sample guarantee.
    """
    if (
        depth.ndim != 2
        or eligible.shape != (len(depth),)
        or eligible.dtype != torch.bool
        or eligible.device != depth.device
        or not math.isfinite(margin_pixels)
        or margin_pixels <= 0
    ):
        raise ValueError("Provide [batch,time] depths and a boolean low-noise mask")
    excess = (depth - margin_pixels / 64).relu().square()
    per_clip = upper_tail(excess, fraction) * eligible
    return upper_tail(per_clip, fraction)
