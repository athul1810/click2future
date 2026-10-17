"""Optional training-only foreground weighting, shared by both video model families."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F


def validate_foreground_weight(weight):
    if (
        isinstance(weight, bool)
        or not isinstance(weight, Real)
        or not math.isfinite(weight)
        or not 1 <= weight <= 64
    ):
        raise ValueError("foreground_weight must be finite and between 1 and 64")


def validate_foreground_margin(margin):
    if isinstance(margin, bool) or not isinstance(margin, int) or not 0 <= margin <= 8:
        raise ValueError("foreground_margin must be an integer between 0 and 8 pixels")


def future_foreground(batch, *, device, margin=0):
    """Union valid future instance masks for loss targets, never model conditioning."""
    validate_foreground_margin(margin)
    masks = torch.as_tensor(batch["instance_masks"], device=device)
    valid = torch.as_tensor(batch["object_valid"], device=device)
    if (
        masks.ndim != 5
        or min(masks.shape) < 1
        or masks.shape[1] < 2
        or masks.dtype != torch.bool
        or valid.dtype != torch.bool
        or valid.shape != (masks.shape[0], masks.shape[2])
        or not valid.any(1).all()
        or (masks & ~valid[:, None, :, None, None]).any()
    ):
        raise ValueError("Loss masks must be boolean [B,T,N,H,W] with valid object slots")
    union = masks[:, 1:].any(dim=2)
    if margin:
        # Pool spatial dimensions only; no dilation across future frames or clips.
        union = F.max_pool2d(union.float(), 2 * margin + 1, stride=1, padding=margin).bool()
    return union.unsqueeze(1)  # [B,1,F,H,W], excludes frame zero


def pixel_mean(error, *, foreground=None, foreground_weight=1.0):
    """Return one normalized loss per clip; weight 1 preserves the original reduction.

    Weights sum to one within each clip, including the channel dimension. Changing
    object area does not change a clip's batch weight. Empty/full foreground masks
    remain finite. Future masks supervise the loss only and are not inference inputs.
    """
    validate_foreground_weight(foreground_weight)
    if error.ndim != 5 or min(error.shape) < 1 or not error.is_floating_point():
        raise ValueError("Pixel errors must be floating [B,C,F,H,W]")
    if error.dtype in {torch.float16, torch.bfloat16}:
        error = error.float()
    if foreground is not None and (
        foreground.shape != (len(error), 1, *error.shape[2:])
        or foreground.device != error.device
        or foreground.dtype != torch.bool
    ):
        raise ValueError("Foreground must be boolean [B,1,F,H,W] on the loss device")
    if foreground_weight == 1:
        return error.flatten(1).mean(1)
    if foreground is None:
        raise ValueError("Foreground weighting requires explicit future loss masks")
    weights = 1 + (foreground_weight - 1) * foreground.to(error.dtype)
    return (error * weights).flatten(1).sum(1) / (error.shape[1] * weights.flatten(1).sum(1))
