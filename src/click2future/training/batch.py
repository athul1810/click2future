"""Convert the NumPy loader contract to model inputs without hidden physics or labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VideoBatch:
    first_frame: Tensor  # [B,3,H,W], [-1,1]
    selected_mask: Tensor  # [B,1,H,W], float 0/1
    force_xy: Tensor  # [B,2], simulator force / force_scale, never clipped
    future: Tensor  # [B,3,F,H,W], [-1,1], training targets only


def prepare_video_batch(batch: dict, *, device="cpu", force_scale=0.5) -> VideoBatch:
    if isinstance(force_scale, bool) or not math.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("force_scale must be finite and positive")
    frames = torch.as_tensor(batch["frames"], device=device)
    mask = torch.as_tensor(batch["selected_mask"], device=device)
    force = torch.as_tensor(batch["force_xy"], device=device)
    if (
        frames.ndim != 5
        or frames.shape[2] != 3
        or frames.shape[0] < 1
        or frames.shape[1] < 2
        or min(frames.shape[-2:]) < 4
    ):
        raise ValueError("Loader frames must be floating [B,T,3,H,W], T >= 2")
    b, _, _, h, w = frames.shape
    duration = torch.as_tensor(batch["force_duration"], device=device, dtype=torch.float64)
    timestamps = torch.as_tensor(batch["timestamps"], device=device, dtype=torch.float64)
    expected_times = torch.arange(frames.shape[1], device=device, dtype=torch.float64) / 8
    if duration.shape != (b,) or not torch.allclose(
        duration, torch.full_like(duration, 0.25), atol=1e-7, rtol=0
    ):
        raise ValueError("The native model contract requires 0.25-second force duration")
    if timestamps.shape != frames.shape[:2] or not torch.allclose(
        timestamps, expected_times[None].expand(b, -1), atol=1e-8, rtol=0
    ):
        raise ValueError("The native model contract requires 8 FPS and pre-action frame zero")
    if (
        not frames.is_floating_point()
        or not torch.isfinite(frames).all()
        or ((frames < 0) | (frames > 1)).any()
    ):
        raise ValueError("Loader RGB must be finite float in [0,1]")
    if mask.shape != (b, 1, h, w) or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Selected masks must be binary [B,1,H,W]")
    if not mask.bool().flatten(1).any(1).all():
        raise ValueError("Every selected-object mask must be nonempty")
    if force.shape != (b, 2) or not torch.isfinite(force).all():
        raise ValueError("Force must be finite [B,2]")
    rgb = frames.float() * 2 - 1
    return VideoBatch(
        rgb[:, 0],
        mask.float(),
        force.float() / force_scale,
        rgb[:, 1:].permute(0, 2, 1, 3, 4).contiguous(),
    )
