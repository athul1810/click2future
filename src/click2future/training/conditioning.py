"""Explicit observable-only model boundary; supervision stays outside it."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Condition:
    first_frame: Tensor
    selected_mask: Tensor
    force_xy: Tensor

    def kwargs(self):
        return {
            "first_frame": self.first_frame,
            "selected_mask": self.selected_mask,
            "force_xy": self.force_xy,
        }


def observable_condition(first_frame: Tensor, selected_mask: Tensor, force_xy: Tensor) -> Condition:
    """RGB in [0,1], binary mask, physical force -> RGB [-1,1], force / 0.5.

    Fixed 0.25 s duration belongs to this model contract. No scene specification,
    hidden coefficient, oracle future or instance labels enter this function.
    Out-of-range actions are rejected rather than silently clipped.
    """
    if first_frame.ndim != 4 or first_frame.shape[1] != 3:
        raise ValueError("first_frame must be [B,3,H,W]")
    batch, _, height, width = first_frame.shape
    if min(batch, height, width) < 1 or selected_mask.shape != (batch, 1, height, width):
        raise ValueError("selected_mask must match nonempty [B,1,H,W]")
    if force_xy.shape != (batch, 2):
        raise ValueError("force_xy must be [B,2]")
    if selected_mask.device != first_frame.device or force_xy.device != first_frame.device:
        raise ValueError("Observable inputs must share a device")
    if not first_frame.is_floating_point() or not force_xy.is_floating_point():
        raise ValueError("RGB and force must be floating point")
    if not torch.isfinite(first_frame).all() or ((first_frame < 0) | (first_frame > 1)).any():
        raise ValueError("RGB must be finite in [0,1]")
    if not torch.isfinite(force_xy).all() or (force_xy.norm(dim=-1) > 0.500001).any():
        raise ValueError("Force magnitude must be finite and at most 0.5 simulator units")
    if not ((selected_mask == 0) | (selected_mask == 1)).all():
        raise ValueError("Mask must be binary")
    if (selected_mask.flatten(1).sum(1) == 0).any():
        raise ValueError("Each selected mask must be nonempty")
    return Condition(first_frame.float() * 2 - 1, selected_mask.float(), force_xy.float() / 0.5)


def training_inputs(batch: dict, device: str | torch.device) -> tuple[Tensor, Condition]:
    """Read only RGB, mask, action and fixed timing from a loader batch."""
    frames = torch.as_tensor(batch["frames"], device=device)
    if frames.ndim != 5 or frames.shape[2] != 3 or frames.shape[1] < 2:
        raise ValueError("frames must be [B,T,3,H,W] with T>=2")
    if not torch.isfinite(frames).all() or ((frames < 0) | (frames > 1)).any():
        raise ValueError("Training RGB must be finite in [0,1]")
    duration = torch.as_tensor(batch["force_duration"], device=device)
    times = torch.as_tensor(batch["timestamps"], device=device)
    expected = torch.arange(frames.shape[1], device=device, dtype=torch.float64) / 8
    if duration.shape != (len(frames),) or not torch.allclose(
        duration.double(), torch.full_like(duration.double(), 0.25), atol=1e-7, rtol=0
    ):
        raise ValueError("This model requires a fixed 0.25 s force duration")
    if times.shape != frames.shape[:2] or not torch.allclose(
        times.double(), expected.expand(len(frames), -1), atol=1e-8, rtol=0
    ):
        raise ValueError("This model requires 8 FPS with pre-action frame 0")
    condition = observable_condition(
        frames[:, 0],
        torch.as_tensor(batch["selected_mask"], device=device),
        torch.as_tensor(batch["force_xy"], device=device),
    )
    return (frames[:, 1:] * 2 - 1).permute(0, 2, 1, 3, 4).contiguous(), condition
