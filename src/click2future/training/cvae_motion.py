"""Optional training-only displacement supervision for the flow CVAE decoder."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .cvae_loss import cvae_loss


def flow_motion_loss(flow, raw):
    """Match mask-average backward flow to object-centre displacement in world units.

    On the training circles/rectangles, rotation's mask-average term is zero up
    to rasterization. This does not assert that the decoded RGB follows the flow.
    Future masks and positions are loss labels, never generator inputs.
    """
    device = flow.device
    masks = torch.as_tensor(raw["instance_masks"], device=device)
    positions = torch.as_tensor(raw["positions"], device=device).float()
    valid = torch.as_tensor(raw["object_valid"], device=device)
    if (
        flow.ndim != 5
        or flow.shape[1] != 2
        or masks.ndim != 5
        or masks.dtype != torch.bool
        or valid.dtype != torch.bool
        or valid.ndim != 2
        or masks.shape[:2] != (len(flow), flow.shape[2] + 1)
        or masks.shape[2] != valid.shape[1]
        or masks.shape[-2:] != flow.shape[-2:]
        or positions.shape != (*masks.shape[:3], 2)
        or valid.shape[0] != len(flow)
        or not valid.any(1).all()
        or not torch.isfinite(flow).all()
        or not torch.isfinite(positions[valid[:, None].expand(-1, len(positions[0]), -1)]).all()
    ):
        raise ValueError(
            "Flow supervision needs matching finite fields, positions and boolean masks"
        )
    weights = masks[:, 1:].float() * valid[:, None, :, None, None]
    area = weights.sum((-2, -1))
    known = (area > 0) & valid[:, None]
    predicted = torch.einsum("bcfhw,bfnhw->bfnc", flow.float() * 0.5, weights)
    predicted = predicted / area.clamp_min(1)[..., None]
    clean = torch.where(valid[:, None, :, None], positions, 0)
    target = clean[:, 1:] - clean[:, :1]
    losses, errors = [], []
    for index in range(len(flow)):
        if known[index].any():
            p, t = predicted[index][known[index]], target[index][known[index]]
            losses.append(F.smooth_l1_loss(p, t, beta=0.05))
            errors.append((p - t).abs().mean())
    if not losses:
        raise ValueError("No observed future objects for flow supervision")
    return torch.stack(losses).mean(), torch.stack(errors).mean()


@dataclass
class MotionObjective:
    pixels: object
    motion: torch.Tensor
    displacement_mae: torch.Tensor
    weight: float

    @property
    def total(self):
        return self.pixels.total + self.weight * self.motion

    def record(self):
        return {
            **self.pixels.record(),
            "loss": float(self.total.detach()),
            "flow_motion_loss": float(self.motion.detach()),
            "flow_displacement_mae": float(self.displacement_mae.detach()),
            "flow_motion_weight": self.weight,
        }


def objective(
    model,
    batch,
    *,
    generator,
    beta,
    reconstruction,
    foreground,
    foreground_weight,
    motion_weight=0,
    raw=None,
):
    fields = []
    handle = None
    if motion_weight:
        handle = model.output.register_forward_hook(
            lambda _m, _a, output: fields.append(output.tanh())
        )
    try:
        output = model(
            batch.first_frame,
            batch.selected_mask,
            batch.force_xy,
            batch.future,
            generator=generator,
        )
    finally:
        if handle is not None:
            handle.remove()
    pixels = cvae_loss(
        output,
        batch.future,
        beta=beta,
        reconstruction=reconstruction,
        foreground=foreground,
        foreground_weight=foreground_weight,
    )
    if not motion_weight:
        return pixels
    if raw is None or len(fields) != 1:
        raise ValueError("Flow supervision requires one captured flow field and label batch")
    motion, mae = flow_motion_loss(fields[0], raw)
    return MotionObjective(pixels, motion, mae, motion_weight)
