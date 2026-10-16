"""Fixed frame-0 object matching for trajectory and collision supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _boolean(value, shape, name, device):
    if not isinstance(value, Tensor) or value.shape != shape or value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean with shape {tuple(shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")


def canonical_object_order(
    first_masks: Tensor, object_valid: Tensor, *, max_objects: int
) -> Tensor:
    """Sort frame-0 mask centres by x, then y. Padding comes last as -1.

    Simulator sampling-order IDs cannot be inferred from RGB. This deterministic
    convention gives heads a visible ordering without future-trajectory matching.
    The caller uses masks for label alignment only, never as extra model inputs.
    """
    if (
        not isinstance(first_masks, Tensor)
        or first_masks.dtype != torch.bool
        or first_masks.ndim != 4
        or min(first_masks.shape) < 1
    ):
        raise ValueError("first_masks must be nonempty boolean [B,N,H,W]")
    batch, objects, height, width = first_masks.shape
    _boolean(object_valid, (batch, objects), "object_valid", first_masks.device)
    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    counts = object_valid.sum(dim=1)
    if (counts < 1).any() or (counts > max_objects).any():
        raise ValueError("Each clip needs 1..max_objects valid objects")
    if first_masks[~object_valid].any() or (first_masks.sum(dim=1) > 1).any():
        raise ValueError("Frame-0 masks must be disjoint, with empty padded slots")
    areas = first_masks.sum(dim=(-1, -2))
    if (areas[object_valid] == 0).any():
        raise ValueError("Every valid frame-0 mask must be nonempty")
    x = torch.arange(width, device=first_masks.device, dtype=torch.float32)
    y = torch.arange(height, device=first_masks.device, dtype=torch.float32)
    centres_x = (first_masks.sum(dim=-2) * x).sum(dim=-1) / areas.clamp_min(1)
    centres_y = (first_masks.sum(dim=-1) * y).sum(dim=-1) / areas.clamp_min(1)
    centres_x = centres_x.masked_fill(~object_valid, torch.inf)
    centres_y = centres_y.masked_fill(~object_valid, torch.inf)
    duplicates = (
        (centres_x[:, :, None] == centres_x[:, None, :])
        & (centres_y[:, :, None] == centres_y[:, None, :])
        & object_valid[:, :, None]
        & object_valid[:, None, :]
    )
    if duplicates.triu(1).any():
        raise ValueError("Frame-0 mask centres must be distinct for unambiguous slot ordering")
    # Two stable sorts give x primary and y secondary, without a hidden-ID tie-break.
    by_y = centres_y.argsort(dim=1, stable=True)
    order = by_y.gather(1, centres_x.gather(1, by_y).argsort(dim=1, stable=True))
    mapping = torch.full((batch, max_objects), -1, device=first_masks.device, dtype=torch.long)
    copied = min(max_objects, objects)
    slots = order[:, :copied]
    mapping[:, :copied] = torch.where(object_valid.gather(1, slots), slots, -1)
    return mapping


@dataclass(frozen=True)
class AuxiliaryTargets:
    positions: Tensor  # [B,T_future,M,2]
    collision_onset: Tensor  # bool [B,T_future,M,M]
    position_valid: Tensor  # bool [B,T_future,M]
    pair_valid: Tensor  # bool [B,T_future,M,M], upper triangle only
    canonical_to_scene: Tensor  # [B,M], -1 for padding


def build_auxiliary_targets(
    positions: Tensor,
    collision_onset: Tensor,
    object_valid: Tensor,
    first_masks: Tensor,
    *,
    max_objects: int,
    future_valid: Tensor | None = None,
) -> AuxiliaryTargets:
    """Reorder labels once from frame 0, remove frame 0 from trajectory targets, and pad.

    Collision index 0 remains the initial-to-first-future interval. A collision
    label is eligible only when both RGB/label endpoints are available. Unknown
    positions and padded slots are replaced with zeros after their masks are set.
    """
    if (
        not isinstance(positions, Tensor)
        or positions.ndim != 4
        or positions.shape[-1] != 2
        or not positions.is_floating_point()
        or min(positions.shape[:3]) < 1
        or positions.shape[1] < 2
    ):
        raise ValueError("positions must be floating [B,T,N,2] with T >= 2")
    batch, frames, objects, _ = positions.shape
    device = positions.device
    _boolean(object_valid, (batch, objects), "object_valid", device)
    _boolean(collision_onset, (batch, frames - 1, objects, objects), "collision_onset", device)
    if (
        first_masks.ndim != 4
        or first_masks.shape[:2] != (batch, objects)
        or first_masks.device != device
    ):
        raise ValueError("first_masks must match label batch/object dimensions and device")
    if future_valid is None:
        future_valid = torch.ones((batch, frames - 1), dtype=torch.bool, device=device)
    _boolean(future_valid, (batch, frames - 1), "future_valid", device)
    full_valid = torch.cat(
        (torch.ones((batch, 1), dtype=torch.bool, device=device), future_valid), dim=1
    )
    valid_positions = full_valid[:, :, None] & object_valid[:, None]
    values = positions[valid_positions]
    if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Available valid positions must be finite and normalized to [0,1]")
    interval_valid = full_valid[:, :-1] & full_valid[:, 1:]
    valid_pairs = (
        interval_valid[:, :, None, None]
        & object_valid[:, None, :, None]
        & object_valid[:, None, None, :]
    )
    if ((collision_onset != collision_onset.transpose(-1, -2)) & valid_pairs).any():
        raise ValueError("Known valid collision matrices must be symmetric")
    if (
        collision_onset.diagonal(dim1=-2, dim2=-1)
        & interval_valid[:, :, None]
        & object_valid[:, None]
    ).any():
        raise ValueError("Known collision diagonals must be false")
    mapping = canonical_object_order(first_masks, object_valid, max_objects=max_objects)
    safe = mapping.clamp_min(0)
    canonical_valid = mapping >= 0
    gathered = positions.detach()[:, 1:].gather(
        2, safe[:, None, :, None].expand(batch, frames - 1, max_objects, 2)
    )
    position_mask = future_valid[:, :, None] & canonical_valid[:, None]
    gathered = torch.where(position_mask[..., None], gathered, 0).float()
    labels = collision_onset.detach().gather(
        2, safe[:, None, :, None].expand(batch, frames - 1, max_objects, objects)
    )
    labels = labels.gather(
        3, safe[:, None, None, :].expand(batch, frames - 1, max_objects, max_objects)
    )
    pairs = canonical_valid[:, :, None] & canonical_valid[:, None, :]
    upper = torch.ones((max_objects, max_objects), dtype=torch.bool, device=device).triu(1)
    pair_mask = pairs[:, None] & interval_valid[:, :, None, None] & upper
    labels = labels & pairs[:, None] & interval_valid[:, :, None, None]
    return AuxiliaryTargets(gathered, labels, position_mask, pair_mask, mapping)


def to_scene_order(
    positions: Tensor, collision_logits: Tensor, canonical_to_scene: Tensor, *, scene_objects: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Restore simulator IDs for head-only evaluation, using the fixed frame-0 mapping."""
    if positions.ndim != 4 or positions.shape[-1] != 2:
        raise ValueError("positions must be [B,F,M,2]")
    if not positions.is_floating_point() or not collision_logits.is_floating_point():
        raise ValueError("Head predictions must be floating point")
    batch, frames, slots, _ = positions.shape
    if (
        collision_logits.shape != (batch, frames, slots, slots)
        or collision_logits.device != positions.device
        or canonical_to_scene.shape != (batch, slots)
        or canonical_to_scene.dtype != torch.long
        or canonical_to_scene.device != positions.device
        or isinstance(scene_objects, bool)
        or not isinstance(scene_objects, int)
        or scene_objects < 1
    ):
        raise ValueError("Head dimensions, mapping, devices and scene_objects must agree")
    restored = positions.new_full((batch, frames, scene_objects, 2), torch.nan)
    logits = collision_logits.new_full((batch, frames, scene_objects, scene_objects), torch.nan)
    valid = torch.zeros((batch, scene_objects), dtype=torch.bool, device=positions.device)
    for b in range(batch):
        known = canonical_to_scene[b] >= 0
        indices = canonical_to_scene[b, known]
        if (
            (canonical_to_scene[b] < -1).any()
            or (indices >= scene_objects).any()
            or len(indices.unique()) != len(indices)
        ):
            raise ValueError("Mapping must contain distinct in-range scene slots")
        restored[b, :, indices] = positions[b, :, known]
        selected = collision_logits[b][:, known][:, :, known]
        logits[b, :, indices[:, None], indices[None, :]] = selected
        valid[b, indices] = True
    return restored, logits, valid
