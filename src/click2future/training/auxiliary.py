"""Masked low-noise trajectory/contact supervision for denoiser features."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from click2future.models.pilot_denoiser import DenoiserOutput

from .targets import AuxiliaryTargets, _boolean


@dataclass(frozen=True)
class AuxiliaryLossConfig:
    trajectory_weight: float = 1.0
    collision_weight: float = 0.1
    smooth_l1_beta: float = 0.05
    max_timestep_fraction: float = 0.25
    collision_positive_weight: float = 1.0

    def validate(self):
        values = (
            self.trajectory_weight,
            self.collision_weight,
            self.smooth_l1_beta,
            self.max_timestep_fraction,
            self.collision_positive_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Loss settings must be finite")
        if min(self.trajectory_weight, self.collision_weight) < 0 or self.smooth_l1_beta <= 0:
            raise ValueError("Loss weights must be nonnegative and Smooth L1 beta positive")
        if not 0 <= self.max_timestep_fraction <= 1 or self.collision_positive_weight < 1:
            raise ValueError("Timestep fraction must be in [0,1] and positive weight >= 1")


@dataclass
class AuxiliaryLoss:
    total: Tensor
    trajectory: Tensor
    collision: Tensor
    statistics: dict

    def record(self) -> dict:
        return {
            **self.statistics,
            "trajectory_loss": float(self.trajectory.detach()),
            "collision_loss": float(self.collision.detach()),
            "weighted_auxiliary_loss": float(self.total.detach()),
        }


def _validate_targets(targets: AuxiliaryTargets):
    p = targets.positions
    if p.ndim != 4 or p.shape[-1] != 2 or min(p.shape[:3]) < 1 or not p.is_floating_point():
        raise ValueError("Target positions must be floating [B,F,M,2]")
    b, f, m, _ = p.shape
    _boolean(targets.position_valid, (b, f, m), "position_valid", p.device)
    _boolean(targets.pair_valid, (b, f, m, m), "pair_valid", p.device)
    _boolean(targets.collision_onset, (b, f, m, m), "collision_onset", p.device)
    if (
        targets.canonical_to_scene.shape != (b, m)
        or targets.canonical_to_scene.dtype != torch.long
        or targets.canonical_to_scene.device != p.device
    ):
        raise ValueError("canonical_to_scene must be long [B,M] on the target device")
    valid = targets.canonical_to_scene >= 0
    if (targets.position_valid & ~valid[:, None]).any():
        raise ValueError("Padded slots cannot have valid positions")
    upper = torch.ones((m, m), dtype=torch.bool, device=p.device).triu(1)
    pairs = valid[:, None, :, None] & valid[:, None, None, :] & upper
    if (targets.pair_valid & ~pairs).any():
        raise ValueError("Pair masks must contain valid unordered pairs only")
    values = p[targets.position_valid]
    if not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Available target coordinates must be finite and normalized")
    return b, f, m


def collision_balance(
    training_targets: Iterable[AuxiliaryTargets], *, max_positive_weight=20.0
) -> dict:
    """Estimate one fixed positive weight from training labels, never a minibatch ratio.

    Only known valid unordered pairs count. Empty, all-negative and all-positive
    training labels use weight 1 and retain an explicit status instead of infinity/zero.
    """
    if not math.isfinite(max_positive_weight) or max_positive_weight < 1:
        raise ValueError("max_positive_weight must be finite and >= 1")
    positives = negatives = batches = 0
    for targets in training_targets:
        _validate_targets(targets)
        values = targets.collision_onset[targets.pair_valid]
        positives += int(values.sum().item())
        negatives += values.numel() - int(values.sum().item())
        batches += 1
    if not batches:
        raise ValueError("Provide at least one training target batch")
    if positives == negatives == 0:
        status = "no_valid_pairs"
    elif positives == 0:
        status = "no_positive_events"
    elif negatives == 0:
        status = "no_negative_intervals"
    else:
        status = "estimated"
    weight = (
        min(max_positive_weight, max(1.0, negatives / positives))
        if positives and negatives
        else 1.0
    )
    return {
        "positive_count": positives,
        "negative_count": negatives,
        "positive_weight": weight,
        "max_positive_weight": max_positive_weight,
        "status": status,
        "rule": "clamp(training negative/positive valid interval counts, 1, max_positive_weight)",
    }


def _zero(tensor: Tensor):
    # An empty slice preserves autograd without touching NaNs in ignored padded entries.
    return tensor.float().reshape(-1)[:0].sum()


def auxiliary_loss(
    output: DenoiserOutput,
    targets: AuxiliaryTargets,
    timesteps: Tensor,
    *,
    num_diffusion_steps: int,
    config: AuxiliaryLossConfig | None = None,
) -> AuxiliaryLoss:
    """Return the weighted auxiliary term to add to the trainer's diffusion objective.

    Supervise clean future labels only at the configured low-timestep fraction.
    Normalize each clip by its own valid coordinates/pairs, then average clips.
    Disabled components and high-noise clips have zero gradient. No diffusion
    scheduler, timestep sampling, or noise loss is implemented here.
    """
    config = config or AuxiliaryLossConfig()
    config.validate()
    batch, frames, slots = _validate_targets(targets)
    if (
        isinstance(num_diffusion_steps, bool)
        or not isinstance(num_diffusion_steps, int)
        or num_diffusion_steps < 2
    ):
        raise ValueError("num_diffusion_steps must be an integer >= 2")
    if (
        timesteps.shape != (batch,)
        or timesteps.dtype not in (torch.int32, torch.int64)
        or timesteps.device != targets.positions.device
        or (timesteps < 0).any()
        or (timesteps >= num_diffusion_steps).any()
    ):
        raise ValueError("timesteps must be valid integer [B] on the target device")
    cutoff = math.floor(config.max_timestep_fraction * (num_diffusion_steps - 1))
    eligible = timesteps <= cutoff
    trajectory_terms, collision_terms = [], []
    coordinate_count = pair_count = positive_count = 0
    trajectory_zero = collision_zero = _zero(output.predicted_noise)
    if config.trajectory_weight:
        predicted = output.positions
        if (
            predicted is None
            or predicted.shape != targets.positions.shape
            or predicted.device != targets.positions.device
            or not predicted.is_floating_point()
        ):
            raise ValueError(
                "Enabled trajectory supervision requires matching floating position heads"
            )
        trajectory_zero = _zero(predicted)
        for b in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            mask = targets.position_valid[b]
            if not mask.any():
                continue
            values = predicted[b][mask].float()
            if not torch.isfinite(values).all():
                raise ValueError("Active trajectory predictions must be finite")
            trajectory_terms.append(
                F.smooth_l1_loss(
                    values,
                    targets.positions[b][mask].float(),
                    beta=config.smooth_l1_beta,
                    reduction="mean",
                )
            )
            coordinate_count += values.numel()
    if config.collision_weight:
        predicted = output.collision_logits
        if (
            predicted is None
            or predicted.shape != (batch, frames, slots, slots)
            or predicted.device != targets.positions.device
            or not predicted.is_floating_point()
        ):
            raise ValueError("Enabled collision supervision requires matching floating logits")
        collision_zero = _zero(predicted)
        positive_weight = targets.positions.new_tensor(
            config.collision_positive_weight, dtype=torch.float32
        )
        for b in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            mask = targets.pair_valid[b]
            if not mask.any():
                continue
            logits = predicted[b][mask].float()
            labels = targets.collision_onset[b][mask].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Active collision logits must be finite")
            collision_terms.append(
                F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=positive_weight, reduction="mean"
                )
            )
            pair_count += logits.numel()
            positive_count += int(labels.sum().item())
    trajectory = torch.stack(trajectory_terms).mean() if trajectory_terms else trajectory_zero
    collision = torch.stack(collision_terms).mean() if collision_terms else collision_zero
    total = config.trajectory_weight * trajectory + config.collision_weight * collision
    if not torch.isfinite(total).item():
        raise ValueError("Auxiliary loss is not finite")
    return AuxiliaryLoss(
        total,
        trajectory,
        collision,
        {
            "loss_version": "1",
            "target_order": "frame0_mask_centroid_x_then_y_v1",
            "config": asdict(config),
            "total_clips": batch,
            "eligible_timestep_clips": int(eligible.sum().item()),
            "trajectory_clips": len(trajectory_terms),
            "collision_clips": len(collision_terms),
            "position_coordinates": coordinate_count,
            "collision_pairs": pair_count,
            "positive_collision_pairs": positive_count,
            "timestep_cutoff_inclusive": cutoff,
            "num_diffusion_steps": num_diffusion_steps,
            "reduction": "mean within each eligible clip, then mean clips per component",
        },
    )
