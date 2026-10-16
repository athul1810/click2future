"""Future-only pixel reconstruction plus mean per-latent KL for the CVAE baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from click2future.models.cvae import CVAEOutput


def diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    if (
        q_mean.ndim != 2
        or min(q_mean.shape) < 1
        or any(
            x.shape != q_mean.shape
            or x.device != q_mean.device
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
            for x in (q_mean, q_logvar, p_mean, p_logvar)
        )
    ):
        raise ValueError("Gaussian parameters must be matching finite floating [B,Z]")
    qm, qv, pm, pv = (x.float() for x in (q_mean, q_logvar, p_mean, p_logvar))
    value = 0.5 * (pv - qv + (qv - pv).exp() + (qm - pm).square() * (-pv).exp() - 1)
    if not torch.isfinite(value).all():
        raise ValueError("Gaussian KL exceeded the finite numeric range")
    return value.clamp_min(0).mean(dim=1)


@dataclass
class CVAELoss:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    beta: float
    pixel_mse: Tensor
    pixel_mae: Tensor
    reconstruction_kind: str

    def record(self):
        return {
            "loss": float(self.total.detach()),
            "future_mse": float(self.pixel_mse.detach()),
            "future_mae": float(self.pixel_mae.detach()),
            "reconstruction_kind": self.reconstruction_kind,
            "kl_mean_per_latent": float(self.kl.detach()),
            "beta": self.beta,
            "frame_zero_scored": False,
        }


def beta_at_step(target: float, step: int, total_steps: int, warmup_fraction: float) -> float:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < 1
        or not math.isfinite(target)
        or target < 0
        or not 0 <= warmup_fraction <= 1
    ):
        raise ValueError("Invalid KL warmup settings")
    warmup_steps = math.ceil(total_steps * warmup_fraction)
    return target * min(1.0, step / warmup_steps) if warmup_steps else target


def cvae_loss(output: CVAEOutput, future: Tensor, *, beta=0.001, reconstruction="l1"):
    if isinstance(beta, bool) or not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    if (
        future.shape != output.reconstruction.shape
        or future.ndim != 5
        or future.shape[1] != 3
        or min(future.shape) < 1
        or not future.is_floating_point()
        or not torch.isfinite(future).all()
        or not torch.isfinite(output.reconstruction).all()
    ):
        raise ValueError("Reconstruction and future must be matching finite [B,3,F,H,W]")
    if reconstruction not in {"l1", "mse"}:
        raise ValueError("Choose l1 or mse reconstruction")
    difference = output.reconstruction.float() - future.float()
    mse = difference.square().flatten(1).mean(1)
    mae = difference.abs().flatten(1).mean(1)
    reconstruction_loss = mae if reconstruction == "l1" else mse
    kl = diagonal_gaussian_kl(
        output.posterior_mean, output.posterior_logvar, output.prior_mean, output.prior_logvar
    )
    if kl.shape != reconstruction_loss.shape:
        raise ValueError("Gaussian and video batch dimensions differ")
    return CVAELoss(
        (reconstruction_loss + beta * kl).mean(),
        reconstruction_loss.mean(),
        kl.mean(),
        beta,
        mse.mean(),
        mae.mean(),
        reconstruction,
    )
