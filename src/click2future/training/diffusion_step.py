"""Shared epsilon objective and E4 auxiliary losses; the trainer owns the scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from click2future.models.pilot_denoiser import DenoiserOutput

from .auxiliary import AuxiliaryLoss, AuxiliaryLossConfig, auxiliary_loss
from .batch import VideoBatch
from .targets import AuxiliaryTargets


@dataclass
class DiffusionLoss:
    total: Tensor
    noise: Tensor
    auxiliary: AuxiliaryLoss | None
    output: DenoiserOutput

    def record(self):
        return {
            "loss": float(self.total.detach()),
            "noise_mse": float(self.noise.detach()),
            "prediction_target": "epsilon",
            "frame_zero_scored": False,
            "auxiliary": self.auxiliary.record() if self.auxiliary else None,
        }


def diffusion_objective(
    model,
    batch: VideoBatch,
    noisy_future: Tensor,
    noise: Tensor,
    timesteps: Tensor,
    *,
    num_diffusion_steps: int,
    targets: AuxiliaryTargets | None = None,
    auxiliary_config: AuxiliaryLossConfig | None = None,
) -> DiffusionLoss:
    """No noise schedule or optimization is inferred from the supplied timestep.

    Pass q(x_t|x_0) and its exact epsilon target from the trainer's schedule.
    E1-E3 omit targets; E4 adds the existing timestep-gated auxiliary term.
    Only first-frame RGB, selected mask, force, and noisy future enter the model.
    """
    if (
        isinstance(num_diffusion_steps, bool)
        or not isinstance(num_diffusion_steps, int)
        or (num_diffusion_steps < 2)
    ):
        raise ValueError("num_diffusion_steps must be an integer >= 2")
    for name, value in (("noisy_future", noisy_future), ("noise", noise)):
        if (
            value.shape != batch.future.shape
            or not value.is_floating_point()
            or value.device != batch.future.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be finite floating [B,3,F,H,W] on the batch device")
    if (
        timesteps.shape != (len(batch.future),)
        or timesteps.dtype != torch.long
        or timesteps.device != batch.future.device
        or (timesteps < 0).any()
        or (timesteps >= num_diffusion_steps).any()
    ):
        raise ValueError("timesteps must be long [B] in the scheduler's range")
    if auxiliary_config is not None and targets is None:
        raise ValueError("Auxiliary settings require explicit targets; omit both for E1-E3")
    output = model(noisy_future, timesteps, batch.first_frame, batch.selected_mask, batch.force_xy)
    if (
        output.predicted_noise.shape != noise.shape
        or not torch.isfinite(output.predicted_noise).all()
    ):
        raise ValueError("The denoiser must return finite noise with the target shape")
    noise_loss = F.mse_loss(output.predicted_noise.float(), noise.float())
    auxiliary = (
        auxiliary_loss(
            output,
            targets,
            timesteps,
            num_diffusion_steps=num_diffusion_steps,
            config=auxiliary_config,
        )
        if targets is not None
        else None
    )
    total = noise_loss + auxiliary.total if auxiliary is not None else noise_loss
    return DiffusionLoss(total, noise_loss, auxiliary, output)
