"""Cosine DDPM epsilon/velocity objectives and deterministic DDIM sampling.

References: DDPM (arXiv:2006.11239), DDIM (arXiv:2010.02502) and
Improved DDPM (arXiv:2102.09672, cosine schedule). No classifier-free guidance.
"""

import math

import torch
from torch import Tensor

from .conditioning import Condition


class Diffusion:
    def __init__(
        self,
        steps: int = 1000,
        device: str | torch.device = "cpu",
        *,
        prediction_type="epsilon",
        representation="rgb",
    ):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("Diffusion needs at least two timesteps")
        self.steps = steps
        if prediction_type not in {"epsilon", "velocity"}:
            raise ValueError("prediction_type must be epsilon or velocity")
        self.prediction_type = prediction_type
        if representation not in {"rgb", "frame0_residual"}:
            raise ValueError("representation must be rgb or frame0_residual")
        self.representation = representation
        time = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
        self.alpha_bar = (1 - betas).cumprod(0).float().to(device)

    def encode_future(self, future: Tensor, condition: Condition) -> Tensor:
        """Training RGB [-1,1] -> diffusion target, without changing the condition.

        Half the difference from the observed first frame stays in [-1,1].
        Static pixels have target zero. Future RGB is supervision only, never
        an added inference input. RGB checkpoints retain the original identity.
        """
        if self.representation == "rgb":
            return future
        return (future - condition.first_frame.unsqueeze(2)) / 2

    def decode_future(self, generated: Tensor, condition: Condition) -> Tensor:
        """Diffusion-space tensor -> RGB [-1,1] before final sampler clipping."""
        if self.representation == "rgb":
            return generated
        return condition.first_frame.unsqueeze(2) + 2 * generated

    def training_target(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        if self.prediction_type == "epsilon":
            return noise
        alpha = self.alpha_bar[timesteps].reshape(-1, *([1] * (clean.ndim - 1)))
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * clean

    def predict_clean(self, noisy: Tensor, timesteps: Tensor, prediction: Tensor) -> Tensor:
        alpha = self.alpha_bar[timesteps].reshape(-1, *([1] * (noisy.ndim - 1)))
        if self.prediction_type == "velocity":
            return alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction
        return (noisy - (1 - alpha).sqrt() * prediction) / alpha.sqrt()

    def add_noise(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        if clean.shape != noise.shape or clean.device != self.alpha_bar.device:
            raise ValueError("Noise shape and schedule device must match the clean tensor")
        if (
            timesteps.shape != (len(clean),)
            or timesteps.dtype != torch.long
            or timesteps.device != clean.device
            or (timesteps < 0).any()
            or (timesteps >= self.steps).any()
        ):
            raise ValueError("Invalid diffusion timesteps")
        alpha = self.alpha_bar[timesteps].reshape(-1, *([1] * (clean.ndim - 1)))
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model,
        condition: Condition,
        *,
        future_frames: int,
        sampling_steps: int = 32,
        seed: int = 0,
    ) -> Tensor:
        """Return float32 future RGB [-1,1], [B,3,F,H,W]; eta=0, clipped x0.

        Uses a private generator. Changing an action with the same seed reuses
        initial noise without changing the training/global RNG state.
        The caller prepends the original frame; frame 0 is never generated.
        """
        if not 2 <= sampling_steps <= self.steps or future_frames < 1:
            raise ValueError("Need 2..diffusion_steps sampling steps and positive future length")
        first = condition.first_frame
        if first.device != self.alpha_bar.device:
            raise ValueError("Condition and diffusion schedule devices must match")
        generator = torch.Generator(device=first.device).manual_seed(seed)
        noisy = torch.randn(
            (len(first), 3, future_frames, *first.shape[-2:]),
            device=first.device,
            generator=generator,
        )
        indices = torch.linspace(self.steps - 1, 0, sampling_steps).round().long().tolist()
        was_training = model.training
        model.eval()
        try:
            for index, timestep in enumerate(indices):
                times = torch.full((len(first),), timestep, device=first.device, dtype=torch.long)
                prediction = model(noisy, times, **condition.kwargs()).predicted_noise.float()
                if prediction.shape != noisy.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Sampler received invalid denoising prediction")
                alpha = self.alpha_bar[timestep]
                clean = self.predict_clean(noisy, times, prediction).clamp(-1, 1)
                # Recompute epsilon after clipping x0 to preserve DDIM consistency.
                epsilon = (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()
                previous = self.alpha_bar[indices[index + 1]] if index + 1 < len(indices) else 1.0
                previous = torch.as_tensor(previous, device=first.device)
                noisy = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
            return self.decode_future(noisy, condition).clamp(-1, 1)
        finally:
            model.train(was_training)
