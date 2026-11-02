"""E9: the E6 pose denoiser with a numeric or object-routed force input."""

import torch

from .object_motion import TrajectoryDiffusion, embedding

ROUTINGS = ("selected", "broadcast")


class RoutedTrajectoryDiffusion(TrajectoryDiffusion):
    """Same parameters as E6; only the path of the force into the object tokens differs.

    "selected" multiplies the force by the observed selected-object flag, exactly as
    in E5/E6, so only the clicked object's token receives it (spatial routing).
    "broadcast" gives the same force to both object tokens, like E2's numeric input:
    the flag stays in the features, so the model must learn which object to move.
    """

    def __init__(self, *, force_routing="selected", **kwargs):
        if force_routing not in ROUTINGS:
            raise ValueError("force_routing must be 'selected' or 'broadcast'")
        super().__init__(**kwargs)
        self.force_routing = force_routing

    def forward(self, noisy, sigma, features, force):
        # Mirrors TrajectoryDiffusion.forward except for the routed force line.
        batch = len(noisy)
        if (
            noisy.shape != (batch, self.future_frames, 2, 4)
            or features.shape != (batch, 2, 11)
            or force.shape != (batch, 2)
            or sigma.shape != (batch,)
            or not torch.isfinite(sigma).all()
            or (sigma <= 0).any()
            or any(
                not value.is_floating_point()
                or not torch.isfinite(value).all()
                or value.device != noisy.device
                for value in (noisy, features, force, sigma)
            )
            or not ((features[:, :, -1] == 0) | (features[:, :, -1] == 1)).all()
            or not (features[:, :, -1].sum(1) == 1).all()
        ):
            raise ValueError("Provide matching motion, observable object features, force and sigma")
        sd = self.sigma_data
        scale = sigma[:, None, None, None]
        denominator = scale.square() + sd * sd
        gate = features[:, :, -1:]
        if self.force_routing == "broadcast":
            gate = torch.ones_like(gate)
        routed_force = gate * force[:, None] / 0.5
        context = self.condition(torch.cat((features, routed_force), dim=-1))
        time = embedding(torch.arange(self.future_frames, device=noisy.device), self.width)
        noise = self.noise(embedding(sigma.log() / 4, self.width))
        tokens = self.motion(noisy / denominator.sqrt()) + context[:, None]
        tokens = tokens + time[None, :, None] + noise[:, None, None]
        tokens = tokens.reshape(batch, self.future_frames * 2, self.width)
        for block in self.blocks:
            tokens = block(tokens)
        residual = self.output(self.norm(tokens)).reshape_as(noisy)
        return sd * sd / denominator * noisy + scale * sd / denominator.sqrt() * residual
