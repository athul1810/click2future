"""CVAE decoder variant that warps observed pixels with a sampled backward flow."""

from dataclasses import replace

import torch
from torch import nn

from .cvae import ConditionalVideoVAE


def warp_first_frame(first_frame, flow):
    """Bilinear border sampling with deterministic gradients to the predicted flow.

    Flow is [B,2,F,H,W] in [-1,1], scaled to half the image width/height.
    Positive horizontal flow moves visible content right. Observed RGB is a
    constant source; gather does not need an atomic backward to the input image.
    """
    if (
        first_frame.ndim != 4
        or first_frame.shape[1] != 3
        or flow.ndim != 5
        or flow.shape[:2] != (len(first_frame), 2)
        or flow.shape[-2:] != first_frame.shape[-2:]
        or flow.shape[2] < 1
        or flow.device != first_frame.device
        or not first_frame.is_floating_point()
        or not flow.is_floating_point()
        or not torch.isfinite(first_frame).all()
        or not torch.isfinite(flow).all()
        or ((first_frame < -1) | (first_frame > 1)).any()
        or ((flow < -1) | (flow > 1)).any()
    ):
        raise ValueError("Provide matching finite RGB [B,3,H,W] and flow [B,2,F,H,W] in [-1,1]")
    b, _, frames, height, width = flow.shape
    if min(b, height, width) < 1:
        raise ValueError("Nonempty images required")
    flow = flow.float()
    x = torch.arange(width, device=flow.device).float()[None, None, None, :]
    y = torch.arange(height, device=flow.device).float()[None, None, :, None]
    x = (x - flow[:, 0] * (width / 2)).clamp(0, width - 1)
    y = (y - flow[:, 1] * (height / 2)).clamp(0, height - 1)
    left, top = x.floor().long(), y.floor().long()
    right, bottom = (left + 1).clamp_max(width - 1), (top + 1).clamp_max(height - 1)
    wx, wy = (x - left).unsqueeze(1), (y - top).unsqueeze(1)
    source = first_frame.detach().float().flatten(2)

    def pixels(rows, columns):
        indices = (rows * width + columns).reshape(b, 1, -1).expand(-1, 3, -1)
        return source.gather(2, indices).reshape(b, 3, frames, height, width)

    return (
        (1 - wy) * ((1 - wx) * pixels(top, left) + wx * pixels(top, right))
        + wy * ((1 - wx) * pixels(bottom, left) + wx * pixels(bottom, right))
    ).clamp(-1, 1)


class FlowVideoVAE(ConditionalVideoVAE):
    """Same prior/posterior, with zero-initialized two-channel displacement output.

    This appearance-preserving decoder cannot invent new surface textures or reveal
    genuinely unseen content. It still has to learn motion and action response.
    """

    def __init__(self, **configuration):
        super().__init__(**configuration)
        self.output = nn.Conv3d(self.base_channels, 2, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def configuration(self):
        return {**super().configuration(), "decoder_kind": "flow"}

    def forward(self, first_frame, selected_mask, force_xy, future, *, generator=None):
        output = super().forward(first_frame, selected_mask, force_xy, future, generator=generator)
        return replace(output, reconstruction=warp_first_frame(first_frame, output.reconstruction))

    @torch.inference_mode()
    def sample(self, first_frame, selected_mask, force_xy, *, seeds):
        flow = super().sample(first_frame, selected_mask, force_xy, seeds=seeds)
        return torch.stack([warp_first_frame(first_frame, field) for field in flow])


def build_cvae(configuration):
    configuration = dict(configuration)
    if "decoder_kind" not in configuration:
        return ConditionalVideoVAE(**configuration)
    if configuration.pop("decoder_kind") != "flow":
        raise ValueError("Optional decoder_kind must be flow; omit it for legacy RGB")
    return FlowVideoVAE(**configuration)
