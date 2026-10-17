"""Observable two-object sprites and rigid-motion decoding for a supervised pilot."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

MOTION_CONTRACT = {
    "position_scale": 0.5,
    "force_scale": 0.5,
    "fps": 8,
    "force_duration": 0.25,
    "object_order": "frame0_centroid_x_then_y",
    "motion_channels": ["dx/0.5", "dy/0.5", "sin_delta_angle", "(cos_delta_angle-1)/2"],
    "observation": "frame0 RGB and selected mask only; exactly two separately visible objects",
}


@dataclass(frozen=True)
class Observation:
    first: np.ndarray
    masks: np.ndarray
    centers: np.ndarray
    features: np.ndarray
    background: np.ndarray
    selected_index: int


def _components(mask):
    remaining = mask.copy()
    height, width = mask.shape
    components = []
    for row, column in zip(*np.nonzero(mask), strict=True):
        if not remaining[row, column]:
            continue
        pending = [(row, column)]
        remaining[row, column] = False
        members = []
        while pending:
            y, x = pending.pop()
            members.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        pending.append((ny, nx))
        if len(members) >= 4:
            component = np.zeros_like(mask)
            y, x = zip(*members, strict=True)
            component[y, x] = True
            components.append(component)
    return components


def observe(first_frame, selected_mask):
    first = np.asarray(first_frame)
    mask = np.asarray(selected_mask)
    if (
        first.dtype != np.uint8
        or first.ndim != 3
        or first.shape[-1] != 3
        or min(first.shape[:2]) < 16
        or first.shape[0] != first.shape[1]
        or mask.shape != first.shape[:2]
        or not ((mask == 0) | (mask == 1)).all()
        or mask.sum() < 4
    ):
        raise ValueError("Provide square tabletop RGB and a nonempty binary selected mask")
    mask = mask.astype(bool)
    height, width = mask.shape
    colors, counts = np.unique(first.reshape(-1, 3), axis=0, return_counts=True)
    background_color = colors[counts.argmax()]
    corners = first[[0, 0, height - 1, height - 1], [0, width - 1, 0, width - 1]]
    colors, counts = np.unique(corners, axis=0, return_counts=True)
    border_color = colors[counts.argmax()]
    foreground = ~np.all(first == background_color, axis=-1)
    foreground &= ~np.all(first == border_color, axis=-1)
    others = _components(foreground & ~mask)
    if len(others) != 1:
        raise ValueError("This pilot requires exactly two separately visible tabletop objects")
    masks = np.stack([mask, others[0]])
    centers_for_order = [tuple(np.nonzero(item)[axis].mean() for axis in (1, 0)) for item in masks]
    order = sorted(range(2), key=lambda index: centers_for_order[index])
    selected_index = order.index(0)
    masks = masks[order]
    centers, descriptors = [], []
    for item in masks:
        y, x = np.nonzero(item)
        x, y = (x + 0.5) / width, (y + 0.5) / height
        center = np.array([x.mean(), y.mean()])
        dx, dy = x - center[0], y - center[1]
        centers.append(center)
        descriptors.append(
            [
                *center,
                float(x.max() - x.min() + 1 / width),
                float(y.max() - y.min() + 1 / height),
                len(x) / (height * width),
                float((dx * dx).mean()),
                float((dy * dy).mean()),
                float((dx * dy).mean()),
            ]
        )
    centers = np.asarray(centers, dtype=np.float32)
    relative = centers[::-1] - centers
    selected = np.zeros((2, 1))
    selected[selected_index] = 1
    features = np.concatenate((np.asarray(descriptors), relative, selected), axis=1).astype(
        np.float32
    )
    background = first.copy()
    background[masks.any(0)] = background_color
    # Complete the observed square border behind any initially covering sprite.
    background[[0, -1]] = border_color
    background[:, [0, -1]] = border_color
    return Observation(first.copy(), masks, centers, features, background, selected_index)


def encode_motion(positions, angles):
    delta = positions[1:] - positions[:1]
    rotation = angles[1:] - angles[:1]
    return np.concatenate(
        (delta / 0.5, np.sin(rotation)[..., None], ((np.cos(rotation) - 1) / 2)[..., None]), axis=-1
    ).astype(np.float32)


def render_motion(observation, motion):
    """Render learned displacement/relative-rotation tokens; never call physics."""
    motion = np.asarray(motion)
    if motion.ndim != 3 or motion.shape[1:] != (2, 4) or not np.isfinite(motion).all():
        raise ValueError("Motion must be finite [F,2,4]")
    height, width = observation.first.shape[:2]
    sprites = [
        Image.fromarray(np.dstack((observation.first, mask.astype(np.uint8) * 255)))
        for mask in observation.masks
    ]
    frames = [observation.first.copy()]
    for state in motion:
        canvas = Image.fromarray(observation.background).convert("RGBA")
        for index, value in enumerate(state):
            dx, dy = value[:2] * 0.5 * np.array([width, height])
            angle = np.arctan2(value[2], value[3] * 2 + 1)
            cosine, sine = np.cos(angle), np.sin(angle)
            cx, cy = observation.centers[index] * [width, height]
            affine = (
                cosine,
                sine,
                cx - cosine * (cx + dx) - sine * (cy + dy),
                -sine,
                cosine,
                cy + sine * (cx + dx) - cosine * (cy + dy),
            )
            warped = sprites[index].transform(
                (width, height), Image.Transform.AFFINE, affine, resample=Image.Resampling.NEAREST
            )
            canvas.alpha_composite(warped)
        frames.append(np.asarray(canvas.convert("RGB"), dtype=np.uint8))
    return np.stack(frames)


def embedding(values, width):
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=values.device) / max(1, width // 2 - 1)
    )
    phase = values[:, None].float() * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), -1)


class TrajectoryDiffusion(nn.Module):
    """EDM-preconditioned object/time attention; poses are training targets only.

    The local force is routed by the observed selected-mask flag. Spatial object
    order is independent of selection, keeping paired sampling noise aligned.
    """

    def __init__(self, *, future_frames=31, width=128, layers=4, sigma_data=0.25):
        super().__init__()
        if (
            future_frames < 1
            or width < 16
            or width % 4
            or layers < 1
            or not math.isfinite(sigma_data)
            or sigma_data <= 0
        ):
            raise ValueError("Invalid trajectory model configuration")
        self.future_frames, self.width, self.sigma_data = future_frames, width, sigma_data
        self.motion = nn.Linear(4, width)
        self.condition = nn.Sequential(nn.Linear(13, width), nn.SiLU(), nn.Linear(width, width))
        self.noise = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    4,
                    width * 4,
                    dropout=0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy, sigma, features, force):
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
        local_force = features[:, :, -1:] * force[:, None] / 0.5
        context = self.condition(torch.cat((features, local_force), dim=-1))
        time = embedding(torch.arange(self.future_frames, device=noisy.device), self.width)
        noise = self.noise(embedding(sigma.log() / 4, self.width))
        tokens = self.motion(noisy / denominator.sqrt()) + context[:, None]
        tokens = tokens + time[None, :, None] + noise[:, None, None]
        tokens = tokens.reshape(batch, self.future_frames * 2, self.width)
        for block in self.blocks:
            tokens = block(tokens)
        residual = self.output(self.norm(tokens)).reshape_as(noisy)
        return sd * sd / denominator * noisy + scale * sd / denominator.sqrt() * residual

    @torch.inference_mode()
    def sample(self, features, force, *, seeds, steps=16, sigma_min=0.002, sigma_max=20.0):
        if (
            features.shape != (1, 2, 11)
            or force.shape != (1, 2)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or not 2 <= steps <= 128
            or not 0 < sigma_min < sigma_max
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
            )
        ):
            raise ValueError("Use one observable condition, distinct seeds and valid EDM settings")
        device = features.device
        ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
        levels = (sigma_max ** (1 / 7) + ramp * (sigma_min ** (1 / 7) - sigma_max ** (1 / 7))) ** 7
        levels = torch.cat((levels, levels.new_zeros(1)))
        samples = []
        was_training = self.training
        self.eval()
        try:
            for seed in seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                current = (
                    torch.randn((1, self.future_frames, 2, 4), device=device, generator=generator)
                    * sigma_max
                )
                for index in range(steps):
                    start, end = float(levels[index]), float(levels[index + 1])
                    sigma = features.new_full((1,), start)
                    denoised = self(current, sigma, features, force)
                    derivative = (current - denoised) / start
                    following = current + (end - start) * derivative
                    if end > 0:
                        second = self(following, features.new_full((1,), end), features, force)
                        derivative_next = (following - second) / end
                        following = current + (end - start) * (derivative + derivative_next) / 2
                    current = following
                if not torch.isfinite(current).all():
                    raise ValueError("Trajectory sampler returned nonfinite poses")
                samples.append(current[0])
        finally:
            self.train(was_training)
        return torch.stack(samples)
