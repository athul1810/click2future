"""Learned local-surface messages around E7, without a contact solver."""

import torch
from torch import nn

from .object_motion import TrajectoryDiffusion, embedding
from .world_motion import WORLD_BOUNDS, WorldTrajectoryDiffusion, sample_world

SURFACE_CONTRACT = "observed-hull-nearest-segment-local-vs-pooled-1"


def check_geometry(geometry, features):
    batch = len(features)
    if (
        set(geometry) != {"hulls", "normals", "axis_valid"}
        or any(geometry[k].shape != (batch, 2, 32, 2) for k in ("hulls", "normals"))
        or geometry["axis_valid"].shape != (batch, 2, 32)
        or geometry["axis_valid"].dtype != torch.bool
        or (geometry["axis_valid"].sum(-1) < 3).any()
        or any(v.device != features.device for v in geometry.values())
        or any(
            not geometry[k].is_floating_point() or not torch.isfinite(geometry[k]).all()
            for k in ("hulls", "normals")
        )
    ):
        raise ValueError("Provide finite observed hulls, normals and valid polygon vertices")
    count = geometry["axis_valid"].sum(-1)
    expected = torch.arange(32, device=features.device)[None, None] < count[..., None]
    if not torch.equal(geometry["axis_valid"], expected):
        raise ValueError("Valid polygon vertices must be a contiguous prefix")
    lengths = geometry["normals"].norm(dim=-1)
    if not torch.allclose(lengths[expected], torch.ones_like(lengths[expected]), atol=1e-5, rtol=0):
        raise ValueError("Valid observed normals must have unit length")


def surface_features(motion, features, geometry, *, local=True):
    """Describe closest surface segments; geometry calculations never move poses.

    Pooled control broadcasts the mean cross-object correspondence before the
    shared point encoder. It keeps the same parameters and own surface detail.
    """
    check_geometry(geometry, features)
    batch, frames = motion.shape[:2]
    if motion.shape != (batch, frames, 2, 4) or not torch.isfinite(motion).all():
        raise ValueError("Provide finite two-object motion tokens")
    valid = geometry["axis_valid"]
    sine, cosine = motion[..., 2], 1 + 2 * motion[..., 3]
    squared_norm = sine.square() + cosine.square()
    norm = squared_norm.clamp_min(1e-16).sqrt()
    safe = squared_norm > 1e-16
    # Same angle interpretation as rigid decoding; this is not a pose correction.
    sine = torch.where(safe, sine / norm.clamp_min(1e-8), torch.zeros_like(sine))
    cosine = torch.where(safe, cosine / norm.clamp_min(1e-8), torch.ones_like(cosine))
    rotation = torch.stack((cosine, -sine, sine, cosine), -1).unflatten(-1, (2, 2))
    offsets = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["hulls"])
    centers = features[:, None, :, :2] + 0.5 * motion[..., :2]
    points = offsets + centers[:, :, :, None]
    normals = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["normals"])
    count = valid.sum(-1)
    next_index = (torch.arange(32, device=motion.device)[None, None] + 1) % count[..., None]
    ends = points.gather(3, next_index[:, None, :, :, None].expand(batch, frames, 2, 32, 2))
    other_start, other_end = points.flip(2), ends.flip(2)
    edges = other_end - other_start
    delta = points[:, :, :, :, None] - other_start[:, :, :, None]
    fraction = (delta * edges[:, :, :, None]).sum(-1) / edges.square().sum(-1)[
        :, :, :, None
    ].clamp_min(1e-12)
    # Clamp only the closest-point coordinate along a segment, not generated motion.
    closest = other_start[:, :, :, None] + fraction.clamp(0, 1)[..., None] * edges[:, :, :, None]
    relative = closest - points[:, :, :, :, None]
    squared = relative.square().sum(-1).masked_fill(~valid.flip(1)[:, None, :, None], torch.inf)
    index = squared.argmin(-1)
    nearest = relative.gather(4, index[..., None, None].expand(batch, frames, 2, 32, 1, 2)).squeeze(
        4
    )
    other_normals = normals.flip(2).gather(3, index[..., None].expand(batch, frames, 2, 32, 2))
    distance = nearest.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    correspondence = torch.cat((nearest, other_normals, distance), -1)
    if not local:
        weights = valid[:, None, :, :, None]
        pooled = (correspondence * weights).sum(3, keepdim=True) / count[:, None, :, None, None]
        correspondence = pooled.expand_as(correspondence)
    velocity = torch.diff(points, dim=1, prepend=points[:, :1])
    other_velocity = velocity.flip(2).gather(3, index[..., None].expand(batch, frames, 2, 32, 2))
    relative_velocity = other_velocity - velocity
    if not local:
        relative_velocity = (relative_velocity * valid[:, None, :, :, None]).sum(
            3, keepdim=True
        ) / count[:, None, :, None, None]
        relative_velocity = relative_velocity.expand_as(velocity)
    boundary = torch.cat((points, 1 - points), -1)
    values = torch.cat(
        (offsets, normals, correspondence, velocity, relative_velocity, boundary), -1
    )
    # Invalid padding is masked before and after the MLP; it cannot affect pooling.
    return values.masked_fill(~valid[:, None, :, :, None], 0), valid


class BoundSurface(nn.Module):
    def __init__(self, model, world, geometry):
        super().__init__()
        self.model, self.world, self.geometry = model, world, geometry
        self.future_frames, self.sigma_data = model.future_frames, model.sigma_data
        self.training = model.training

    def forward(self, noisy, sigma, features, force):
        return self.model(noisy, sigma, features, force, self.world, self.geometry)


class SurfaceTrajectoryDiffusion(nn.Module):
    """Both arms refine all E7 weights; point correspondence is the only switch."""

    def __init__(self, *, local_surfaces=True, surface_width=64, **backbone_config):
        super().__init__()
        if (
            not isinstance(local_surfaces, bool)
            or isinstance(surface_width, bool)
            or not isinstance(surface_width, int)
            or not 16 <= surface_width <= 128
        ):
            raise ValueError("Use a boolean correspondence switch and bounded surface width")
        if (
            backbone_config.get("world_condition") is not True
            or backbone_config.get("relational") is not True
        ):
            raise ValueError("Surface model requires conditioned relational E7")
        self.backbone = WorldTrajectoryDiffusion(**backbone_config)
        self.local_surfaces = local_surfaces
        self.future_frames, self.sigma_data = self.backbone.future_frames, self.backbone.sigma_data
        self.world_condition = self.relational = True
        # 17 geometry channels + own/other world(6), local force(2), noise(16).
        self.points = nn.Sequential(
            nn.Linear(41, surface_width),
            nn.SiLU(),
            nn.Linear(surface_width, surface_width),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * surface_width, surface_width), nn.SiLU(), nn.Linear(surface_width, 4)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy, sigma, features, force, world, geometry):
        base = self.backbone(noisy, sigma, features, force, world)
        shape, valid = surface_features(base, features, geometry, local=self.local_surfaces)
        bounds = world.new_tensor(WORLD_BOUNDS)
        normalized = 2 * (world - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
        context = torch.cat(
            (normalized, normalized.flip(1), features[..., -1:] * force[:, None] / 0.5), -1
        )
        noise = embedding(sigma.log() / 4, 16)
        tokens = torch.cat(
            (
                shape,
                context[:, None, :, None].expand(-1, self.future_frames, -1, 32, -1),
                noise[:, None, None, None].expand(-1, self.future_frames, 2, 32, -1),
            ),
            -1,
        )
        encoded = self.points(tokens)
        weight = valid[:, None, :, :, None]
        mean = (encoded * weight).sum(3) / valid.sum(-1)[:, None, :, None]
        maximum = encoded.masked_fill(~weight, -torch.inf).amax(3)
        correction = self.output(torch.cat((mean, maximum), -1))
        scale = sigma[:, None, None, None]
        return (
            base
            + scale * self.sigma_data / (scale.square() + self.sigma_data**2).sqrt() * correction
        )

    @torch.inference_mode()
    def sample(self, features, force, geometry, *, seeds, **kwargs):
        check_geometry(geometry, features)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Use distinct sampling seeds")
        results = []
        for seed in seeds:
            bound = BoundSurface(self, sample_world(seed, features.device), geometry)
            results.append(
                TrajectoryDiffusion.sample(bound, features, force, seeds=[seed], **kwargs)
            )
        return torch.cat(results)
