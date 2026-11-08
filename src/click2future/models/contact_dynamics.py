"""Experimental learned contact impulses with explicit inertial state updates.

This is a restricted two-body neural dynamics candidate, not a RiCo reproduction.
Observed hulls determine local neighborhoods, not a collision response or pose
projection. Learned responses may still penetrate or add energy.
"""

import torch
from torch import nn

from click2future.training.contact_geometry import transformed_hulls

from .world_motion import sample_world

CONTACT_DYNAMICS = "local-impulse-finite-difference-dynamics-1"
DT = 1 / 8


def motion_tokens(state):
    return torch.cat(
        (state[..., :2] / 0.5, state[..., 2:].sin(), (state[..., 2:].cos() - 1) / 2), -1
    )


def free_acceleration(force, features, world, interval):
    """Exact finite difference of native 240-Hz force-only positions at 8 fps."""
    coefficient = torch.where(
        interval == 0,
        DT * (DT - 1 / 240) / 2,
        torch.where(interval == 1, DT**2, torch.where(interval == 2, DT * (DT + 1 / 240) / 2, 0)),
    )
    return force[:, None] * features[..., -1:] / world[..., :1] * coefficient[:, None, None]


def _mlp(inputs, outputs, width):
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.SiLU(),
        nn.Linear(width, width),
        nn.SiLU(),
        nn.Linear(width, outputs),
    )


class ContactDynamics(nn.Module):
    """Predict local momentum/displacement increments, then integrate 31 steps.

    Pair impulses are equal and opposite. When walls are out of range, their
    contribution to total finite-difference linear momentum is exactly zero.
    Rest is exact by a homogeneous motion scale. These properties alone do not
    guarantee nonpenetration, energy dissipation or correct collision timing.
    """

    future_frames = 31

    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.pair = _mlp(38, 4, width)
        self.wall = _mlp(21, 3, width)
        for net in (self.pair, self.wall):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def advance(self, state, delta, features, force, world, geometry, interval):
        """State/delta are [B,2,3] relative x/y/angle and their previous increments."""
        points, rotation = transformed_hulls(motion_tokens(state)[:, None], features, geometry)
        points, rotation = points[:, 0], rotation[:, 0]
        axes = torch.einsum("bnij,bnvj->bnvi", rotation, geometry["normals"]).flatten(1, 2)
        projection = torch.einsum("bnvi,bai->bnva", points, axes)
        low, high = projection.amin(-2), projection.amax(-2)
        overlap = torch.minimum(high[:, 0] - low[:, 1], high[:, 1] - low[:, 0])
        overlap = overlap.masked_fill(~geometry["axis_valid"].flatten(1), torch.inf)
        penetration, axis_index = overlap.min(-1)
        normal = axes[torch.arange(len(state), device=state.device), axis_index]
        relative = features[:, 1, :2] + state[:, 1, :2] - features[:, 0, :2] - state[:, 0, :2]
        normal = normal * torch.where((normal * relative).sum(-1, keepdim=True) < 0, -1, 1)
        radius = geometry["hulls"].norm(dim=-1).amax(-1).clamp_min(1 / 64)
        free = free_acceleration(force, features, world, interval)
        speed = delta[..., :2].norm(dim=-1) + delta[..., 2].abs() * radius + free.norm(dim=-1)
        active = (interval < 2).to(state.dtype)[:, None, None]
        local_force = features[..., -1:] * force[:, None] * active
        body = torch.cat(
            (
                features[..., 2:8] * 64,
                state[..., 2:].sin(),
                state[..., 2:].cos(),
                world,
                delta[..., :2] / DT,
                delta[..., 2:] / DT,
            ),
            -1,
        )
        phase = active[:, 0]
        first = torch.cat(
            (
                body[:, 0],
                body[:, 1],
                relative,
                penetration[:, None] * 64,
                normal,
                local_force.flatten(1),
                phase,
            ),
            -1,
        )
        second = torch.cat(
            (
                body[:, 1],
                body[:, 0],
                -relative,
                penetration[:, None] * 64,
                -normal,
                local_force.flip(1).flatten(1),
                phase,
            ),
            -1,
        )
        a, b = self.pair(first).tanh(), self.pair(second).tanh()
        nearby = penetration >= -(speed.sum(1) + 1 / 64)
        reduced_mass = 1 / world[..., 0].reciprocal().sum(1)
        scale = speed.sum(1) * reduced_mass * nearby
        impulse = (a[:, :2] - b[:, :2]) * scale[:, None]
        pair_position = torch.stack((-impulse, impulse), 1) / world[..., :1]
        angular = torch.stack((a[:, 2] + b[:, 3], a[:, 3] + b[:, 2]), 1)
        angular = angular * (speed.sum(1) * nearby)[:, None] / radius
        wall_gap = torch.cat((points.amin(-2), 1 - points.amax(-2)), -1)
        wall_input = torch.cat(
            (body, wall_gap * 64, local_force, phase[:, None].expand(-1, 2, -1)), -1
        )
        near_wall = wall_gap.amin(-1) <= speed + 1 / 64
        wall = self.wall(wall_input).tanh() * (speed * near_wall)[..., None]
        correction = torch.cat(
            (pair_position + wall[..., :2], (angular + wall[..., 2] / radius)[..., None]), -1
        )
        change = delta + torch.cat((free, torch.zeros_like(free[..., :1])), -1) + correction
        return state + change, change

    def rollout(self, features, force, world, geometry):
        state = features.new_zeros((len(features), 2, 3))
        delta = torch.zeros_like(state)
        result = []
        for frame in range(self.future_frames):
            interval = torch.full((len(features),), frame, device=features.device)
            state, delta = self.advance(state, delta, features, force, world, geometry, interval)
            result.append(state)
        return torch.stack(result, 1)

    @torch.inference_mode()
    def sample(self, features, force, *, geometry, seeds):
        if features.shape != (1, 2, 11) or force.shape != (1, 2) or not seeds:
            raise ValueError("Use one observable condition and nonempty sampling seeds")
        world = torch.cat([sample_world(seed, features.device) for seed in seeds])
        return motion_tokens(
            self.rollout(
                features.expand(len(seeds), -1, -1),
                force.expand(len(seeds), -1),
                world,
                {
                    key: value.expand(len(seeds), *value.shape[1:])
                    for key, value in geometry.items()
                },
            )
        )
