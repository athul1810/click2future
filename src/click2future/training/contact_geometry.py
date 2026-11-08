"""Differentiable convex-silhouette training losses; never a sampling correction."""

from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.rigid_geometry import _hull
from click2future.models.object_motion import observe


def observed_hulls(observation, capacity=32):
    points, normals, valid = [], [], []
    for mask, center in zip(observation.masks, observation.centers, strict=True):
        y, x = np.nonzero(mask)
        polygon = _hull(np.stack((x + 0.5, y + 0.5), -1))
        if len(polygon) > capacity:
            raise ValueError("Observed hull exceeds the declared geometry capacity")
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.stack((-edges[:, 1], edges[:, 0]), -1)
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        size = np.array(observation.first.shape[:2][::-1])
        polygon = polygon / size - center
        points.append(np.pad(polygon, ((0, capacity - len(polygon)), (0, 0)), mode="edge"))
        normals.append(np.pad(axes, ((0, capacity - len(axes)), (0, 0))))
        valid.append(np.arange(capacity) < len(axes))
    return {
        "hulls": np.asarray(points, np.float32),
        "normals": np.asarray(normals, np.float32),
        "axis_valid": np.asarray(valid, bool),
    }


def prepare_contact_geometry(root, data):
    """Build visible geometry for precisely the already selected train/val examples."""
    frames = {}
    for row in data.values():
        for example in row["examples"]:
            sample = (str(Path(example.get("root", root)).resolve()), example["sample_id"])
            if sample not in frames:
                reference = load_rollout(*sample)
                frames[sample] = reference["frames"][0], reference["selected_mask"]
    result = {}
    for split, row in data.items():
        values = {key: [] for key in ("hulls", "normals", "axis_valid")}
        for index, example in enumerate(row["examples"]):
            image, mask = frames[
                (str(Path(example.get("root", root)).resolve()), example["sample_id"])
            ]
            image = np.rot90(image, example["quarter_turns"]).copy()
            mask = np.rot90(mask, example["quarter_turns"]).copy()
            if example["mirror"]:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            observation = observe(image, mask)
            if not np.array_equal(observation.features, row["features"][index]):
                raise ValueError("Geometry and training observable identities differ")
            for key, value in observed_hulls(observation).items():
                values[key].append(value)
        result[split] = {key: np.stack(value) for key, value in values.items()}
    return result


def transformed_hulls(motion, features, geometry):
    """Transform the observed hulls without modifying predicted poses."""
    sine, cosine = motion[..., 2], 1 + 2 * motion[..., 3]
    degenerate = sine.square() + cosine.square() < 1e-16
    sine = torch.where(degenerate, torch.zeros_like(sine), sine)
    cosine = torch.where(degenerate, torch.ones_like(cosine), cosine)
    length = (sine.square() + cosine.square()).sqrt()
    sine, cosine = sine / length, cosine / length
    rotation = torch.stack((cosine, -sine, sine, cosine), -1).unflatten(-1, (2, 2))
    points = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["hulls"])
    points = points + (features[:, None, :, :2] + 0.5 * motion[..., :2])[:, :, :, None]
    return points, rotation


def canvas_excess(motion, features, geometry):
    """Maximum observed-hull excursion beyond the native [0,1] canvas per object.

    This is a training geometry signal, not the historically uncalibrated
    one-pixel interior wall score. Raster tolerance is declared separately.
    """
    points, _ = transformed_hulls(motion, features, geometry)
    return torch.cat((-points, points - 1), dim=-1).amax(dim=(-1, -2)).clamp_min(0)


def signed_penetration(motion, features, geometry):
    """Signed SAT translation in normalized image units; positive is penetration.

    This uses observed hulls before compositing. Negative values are separating
    projections, not Euclidean distance. No pose is moved or projected.
    """
    points, rotation = transformed_hulls(motion, features, geometry)
    axes = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["normals"]).flatten(2, 3)
    projections = torch.einsum("bfnvi,bfai->bfnva", points, axes)
    low, high = projections.amin(-2), projections.amax(-2)
    distance = torch.minimum(high[:, :, 0] - low[:, :, 1], high[:, :, 1] - low[:, :, 0])
    distance = distance.masked_fill(~geometry["axis_valid"].flatten(1)[:, None], torch.inf)
    return distance.amin(-1)


def contact_dynamics(predicted_depth, true_depth, window_pixels):
    """Per-sample contact-transition and signed-normal-speed errors."""
    with torch.no_grad():
        near_contact = true_depth.abs() <= window_pixels / 64
        transition = near_contact[:, 1:] != near_contact[:, :-1]
        contact_neighborhood = near_contact[:, 1:] | near_contact[:, :-1]

    depth_error = (predicted_depth - true_depth).square()
    transition_error = (depth_error[:, 1:] + depth_error[:, :-1]) / 2
    transition_error = (transition_error * transition).sum(1) / transition.sum(1).clamp_min(1)

    predicted_speed = predicted_depth[:, 1:] - predicted_depth[:, :-1]
    true_speed = true_depth[:, 1:] - true_depth[:, :-1]
    normal_velocity_error = ((predicted_speed - true_speed).square() * contact_neighborhood).sum(1)
    normal_velocity_error = normal_velocity_error / contact_neighborhood.sum(1).clamp_min(1)
    return transition_error, normal_velocity_error


def contact_objective(model, clean, features, force, sigma, noise, geometry, config):
    prediction = model(clean + sigma[:, None, None, None] * noise, sigma, features, force)
    error = prediction - clean
    weight = (sigma.square() + model.sigma_data**2) / (sigma * model.sigma_data).square()
    point = error.square().flatten(1).mean(1)
    velocity = (error[:, 1:] - error[:, :-1]).square().flatten(1).mean(1)
    acceleration = (error[:, 2:] - 2 * error[:, 1:-1] + error[:, :-2]).square().flatten(1).mean(1)
    zero = (force == 0).all(-1)
    denoising = (weight * (point + config.temporal_weight * velocity)).mean()
    if config.stationary_tokens:
        # Ground-truth rest is training supervision, never an inference-time mask.
        stationary_mask = clean.abs().amax(-1) <= 1e-7
        stationary_error = (error.square().mean(-1) * stationary_mask).mean((1, 2))
        stationary = (weight * stationary_error).mean()
    else:
        stationary = (weight * point * zero).mean()
    predicted_depth = signed_penetration(prediction, features, geometry)
    with torch.no_grad():
        true_depth = signed_penetration(clean, features, geometry)
        near_contact = true_depth.abs() <= config.contact_window_pixels / 64
    excess = (predicted_depth - config.penetration_margin_pixels / 64).relu().square().mean(1)
    gap = ((predicted_depth - true_depth).square() * near_contact).sum(1)
    gap = gap / near_contact.sum(1).clamp_min(1)
    low_noise = (sigma <= config.physics_sigma_max).float()
    geometry_loss = (excess * low_noise).mean()
    contact_loss = (gap * low_noise).mean()
    acceleration_loss = (weight * acceleration).mean()
    loss = (
        denoising
        + config.stationary_weight * stationary
        + config.penetration_weight * geometry_loss
        + config.contact_weight * contact_loss
        + config.acceleration_weight * acceleration_loss
    )
    parts = {
        "denoising": denoising.detach(),
        "stationary": stationary.detach(),
        "penetration": geometry_loss.detach(),
        "contact_gap": contact_loss.detach(),
        "acceleration": acceleration_loss.detach(),
    }
    if config.boundary_weight > 0:
        depth = canvas_excess(prediction, features, geometry)
        boundary = (depth - config.boundary_margin_pixels / 64).relu().square().mean((1, 2))
        boundary_loss = (boundary * low_noise).mean()
        loss = loss + config.boundary_weight * boundary_loss
        parts["canvas_excess"] = boundary_loss.detach()
    if config.risk_penetration_weight > 0:
        from .contact_risk import penetration_risk

        risk = penetration_risk(
            predicted_depth,
            sigma <= config.physics_sigma_max,
            margin_pixels=config.penetration_margin_pixels,
            fraction=config.risk_tail_fraction,
        )
        loss = loss + config.risk_penetration_weight * risk
        parts["tail_penetration"] = risk.detach()
    transition_weight = getattr(config, "contact_transition_weight", 0.0)
    velocity_weight = getattr(config, "normal_velocity_weight", 0.0)
    if transition_weight > 0 or velocity_weight > 0:
        transition, normal_velocity = contact_dynamics(
            predicted_depth, true_depth, config.contact_window_pixels
        )
        if transition_weight > 0:
            transition_loss = (transition * low_noise).mean()
            loss = loss + transition_weight * transition_loss
            parts["contact_transition"] = transition_loss.detach()
        if velocity_weight > 0:
            velocity_loss = (normal_velocity * low_noise).mean()
            loss = loss + velocity_weight * velocity_loss
            parts["normal_velocity"] = velocity_loss.detach()
    return loss, parts
