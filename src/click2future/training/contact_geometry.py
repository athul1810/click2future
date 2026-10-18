"""Differentiable convex-silhouette training losses; never a sampling correction."""

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
            sample = example["sample_id"]
            if sample not in frames:
                reference = load_rollout(root, sample)
                frames[sample] = reference["frames"][0], reference["selected_mask"]
    result = {}
    for split, row in data.items():
        values = {key: [] for key in ("hulls", "normals", "axis_valid")}
        for index, example in enumerate(row["examples"]):
            image, mask = frames[example["sample_id"]]
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


def signed_penetration(motion, features, geometry):
    """Signed SAT translation in normalized image units; positive is penetration.

    This uses observed hulls before compositing. Negative values are separating
    projections, not Euclidean distance. No pose is moved or projected.
    """
    sine, cosine = motion[..., 2], 1 + 2 * motion[..., 3]
    degenerate = sine.square() + cosine.square() < 1e-16
    sine = torch.where(degenerate, torch.zeros_like(sine), sine)
    cosine = torch.where(degenerate, torch.ones_like(cosine), cosine)
    length = (sine.square() + cosine.square()).sqrt()
    sine, cosine = sine / length, cosine / length
    rotation = torch.stack((cosine, -sine, sine, cosine), -1).unflatten(-1, (2, 2))
    points = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["hulls"])
    points = points + (features[:, None, :, :2] + 0.5 * motion[..., :2])[:, :, :, None]
    axes = torch.einsum("bfnij,bnvj->bfnvi", rotation, geometry["normals"]).flatten(2, 3)
    projections = torch.einsum("bfnvi,bfai->bfnva", points, axes)
    low, high = projections.amin(-2), projections.amax(-2)
    distance = torch.minimum(high[:, :, 0] - low[:, :, 1], high[:, :, 1] - low[:, :, 0])
    distance = distance.masked_fill(~geometry["axis_valid"].flatten(1)[:, None], torch.inf)
    return distance.amin(-1)


def contact_objective(model, clean, features, force, sigma, noise, geometry, config):
    prediction = model(clean + sigma[:, None, None, None] * noise, sigma, features, force)
    error = prediction - clean
    weight = (sigma.square() + model.sigma_data**2) / (sigma * model.sigma_data).square()
    point = error.square().flatten(1).mean(1)
    velocity = (error[:, 1:] - error[:, :-1]).square().flatten(1).mean(1)
    acceleration = (error[:, 2:] - 2 * error[:, 1:-1] + error[:, :-2]).square().flatten(1).mean(1)
    zero = (force == 0).all(-1)
    denoising = (weight * (point + config.temporal_weight * velocity)).mean()
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
    return loss, {
        "denoising": denoising.detach(),
        "stationary": stationary.detach(),
        "penetration": geometry_loss.detach(),
        "contact_gap": contact_loss.detach(),
        "acceleration": acceleration_loss.detach(),
    }
