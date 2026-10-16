"""Translation-compensated RGB error and acceleration away from reference contacts."""

from __future__ import annotations

import numpy as np

from .metrics import _boolean, _ratio
from .sequences import finite_norm, timestamps_input, tracks_input


def _warp_error(first, second, source_mask, target_mask, displacement):
    height, width = source_mask.shape
    y, x = np.nonzero(target_mask)
    target_count = len(y)
    dx, dy = displacement * [width, height]
    if abs(dx) > width or abs(dy) > height:
        return None, 0, target_count
    source_x, source_y = x - dx, y - dy
    x0, y0 = np.floor(source_x).astype(int), np.floor(source_y).astype(int)
    inside = (x0 >= 0) & (y0 >= 0) & (x0 + 1 < width) & (y0 + 1 < height)
    x, y, source_x, source_y, x0, y0 = (
        value[inside] for value in (x, y, source_x, source_y, x0, y0)
    )
    # Every bilinear tap must be inside the source object, excluding silhouette mixing.
    interior = (
        source_mask[y0, x0]
        & source_mask[y0, x0 + 1]
        & source_mask[y0 + 1, x0]
        & source_mask[y0 + 1, x0 + 1]
    )
    x, y, source_x, source_y, x0, y0 = (
        value[interior] for value in (x, y, source_x, source_y, x0, y0)
    )
    if not len(x):
        return None, 0, target_count
    wx, wy = (source_x - x0)[:, None], (source_y - y0)[:, None]
    warped = (
        first[y0, x0].astype(float) * (1 - wx) * (1 - wy)
        + first[y0, x0 + 1] * wx * (1 - wy)
        + first[y0 + 1, x0] * (1 - wx) * wy
        + first[y0 + 1, x0 + 1] * wx * wy
    )
    return float(np.abs(warped - second[y, x]).mean() / 255), len(x), target_count


def temporal_metrics(
    videos,
    positions,
    masks,
    tracked,
    object_valid,
    timestamps,
    *,
    collision_onset,
    wall_contact_onset,
    selected_object: int,
    force_duration: float,
    contact_margin: int = 1,
) -> dict:
    """Only future-to-future transitions/triples are scored. Static output can score well.

    RGB compensation uses each tracked object's centroid translation. Background is
    scored only when all objects are tracked at both endpoints. Acceleration windows
    use fixed reference contact/force exclusions shared across compared models.
    """
    positions, known, valid = tracks_input(positions, tracked, object_valid)
    samples, frames, objects = positions.shape[:3]
    videos = np.asarray(videos)
    if (
        videos.dtype != np.uint8
        or videos.ndim != 5
        or videos.shape[:2] != (samples, frames)
        or videos.shape[-1] != 3
        or min(videos.shape[2:4]) < 1
    ):
        raise ValueError("videos must be matching uint8 [K,T,H,W,3]")
    height, width = videos.shape[2:4]
    masks = _boolean(masks, (samples, frames, objects, height, width), "masks")
    if np.any(known & ~masks.any(axis=(-1, -2))):
        raise ValueError("Tracked objects require nonempty masks")
    if np.any(masks.sum(axis=2) > 1):
        raise ValueError("Tracked pixel masks must be disjoint")
    times = timestamps_input(timestamps, frames)
    contacts = _boolean(collision_onset, (frames - 1, objects, objects), "collision_onset")
    walls = _boolean(wall_contact_onset, (frames - 1, objects, 4), "wall_contact_onset")
    if (
        not np.array_equal(contacts, contacts.swapaxes(-1, -2))
        or np.diagonal(contacts, axis1=-2, axis2=-1).any()
    ):
        raise ValueError("Contact labels must be symmetric with false diagonals")
    if (
        isinstance(selected_object, bool)
        or not isinstance(selected_object, (int, np.integer))
        or not 0 <= selected_object < objects
        or not valid[selected_object]
    ):
        raise ValueError("selected_object must identify a valid slot")
    if not np.isfinite(force_duration) or force_duration < 0:
        raise ValueError("force_duration must be finite and nonnegative")
    if (
        isinstance(contact_margin, bool)
        or not isinstance(contact_margin, int)
        or contact_margin < 0
    ):
        raise ValueError("contact_margin must be a nonnegative interval count")
    object_contacts = (contacts & valid[None, :, None] & valid[None, None, :]).any(axis=2)
    object_contacts |= walls.any(axis=2)
    excluded = object_contacts.copy()
    for offset in range(1, min(contact_margin, frames - 2) + 1):
        excluded[offset:] |= object_contacts[:-offset]
        excluded[:-offset] |= object_contacts[offset:]
    excluded[:, selected_object] |= times[:-1] < force_duration
    records = []
    for k in range(samples):
        object_errors, background_errors, acceleration = [], [], []
        candidate_pixels = scored_pixels = background_pixels = 0
        observed_object_transitions = observed_background_transitions = 0
        jitter_missing = jitter_excluded = 0
        for i in range(1, frames - 1):
            for obj in np.flatnonzero(valid):
                if not known[k, i : i + 2, obj].all():
                    continue
                error, count, total = _warp_error(
                    videos[k, i],
                    videos[k, i + 1],
                    masks[k, i, obj],
                    masks[k, i + 1, obj],
                    positions[k, i + 1, obj] - positions[k, i, obj],
                )
                candidate_pixels += total
                scored_pixels += count
                if error is not None:
                    object_errors.append(error)
                    observed_object_transitions += 1
            if known[k, i : i + 2, valid].all():
                background = ~(masks[k, i, valid].any(axis=0) | masks[k, i + 1, valid].any(axis=0))
                if background.any():
                    difference = np.abs(videos[k, i].astype(float) - videos[k, i + 1])
                    background_errors.append(float(difference[background].mean() / 255))
                    background_pixels += int(background.sum())
                    observed_background_transitions += 1
        for i in range(1, frames - 2):
            for obj in np.flatnonzero(valid):
                if excluded[i : i + 2, obj].any():
                    jitter_excluded += 1
                elif not known[k, i : i + 3, obj].all():
                    jitter_missing += 1
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        velocity_left = (positions[k, i + 1, obj] - positions[k, i, obj]) / (
                            times[i + 1] - times[i]
                        )
                        velocity_right = (positions[k, i + 2, obj] - positions[k, i + 1, obj]) / (
                            times[i + 2] - times[i + 1]
                        )
                        value = 2 * (velocity_right - velocity_left) / (times[i + 2] - times[i])
                    acceleration.append(float(finite_norm(value)))
        records.append(
            {
                "sample_index": k,
                "object_translation_mae_over_observed": float(np.mean(object_errors))
                if object_errors
                else None,
                "scored_object_transitions": observed_object_transitions,
                "expected_object_transitions": max(0, frames - 2) * int(valid.sum()),
                "candidate_target_pixels": candidate_pixels,
                "scored_target_pixels": scored_pixels,
                "warp_valid_pixel_fraction": _ratio(scored_pixels, candidate_pixels),
                "background_mae_over_observed": float(np.mean(background_errors))
                if background_errors
                else None,
                "scored_background_transitions": observed_background_transitions,
                "expected_background_transitions": max(0, frames - 2),
                "scored_background_pixels": background_pixels,
                "mean_acceleration_norm_over_scored": float(np.mean(acceleration))
                if acceleration
                else None,
                "expected_acceleration_windows": max(0, frames - 3) * int(valid.sum()),
                "scored_acceleration_windows": len(acceleration),
                "excluded_reference_contact_or_force_windows": jitter_excluded,
                "missing_acceleration_windows": jitter_missing,
            }
        )
    return {
        "temporal_metric_version": "1",
        "pixel_error_units": "RGB absolute error / 255",
        "acceleration_units": "normalized_image_coordinates / second^2",
        "frame_zero_scored": False,
        "compensation": "centroid translation with bilinear RGB taps",
        "contact_margin_intervals": contact_margin,
        "acceleration_exclusion": (
            "fixed reference object/wall contacts plus force on selected object"
        ),
        "aggregation": (
            "equal weight per observed object/transition; counts required alongside means"
        ),
        "limitations": [
            "Translation does not model rotation or recover occluded pixels",
            "Static output can score zero; read alongside action/quality scores",
            "Unmatched generated contacts may appear in acceleration diagnostics",
        ],
        "per_sample": records,
    }
