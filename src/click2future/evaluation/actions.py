"""Action-control metrics on a fixed, contact-free initial window."""

from __future__ import annotations

import numpy as np

from .metrics import _boolean, _positions, _ratio


def _inputs(positions, tracked, object_valid, selected_object, end_frame, motion_threshold):
    positions = _positions(positions, "positions")
    if positions.ndim != 4 or positions.shape[-1] != 2 or min(positions.shape[:3]) < 1:
        raise ValueError("positions must be nonempty [K,T,N,2]")
    samples, frames, objects, _ = positions.shape
    valid = _boolean(object_valid, (objects,), "object_valid")
    tracked = _boolean(tracked, (samples, frames, objects), "tracked")
    if (
        isinstance(selected_object, bool)
        or not isinstance(selected_object, (int, np.integer))
        or not 0 <= selected_object < objects
        or not valid[selected_object]
    ):
        raise ValueError("selected_object must identify a valid slot")
    if (
        isinstance(end_frame, bool)
        or not isinstance(end_frame, (int, np.integer))
        or not 1 <= end_frame < frames
    ):
        raise ValueError("end_frame must be an existing future frame")
    if not np.isfinite(motion_threshold) or motion_threshold <= 0:
        raise ValueError("motion_threshold must be finite and positive")
    if not np.isfinite(positions[tracked & valid[None, None]]).all():
        raise ValueError("Tracked valid positions must be finite")
    return positions[:, : end_frame + 1], tracked[:, : end_frame + 1], valid


def _force(force_xy):
    force = _positions(force_xy, "force_xy")
    if force.shape != (2,) or not np.isfinite(force).all():
        raise ValueError("force_xy must be a finite vector of length 2")
    norm = float(np.hypot(*force))
    if not np.isfinite(norm):
        raise ValueError("Force magnitude overflowed")
    return force, norm


def _displacement(values):
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = values - values[0]
        distances = np.linalg.norm(displacement, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("Displacement overflowed")
    return displacement, distances


def action_metrics(
    positions,
    tracked,
    object_valid,
    *,
    selected_object: int,
    force_xy,
    end_frame: int,
    motion_threshold: float,
) -> dict:
    """Score direction or zero-force drift; missing/stationary outputs keep their counts.

    The caller must predefine a contact-free, initially stationary control window.
    Thresholds and positions use normalized image coordinates. Zero-force drift
    measures the maximum excursion of ANY valid object, not only endpoint drift.
    """
    positions, tracked, valid = _inputs(
        positions, tracked, object_valid, selected_object, end_frame, motion_threshold
    )
    force, norm = _force(force_xy)
    records = []
    for index in range(len(positions)):
        record = {"sample_index": index, "status": "missing_track"}
        if norm == 0:
            if tracked[index, :, valid].all():
                _, distances = _displacement(positions[index][:, valid])
                peak = float(distances[1:].max())
                record.update(
                    status="scored",
                    max_drift=peak,
                    exceeds_motion_threshold=peak >= motion_threshold,
                )
        elif tracked[index, :, selected_object].all():
            delta, distances = _displacement(positions[index, :, selected_object])
            distance = float(distances[-1])
            projection = float(np.dot(delta[-1], force / norm))
            record.update(
                status="stationary" if distance < motion_threshold else "moving",
                displacement=distance,
                projection_along_force=projection,
                direction_error_degrees=None,
            )
            if distance >= motion_threshold:
                cosine = np.clip(projection / distance, -1, 1)
                record["direction_error_degrees"] = float(np.degrees(np.arccos(cosine)))
        records.append(record)
    complete = [row for row in records if row["status"] != "missing_track"]
    angles = [row["direction_error_degrees"] for row in records if row["status"] == "moving"]
    result = {
        "units": "normalized_image_coordinates",
        "end_frame": int(end_frame),
        "motion_threshold": motion_threshold,
        "force_xy": force.tolist(),
        "total_samples": len(records),
        "complete_samples": len(complete),
        "missing_samples": len(records) - len(complete),
        "per_sample": records,
    }
    if norm == 0:
        drifts = [row["max_drift"] for row in complete]
        result.update(
            kind="zero_force",
            mean_max_drift=float(np.mean(drifts)) if drifts else None,
            drifting_samples=sum(row["exceeds_motion_threshold"] for row in complete),
            drift_rate_over_complete=_ratio(
                sum(row["exceeds_motion_threshold"] for row in complete), len(complete)
            ),
        )
    else:
        stationary = sum(row["status"] == "stationary" for row in complete)
        result.update(
            kind="nonzero_force",
            stationary_samples=stationary,
            stationary_rate_over_complete=_ratio(stationary, len(complete)),
            moving_samples=len(angles),
            mean_direction_error_degrees=float(np.mean(angles)) if angles else None,
        )
    return result


def reversal_metrics(
    forward,
    reversed_positions,
    forward_tracked,
    reversed_tracked,
    object_valid,
    *,
    selected_object: int,
    force_xy,
    end_frame: int,
    motion_threshold: float,
) -> dict:
    """Pair samples by seed/order, with the same initial state and opposite forces.

    Correct responses must move along +force and -force respectively. No equal
    displacement-magnitude requirement is imposed. Stationary/missing pairs are
    counted separately and never assigned a favourable angle.
    """
    first, known_first, valid = _inputs(
        forward, forward_tracked, object_valid, selected_object, end_frame, motion_threshold
    )
    second, known_second, _ = _inputs(
        reversed_positions,
        reversed_tracked,
        object_valid,
        selected_object,
        end_frame,
        motion_threshold,
    )
    if first.shape != second.shape:
        raise ValueError("Forward and reversed arrays must have matching shapes")
    force, norm = _force(force_xy)
    if norm == 0:
        raise ValueError("Reversal requires a nonzero force")
    shared_initial = known_first[:, 0] & known_second[:, 0] & valid
    if not np.array_equal(first[:, 0][shared_initial], second[:, 0][shared_initial]):
        raise ValueError("Paired samples must have identical initial positions")
    records = []
    for index in range(len(first)):
        row = {"sample_index": index, "status": "missing_track"}
        if (
            known_first[index, :, selected_object].all()
            and known_second[index, :, selected_object].all()
        ):
            p, lengths_p = _displacement(first[index, :, selected_object])
            q, lengths_q = _displacement(second[index, :, selected_object])
            if min(lengths_p[-1], lengths_q[-1]) < motion_threshold:
                row["status"] = "stationary"
            else:
                cosine = np.clip(np.dot(p[-1] / lengths_p[-1], -q[-1] / lengths_q[-1]), -1, 1)
                row.update(
                    status="moving",
                    reversal_error_degrees=float(np.degrees(np.arccos(cosine))),
                    correct_force_signs=bool(
                        np.dot(p[-1], force / norm) >= motion_threshold
                        and np.dot(q[-1], force / norm) <= -motion_threshold
                    ),
                )
        records.append(row)
    moving = [row for row in records if row["status"] == "moving"]
    complete = [row for row in records if row["status"] != "missing_track"]
    return {
        "total_pairs": len(records),
        "complete_pairs": len(complete),
        "missing_pairs": len(records) - len(complete),
        "moving_pairs": len(moving),
        "stationary_pairs": len(complete) - len(moving),
        "mean_reversal_error_degrees": float(
            np.mean([row["reversal_error_degrees"] for row in moving])
        )
        if moving
        else None,
        "correct_force_sign_rate_over_complete": _ratio(
            sum(row["correct_force_signs"] for row in moving), len(complete)
        ),
        "end_frame": int(end_frame),
        "motion_threshold": motion_threshold,
        "per_sample": records,
    }
