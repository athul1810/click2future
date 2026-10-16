"""Normalized-coordinate trajectory errors and object-pair collision metrics.

Inputs retain fixed frame-0 object identities. These functions do not extract tracks
or contacts from pixels. Frame 0 is excluded from trajectory scores; collision index
zero is the interval from frame 0 to frame 1.
"""

from __future__ import annotations

import numpy as np


def _boolean(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return array


def _positions(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric coordinates")
    return array.astype(np.float64)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def trajectory_metrics(
    predicted, reference, object_valid, tracked, *, selected_object: int
) -> dict:
    """Score `[K,T,N,2]` samples against `[T,N,2]`, including but not scoring frame 0.

    `tracked` is boolean `[K,T,N]`. A sample's ADE/FDE is unavailable if any
    requested future object observation is missing. Padding may contain NaNs;
    coordinates marked tracked for valid objects must be finite. Best-of-K is
    reported only when all K samples are complete for the requested object set.
    """
    predicted = _positions(predicted, "predicted")
    reference = _positions(reference, "reference")
    if reference.ndim != 3 or reference.shape[-1] != 2:
        raise ValueError("reference must have shape [T,N,2]")
    frames, objects, _ = reference.shape
    if frames < 2 or objects < 1:
        raise ValueError("At least two frames and one object slot are required")
    if predicted.ndim != 4 or predicted.shape[1:] != reference.shape or len(predicted) < 1:
        raise ValueError("predicted must have shape [K,T,N,2] with K >= 1")
    samples = len(predicted)
    valid = _boolean(object_valid, (objects,), "object_valid")
    tracked = _boolean(tracked, (samples, frames, objects), "tracked")
    if (
        isinstance(selected_object, bool)
        or not isinstance(selected_object, (int, np.integer))
        or not 0 <= selected_object < objects
        or not valid[selected_object]
    ):
        raise ValueError("selected_object must identify a valid object slot")
    reference_valid = reference[:, valid]
    if not np.isfinite(reference_valid).all() or np.any(
        (reference_valid < 0) | (reference_valid > 1)
    ):
        raise ValueError("Valid reference coordinates must be finite and normalized to [0,1]")
    known = tracked & valid[None, None, :]
    if not np.isfinite(predicted[known]).all():
        raise ValueError("Tracked coordinates for valid objects must be finite")

    def score(indices: np.ndarray) -> dict:
        future = predicted[:, 1:, indices, :]
        target = reference[None, 1:, indices, :]
        available = tracked[:, 1:, indices]
        complete = available.all(axis=(1, 2))
        per_sample = []
        for index in range(samples):
            if complete[index]:
                with np.errstate(over="ignore", invalid="ignore"):
                    distances = np.linalg.norm(future[index] - target[0], axis=-1)
                    ade, fde = float(distances.mean()), float(distances[-1].mean())
                if not np.isfinite([ade, fde]).all():
                    raise ValueError("Trajectory errors exceed the supported numeric range")
            else:
                ade = fde = None
            per_sample.append({"sample_index": index, "ade": ade, "fde": fde})
        successful = [record for record in per_sample if record["ade"] is not None]
        best = None
        if complete.all():
            ade_index = min(range(samples), key=lambda k: per_sample[k]["ade"])
            fde_index = min(range(samples), key=lambda k: per_sample[k]["fde"])
            best = {
                "k": samples,
                "min_ade": per_sample[ade_index]["ade"],
                "min_ade_sample_index": ade_index,
                "fde_at_min_ade": per_sample[ade_index]["fde"],
                "min_fde": per_sample[fde_index]["fde"],
                "min_fde_sample_index": fde_index,
            }
        outside = ((future < 0) | (future > 1)).any(axis=-1) & available
        scored_observations = int(available.sum())
        total_observations = int(available.size)
        return {
            "per_sample": per_sample,
            "complete_samples": int(complete.sum()),
            "total_samples": samples,
            "mean_ade_over_complete_samples": (
                sum(record["ade"] / len(successful) for record in successful)
                if successful
                else None
            ),
            "mean_fde_over_complete_samples": (
                sum(record["fde"] / len(successful) for record in successful)
                if successful
                else None
            ),
            "best_of_k": best,
            "expected_future_observations": total_observations,
            "tracked_future_observations": scored_observations,
            "missing_track_rate": (total_observations - scored_observations) / total_observations,
            "out_of_bounds_rate_over_tracked": _ratio(int(outside.sum()), scored_observations),
        }

    return {
        "units": "normalized_image_coordinates",
        "frame_zero_scored": False,
        "missing_track_policy": "complete_sample_only; best_of_k_requires_all_samples_complete",
        "all_objects": score(np.flatnonzero(valid)),
        "selected_object": score(np.array([selected_object])),
    }


def _event_matches(predicted: np.ndarray, reference: np.ndarray, tolerance: int) -> int:
    """Maximum one-to-one matches between sorted events under an absolute time window."""
    i = j = matched = 0
    while i < len(predicted) and j < len(reference):
        if predicted[i] < reference[j] - tolerance:
            i += 1
        elif reference[j] < predicted[i] - tolerance:
            j += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def _classification(tp: int, fp: int, fn: int) -> dict:
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def collision_metrics(predicted, reference, object_valid, observed, *, tolerance: int) -> dict:
    """Score onset booleans `[K,T-1,N,N]` against `[T-1,N,N]`.

    Only valid unordered pairs count. Interval scores use known bins without a time
    tolerance. Event scores use complete pair timelines with one-to-one matching.
    Encounter scores ask whether each pair had any onset over the complete clip;
    they do not measure event count or timing.
    Missing observations are counted and never interpreted as negative predictions.
    Undefined ratios are None, including F1 when both event sets are empty.
    """
    reference = np.asarray(reference)
    predicted = np.asarray(predicted)
    if reference.ndim != 3 or reference.shape[1] != reference.shape[2]:
        raise ValueError("reference collisions must have shape [T-1,N,N]")
    intervals, objects, _ = reference.shape
    if intervals < 1 or objects < 1:
        raise ValueError("At least one interval and one object slot are required")
    if predicted.ndim != 4 or predicted.shape[1:] != reference.shape or len(predicted) < 1:
        raise ValueError("predicted collisions must have shape [K,T-1,N,N] with K >= 1")
    samples = len(predicted)
    reference = _boolean(reference, reference.shape, "reference collisions")
    predicted = _boolean(predicted, predicted.shape, "predicted collisions")
    valid = _boolean(object_valid, (objects,), "object_valid")
    observed = _boolean(observed, predicted.shape, "observed collisions")
    if not valid.any():
        raise ValueError("At least one object slot must be valid")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, np.integer)) or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer number of intervals")
    valid_pairs = valid[:, None] & valid[None, :]
    if np.any(np.diagonal(reference, axis1=-2, axis2=-1)[..., valid]):
        raise ValueError("Reference collision diagonal must be false for valid objects")
    if np.any(np.diagonal(predicted, axis1=-2, axis2=-1)[..., valid]):
        raise ValueError("Predicted collision diagonal must be false for valid objects")
    for array, name in [(reference, "reference"), (predicted, "predicted"), (observed, "observed")]:
        if np.any((array != array.swapaxes(-1, -2))[..., valid_pairs]):
            raise ValueError(f"{name} collisions must be symmetric for valid object pairs")
    pairs = np.triu(valid_pairs, k=1)
    truth = reference[:, pairs]
    guesses = predicted[:, :, pairs]
    known = observed[:, :, pairs]
    pair_count = int(pairs.sum())
    records = []
    for index in range(samples):
        estimate, available = guesses[index], known[index]
        tp = int((estimate & truth & available).sum())
        fp = int((estimate & ~truth & available).sum())
        fn = int((~estimate & truth & available).sum())
        tn = int((~estimate & ~truth & available).sum())
        complete_pairs = available.all(axis=0)
        predicted_encounter = estimate[:, complete_pairs].any(axis=0)
        reference_encounter = truth[:, complete_pairs].any(axis=0)
        encounter_tp = int((predicted_encounter & reference_encounter).sum())
        encounter_fp = int((predicted_encounter & ~reference_encounter).sum())
        encounter_fn = int((~predicted_encounter & reference_encounter).sum())
        encounter_tn = int((~predicted_encounter & ~reference_encounter).sum())
        scored_pairs = int(complete_pairs.sum())
        matched = predicted_events = reference_events = 0
        for pair in np.flatnonzero(complete_pairs):
            p = np.flatnonzero(estimate[:, pair])
            q = np.flatnonzero(truth[:, pair])
            matched += _event_matches(p, q, tolerance)
            predicted_events += len(p)
            reference_events += len(q)
        records.append(
            {
                "sample_index": index,
                "interval": {
                    **_classification(tp, fp, fn),
                    "true_negative": tn,
                    "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
                    "positive_prevalence": _ratio(tp + fn, tp + fp + fn + tn),
                    "scored_bins": int(available.sum()),
                    "expected_bins": intervals * pair_count,
                    "missing_bins": int((~available).sum()),
                },
                "event": {
                    **_classification(
                        matched, predicted_events - matched, reference_events - matched
                    ),
                    "complete_pair_timelines": int(complete_pairs.sum()),
                    "expected_pair_timelines": pair_count,
                },
                "encounter": {
                    **_classification(encounter_tp, encounter_fp, encounter_fn),
                    "true_negative": encounter_tn,
                    "accuracy": _ratio(encounter_tp + encounter_tn, scored_pairs),
                    "positive_prevalence": _ratio(encounter_tp + encounter_fn, scored_pairs),
                    "scored_pairs": scored_pairs,
                    "expected_pairs": pair_count,
                    "missing_pairs": pair_count - scored_pairs,
                },
            }
        )
    return {
        "collision_metric_version": "2",
        "tolerance_intervals": int(tolerance),
        "pair_rule": "valid_upper_triangle; object_object_only",
        "undefined_ratio": "null",
        "missing_policy": "known_interval_bins; complete_pair_timelines_for_event_and_encounter",
        "encounter_definition": "any_contact_onset_per_pair_over_clip; no_timing_or_count_score",
        "per_sample": records,
    }
