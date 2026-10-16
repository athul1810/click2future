"""Whole-trajectory ensemble spread and finite-reference-set coverage."""

from __future__ import annotations

import itertools

import numpy as np

from .metrics import _positions
from .sequences import finite_norm, tracks_input


def ensemble_metrics(positions, tracked, object_valid, references, *, selected_object: int) -> dict:
    """Compare K pixel-derived samples with R simulator futures of one visible condition.

    Pairwise spread uses complete whole trajectories. Coverage/support distances are
    available only if every K sample is complete, so missing samples cannot improve
    best-of-K. References use exact oracle positions, never inference conditioning.
    """
    positions, known, valid = tracks_input(positions, tracked, object_valid)
    references = _positions(references, "references")
    if references.ndim != 4 or len(references) < 1 or references.shape[1:] != positions.shape[1:]:
        raise ValueError("references must have matching shape [R,T,N,2] with R >= 1")
    if (
        isinstance(selected_object, bool)
        or not isinstance(selected_object, (int, np.integer))
        or not 0 <= selected_object < len(valid)
        or not valid[selected_object]
    ):
        raise ValueError("selected_object must identify a valid slot")
    truth = references[:, :, valid]
    if not np.isfinite(truth).all() or np.any((truth < 0) | (truth > 1)):
        raise ValueError("Valid reference positions must be finite and normalized")
    if not np.all(truth[:, 0] == truth[0, 0]):
        raise ValueError("Reference futures must share identical initial positions")
    samples, frames = positions.shape[:2]

    def summarize(indices):
        future = positions[:, 1:, indices]
        targets = references[:, 1:, indices]
        complete = known[:, 1:, indices].all(axis=(1, 2))

        def distance(first, second):
            with np.errstate(over="ignore", invalid="ignore"):
                difference = first - second
            return float(finite_norm(difference).mean())

        pairs = [
            {
                "first": int(i),
                "second": int(j),
                "mean_distance": distance(future[i], future[j]),
                "endpoint_distance": distance(future[i, -1:], future[j, -1:]),
            }
            for i, j in itertools.combinations(np.flatnonzero(complete), 2)
        ]
        reference_pairs = [
            distance(targets[i], targets[j])
            for i, j in itertools.combinations(range(len(targets)), 2)
        ]
        matrix = None
        if complete.all():
            matrix = np.array(
                [[distance(sample, target) for target in targets] for sample in future]
            )
        return {
            "complete_samples": int(complete.sum()),
            "total_samples": samples,
            "expected_sample_pairs": samples * (samples - 1) // 2,
            "scored_sample_pairs": len(pairs),
            "mean_pairwise_trajectory_distance": float(np.mean([p["mean_distance"] for p in pairs]))
            if pairs
            else None,
            "mean_pairwise_endpoint_distance": float(
                np.mean([p["endpoint_distance"] for p in pairs])
            )
            if pairs
            else None,
            "per_sample_pair": pairs,
            "reference_futures": len(targets),
            "reference_pairs": len(reference_pairs),
            "mean_reference_pairwise_trajectory_distance": float(np.mean(reference_pairs))
            if reference_pairs
            else None,
            "coverage_requires_all_samples_complete": True,
            "mean_reference_to_nearest_sample": float(matrix.min(axis=0).mean())
            if matrix is not None
            else None,
            "mean_sample_to_nearest_reference": float(matrix.min(axis=1).mean())
            if matrix is not None
            else None,
            "reference_to_nearest_sample": matrix.min(axis=0).tolist()
            if matrix is not None
            else None,
            "sample_to_nearest_reference": matrix.min(axis=1).tolist()
            if matrix is not None
            else None,
        }

    return {
        "ensemble_metric_version": "1",
        "units": "normalized_image_coordinates",
        "frame_zero_scored": False,
        "future_frames": frames - 1,
        "distance": "Mean Euclidean distance over a whole future trajectory and fixed object slots",
        "missing_policy": "complete_trajectory_pairs; coverage_requires_all_K_complete",
        "limitation": "Nearest distance to a finite reference set does not prove physical validity",
        "all_objects": summarize(np.flatnonzero(valid)),
        "selected_object": summarize(np.array([selected_object])),
    }


def collision_frequency(predicted, observed, references, object_valid) -> dict:
    """Compare any-event rates for each unordered pair; complete K timelines required."""
    from .metrics import collision_metrics

    predicted, observed, references = map(np.asarray, (predicted, observed, references))
    if references.ndim != 4 or len(references) < 1:
        raise ValueError("Reference collision futures must be [R,T-1,N,N]")
    for reference in references:
        collision_metrics(predicted, reference, object_valid, observed, tolerance=0)
    valid = np.asarray(object_valid)
    records = []
    for first, second in itertools.combinations(np.flatnonzero(valid), 2):
        complete = observed[:, :, first, second].all(axis=1)
        probability = (
            float(predicted[:, :, first, second].any(axis=1).mean()) if complete.all() else None
        )
        reference_probability = float(references[:, :, first, second].any(axis=1).mean())
        records.append(
            {
                "first": int(first),
                "second": int(second),
                "complete_sample_timelines": int(complete.sum()),
                "expected_sample_timelines": len(predicted),
                "sample_any_event_rate": probability,
                "reference_any_event_rate": reference_probability,
                "sample_bernoulli_variance": probability * (1 - probability)
                if probability is not None
                else None,
            }
        )
    return {
        "event": "at least one object-pair contact onset during the future",
        "source": "video_contact_estimates; interpret using the declared estimator validation",
        "sample_count": len(predicted),
        "reference_count": len(references),
        "per_pair": records,
    }
