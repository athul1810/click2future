"""Fair energy score for trajectory ensembles, averaged over all references.

    ES(X, y) = mean_i d(x_i, y) - 1/2 * mean_{i != j} d(x_i, x_j)

Lower is better. The distinct-pair estimate removes finite-ensemble bias for
independent samples (Ferro, 2014). Its expectation is minimized by the reference
distribution under the proper energy score (Gneiting and Raftery, 2007).

Use displacement from frame zero in native pixels. Distance is the RMS over
frames of Euclidean distance in concatenated object coordinates, a fixed
multiple of their Euclidean norm. Average each condition over its references;
paired intervals resample whole scene groups."""

import itertools

import numpy as np

ENERGY_SCORE_CONTRACT = "fair-energy-score-rms-displacement-pixels-1"


def rms_distance(first, second):
    """RMS over frames of the Euclidean distance; leading axis is frames."""
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    if difference.ndim < 2 or not np.isfinite(difference).all():
        raise ValueError("Provide finite trajectories with a leading frame axis")
    return float(np.sqrt(np.square(difference.reshape(len(difference), -1)).sum(-1).mean()))


def mean_pairwise(trajectories):
    """Mean distance over distinct unordered pairs."""
    pairs = list(itertools.combinations(range(len(trajectories)), 2))
    if not pairs:
        raise ValueError("Need at least two trajectories for a pairwise distance")
    return float(np.mean([rms_distance(trajectories[i], trajectories[j]) for i, j in pairs]))


def energy_score(samples, references):
    """Fair energy score of ``samples`` [K, T, ...] averaged over ``references`` [R, T, ...]."""
    samples = np.asarray(samples, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if samples.ndim < 3 or len(samples) < 2 or len(references) < 1:
        raise ValueError("Use at least two samples and one reference")
    if samples.shape[1:] != references.shape[1:]:
        raise ValueError("Samples and references must share their trajectory shape")
    spread = mean_pairwise(samples)
    accuracy = float(np.mean([np.mean([rms_distance(x, y) for x in samples]) for y in references]))
    return {
        "score": accuracy - 0.5 * spread,
        "accuracy": accuracy,
        "sample_spread": spread,
        "reference_spread": mean_pairwise(references) if len(references) > 1 else None,
        "samples": len(samples),
        "references": len(references),
    }


def paired_scene_interval(differences, *, draws=10000, seed=20261109):
    """Bootstrap 95% interval for the mean of per-scene-group differences."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Provide at least two finite scene-group differences")
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, (draws, len(differences)), replace=True).mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()
