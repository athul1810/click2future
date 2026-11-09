"""D031: multi-reference energy score for sampled trajectory futures.

The strict trajectory gate compares every sample with one reference. D024 and D030
show that even the simulator with true geometry and prior-sampled hidden physics fails
that gate, so it partly measures hidden-physics uncertainty. Each controlled condition
stores several references that differ only in hidden physics. The energy score
(Gneiting and Raftery, 2007) compares the whole sample set with each of them:

    ES(X, y) = mean_i d(x_i, y) - 1/2 * mean_{i != j} d(x_i, x_j)

It is a proper scoring rule, so in expectation it is lowest when the samples come
from the same distribution as the references. The pairwise term uses distinct pairs
only (the "fair" form of Ferro, 2014), which removes the small-ensemble bias for K
samples. The score of a condition averages ES over its references. Lower is better.

Trajectories are displacements from frame zero in native pixels. The distance d is
the root mean square over frames of the Euclidean distance, summed over the objects
included. This is a fixed multiple of the Euclidean norm, so the rule stays proper.
The score is descriptive: it changes no gate, scorer or recorded result.
"""

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
