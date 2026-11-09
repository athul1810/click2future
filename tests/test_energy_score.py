import numpy as np
import pytest

from click2future.evaluation.energy_score import (
    energy_score,
    mean_pairwise,
    paired_scene_interval,
    rms_distance,
)


def test_rms_distance_is_root_mean_square_over_frames():
    first = np.zeros((4, 2))
    second = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert rms_distance(first, second) == pytest.approx(np.sqrt(25 / 4))
    both = np.zeros((2, 2, 2))
    moved = both.copy()
    moved[:, 1] = [3.0, 4.0]
    assert rms_distance(both, moved) == pytest.approx(5.0)


def test_known_values_and_decomposition():
    samples = np.array([[[0.0, 0.0]], [[2.0, 0.0]]])
    references = np.array([[[1.0, 0.0]], [[1.0, 0.0]]])
    result = energy_score(samples, references)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["sample_spread"] == pytest.approx(2.0)
    assert result["score"] == pytest.approx(0.0)
    assert result["reference_spread"] == pytest.approx(0.0)
    single = energy_score(samples, references[:1])
    assert single["reference_spread"] is None


def test_matching_distribution_beats_collapsed_and_shifted_samples():
    rng = np.random.default_rng(3)
    totals = {"matched": 0.0, "collapsed": 0.0, "shifted": 0.0, "too_wide": 0.0}
    for _ in range(400):
        references = rng.normal(size=(4, 5, 2))
        totals["matched"] += energy_score(rng.normal(size=(4, 5, 2)), references)["score"]
        point = np.repeat(rng.normal(size=(1, 5, 2)) * 0.0, 4, axis=0)
        point = point + rng.normal(scale=1e-6, size=point.shape)
        totals["collapsed"] += energy_score(point, references)["score"]
        totals["shifted"] += energy_score(rng.normal(1.0, 1.0, (4, 5, 2)), references)["score"]
        totals["too_wide"] += energy_score(rng.normal(0.0, 2.5, (4, 5, 2)), references)["score"]
    assert totals["matched"] < min(totals["collapsed"], totals["shifted"], totals["too_wide"])


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        energy_score(np.zeros((1, 3, 2)), np.zeros((2, 3, 2)))
    with pytest.raises(ValueError):
        energy_score(np.zeros((2, 3, 2)), np.zeros((2, 4, 2)))
    with pytest.raises(ValueError):
        rms_distance(np.array([np.nan, 0.0]), np.zeros(2))
    with pytest.raises(ValueError):
        mean_pairwise(np.zeros((1, 3, 2)))


def test_paired_scene_interval_is_seeded_and_brackets_mean():
    differences = [0.1, 0.2, 0.15, 0.05]
    first = paired_scene_interval(differences)
    assert first == paired_scene_interval(differences)
    assert first[0] <= np.mean(differences) <= first[1]
    with pytest.raises(ValueError):
        paired_scene_interval([0.1])
