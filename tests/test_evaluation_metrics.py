"""Analytical controls for trajectory errors, failure accounting, and collision events."""

import numpy as np
import pytest

from click2future.evaluation import collision_metrics, trajectory_metrics


def test_exact_shifted_and_static_trajectories_exclude_frame_zero():
    reference = np.array([[[0.1, 0.1]], [[0.2, 0.1]], [[0.3, 0.1]]])
    predicted = np.stack([reference, reference + [0.3, 0.4], np.repeat(reference[:1], 3, axis=0)])
    predicted[:, 0] = 99  # Frame 0 does not enter any score or future failure rate.
    result = trajectory_metrics(
        predicted, reference, np.array([True]), np.ones((3, 3, 1), bool), selected_object=0
    )["all_objects"]
    np.testing.assert_allclose([r["ade"] for r in result["per_sample"]], [0, 0.5, 0.15])
    np.testing.assert_allclose([r["fde"] for r in result["per_sample"]], [0, 0.5, 0.2])
    assert result["best_of_k"]["min_ade"] == 0
    assert result["out_of_bounds_rate_over_tracked"] == 0


def test_best_of_k_selects_whole_samples_and_separates_fde_rule():
    reference = np.zeros((3, 1, 2))
    predicted = np.zeros((2, 3, 1, 2))
    predicted[0, 1:, 0, 0] = [0.1, 0.6]
    predicted[1, 1:, 0, 0] = [0.9, 0.0]
    result = trajectory_metrics(
        predicted, reference, np.array([True]), np.ones((2, 3, 1), bool), selected_object=0
    )["all_objects"]
    assert result["best_of_k"]["min_ade"] == pytest.approx(0.35)
    assert result["best_of_k"]["min_ade_sample_index"] == 0
    assert result["best_of_k"]["fde_at_min_ade"] == pytest.approx(0.6)
    assert result["best_of_k"]["min_fde"] == 0
    assert result["best_of_k"]["min_fde_sample_index"] == 1


def test_best_of_k_does_not_mix_objects_from_different_samples():
    reference = np.zeros((2, 2, 2))
    predicted = np.zeros((2, 2, 2, 2))
    predicted[0, 1, 0, 0] = 1
    predicted[1, 1, 1, 0] = 1
    result = trajectory_metrics(
        predicted, reference, np.array([True, True]), np.ones((2, 2, 2), bool), selected_object=0
    )
    assert result["all_objects"]["best_of_k"]["min_ade"] == 0.5
    assert result["selected_object"]["best_of_k"]["min_ade"] == 0


def test_missing_tracks_are_counted_and_padding_is_ignored():
    reference = np.zeros((3, 3, 2))
    reference[:, 2] = np.nan
    predicted = np.repeat(reference[None], 2, axis=0)
    tracked = np.ones((2, 3, 3), bool)
    predicted[0, 1, 1] = np.nan
    tracked[0, 1, 1] = False
    result = trajectory_metrics(
        predicted, reference, np.array([True, True, False]), tracked, selected_object=0
    )
    all_objects = result["all_objects"]
    assert all_objects["expected_future_observations"] == 8
    assert all_objects["tracked_future_observations"] == 7
    assert all_objects["missing_track_rate"] == 1 / 8
    assert all_objects["complete_samples"] == 1
    assert all_objects["per_sample"][0]["ade"] is None
    assert all_objects["per_sample"][0]["fde"] is None
    assert all_objects["best_of_k"] is None
    assert result["selected_object"]["complete_samples"] == 2


def test_corrupt_tracked_coordinates_fail_but_out_of_bounds_remain_scored():
    reference = np.zeros((2, 1, 2))
    predicted = np.zeros((1, 2, 1, 2))
    tracked = np.ones((1, 2, 1), bool)
    predicted[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        trajectory_metrics(predicted, reference, np.array([True]), tracked, selected_object=0)
    predicted[0, 1, 0, 0] = 2
    result = trajectory_metrics(predicted, reference, np.array([True]), tracked, selected_object=0)[
        "all_objects"
    ]
    assert result["per_sample"][0]["ade"] == 2
    assert result["out_of_bounds_rate_over_tracked"] == 1


def test_all_missing_tracks_do_not_produce_zero_error():
    result = trajectory_metrics(
        np.full((1, 2, 1, 2), np.nan),
        np.zeros((2, 1, 2)),
        np.array([True]),
        np.zeros((1, 2, 1), bool),
        selected_object=0,
    )["all_objects"]
    assert result["mean_ade_over_complete_samples"] is None
    assert result["missing_track_rate"] == 1
    assert result["out_of_bounds_rate_over_tracked"] is None


def events(times, intervals=5, objects=2):
    result = np.zeros((intervals, objects, objects), bool)
    result[times, 0, 1] = True
    result[times, 1, 0] = True
    return result


def evaluate_events(prediction, truth, tolerance=0, observed=None, object_valid=None):
    if observed is None:
        observed = np.ones((1, *prediction.shape), bool)
    if object_valid is None:
        object_valid = np.ones(truth.shape[-1], bool)
    return collision_metrics(prediction[None], truth, object_valid, observed, tolerance=tolerance)[
        "per_sample"
    ][0]


def test_shifted_collision_is_wrong_interval_but_matches_with_tolerance():
    result = evaluate_events(events([2]), events([1]), tolerance=1)
    assert result["interval"]["accuracy"] == 3 / 5
    assert result["interval"]["scored_bins"] == 5  # No diagonal or duplicate-pair bins.
    assert result["interval"]["positive_prevalence"] == 1 / 5
    assert result["interval"]["f1"] == 0
    assert result["event"]["f1"] == 1
    assert evaluate_events(events([2]), events([1]), tolerance=0)["event"]["f1"] == 0


def test_duplicate_predictions_cannot_match_one_reference_twice():
    result = evaluate_events(events([1, 2, 3]), events([2]), tolerance=1)["event"]
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 0
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1
    assert result["f1"] == 0.5


def test_encounter_is_separate_from_inaccurate_timing_and_duplicate_events():
    result = evaluate_events(events([0, 1, 2]), events([4]), tolerance=0)
    assert result["event"]["f1"] == 0
    assert result["encounter"]["true_positive"] == 1
    assert result["encounter"]["f1"] == 1
    assert result["encounter"]["accuracy"] == 1
    assert result["encounter"]["scored_pairs"] == 1


@pytest.mark.parametrize(
    "guessed,truth,key",
    [
        (True, False, "false_positive"),
        (False, True, "false_negative"),
        (False, False, "true_negative"),
    ],
)
def test_encounter_counts_absence_and_false_predictions(guessed, truth, key):
    result = evaluate_events(events([1] if guessed else []), events([2] if truth else []))
    encounter = result["encounter"]
    assert encounter[key] == 1 and encounter["expected_pairs"] == 1
    if not guessed and not truth:
        assert encounter["accuracy"] == 1 and encounter["f1"] is None


def test_ordered_matching_finds_both_matches_in_overlapping_windows():
    result = evaluate_events(events([1, 2]), events([0, 1]), tolerance=1)["event"]
    assert result["true_positive"] == 2
    assert result["f1"] == 1


def test_empty_event_convention_is_explicit():
    result = evaluate_events(events([]), events([]))
    assert result["interval"]["accuracy"] == 1
    assert result["interval"]["positive_prevalence"] == 0
    assert result["event"]["precision"] is None
    assert result["event"]["recall"] is None
    assert result["event"]["f1"] is None
    assert result["encounter"]["scored_pairs"] == 1
    assert result["encounter"]["missing_pairs"] == 0
    assert result["encounter"]["f1"] is None
    hallucination = evaluate_events(events([1]), events([]))["event"]
    assert hallucination["precision"] == 0
    assert hallucination["recall"] is None
    assert hallucination["f1"] == 0


def test_missing_collision_bins_are_not_counted_as_negatives():
    observed = np.ones((1, 5, 2, 2), bool)
    observed[0, 2, 0, 1] = observed[0, 2, 1, 0] = False
    result = evaluate_events(events([]), events([2]), observed=observed)
    assert result["interval"]["scored_bins"] == 4
    assert result["interval"]["missing_bins"] == 1
    assert result["event"]["complete_pair_timelines"] == 0
    assert result["event"]["expected_pair_timelines"] == 1
    assert result["event"]["f1"] is None
    assert result["encounter"]["scored_pairs"] == 0
    assert result["encounter"]["missing_pairs"] == 1
    assert result["encounter"]["f1"] is None


def test_padding_and_single_object_have_no_extra_collision_pairs():
    truth = events([1], objects=3)
    truth[:, 2, 2] = True  # Invalid padded slots do not enter validation or scores.
    result = evaluate_events(truth, truth, object_valid=np.array([True, True, False]))
    assert result["interval"]["scored_bins"] == 5
    assert result["event"]["true_positive"] == 1
    one_object = evaluate_events(events([]), events([]), object_valid=np.array([True, False]))
    assert one_object["interval"]["expected_bins"] == 0
    assert one_object["interval"]["accuracy"] is None
    assert one_object["encounter"]["expected_pairs"] == 0
    assert one_object["encounter"]["accuracy"] is None


@pytest.mark.parametrize("kind", ["asymmetric", "diagonal", "probabilities"])
def test_invalid_collision_contract_is_rejected(kind):
    prediction = events([])
    if kind == "asymmetric":
        prediction[0, 0, 1] = True
    elif kind == "diagonal":
        prediction[0, 0, 0] = True
    else:
        prediction = prediction.astype(float)
    with pytest.raises(ValueError):
        evaluate_events(prediction, events([]))


def test_extreme_coordinates_fail_instead_of_exporting_infinite_scores():
    with pytest.raises(ValueError, match="numeric range"):
        trajectory_metrics(
            np.full((1, 2, 1, 2), 1e308),
            np.zeros((2, 1, 2)),
            np.array([True]),
            np.ones((1, 2, 1), bool),
            selected_object=0,
        )
