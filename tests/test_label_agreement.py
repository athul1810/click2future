import numpy as np

from click2future.evaluation.collision_audit import event_summary
from click2future.evaluation.label_agreement import compare_labels


def timeline(*onsets, length=31):
    values = np.zeros(length, dtype=bool)
    values[list(onsets)] = True
    return values


def test_identical_labels_agree_exactly():
    rows = compare_labels(timeline(5, 20), [timeline(5, 20)], tolerance=1)
    summary = event_summary(rows)
    assert (summary["true_positive"], summary["false_positive"], summary["false_negative"]) == (
        2,
        0,
        0,
    )
    assert summary["f1"] == 1.0
    assert summary["missing_timelines"] == 0


def test_tolerance_bounds_matching():
    within = compare_labels(timeline(10), [timeline(11)], tolerance=1)[0]
    assert within["event"] == {"true_positive": 1, "false_positive": 0, "false_negative": 0}
    assert within["matches"][0]["offset"] == 1
    beyond = compare_labels(timeline(10), [timeline(12)], tolerance=1)[0]
    assert beyond["event"] == {"true_positive": 0, "false_positive": 1, "false_negative": 1}


def test_alternatives_are_pooled_and_empty_pairs_stay_undefined():
    rows = compare_labels(timeline(4), [timeline(4), timeline(), timeline(4, 9)], tolerance=0)
    summary = event_summary(rows)
    assert summary["expected_timelines"] == 3
    assert (summary["true_positive"], summary["false_positive"], summary["false_negative"]) == (
        2,
        1,
        1,
    )
    empty = event_summary(compare_labels(timeline(), [timeline()], tolerance=1))
    assert empty["f1"] is None
