"""Calibration selection must use observable positives and account for missing pairs."""

import numpy as np
import pytest

from click2future.evaluation.calibrate import select_gap, summarize
from click2future.evaluation.tracking import Tracks


def candidate(gap, tp, fp, fn):
    return {
        "max_gap_pixels": gap,
        "metrics": {
            "collision": {
                "event": {
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
                }
            }
        },
    }


def test_selection_uses_event_f1_and_smaller_gap_for_ties():
    assert select_gap([candidate(1, 1, 0, 0), candidate(0, 1, 0, 0), candidate(0.5, 1, 3, 0)]) == 0


def test_no_validation_reference_events_cannot_select_a_threshold():
    with pytest.raises(ValueError, match="insufficient"):
        select_gap([candidate(0, 0, 0, 0), candidate(1, 0, 1, 0)])


def test_summary_reports_events_excluded_by_missing_tracks():
    masks = np.zeros((2, 2, 8, 8), bool)
    masks[:, 0, 1:3, 1:3] = True
    masks[:, 1, 5:7, 5:7] = True
    positions = np.zeros((2, 2, 2))
    tracked = np.array([[True, True], [True, False]])
    tracks = Tracks(
        positions, tracked, masks, np.array([["ok", "ok"], ["ok", "missing_or_ambiguous"]])
    )
    reference = {
        "object_valid": np.ones(2, bool),
        "positions": positions,
        "collision_onset": np.array([[[False, True], [True, False]]]),
    }
    result = summarize([(reference, tracks)], 0, 1)
    assert result["missing_track_rate"] == 0.5
    assert result["reference_events_all_pairs"] == 1
    assert result["reference_events_excluded_incomplete_pairs"] == 1
    assert result["collision"]["event"]["f1"] is None
    assert result["collision"]["event"]["complete_pair_timelines"] == 0


def test_motion_selection_uses_validation_counts_and_deterministic_ties():
    from click2future.evaluation.calibrate import select_motion

    candidates = [
        dict(candidate(0, 2, 1, 1), motion_window=5, motion_threshold=0.1),
        dict(candidate(0, 2, 1, 1), motion_window=3, motion_threshold=0.3),
        dict(candidate(0, 2, 4, 1), motion_window=3, motion_threshold=0.1),
    ]
    assert select_motion(candidates)["motion_threshold"] == 0.3


def test_calibration_rejects_overlapping_scene_groups(tmp_path):
    from click2future.evaluation.calibrate import main

    with pytest.raises(SystemExit):
        main(
            [
                "--validation-start",
                "20000",
                "--held-out-start",
                "20002",
                "--output",
                str(tmp_path / "report.json"),
                "--config-output",
                str(tmp_path / "config.yaml"),
            ]
        )
    assert not (tmp_path / "report.json").exists()
