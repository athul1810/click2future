"""Collision diagnostics retain uncertainty, missing timelines and raster margins."""

from functools import lru_cache

import numpy as np
import pytest

from click2future.evaluation import collision_audit
from click2future.evaluation.collision_audit import (
    event_details,
    event_summary,
    penetration_details,
)


def test_event_offsets_and_repeated_contacts_use_one_to_one_matches():
    p, q = np.zeros(15, bool), np.zeros(15, bool)
    p[[3, 11]], q[[2, 7, 10]] = True, True
    r = event_details(p, q, np.ones(15, bool), tolerance=1)
    assert r["matches"] == [
        {"predicted": 3, "reference": 2, "offset": 1},
        {"predicted": 11, "reference": 10, "offset": 1},
    ]
    assert r["unmatched_reference"] == [7] and r["unmatched_predicted"] == []
    assert event_summary([r])["f1"] == pytest.approx(4 / 5)


def optimum_matches(a, b, tolerance):
    @lru_cache(None)
    def optimum(i, j):
        if i == len(a) or j == len(b):
            return 0
        score = max(optimum(i + 1, j), optimum(i, j + 1))
        if abs(a[i] - b[j]) <= tolerance:
            score = max(score, 1 + optimum(i + 1, j + 1))
        return score

    return optimum(0, 0)


def test_matching_agrees_with_exhaustive_optimum_in_ambiguous_windows():
    for pbits in range(16):
        for qbits in range(16):
            p = np.array([bool(pbits & (1 << i)) for i in range(4)])
            q = np.array([bool(qbits & (1 << i)) for i in range(4)])
            a, b = np.flatnonzero(p), np.flatnonzero(q)
            for tolerance in (0, 1, 2):
                r = event_details(p, q, np.ones(4, bool), tolerance=tolerance)
                assert r["event"]["true_positive"] == optimum_matches(a, b, tolerance)


def test_missing_timeline_cannot_become_a_negative_or_matching_failure():
    p = np.array([False, True, False])
    q = np.array([False, False, True])
    r = event_details(p, q, np.array([True, False, True]), tolerance=1)
    assert r["predicted_intervals"] == [1] and r["reference_intervals"] == [2]
    assert r["event"] is r["matches"] is r["unmatched_reference"] is None
    s = event_summary([r])
    assert s["missing_timelines"] == 1 and s["complete_timelines"] == 0
    assert s["f1"] is None and s["true_positive"] == s["false_positive"] == 0


def test_empty_event_sets_are_undefined_not_perfect():
    r = event_details(np.zeros(3, bool), np.zeros(3, bool), np.ones(3, bool), tolerance=1)
    assert r["complete"] and event_summary([r])["f1"] is None


def test_penetration_keeps_native_frames_and_consecutive_duration():
    r = penetration_details([0, 1, 1.01, 2, 0, 1.5])
    assert r["frames_above_one_pixel"] == [3, 4, 6]
    assert r["longest_consecutive_frames_above_one_pixel"] == 2
    assert r["maximum_depth_pixels"] == 2 and not r["frame_zero_scored"]


@pytest.mark.parametrize("depths", [[], [np.nan], [-1], [[0, 1]]])
def test_invalid_depths_are_refused(depths):
    with pytest.raises(ValueError, match="finite"):
        penetration_details(depths)


def test_changed_collision_config_is_rejected_before_loading_reference_arrays(
    tmp_path, monkeypatch
):
    config = tmp_path / "config.yaml"
    config.write_text("event_tolerance_intervals: 99\n")
    monkeypatch.setattr(
        collision_audit,
        "verify_inputs",
        lambda *args: ({"input_sha256": {"config": "unchanged historical digest"}}, {}, {}, {}),
    )
    with pytest.raises(ValueError, match="historical validation plan"):
        collision_audit.audit("plan", "predictions", tmp_path, "weights", tmp_path, config)


def test_changed_metric_source_cannot_silently_rescore_old_videos(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("event_tolerance_intervals: 1\n")
    monkeypatch.setattr(
        collision_audit,
        "verify_inputs",
        lambda *args: (
            {
                "input_sha256": {"config": collision_audit.sha256(config)},
                "evaluator_source_sha256": {"contacts.py": "different historical implementation"},
            },
            {},
            {},
            {},
        ),
    )
    with pytest.raises(ValueError, match="metric source differs"):
        collision_audit.audit("plan", "predictions", tmp_path, "weights", tmp_path, config)


def test_cli_preserves_existing_output_before_any_inputs_load(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text("retained earlier attempt")
    args = [
        part
        for key in ("plan", "predictions", "dataset", "checkpoint", "poses-dir", "config")
        for part in ("--" + key, "unused")
    ]
    with pytest.raises(SystemExit):
        collision_audit.main([*args, "--output", str(path)])
    assert path.read_text() == "retained earlier attempt"
