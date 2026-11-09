"""Pose-based contact timing: geometry, onset rule and first-contact scoring."""

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/pose_contact.py"))


def square(x, y, half=2.0):
    return np.array(
        [[x - half, y - half], [x + half, y - half], [x + half, y + half], [x - half, y + half]]
    )


def test_signed_distance_reports_gap_touch_and_penetration():
    distance = MODULE["signed_distance"]
    assert distance(square(0, 0), square(5, 0)) == pytest.approx(1.0)
    assert distance(square(0, 0), square(4, 0)) == pytest.approx(0.0)
    assert distance(square(0, 0), square(3.5, 0)) == pytest.approx(-0.5)


def test_onset_is_a_rising_edge_into_touch():
    onset = MODULE["onsets"]([2.0, 0.5, 0.0, -0.3, 1.0, 0.04], 0.05)
    assert onset.tolist() == [False, True, False, False, True]


def test_encoded_simulator_poses_move_outlines_to_their_true_place():
    resolution, offset = 64, 0.2
    start = (0.3, 0.4, 0.0)
    now = (0.35, 0.42, 0.6)
    poses = np.array([[start, (0.7, 0.6, 0.0)], [now, (0.7, 0.6, 0.0)]])
    polygon = square(start[0] * resolution + offset, start[1] * resolution + offset)
    centres = [polygon.mean(0) + [0.7, -0.4], np.array([45.0, 38.0])]
    motion = MODULE["encode"](poses, centres, [0, 1], resolution, offset)
    moved = MODULE["move"]([polygon, square(45, 38)], centres, motion[0], resolution)[0]
    c, s = math.cos(now[2]), math.sin(now[2])
    local = square(0, 0)
    expected = local @ np.array([[c, -s], [s, c]]).T + [
        now[0] * resolution + offset,
        now[1] * resolution + offset,
    ]
    np.testing.assert_allclose(moved, expected, atol=1e-9)


def test_first_contact_scores_occurrence_range_and_nearest_error():
    stats = MODULE["first_contact_stats"](
        [
            (5, [4, 9]),  # inside the range, nearest error 1
            (12, [4, 9]),  # outside range +- 1, nearest error 3
            (None, [None, None]),  # agreed no contact
            (3, [None, 6]),  # references disagree: occurrence unscored, still timed
        ]
    )
    assert stats["samples"] == 4
    assert stats["occurrence_scored"] == 3
    assert stats["occurrence_agreement"] == pytest.approx(1.0)
    assert stats["timed"] == 3
    assert stats["within_reference_range_plus_minus_one_interval"] == pytest.approx(1 / 3)
    assert stats["mean_nearest_reference_error_intervals"] == pytest.approx((1 + 3 + 3) / 3)


def test_held_out_folders_require_their_dataset(tmp_path):
    with pytest.raises(SystemExit):
        MODULE["main"](
            [
                "--dataset",
                str(tmp_path),
                "--held-out",
                f"arm={tmp_path}",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert not (tmp_path / "out.json").exists()


def test_empty_reference_set_is_unscorable_instead_of_a_no_contact_vote():
    stats = MODULE["first_contact_stats"]([(None, []), (3, []), (5, [4, 9])])
    assert stats["samples"] == 3
    assert stats["occurrence_scored"] == 1
    assert stats["occurrence_agreement"] == 1.0
    assert stats["timed"] == 1


def test_evaluate_rejects_a_plan_from_another_split(tmp_path):
    (tmp_path / "plan.json").write_text('{"split": "id_test", "cases": []}')
    with pytest.raises(ValueError, match="validation"):
        MODULE["evaluate"](tmp_path, tmp_path, 0.05, 0.2)


def test_d028_layout_reads_shared_plan_and_per_arm_pixel_counts(tmp_path):
    compare = tmp_path / "compare"
    (compare / "projected").mkdir(parents=True)
    (compare / "plan.json").write_text('{"split": "validation", "cases": []}')
    (compare / "results.json").write_text(
        '{"models": {"projected": {"contact_counts": {"tp": 21, "fp": 94, "fn": 103}}}}'
    )
    events = MODULE["_pixel_events"](compare / "projected", True)
    assert events["true_positive"] == 21
    assert events["f1"] == pytest.approx(42 / (42 + 94 + 103))
    with pytest.raises(ValueError, match="id_test"):
        MODULE["evaluate"](compare / "projected", tmp_path, 0.05, 0.2, "id_test")
