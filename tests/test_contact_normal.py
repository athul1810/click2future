"""Contact-normal impulses, identity invariance, and calibration cache parity."""

import numpy as np
import pytest

from click2future.evaluation.contacts import (
    contacts_from_geometry,
    estimate_contacts,
    prepare_contact_geometry,
)


def fixture_masks():
    masks = np.zeros((8, 2, 24, 40), bool)
    for t, (first, second) in enumerate(
        zip([1, 3, 5, 6, 6, 6, 6, 6], [9, 9, 9, 10, 12, 14, 16, 18], strict=True)
    ):
        masks[t, 0, 10:14, first : first + 4] = True
        masks[t, 1, 10:14, second : second + 4] = True
    return masks, np.ones((8, 2), bool), np.ones(2, bool)


def test_normal_impulse_detects_a_bounce_and_is_identity_and_mirror_invariant():
    masks, known, valid = fixture_masks()
    options = dict(max_gap_pixels=0, motion_threshold=0.08, motion_window=3, motion_rule="normal")
    result = estimate_contacts(masks, known, valid, **options)
    events = np.flatnonzero(result.onset[:, 0, 1])
    assert len(events) == 1 and 1 <= events[0] <= 3
    reverse = estimate_contacts(masks[:, ::-1], known, valid, **options)
    mirror = estimate_contacts(masks[..., ::-1], known, valid, **options)
    assert np.array_equal(result.onset, reverse.onset)
    assert np.array_equal(result.onset, mirror.onset)


@pytest.mark.parametrize("rule", ["normal", "opposing"])
def test_cached_distances_match_direct_scoring_and_missing_windows(rule):
    masks, known, valid = fixture_masks()
    known[4, 1] = False
    geometry = prepare_contact_geometry(masks, known, valid)
    for gap in (0, 1, 2):
        options = dict(max_gap_pixels=gap, motion_threshold=0.08, motion_rule=rule)
        cached = contacts_from_geometry(geometry, **options)
        direct = estimate_contacts(masks, known, valid, **options)
        for key in ("state", "onset", "observed"):
            assert np.array_equal(getattr(cached, key), getattr(direct, key))
        assert not cached.observed[2, 0, 1]


def test_invalid_motion_rule_and_reused_held_out_ranges_fail(tmp_path):
    masks, known, valid = fixture_masks()
    with pytest.raises(ValueError):
        estimate_contacts(masks, known, valid, max_gap_pixels=0, motion_rule="unknown")
    from click2future.evaluation.calibrate import main

    with pytest.raises(SystemExit):
        main(
            [
                "--method",
                "normal_motion",
                "--held-out-start",
                "110000",
                "--output",
                str(tmp_path / "record.json"),
                "--config-output",
                str(tmp_path / "config.yaml"),
            ]
        )
    assert not (tmp_path / "record.json").exists()
