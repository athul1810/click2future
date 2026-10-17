"""Geometry audits expose compositing overlap and keep off-canvas failures."""

import numpy as np
import pytest

from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.models.object_motion import observe, render_motion


def fixture_observation():
    image = np.full((64, 64, 3), 210, np.uint8)
    image[[0, -1]], image[:, [0, -1]] = 70, 70
    image[20:28, 10:18] = [220, 30, 30]
    image[20:28, 30:38] = [30, 120, 230]
    mask = np.zeros((64, 64), bool)
    mask[20:28, 10:18] = True
    return observe(image, mask)


def test_identity_has_no_overlap_or_boundary_failure_and_excludes_frame_zero():
    report = geometry_diagnostics(fixture_observation(), np.zeros((2, 3, 2, 4)))
    assert report["expected_pair_frames"] == 6
    assert report["expected_object_frames"] == 12
    assert report["overlap_frames"] == report["outside_interior_object_frames"] == 0
    assert not report["frame_zero_scored"]


def test_hidden_sprite_overlap_matches_actual_compositing():
    obs = fixture_observation()
    motion = np.zeros((1, 3, 2, 4))
    motion[:, :, 0, 0] = 20 / 32
    report = geometry_diagnostics(obs, motion)
    clip = report["clips"][0]
    assert clip["overlap_pixels_per_frame"] == [64] * 3
    assert clip["maximum_overlap_fraction"] == 1
    assert report["clips_with_overlap_over_ten_percent"] == 1
    frames = render_motion(obs, motion[0])
    # The later blue sprite hides every red pixel. A successful RGB rendering
    # does not remove the underlying overlap from this separate pose diagnostic.
    assert not np.all(frames[1:] == [220, 30, 30], axis=-1).any()
    assert np.all(frames[1:] == [30, 120, 230], axis=-1).sum() == 64 * 3


def test_off_canvas_object_is_counted_even_when_no_silhouette_survives():
    motion = np.zeros((1, 2, 2, 4))
    motion[:, :, 0, 0] = -3
    report = geometry_diagnostics(fixture_observation(), motion)
    assert report["outside_interior_object_frames"] == 2
    assert report["clips_outside_interior"] == 1
    assert report["clips"][0]["maximum_outside_pixel_fraction"] == 1
    assert report["expected_object_frames"] == 4


def test_rotation_and_translation_do_not_mutate_inputs():
    obs = fixture_observation()
    motion = np.zeros((1, 2, 2, 4))
    motion[:, :, 0, 2:] = [1, -0.5]
    motion[:, :, 0, 1] = -1
    original = motion.copy()
    first = obs.first.copy()
    assert geometry_diagnostics(obs, motion)["outside_interior_object_frames"] == 2
    np.testing.assert_array_equal(motion, original)
    np.testing.assert_array_equal(obs.first, first)


@pytest.mark.parametrize(
    "motion", [np.zeros((0, 2, 2, 4)), np.zeros((1, 2, 3, 4)), np.full((1, 2, 2, 4), np.nan)]
)
def test_invalid_geometry_inputs_fail(motion):
    with pytest.raises(ValueError, match="finite future motions"):
        geometry_diagnostics(fixture_observation(), motion)
