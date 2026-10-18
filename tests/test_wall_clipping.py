"""Padded diagnostics distinguish lost canvas pixels from a smaller learned sprite."""

import numpy as np
import pytest

from click2future.evaluation.wall_clipping import sprite_areas
from click2future.models.object_motion import observe


def observation():
    image = np.full((64, 64, 3), 210, np.uint8)
    image[[0, -1]], image[:, [0, -1]] = 70, 70
    image[20:28, 10:18] = [220, 30, 30]
    image[20:28, 30:38] = [30, 120, 230]
    selected = np.zeros((64, 64), bool)
    selected[20:28, 10:18] = True
    return observe(image, selected)


def test_stationary_sprite_preserves_area():
    result = sprite_areas(observation(), np.zeros(4), 0)
    assert result == {
        "native_amodal_area": 64,
        "unclipped_amodal_area": 64,
        "clipped_fraction": 0,
    }


@pytest.mark.parametrize("pixels", [50, -14])
def test_half_square_lost_at_either_canvas_edge_is_measured_without_repair(pixels):
    obs = observation()
    before = obs.first.copy()
    result = sprite_areas(obs, np.array([pixels / 32, 0, 0, 0]), 0)
    assert result["native_amodal_area"] == 32
    assert result["unclipped_amodal_area"] == 64
    assert result["clipped_fraction"] == 0.5
    np.testing.assert_array_equal(obs.first, before)


def test_fully_off_canvas_sprite_has_no_visible_pixels():
    result = sprite_areas(observation(), np.array([80 / 32, 0, 0, 0]), 0)
    assert result["native_amodal_area"] == 0
    assert result["unclipped_amodal_area"] == 64
    assert result["clipped_fraction"] == 1


def test_insufficient_padding_cannot_silently_truncate_the_control():
    with pytest.raises(ValueError, match="too small"):
        sprite_areas(observation(), np.array([100 / 32, 0, 0, 0]), 0, padding=8)


@pytest.mark.parametrize("value", [np.zeros(3), np.array([np.nan, 0, 0, 0])])
def test_invalid_motion_is_refused(value):
    with pytest.raises(ValueError, match="finite"):
        sprite_areas(observation(), value, 0)
