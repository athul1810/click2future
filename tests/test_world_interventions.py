"""The latent-mass diagnostic uses signed force projection, not absolute travel."""

import numpy as np
import pytest

from click2future.evaluation.world_interventions import mass_effects


def test_mass_intervention_keeps_equal_wrong_and_correct_effects_separate():
    heavy = np.zeros((3, 4, 2, 4), np.float32)
    light = heavy.copy()
    light[:, -1, 1, 0] = [-0.1, 0, 0.1]
    result = mass_effects(light, heavy, [-0.25, 0], 1)
    assert result["samples"] == 3
    assert result["lighter_further"] == result["equal"] == result["lighter_shorter"] == 1
    assert result["passive_maximum_excursion_pixels"]["light"] == [0, 0, 0]
    np.testing.assert_allclose(result["light_minus_heavy_endpoint_pixels"], [3.2, 0, -3.2])
    with pytest.raises(ValueError):
        mass_effects(light, heavy, [0, 0], 1)
