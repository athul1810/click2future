import numpy as np
from test_physics_selection import observation

from click2future.models.projected_selection import select_projected


def test_projected_selection_prefers_small_repairs_then_preserves_diversity():
    obs = observation()
    raw = np.zeros((7, 4, 2, 4), dtype=np.float32)
    projected = raw.copy()
    projected[0, :, obs.selected_index, 0] = 0.25
    for index, distance in enumerate((0, 1, 2, 3, 4, 5), start=1):
        raw[index, :, obs.selected_index, 1] = distance / 32
        projected[index] = raw[index]

    result = select_projected(obs, raw, projected, [0.15, 0], output_count=3)

    assert result["quality_ranking"][0] == 1
    assert {1, 6} <= set(result["selected_indices"])
    assert 0 not in result["selected_indices"]
    assert result["scores"][0]["mean_projection_pixels"] > 0
