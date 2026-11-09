import numpy as np
from PIL import Image, ImageDraw

from click2future.models.object_motion import observe
from click2future.models.physics_selection import score_physics, select_physics_guided


def observation():
    image = Image.new("RGB", (64, 64), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 63, 63), outline=(40, 45, 50))
    draw.ellipse((12, 28, 20, 36), fill=(170, 60, 50))
    draw.ellipse((44, 28, 52, 36), fill=(50, 100, 190))
    frame = np.asarray(image)
    return observe(frame, np.all(frame == [170, 60, 50], axis=-1))


def test_physics_score_detects_reference_free_failures():
    obs = observation()
    motions = np.zeros((4, 4, 2, 4), dtype=np.float32)
    motions[1, :, obs.selected_index, 0] = -0.25
    motions[2, :, obs.selected_index, 0] = 1.5
    motions[3, :, obs.selected_index, 0] = 0.875
    scores = score_physics(obs, motions, [0.15, 0])
    assert scores[0]["total"] == 0
    assert scores[1]["force_opposition_pixels"] > 0
    assert scores[2]["boundary_excess_pixels"] > 0
    assert scores[3]["penetration_excess_pixels"] > 0


def test_selection_keeps_the_best_candidate_and_diverse_clean_futures():
    obs = observation()
    motions = np.zeros((6, 4, 2, 4), dtype=np.float32)
    for index, distance in enumerate((0, 1, 2, 3, 4)):
        motions[index, :, obs.selected_index, 1] = distance / 32
    motions[5, :, obs.selected_index, 0] = -0.5
    result = select_physics_guided(obs, motions, [0.15, 0], output_count=3, shortlist_size=6)
    assert result["selected_indices"][0] == 0
    assert set(result["selected_indices"]) == {0, 2, 4}
    assert 5 not in result["selected_indices"]


def test_selection_retains_every_hard_valid_candidate_when_there_are_too_few():
    obs = observation()
    motions = np.zeros((5, 4, 2, 4), dtype=np.float32)
    for index in range(2, 5):
        motions[index, :, obs.selected_index, 0] = 1.5 + index / 10
    result = select_physics_guided(obs, motions, [0.15, 0], output_count=4)
    assert {0, 1} <= set(result["selected_indices"])
    assert result["hard_valid_candidates"] == 2
