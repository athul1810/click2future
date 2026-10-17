"""Pixel-only tracking and contact controls, including failures and identity crossings."""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from click2future.data import save_rollout
from click2future.evaluation.contacts import estimate_contacts
from click2future.evaluation.tracking import TrackerConfig, track_video
from click2future.evaluation.video import evaluate_videos, main
from click2future.sim import make_scene, run_rollout


def rectangles(positions, colors=((220, 30, 30), (30, 150, 230)), size=5):
    frames = np.full((len(positions), 32, 32, 3), 210, np.uint8)
    masks = np.zeros((len(positions), 2, 32, 32), bool)
    for t, locations in enumerate(positions):
        for obj, (x, y) in enumerate(locations):
            masks[t, obj, y : y + size, x : x + size] = True
            frames[t, y : y + size, x : x + size] = colors[obj]
    return frames, masks


def test_centres_and_identity_survive_crossing_paths():
    frames, masks = rectangles([[(2, 2), (22, 22)], [(12, 2), (12, 22)], [(22, 2), (2, 22)]])
    result = track_video(frames, masks[0], np.ones(2, bool))
    assert result.tracked.all()
    np.testing.assert_allclose(result.positions[:, 0, 0], (np.array([2, 12, 22]) + 2.5) / 32)
    np.testing.assert_allclose(result.positions[:, 1, 0], (np.array([22, 12, 2]) + 2.5) / 32)


def test_color_tolerance_is_explicit():
    frames, masks = rectangles([[(2, 2), (22, 22)]] * 2)
    frames[1][masks[1, 0]] += np.array([5, 5, 5], np.uint8)
    result = track_video(frames, masks[0], np.ones(2, bool))
    assert result.tracked.all()
    strict = track_video(
        frames, masks[0], np.ones(2, bool), replace(TrackerConfig(), color_tolerance=0)
    )
    assert not strict.tracked[1, 0]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "area", "same_color"])
def test_corrupt_or_ambiguous_objects_are_not_silently_tracked(failure):
    frames, masks = rectangles([[(2, 2), (22, 22)]] * 2)
    if failure == "missing":
        frames[1][masks[1, 0]] = 210
    elif failure == "duplicate":
        frames[1, 12:17, 12:17] = [220, 30, 30]
    elif failure == "area":
        frames[1, 2:13, 2:13] = [220, 30, 30]
    else:
        frames[:, 22:27, 22:27] = [220, 30, 30]
    result = track_video(frames, masks[0], np.ones(2, bool))
    assert not result.tracked[1, 0]
    assert np.isnan(result.positions[1, 0]).all()
    assert not result.masks[1, 0].any()


def test_isolated_specks_do_not_reject_a_visible_object():
    frames, masks = rectangles([[(2, 2), (22, 22)]] * 2)
    clean = track_video(frames, masks[0], np.ones(2, bool))
    # Scattered single pixels in the object's colour, like background noise near its palette.
    for y, x in [(14, 0), (14, 4), (0, 14), (4, 18), (30, 9), (18, 30), (9, 30), (31, 31)]:
        frames[1, y, x] = [220, 30, 30]
    result = track_video(frames, masks[0], np.ones(2, bool))
    assert result.tracked.all()
    np.testing.assert_array_equal(result.positions, clean.positions)
    assert not result.masks[1, 0, 14, 0]


def test_fragment_at_least_min_pixels_is_still_rejected():
    frames, masks = rectangles([[(2, 2), (22, 22)]] * 2)
    # Six pixels is above min_pixels and leaves the main piece below dominant_fraction.
    frames[1, 12:14, 12:15] = [220, 30, 30]
    result = track_video(frames, masks[0], np.ones(2, bool))
    assert not result.tracked[1, 0]
    assert result.reasons[1, 0] == "fragmented_or_duplicate"


@pytest.mark.parametrize("min_pixels", [2, 4, 6])
@pytest.mark.parametrize("at_cutoff", [False, True])
def test_fragment_cutoff_uses_configured_minimum(min_pixels, at_cutoff):
    frames, masks = rectangles([[(2, 2), (22, 22)]] * 2, size=3)
    config = replace(TrackerConfig(), min_pixels=min_pixels)
    clean = track_video(frames, masks[0], np.ones(2, bool), config)
    fragment_size = min_pixels if at_cutoff else min_pixels - 1
    frames[1, 12, 12 : 12 + fragment_size] = [220, 30, 30]
    result = track_video(frames, masks[0], np.ones(2, bool), config)
    if at_cutoff:
        assert not result.tracked[1, 0]
        assert result.reasons[1, 0] == "fragmented_or_duplicate"
    else:
        np.testing.assert_array_equal(result.tracked, clean.tracked)
        np.testing.assert_array_equal(result.positions, clean.positions)
        np.testing.assert_array_equal(result.masks, clean.masks)


def test_padding_is_ignored():
    frames, masks = rectangles([[(2, 2), (22, 22)]])
    initial = np.concatenate([masks[0], np.zeros((1, 32, 32), bool)])
    result = track_video(frames, initial, np.array([True, True, False]))
    assert result.tracked[0, :2].all()
    assert not result.tracked[0, 2]
    assert result.reasons[0, 2] == "padding"


def test_contact_onsets_do_not_repeat_during_persistent_contact():
    _, masks = rectangles(
        [[(2, 2), (15, 2)], [(2, 2), (7, 2)], [(2, 2), (7, 2)], [(2, 2), (15, 2)], [(2, 2), (7, 2)]]
    )
    result = estimate_contacts(masks, np.ones((5, 2), bool), np.ones(2, bool), max_gap_pixels=0)
    np.testing.assert_array_equal(result.onset[:, 0, 1], [True, False, False, True])
    assert np.array_equal(result.onset, result.onset.swapaxes(1, 2))
    assert not np.diagonal(result.onset, axis1=1, axis2=2).any()


def test_contact_gap_and_missing_observation_semantics():
    _, masks = rectangles([[(2, 2), (15, 2)], [(2, 2), (8, 2)], [(2, 2), (8, 2)]])
    tracked = np.ones((3, 2), bool)
    tight = estimate_contacts(masks, tracked, np.ones(2, bool), max_gap_pixels=0)
    wider = estimate_contacts(masks, tracked, np.ones(2, bool), max_gap_pixels=1)
    assert not tight.onset.any()
    assert wider.onset[0, 0, 1]
    tracked[1, 0] = False
    missing = estimate_contacts(masks, tracked, np.ones(2, bool), max_gap_pixels=1)
    assert not missing.observed.any()
    assert not missing.onset.any()


def test_initial_contact_is_not_a_future_onset():
    _, masks = rectangles([[(2, 2), (7, 2)]] * 2)
    result = estimate_contacts(masks, np.ones((2, 2), bool), np.ones(2, bool), max_gap_pixels=0)
    assert result.state[:, 0, 1].all()
    assert not result.onset.any()


@pytest.mark.parametrize(
    "setting",
    [
        dict(color_tolerance=-1),
        dict(min_area_ratio=2),
        dict(color_margin=0),
        dict(dominant_fraction=0.5),
    ],
)
def test_invalid_tracker_settings_fail(setting):
    with pytest.raises(ValueError):
        TrackerConfig(**setting).validate()


@pytest.mark.parametrize("motion_threshold", [None, 0.3])
def test_simulator_video_extraction_does_not_use_future_oracle_masks_or_positions(motion_threshold):
    reference = run_rollout(make_scene(31, object_count=3), frame_count=8)
    videos = reference["frames"][None]
    first, arrays = evaluate_videos(
        reference,
        videos,
        tracker=TrackerConfig(),
        max_gap_pixels=0,
        event_tolerance=1,
        run_id="test",
        motion_threshold=motion_threshold,
    )
    corrupted = copy.deepcopy(reference)
    corrupted["instance_masks"][1:] = False
    corrupted["positions"][1:] = 0.9
    corrupted["collision_onset"][:] = False
    _, second = evaluate_videos(
        corrupted,
        videos,
        tracker=TrackerConfig(),
        max_gap_pixels=0,
        event_tolerance=1,
        run_id="test",
        motion_threshold=motion_threshold,
    )
    for key in [
        "predicted_positions",
        "tracked",
        "predicted_collision_onset",
        "collision_observed",
    ]:
        np.testing.assert_array_equal(arrays[key], second[key])
    assert first["source"] == "video_tracking"
    assert first["trajectory"]["all_objects"]["mean_ade_over_complete_samples"] < 0.025


def test_changed_first_frame_is_rejected():
    reference = run_rollout(make_scene(31), frame_count=2)
    videos = reference["frames"][None].copy()
    videos[0, 0, 1, 1] = 0
    with pytest.raises(ValueError, match="first frame"):
        evaluate_videos(
            reference,
            videos,
            tracker=TrackerConfig(),
            max_gap_pixels=0,
            event_tolerance=1,
            run_id="test",
        )


def test_video_cli_exports_scores_and_extracted_arrays(tmp_path, capsys):
    import yaml

    reference = run_rollout(make_scene(31), frame_count=4)
    save_rollout(reference, tmp_path, "reference")
    video_path = tmp_path / "videos.npz"
    np.savez_compressed(video_path, videos=reference["frames"][None])
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "resolution_hw": [64, 64],
                "tracker": {},
                "max_gap_pixels": 0,
                "event_tolerance_intervals": 1,
            }
        )
    )
    output, extracted = tmp_path / "report.json", tmp_path / "extracted.npz"
    args = [
        "--reference-dir",
        str(tmp_path),
        "--sample-id",
        "reference",
        "--videos",
        str(video_path),
        "--config",
        str(config),
        "--run-id",
        "control",
        "--output",
        str(output),
        "--extracted-output",
        str(extracted),
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.read_text())
    assert len(report["input_sha256"]) == 4
    with np.load(extracted, allow_pickle=False) as arrays:
        assert arrays["predicted_positions"].shape == (1, 4, 2, 2)


def test_motion_contact_uses_impulse_instead_of_first_pixel_proximity():
    # Pixel regions remain near for several frames before momentum transfers.
    _, masks = rectangles(
        [
            [(x, 2), (y, 2)]
            for x, y in [(2, 10), (4, 10), (6, 10), (8, 10), (8, 12), (8, 14), (8, 16)]
        ],
        size=2,
    )
    result = estimate_contacts(
        masks,
        np.ones((7, 2), bool),
        np.ones(2, bool),
        max_gap_pixels=0,
        motion_threshold=0.3,
        motion_window=3,
    )
    events = np.flatnonzero(result.onset[:, 0, 1])
    assert len(events) == 1
    assert abs(events[0] - 3) <= 1
    assert result.observed[:, 0, 1].all()


def test_motion_contact_rejects_near_pass_without_impulse():
    _, masks = rectangles([[(x, 2), (10, 4)] for x in range(2, 17, 2)], size=2)
    result = estimate_contacts(
        masks,
        np.ones((8, 2), bool),
        np.ones(2, bool),
        max_gap_pixels=0,
        motion_threshold=0.1,
        motion_window=3,
    )
    assert result.state.any()  # Raster proximity alone would report a contact.
    assert not result.onset.any()


def test_missing_track_masks_entire_motion_context():
    _, masks = rectangles([[(2, 2), (7, 2)]] * 10)
    tracked = np.ones((10, 2), bool)
    tracked[4, 0] = False
    result = estimate_contacts(
        masks, tracked, np.ones(2, bool), max_gap_pixels=0, motion_threshold=0.1, motion_window=3
    )
    assert not result.observed[1:7, 0, 1].any()
    assert result.observed[0, 0, 1]
    assert not result.onset.any()


@pytest.mark.parametrize(
    "options",
    [
        dict(motion_threshold=0),
        dict(motion_threshold=float("nan")),
        dict(motion_threshold=0.1, motion_window=1),
    ],
)
def test_invalid_motion_contact_settings_fail(options):
    _, masks = rectangles([[(2, 2), (7, 2)]] * 2)
    with pytest.raises(ValueError):
        estimate_contacts(
            masks, np.ones((2, 2), bool), np.ones(2, bool), max_gap_pixels=0, **options
        )
