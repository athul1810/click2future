"""Analytical temporal/ensemble controls and full RGB-to-report integration."""

import copy
import json

import numpy as np
import pytest

from click2future.data import save_rollout
from click2future.evaluation.diversity import collision_frequency, ensemble_metrics
from click2future.evaluation.futures import evaluate_futures, main
from click2future.evaluation.temporal import temporal_metrics
from click2future.evaluation.tracking import TrackerConfig
from click2future.sim import make_scene, run_rollout


def moving_square(xs=(2, 3, 4, 5, 6, 7)):
    frames = len(xs)
    videos = np.full((1, frames, 32, 32, 3), 20, np.uint8)
    masks = np.zeros((1, frames, 1, 32, 32), bool)
    positions = np.zeros((1, frames, 1, 2))
    for t, x in enumerate(xs):
        masks[0, t, 0, 12:18, x : x + 6] = True
        videos[0, t][masks[0, t, 0]] = 180
        positions[0, t, 0] = [(x + 3) / 32, 15 / 32]
    return dict(
        videos=videos,
        positions=positions,
        masks=masks,
        tracked=np.ones((1, frames, 1), bool),
        object_valid=np.ones(1, bool),
        timestamps=np.arange(frames) / 10,
        collision_onset=np.zeros((frames - 1, 1, 1), bool),
        wall_contact_onset=np.zeros((frames - 1, 1, 4), bool),
        selected_object=0,
        force_duration=0,
        contact_margin=0,
    )


def temporal(**kwargs):
    return temporal_metrics(**kwargs)["per_sample"][0]


def test_translation_compensation_and_constant_velocity():
    result = temporal(**moving_square())
    assert result["object_translation_mae_over_observed"] == 0
    assert result["background_mae_over_observed"] == 0
    assert result["mean_acceleration_norm_over_scored"] == pytest.approx(0, abs=1e-12)
    assert result["scored_object_transitions"] == 4
    assert result["scored_acceleration_windows"] == 3
    assert 0 < result["warp_valid_pixel_fraction"] < 1


def test_flicker_is_detected_after_motion_compensation():
    data = moving_square()
    for t in (1, 3, 5):
        data["videos"][0, t][data["masks"][0, t, 0]] += 20
    assert temporal(**data)["object_translation_mae_over_observed"] == pytest.approx(20 / 255)
    data = moving_square()
    for t in (1, 3, 5):
        data["videos"][0, t][~data["masks"][0, t, 0]] += 10
    assert temporal(**data)["background_mae_over_observed"] == pytest.approx(10 / 255)


def test_jitter_and_reference_contact_exclusions():
    assert temporal(**moving_square((2, 3, 5, 4, 6, 5)))["mean_acceleration_norm_over_scored"] > 0
    collision = moving_square((2, 4, 6, 6, 4, 2))
    assert temporal(**collision)["mean_acceleration_norm_over_scored"] > 0
    collision["wall_contact_onset"][2, 0, 0] = True
    result = temporal(**collision)
    assert result["mean_acceleration_norm_over_scored"] == pytest.approx(0, abs=1e-12)
    assert result["excluded_reference_contact_or_force_windows"] == 2
    assert result["scored_acceleration_windows"] == 1


def test_action_windows_and_frame_zero_are_excluded():
    data = moving_square((2, 3, 5, 8, 11, 14))
    data["force_duration"] = 0.3
    result = temporal(**data)
    assert result["excluded_reference_contact_or_force_windows"] == 2
    assert result["mean_acceleration_norm_over_scored"] == pytest.approx(0, abs=1e-12)
    data["videos"][:, 0] = 255
    assert temporal(**data) == result


def test_missing_tracks_cannot_turn_into_good_temporal_scores():
    data = moving_square()
    data["tracked"][0, 3, 0] = False
    data["positions"][0, 3, 0] = np.nan
    data["masks"][0, 3, 0] = False
    result = temporal(**data)
    assert result["scored_object_transitions"] == 2
    assert result["scored_background_transitions"] == 2
    assert result["mean_acceleration_norm_over_scored"] is None
    assert result["missing_acceleration_windows"] == 3
    data["tracked"][:, 1:] = False
    data["masks"][:, 1:] = False
    result = temporal(**data)
    assert result["object_translation_mae_over_observed"] is None
    assert result["background_mae_over_observed"] is None
    json.dumps(result, allow_nan=False)


def test_static_output_can_be_temporally_consistent():
    result = temporal(**moving_square((2, 2, 2, 2, 2, 2)))
    assert result["mean_acceleration_norm_over_scored"] == 0
    assert result["object_translation_mae_over_observed"] == 0


def test_irregular_timestamps_use_real_time_for_acceleration():
    data = moving_square()
    data["timestamps"] = np.array([0, 0.1, 0.2, 0.4, 0.7, 1.0])
    data["positions"][0, :, 0, 0] = 0.1 + 0.05 * data["timestamps"]
    assert temporal(**data)["mean_acceleration_norm_over_scored"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [("timestamps", [0, 1, 1, 2, 3, 4]), ("contact_margin", -1), ("force_duration", -1)],
)
def test_temporal_rejects_bad_contracts(field, value):
    data = moving_square()
    data[field] = value
    with pytest.raises(ValueError):
        temporal(**data)


def ensemble(offsets=(0, 0.1, 0.2), objects=1):
    positions = np.full((len(offsets), 4, objects, 2), 0.5)
    for k, offset in enumerate(offsets):
        positions[k, 1:, 0, 0] += offset
    return positions, np.ones(positions.shape[:-1], bool), np.ones(objects, bool)


def diversity(p, known, valid, refs=None):
    return ensemble_metrics(p, known, valid, p.copy() if refs is None else refs, selected_object=0)


def test_whole_trajectory_spread_and_reference_coverage():
    p, known, valid = ensemble()
    result = diversity(p, known, valid)["all_objects"]
    assert result["scored_sample_pairs"] == 3
    assert result["mean_pairwise_trajectory_distance"] == pytest.approx(0.4 / 3)
    assert result["mean_pairwise_endpoint_distance"] == pytest.approx(0.4 / 3)
    assert result["mean_reference_to_nearest_sample"] == 0
    assert result["mean_sample_to_nearest_reference"] == 0
    assert result["mean_reference_pairwise_trajectory_distance"] == pytest.approx(0.4 / 3)


def test_coverage_does_not_select_different_samples_per_frame():
    p, known, valid = ensemble((0, 0))
    p[0, 2:, 0, 0] += 0.3
    p[1, 1, 0, 0] += 0.3
    reference = np.full((1, 4, 1, 2), 0.5)
    result = diversity(p, known, valid, reference)["all_objects"]
    assert result["mean_reference_to_nearest_sample"] == pytest.approx(0.1)


def test_duplicate_and_single_sample_have_different_pair_denominators():
    p, known, valid = ensemble((0, 0, 0, 0))
    assert diversity(p, known, valid)["all_objects"]["mean_pairwise_trajectory_distance"] == 0
    result = diversity(p[:1], known[:1], valid)["all_objects"]
    assert result["mean_pairwise_trajectory_distance"] is None
    assert result["expected_sample_pairs"] == 0
    assert result["mean_reference_to_nearest_sample"] == 0


def test_missing_nonselected_object_blocks_whole_scene_coverage_only():
    p, known, valid = ensemble(objects=2)
    refs = p.copy()
    known[0, 2, 1] = False
    p[0, 2, 1] = np.nan
    result = diversity(p, known, valid, refs)
    assert result["all_objects"]["mean_reference_to_nearest_sample"] is None
    assert result["all_objects"]["scored_sample_pairs"] == 1
    assert result["selected_object"]["mean_reference_to_nearest_sample"] == 0
    valid[1] = False
    result = diversity(p, known, valid, refs)
    assert result["all_objects"]["mean_reference_to_nearest_sample"] == 0


def test_reference_futures_must_share_initial_state():
    p, known, valid = ensemble()
    refs = p.copy()
    refs[1, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="initial positions"):
        diversity(p, known, valid, refs)


def test_collision_frequency_requires_complete_pair_timelines():
    predicted = np.zeros((4, 3, 2, 2), bool)
    predicted[:2, 1, 0, 1] = predicted[:2, 1, 1, 0] = True
    observed = np.ones_like(predicted)
    refs = predicted[:2].copy()
    result = collision_frequency(predicted, observed, refs, np.ones(2, bool))["per_pair"][0]
    assert result["sample_any_event_rate"] == 0.5
    assert result["reference_any_event_rate"] == 1
    assert result["sample_bernoulli_variance"] == 0.25
    observed[0, 0, 0, 1] = observed[0, 0, 1, 0] = False
    result = collision_frequency(predicted, observed, refs, np.ones(2, bool))["per_pair"][0]
    assert result["sample_any_event_rate"] is None
    assert result["complete_sample_timelines"] == 3


def reference_variants():
    return [
        run_rollout(make_scene(70000, object_count=2, physics_seed=seed), frame_count=8)
        for seed in (1, 2)
    ]


def test_rgb_futures_pipeline_counts_missing_frames_and_rejects_changed_action():
    refs = reference_variants()
    videos = np.stack([r["frames"] for r in refs])
    settings = dict(
        tracker=TrackerConfig(),
        contact_settings={"max_gap_pixels": 0},
        source="simulator_control",
        run_id="test",
    )
    result = evaluate_futures(refs, videos, **settings)
    assert result["ensemble"]["all_objects"]["scored_sample_pairs"] == 1
    assert result["temporal"]["per_sample"][0]["scored_object_transitions"] > 0
    valid = np.ones(videos.shape[:2], bool)
    valid[0, 3] = False
    result = evaluate_futures(refs, videos, frame_valid=valid, **settings)
    assert result["ensemble"]["all_objects"]["mean_reference_to_nearest_sample"] is None
    assert result["tracking_reasons_future"]["missing_rgb"] == 2
    changed = copy.deepcopy(refs)
    changed[1]["scene_spec"]["action"]["force_xy"][0] += 0.1
    with pytest.raises(ValueError, match="share visible"):
        evaluate_futures(changed, videos, **settings)


def test_futures_cli(tmp_path):
    refs = reference_variants()
    for index, ref in enumerate(refs):
        save_rollout(ref, tmp_path, f"ref{index}")
    videos = tmp_path / "samples.npz"
    np.savez_compressed(videos, videos=np.stack([r["frames"] for r in refs]))
    output = tmp_path / "scores.json"
    args = [
        "--reference-dir",
        str(tmp_path),
        "--reference-ids",
        "ref0",
        "ref1",
        "--videos",
        str(videos),
        "--source",
        "simulator_control",
        "--run-id",
        "cli-test",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["reference_count"] == 2
    assert len(result["input_sha256"]) == 6
    assert result["ensemble"]["all_objects"]["total_samples"] == 2
    with pytest.raises(SystemExit):
        main(args + ["--output", str(videos)])
    videos.write_bytes(b"corrupt input")
    assert main(args) == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_temporal_exclusions_include_all_declared_reference_futures():
    refs = reference_variants()
    videos = np.stack([r["frames"] for r in refs])
    settings = dict(
        tracker=TrackerConfig(),
        contact_settings={"max_gap_pixels": 0},
        source="simulator_control",
        run_id="reference-union-test",
        contact_margin=0,
    )
    baseline = evaluate_futures(refs, videos, **settings)
    changed = copy.deepcopy(refs)
    changed[1]["wall_contact_onset"][3, 0, 0] = True
    result = evaluate_futures(changed, videos, **settings)
    before = baseline["temporal"]["per_sample"][0]["excluded_reference_contact_or_force_windows"]
    after = result["temporal"]["per_sample"][0]["excluded_reference_contact_or_force_windows"]
    assert after > before
    assert result["ensemble"] == baseline["ensemble"]
