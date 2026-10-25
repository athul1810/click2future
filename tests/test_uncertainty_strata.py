"""Late drift, missing tracks and incomplete coverage remain visible."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from click2future.evaluation.futures import evaluate_futures
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import stratify, zero_horizon
from click2future.sim import run_rollout


@pytest.fixture
def reference():
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    scene["action"]["force_xy"] = [0, 0]
    return run_rollout(scene, frame_count=8, resolution=64, fps=8)


def test_late_motion_fails_full_horizon_while_static_copy_passes(reference):
    videos = reference["frames"][None].copy()
    assert zero_horizon(reference, videos)["drifting_samples"] == 0
    videos[:, -1] = np.roll(videos[:, -1], 3, axis=2)
    report = zero_horizon(reference, videos)
    assert report["end_frame"] == 7 and report["drifting_samples"] == 1
    assert report["mean_max_drift"] == pytest.approx(3 / 64)


def test_missing_zero_force_object_is_not_scored_as_stationary(reference):
    videos = reference["frames"][None].copy()
    videos[:, -1] = 210
    report = zero_horizon(reference, videos)
    assert report["missing_samples"] == 1 and report["complete_samples"] == 0
    assert report["drift_rate_over_complete"] is None


def test_moving_or_nonzero_force_reference_is_not_stationarity_control(reference):
    changed = dict(reference, force_xy=np.array([0.1, 0]))
    with pytest.raises(ValueError, match="stationary reference"):
        zero_horizon(changed, reference["frames"][None])


@pytest.fixture
def comparison(reference, tmp_path):
    cases, reports = [], []
    for index, family in enumerate(("zero", "collision")):
        group = "scene-0"
        ref = copy.deepcopy(reference)
        ref["scene_spec"].update(scene_group_id=group, condition_id=group + "-" + family)
        videos = ref["frames"][None].copy()
        if family == "collision":
            videos[:, -1] = 210
        report = evaluate_futures(
            [ref],
            videos,
            tracker=TrackerConfig(),
            contact_settings={"max_gap_pixels": 1},
            source="generated_video",
            run_id="fixture",
        )
        name = f"sample-{index}"
        report.update(
            reference_ids=[name],
            input_sha256={
                "videos": "a" * 64,
                "config": "b" * 64,
                name + ".npz": "c" * 64,
                name + ".json": "d" * 64,
            },
        )
        filename = family + "-futures.json"
        (tmp_path / filename).write_text(json.dumps(report))
        cases.append(
            {
                "scene_group_id": group,
                "condition_id": group + "-" + family,
                "sample_id": name,
                "reference_ids": [name],
            }
        )
        reports.append({"case_index": index, "futures": filename})
    manifest = {
        "schema_version": 1,
        "source": "video_tracking",
        "suite": "validation",
        "dataset_id": "fixture",
        "protocol_id": "fixture",
        "sampling_seeds": [0],
        "cases": cases,
        "runs": [
            {
                "method": "E5",
                "run_id": "fixture",
                "training_seed": 0,
                "checkpoint_sha256": "e" * 64,
                "reports": reports,
            }
        ],
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("method", ["E5", "E6", "E7"])
def test_strata_keep_failed_family_and_undefined_reference_spread(comparison, method):
    manifest = json.loads(comparison.read_text())
    manifest["runs"][0]["method"] = method
    comparison.write_text(json.dumps(manifest))
    result = stratify(comparison, bootstrap=10)
    assert result["total_case_count"] == 2
    assert sum(row["case_count"] for row in result["action_strata"].values()) == 2
    failed = result["action_strata"]["collision"]["methods"][0]
    assert failed["counts"]["ensemble.all_objects.complete_samples"] == 0
    assert failed["metrics"]["all_objects.mean_reference_to_nearest_sample"]["estimate"] is None
    assert failed["collision_frequency"]["mean_generated_any_contact_rate_if_complete"] is None
    zero = result["action_strata"]["zero"]["methods"][0]
    assert (
        zero["metrics"]["all_objects.mean_reference_pairwise_trajectory_distance"]["estimate"]
        is None
    )


def test_missing_report_rejects_stratification_instead_of_dropping_case(comparison):
    comparison.with_name("collision-futures.json").unlink()
    with pytest.raises(FileNotFoundError):
        stratify(comparison, bootstrap=10)


def test_final_test_reports_cannot_become_validation_strata(comparison):
    manifest = json.loads(comparison.read_text())
    manifest["suite"] = "id_test"
    comparison.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="validation RGB"):
        stratify(comparison, bootstrap=10)
