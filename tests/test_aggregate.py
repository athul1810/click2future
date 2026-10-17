"""Scene dependence, missing cases, and provenance must survive table generation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from click2future.evaluation.aggregate import aggregate, csv_table, main
from click2future.evaluation.metrics import collision_metrics, trajectory_metrics


def make_report(index, error, run_id="run-0", *, missing=False):
    reference = np.full((3, 2, 2), 0.2)
    predicted = np.broadcast_to(reference, (2, *reference.shape)).copy()
    predicted[:, 1:, :, 0] += error
    tracked = np.ones((2, 3, 2), bool)
    if missing:
        tracked[:, 1:, 1] = False
    valid = np.ones(2, bool)
    contacts = np.zeros((2, 2, 2), bool)
    contacts[0, 0, 1] = contacts[0, 1, 0] = True
    guesses = np.stack([contacts, np.zeros_like(contacts)])
    return {
        "status": "passed",
        "source": "video_tracking",
        "metric_version": "1",
        "run_id": run_id,
        "sample_id": f"reference-{index}",
        "scene_group_id": f"scene-{index // 2}",
        "condition_id": f"condition-{index // 2}",
        "extractor_version": "test-1",
        "settings": {"tracker": "fixed-test"},
        "timestamps_seconds": [0, 0.125, 0.25],
        "input_sha256": {"videos": "a" * 64, "reference_arrays": str(index) * 64},
        "trajectory": trajectory_metrics(predicted, reference, valid, tracked, selected_object=0),
        "collision": collision_metrics(
            guesses, contacts, valid, np.ones_like(guesses), tolerance=1
        ),
    }


@pytest.fixture
def comparison(tmp_path):
    reports = []
    cases = []
    for index, error in enumerate((0.1, 0.3, 0.6)):
        report = make_report(index, error)
        name = f"case-{index}.json"
        (tmp_path / name).write_text(json.dumps(report))
        reports.append({"case_index": index, "trajectory": name})
        cases.append({key: report[key] for key in ("scene_group_id", "condition_id", "sample_id")})
    manifest = {
        "schema_version": 1,
        "source": "video_tracking",
        "dataset_id": "test-data",
        "protocol_id": "test-protocol",
        "suite": "id",
        "sampling_seeds": [10, 11],
        "cases": cases,
        "runs": [
            {
                "method": "E2",
                "run_id": "run-0",
                "training_seed": 1,
                "checkpoint_sha256": "c" * 64,
                "reports": reports,
            }
        ],
    }
    return manifest, tmp_path


def result(comparison):
    manifest, folder = comparison
    return aggregate(manifest, folder, bootstrap=100, seed=13)


def test_equal_scene_weight_and_collision_count_pooling(comparison):
    report = result(comparison)
    row = report["methods"][0]
    # Scene 0 has two variants averaging .2; scene 1 has .6. Equal scenes gives .4.
    assert row["metrics"]["all_objects.mean_ade"]["estimate"] == pytest.approx(0.4)
    assert row["metrics"]["all_objects.min_ade"]["estimate"] == pytest.approx(0.4)
    assert row["collision_micro"]["collision.event.f1"]["estimate"] == pytest.approx(2 / 3)
    assert row["counts"]["collision.event.true_positive"] == 3
    assert row["counts"]["collision.event.false_negative"] == 3
    assert row["collision_micro"]["collision.encounter.f1"]["estimate"] == pytest.approx(2 / 3)
    assert row["counts"]["collision.encounter.expected_pairs"] == 6
    assert "collision.encounter.accuracy" in csv_table(report)
    assert row["counts"]["all_objects.total_samples"] == 6
    assert len(report["provenance"]) == 3
    assert "all_objects.mean_ade" in csv_table(report)
    assert "count.collision.interval.missing_bins" in csv_table(report)
    assert report == result(comparison)


def test_legacy_collision_reports_still_aggregate_without_encounter_claims(comparison):
    _, folder = comparison
    for path in folder.glob("case-*.json"):
        report = json.loads(path.read_text())
        report["collision"].pop("collision_metric_version")
        report["collision"].pop("encounter_definition")
        for sample in report["collision"]["per_sample"]:
            sample.pop("encounter")
        path.write_text(json.dumps(report))
    row = result(comparison)["methods"][0]
    assert row["collision_micro"]["collision.event.f1"]["estimate"] == pytest.approx(2 / 3)
    assert not any("encounter" in key for key in row["collision_micro"])


def test_mixed_collision_versions_cannot_share_a_table(comparison):
    _, folder = comparison
    path = folder / "case-0.json"
    report = json.loads(path.read_text())
    report["collision"].pop("collision_metric_version")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Incompatible trajectory"):
        result(comparison)


def test_missing_case_invalidates_its_group_without_hiding_denominators(comparison):
    manifest, folder = comparison
    (folder / "case-0.json").write_text(json.dumps(make_report(0, 0.1, missing=True)))
    row = result(comparison)["methods"][0]
    metric = row["metrics"]["all_objects.mean_ade"]
    assert metric["estimate"] == pytest.approx(0.6)
    assert metric["scored_scene_groups"] == 1
    assert metric["expected_scene_groups"] == 2
    assert metric["ci"] is None
    assert row["counts"]["all_objects.complete_samples"] == 4
    assert row["counts"]["all_objects.total_samples"] == 6


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "failed",
        "settings",
        "source",
        "k",
        "identity",
        "time",
        "unhashed",
        "unknown-family",
    ],
)
def test_incomparable_or_incomplete_inputs_rejected(comparison, change):
    manifest, folder = comparison
    run = manifest["runs"][0]
    report = json.loads((folder / "case-0.json").read_text())
    if change == "missing":
        run["reports"].pop()
    elif change == "duplicate":
        run["reports"][0]["case_index"] = 1
    elif change == "failed":
        report = {"status": "failed", "error": "tracker crashed"}
    elif change == "settings":
        report["settings"]["tracker"] = "changed"
    elif change == "source":
        report["source"] = "auxiliary_head"
    elif change == "k":
        manifest["sampling_seeds"].append(12)
    elif change == "identity":
        report["sample_id"] = "another-reference"
    elif change == "time":
        report["timestamps_seconds"] = [0, 0.5, 1]
    elif change == "unhashed":
        report.pop("input_sha256")
    else:
        run["reports"][0]["anything"] = "some-file"
    (folder / "case-0.json").write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError)):
        result(comparison)


def test_reference_hash_must_match_across_runs(comparison):
    manifest, folder = comparison
    second = copy.deepcopy(manifest["runs"][0])
    second.update(run_id="run-1", method="E3")
    for index, entry in enumerate(second["reports"]):
        report = make_report(index, 0.1, run_id="run-1")
        report["input_sha256"]["reference_arrays"] = "changed"
        entry["trajectory"] = f"other-{index}.json"
        (folder / entry["trajectory"]).write_text(json.dumps(report))
    manifest["runs"].append(second)
    with pytest.raises(ValueError, match="Reference inputs differ"):
        result(comparison)


def test_null_event_score_stays_null_instead_of_perfect(comparison):
    _, folder = comparison
    for path in folder.glob("case-*.json"):
        report = json.loads(path.read_text())
        for sample in report["collision"]["per_sample"]:
            for name in ("true_positive", "false_positive", "false_negative"):
                sample["event"][name] = 0
        path.write_text(json.dumps(report))
    row = result(comparison)["methods"][0]
    assert row["collision_micro"]["collision.event.f1"]["estimate"] is None
    assert row["collision_micro"]["collision.event.f1"]["ci"] is None


def test_repeated_training_seed_is_not_an_independent_run(comparison):
    manifest, folder = comparison
    second = copy.deepcopy(manifest["runs"][0])
    second.update(run_id="run-1")
    for index, entry in enumerate(second["reports"]):
        report = make_report(index, 0.1, run_id="run-1")
        entry["trajectory"] = f"repeat-{index}.json"
        (folder / entry["trajectory"]).write_text(json.dumps(report))
    manifest["runs"].append(second)
    with pytest.raises(ValueError, match="Repeated training seeds"):
        result(comparison)


def test_head_table_has_its_own_source(comparison):
    manifest, folder = comparison
    manifest["source"] = "auxiliary_head"
    for path in folder.glob("case-*.json"):
        report = json.loads(path.read_text())
        report["source"] = "auxiliary_head"
        report["input_sha256"] = "a" * 64
        report.pop("extractor_version")
        path.write_text(json.dumps(report))
    assert result(comparison)["source"] == "auxiliary_head"


def test_real_rgb_reports_aggregate_together_and_detect_cross_family_corruption(tmp_path):
    pytest.importorskip("skimage")
    from click2future.evaluation.aggregate_controls import check

    folder = tmp_path / "controls"
    root = Path(__file__).resolve().parents[1]
    record = check(folder, root / "configs/evaluation/pixel.yaml")
    assert record["status"] == "passed"
    assert record["report_inputs"] == 24
    manifest = json.loads((folder / "comparison.json").read_text())
    # Individual reports can be valid while referring to different input video batches.
    entry = manifest["runs"][0]["reports"][0]
    path = folder / entry["images"]
    images = json.loads(path.read_text())
    images["input_sha256"]["videos"] = "b" * 64
    path.write_text(json.dumps(images))
    with pytest.raises(ValueError, match="same video batch"):
        aggregate(manifest, folder, bootstrap=10)


def test_cli_exports_and_preserves_inputs_on_error(comparison):
    manifest, folder = comparison
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    output, csv = folder / "summary.json", folder / "table.csv"
    args = [
        "--manifest",
        str(path),
        "--output",
        str(output),
        "--csv",
        str(csv),
        "--bootstrap",
        "20",
    ]
    assert main(args) == 0
    assert json.loads(output.read_text())["manifest_sha256"]
    assert len(csv.read_text().splitlines()) > 10
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(["--manifest", str(path), "--output", str(path), "--csv", str(csv)])
    assert path.read_bytes() == before
    output.write_text("existing table")
    manifest["runs"][0]["reports"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_text() == "existing table"
