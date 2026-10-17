"""Frozen inputs, matched runs, and missing observations in the batch evaluator."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from click2future.data import generate_controlled_dataset, generate_dataset, load_rollout
from click2future.evaluation.suite import main, prepare_plan, run_suite
from click2future.evaluation.video import main as video_main

pytest.importorskip("skimage")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite-data")
    generate_dataset(
        root,
        group_count=20,
        variants_per_group=2,
        seed=180000,
        config={"frame_count": 4, "resolution": 64, "fps": 8},
    )
    return root


@pytest.fixture
def prepared(dataset, tmp_path):
    config = tmp_path / "pixel.yaml"
    config.write_bytes(
        (Path(__file__).resolve().parents[1] / "configs/evaluation/pixel.yaml").read_bytes()
    )
    plan = prepare_plan(dataset, config, metrics=("ssim",))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    predictions = {"source": "simulator_control", "sampling_seeds": [0, 1, 2, 3], "runs": []}
    for name in ("oracle", "static"):
        run = {"method": name, "run_id": name, "videos": {}}
        for index, case in enumerate(plan["cases"]):
            ref = load_rollout(dataset, case["sample_id"])
            clip = ref["frames"] if name == "oracle" else np.repeat(ref["frames"][:1], 4, axis=0)
            videos = np.repeat(clip[None], 4, axis=0)
            file = tmp_path / f"{name}-{index}.npz"
            np.savez_compressed(file, videos=videos)
            run["videos"][case["sample_id"]] = file.name
        predictions["runs"].append(run)
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions))
    return plan_path, prediction_path, plan, predictions


def test_plan_covers_all_held_out_conditions_and_all_reference_variants(prepared, dataset):
    _, _, plan, _ = prepared
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    expected = {row["sample_id"] for row in manifest["samples"] if row["split"] == "test"}
    actual = {name for case in plan["cases"] for name in case["reference_ids"]}
    assert actual == expected and len(plan["cases"]) == 2
    assert all(len(case["reference_ids"]) == 2 for case in plan["cases"])
    assert plan["sampling_seeds"] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="held-out"):
        prepare_plan(dataset, Path(plan["config"]), split="train")


def test_controlled_plan_keeps_all_action_families_and_physics_references(tmp_path):
    root = tmp_path / "controlled"
    result = generate_controlled_dataset(root, group_count=10, seed=194010)
    config_dir = Path(__file__).resolve().parents[1] / "configs/evaluation"
    plan = prepare_plan(
        root, config_dir / "pixel.yaml", metrics=("ssim",), protocol=config_dir / "comparison.yaml"
    )
    expected = {row["sample_id"] for row in result["manifest"]["samples"] if row["split"] == "test"}
    assert {name for case in plan["cases"] for name in case["reference_ids"]} == expected
    assert len(plan["cases"]) == 6
    assert sorted(len(case["reference_ids"]) for case in plan["cases"]) == [1, 4, 4, 4, 4, 4]
    assert len({case["scene_group_id"] for case in plan["cases"]}) == 1


def test_two_run_pipeline_exports_all_metric_families_and_counts(prepared, tmp_path):
    plan_path, prediction_path, plan, _ = prepared
    folder = tmp_path / "results"
    report = run_suite(plan_path, prediction_path, folder, bootstrap=100)
    assert report["status"] == "passed" and report["source"] == "simulator_control"
    assert len(report["provenance"]) == 2 * len(plan["cases"]) * 3
    oracle = next(row for row in report["methods"] if row["method"] == "oracle")
    assert oracle["metrics"]["image.ssim"]["estimate"] == pytest.approx(1)
    assert oracle["counts"]["all_objects.total_samples"] == 8
    assert oracle["metrics"]["all_objects.mean_ade"]["estimate"] < 0.02
    assert oracle["counts"]["ensemble.all_objects.reference_futures"] == 4
    assert (folder / "summary.csv").is_file() and not (folder / "failure.json").exists()
    manifest = json.loads((folder / "comparison.json").read_text())
    assert all(
        set(entry) == {"case_index", "trajectory", "images", "futures"}
        for run in manifest["runs"]
        for entry in run["reports"]
    )
    with pytest.raises(FileExistsError):
        run_suite(plan_path, prediction_path, folder, bootstrap=100)


@pytest.mark.parametrize(
    "mutation", ["case", "seeds", "boolean_seed", "run_id", "source", "plan", "config"]
)
def test_changed_or_incomplete_inputs_cannot_produce_a_comparison(prepared, tmp_path, mutation):
    plan_path, prediction_path, plan, predictions = prepared
    changed = copy.deepcopy(predictions)
    if mutation == "case":
        changed["runs"][0]["videos"].pop(plan["cases"][0]["sample_id"])
    elif mutation == "seeds":
        changed["sampling_seeds"] = [4, 5, 6, 7]
    elif mutation == "boolean_seed":
        changed["sampling_seeds"] = [False, 1, 2, 3]
    elif mutation == "run_id":
        changed["runs"][1]["run_id"] = changed["runs"][0]["run_id"]
    elif mutation == "source":
        changed["runs"][0]["method"] = "E3"
    elif mutation == "plan":
        plan["cases"].pop()
        plan_path.write_text(json.dumps(plan))
    else:
        Path(plan["config"]).write_text("changed after freezing")
    prediction_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        run_suite(plan_path, prediction_path, tmp_path / "bad")
    assert not (tmp_path / "bad" / "summary.json").exists()


def test_missing_rgb_is_counted_by_all_families_and_the_video_cli(prepared, dataset, tmp_path):
    plan_path, prediction_path, plan, predictions = prepared
    case = plan["cases"][0]
    video_path = tmp_path / predictions["runs"][0]["videos"][case["sample_id"]]
    with np.load(video_path) as archive:
        videos = archive["videos"]
    valid = np.ones(videos.shape[:2], dtype=bool)
    valid[0, 2] = False
    np.savez_compressed(video_path, videos=videos, frame_valid=valid)
    folder = tmp_path / "missing"
    report = run_suite(plan_path, prediction_path, folder, bootstrap=100)
    oracle = next(row for row in report["methods"] if row["method"] == "oracle")
    assert oracle["counts"]["all_objects.tracked_future_observations"] == 46
    assert oracle["counts"]["all_objects.expected_future_observations"] == 48
    assert oracle["counts"]["image.missing_future_frames"] == 1
    future = json.loads((folder / "oracle-00000-futures.json").read_text())
    assert future["tracking_reasons_future"]["missing_rgb"] == 2
    output = tmp_path / "single.json"
    assert (
        video_main(
            [
                "--reference-dir",
                str(dataset),
                "--sample-id",
                case["sample_id"],
                "--videos",
                str(video_path),
                "--config",
                plan["config"],
                "--run-id",
                "missing-cli",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    single = json.loads(output.read_text())
    assert single["tracking_reasons_future"]["missing_rgb"] == 2


def test_bad_video_records_failure_and_never_writes_a_success_table(prepared, tmp_path):
    plan_path, prediction_path, plan, predictions = prepared
    name = predictions["runs"][0]["videos"][plan["cases"][0]["sample_id"]]
    np.savez_compressed(tmp_path / name, videos=np.zeros((3, 4, 64, 64, 3), np.uint8))
    output = tmp_path / "failure"
    with pytest.raises(ValueError, match="Prediction K"):
        run_suite(plan_path, prediction_path, output)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    assert not (output / "summary.json").exists()


def test_model_runs_require_checkpoint_provenance(prepared, tmp_path):
    plan_path, prediction_path, _, predictions = prepared
    predictions["source"] = "generated_video"
    for index, run in enumerate(predictions["runs"]):
        run.update(method=f"E{index + 2}", training_seed=0, checkpoint="missing.pt")
    prediction_path.write_text(json.dumps(predictions))
    with pytest.raises(FileNotFoundError):
        run_suite(plan_path, prediction_path, tmp_path / "no-checkpoint")


def test_e5_keeps_generated_video_provenance_and_shared_metrics(prepared, tmp_path):
    plan_path, prediction_path, _, predictions = prepared
    predictions["source"] = "generated_video"
    checkpoint = tmp_path / "unit-fixture.pt"
    checkpoint.write_bytes(b"unit fixture; supplied video is a simulator control")
    run = predictions["runs"][0]
    run.update(method="E5", training_seed=19, checkpoint=checkpoint.name)
    predictions["runs"] = [run]
    prediction_path.write_text(json.dumps(predictions))
    report = run_suite(plan_path, prediction_path, tmp_path / "e5-unit", bootstrap=100)
    assert report["source"] == "video_tracking"
    assert report["methods"][0]["method"] == "E5"
    assert report["methods"][0]["metrics"]["image.ssim"]["estimate"] == pytest.approx(1)


def test_changed_evaluator_code_invalidates_the_plan(prepared, tmp_path, monkeypatch):
    from click2future.evaluation import suite

    plan_path, prediction_path, _, _ = prepared
    monkeypatch.setattr(suite, "_evaluator_sources", lambda: {"video.py": "0" * 64})
    with pytest.raises(ValueError, match="Evaluator source changed"):
        run_suite(plan_path, prediction_path, tmp_path / "changed-code")


def test_prepare_cli_refuses_to_replace_an_existing_plan(prepared, dataset):
    plan_path, _, plan, _ = prepared
    import yaml

    protocol = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/evaluation/comparison.yaml").read_text()
    )
    protocol["total_frames"] = 4
    protocol_path = plan_path.parent / "comparison.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol))
    before = plan_path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(
            [
                "prepare",
                "--dataset",
                str(dataset),
                "--config",
                plan["config"],
                "--metrics",
                "ssim",
                "--protocol",
                str(protocol_path),
                "--output",
                str(plan_path),
            ]
        )
    assert error.value.code == 1 and plan_path.read_bytes() == before
