"""Relocated audit inputs retain frozen validation and artifact identities."""

import json

import pytest

from click2future.evaluation.suite import _digest
from click2future.evaluation.trajectory_audit import verify_inputs
from click2future.training.train_cvae import sha256


@pytest.fixture
def inputs(tmp_path):
    data, predictions = tmp_path / "data", tmp_path / "predictions"
    data.mkdir()
    predictions.mkdir()
    manifest = {
        "samples": [
            {
                "sample_id": "one",
                "split": "validation",
                "scene_group_id": "scene",
                "condition_id": "forward",
            }
        ]
    }
    (data / "dataset_manifest.json").write_text(json.dumps(manifest))
    for name in ("splits.json", "one.json", "one.npz"):
        (data / name).write_bytes(b"transport fixture only; no arrays should load")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"transport fixture only; no model should load")
    (predictions / "one.npz").write_bytes(b"frozen pixel archive transport fixture")
    plan = {
        "schema_version": 1,
        "split": "validation",
        "sampling_seeds": [0, 1, 2, 3],
        "cases": [
            {
                "sample_id": "one",
                "scene_group_id": "scene",
                "condition_id": "forward",
                "reference_ids": ["one"],
            }
        ],
        "input_sha256": {
            key: sha256(data / name)
            for key, name in (
                ("dataset_manifest", "dataset_manifest.json"),
                ("splits", "splits.json"),
                ("one.npz", "one.npz"),
                ("one.json", "one.json"),
            )
        },
    }
    plan["plan_sha256"] = _digest(plan)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    run = {
        "method": "E5",
        "checkpoint_sha256": sha256(checkpoint),
        "untrained_fixture": False,
        "videos": {"one": "one.npz"},
        "video_sha256": {"one": sha256(predictions / "one.npz")},
    }
    recorded = {
        "source": "generated_video",
        "runs": [run],
        "evaluation_plan_sha256": plan["plan_sha256"],
        "sampling_seeds": [0, 1, 2, 3],
    }
    (predictions / "predictions.json").write_text(json.dumps(recorded))
    (predictions / "sampling.json").write_text(
        json.dumps({**recorded, "sampling_steps": 16, "network_evaluations_per_sample": 31})
    )
    return tmp_path / "plan.json", predictions / "predictions.json", data, checkpoint


def test_relocated_inputs_are_verified_before_arrays_or_models_load(inputs):
    plan, run, paths, hashes = verify_inputs(*inputs)
    assert plan["split"] == "validation" and run["method"] == "E5"
    assert set(paths) == {"one"} and len(hashes) == 9


@pytest.mark.parametrize(
    "corruption", ["test_split", "plan", "pixels", "missing", "path", "reference", "sampler"]
)
def test_invalid_or_incomplete_audit_cannot_relabel_saved_evidence(inputs, corruption):
    plan_path, prediction_path, data, _ = inputs
    plan = json.loads(plan_path.read_text())
    predictions = json.loads(prediction_path.read_text())
    if corruption in ("test_split", "plan"):
        plan["split"] = "id_test"
        if corruption == "test_split":
            plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        plan_path.write_text(json.dumps(plan))
    elif corruption == "pixels":
        prediction_path.with_name("one.npz").write_bytes(b"changed pixels")
    elif corruption == "reference":
        (data / "one.npz").write_bytes(b"changed reference")
    elif corruption == "sampler":
        path = prediction_path.with_name("sampling.json")
        sampling = json.loads(path.read_text())
        sampling["sampling_steps"] = 32
        path.write_text(json.dumps(sampling))
    else:
        if corruption == "missing":
            predictions["runs"][0]["videos"] = {}
        else:
            predictions["runs"][0]["videos"]["one"] = "../outside.npz"
        prediction_path.write_text(json.dumps(predictions))
        sampling_path = prediction_path.with_name("sampling.json")
        sampling = json.loads(sampling_path.read_text())
        sampling["runs"] = predictions["runs"]
        sampling_path.write_text(json.dumps(sampling))
    with pytest.raises(ValueError):
        verify_inputs(*inputs)
