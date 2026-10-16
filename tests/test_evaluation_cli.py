"""Exercise JSON export with controls, corrupt archives, and separate metric sources."""

import hashlib
import json

import numpy as np
import pytest

from click2future.evaluation.__main__ import main


@pytest.fixture
def fixture_archive(tmp_path):
    arrays = {
        "predicted_positions": np.zeros((2, 3, 2, 2)),
        "reference_positions": np.zeros((3, 2, 2)),
        "object_valid": np.ones(2, bool),
        "tracked": np.ones((2, 3, 2), bool),
        "predicted_collision_onset": np.zeros((2, 2, 2, 2), bool),
        "reference_collision_onset": np.zeros((2, 2, 2), bool),
        "collision_observed": np.ones((2, 2, 2, 2), bool),
        "timestamps": np.array([0, 0.125, 0.25]),
    }
    path = tmp_path / "control.npz"
    np.savez_compressed(path, **arrays)
    return path, arrays


def arguments(path, output, source="control_fixture"):
    return [
        "--input",
        str(path),
        "--output",
        str(output),
        "--source",
        source,
        "--run-id",
        "test-controls",
        "--sample-id",
        "reference-0",
        "--scene-group-id",
        "scene-0",
        "--condition-id",
        "condition-0",
        "--selected-object",
        "0",
        "--event-tolerance",
        "1",
    ]


@pytest.mark.parametrize("source", ["control_fixture", "video_tracking", "auxiliary_head"])
def test_report_preserves_source_and_denominators(fixture_archive, tmp_path, capsys, source):
    path, _ = fixture_archive
    output = tmp_path / "reports" / "metrics.json"
    args = arguments(path, output, source)
    if source == "video_tracking":
        args += ["--extractor-version", "test-tracker-v1"]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["source"] == source
    assert report["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["trajectory"]["all_objects"]["complete_samples"] == 2
    assert report["collision"]["per_sample"][0]["event"]["f1"] is None
    assert "NaN" not in output.read_text()


def test_video_tracking_needs_extractor_provenance(fixture_archive, tmp_path, capsys):
    path, _ = fixture_archive
    assert main(arguments(path, tmp_path / "failed.json", "video_tracking")) == 1
    assert "extractor-version" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("corruption", ["time", "samples", "empty", "pickle", "archive"])
def test_bad_inputs_export_failure_records(fixture_archive, tmp_path, capsys, corruption):
    path, arrays = fixture_archive
    if corruption == "time":
        arrays["timestamps"][2] = 0.125
    elif corruption == "samples":
        arrays["predicted_collision_onset"] = arrays["predicted_collision_onset"][:1]
    elif corruption == "empty":
        arrays["reference_positions"] = np.zeros((0, 2, 2))
    elif corruption == "pickle":
        arrays["predicted_positions"] = np.array([{"bad": "object array"}], dtype=object)
    np.savez_compressed(path, **arrays)
    if corruption == "archive":
        path.write_bytes(b"not an NPZ archive")
    assert main(arguments(path, tmp_path / "failed.json")) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert "trajectory" not in report


def test_output_cannot_overwrite_input(fixture_archive):
    path, _ = fixture_archive
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main(arguments(path, path))
    assert error.value.code == 2
    assert path.read_bytes() == before
