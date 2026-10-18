"""Saved-artifact provenance and unchanged historical records."""

import hashlib
import json

import numpy as np
import pytest

from click2future.data import save_rollout
from click2future.sim import run_rollout
from click2future.training.rescore_saved import digest, rescore


@pytest.fixture
def saved(tmp_path):
    from pathlib import Path

    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    reference = run_rollout(scene, frame_count=4, resolution=64, fps=8)
    save_rollout(reference, tmp_path / "data", "one")
    videos = np.repeat(reference["frames"][None], 4, axis=0)
    np.savez_compressed(tmp_path / "videos.npz", videos=videos)
    historical = b'{\r\n  "extractor_version": "old"\r\n}\r\n'
    (tmp_path / "old.json").write_bytes(historical)
    spec = {
        "dataset": "data",
        "sample_id": "one",
        "reference_sha256": {e: digest(tmp_path / "data" / ("one." + e)) for e in ("npz", "json")},
        "historical_records_sha256": {"old.json": digest(tmp_path / "old.json", text=True)},
        "videos": [
            {
                "id": "one",
                "file": "videos.npz",
                "file_sha256": digest(tmp_path / "videos.npz"),
                "seeds": [0, 1, 2, 3],
            }
        ],
    }
    return tmp_path, spec, historical


def test_saved_rescore_retains_pixels_reference_and_historical_bytes(saved):
    root, spec, old = saved
    report = rescore(spec, root=root)
    assert report["records"][0]["declared_training_gate_passed"]
    assert report["records"][0]["scores"]["all_objects"]["tracked_future_observations"] == 24
    assert (root / "old.json").read_bytes() == old
    (root / "old.json").write_bytes(old.replace(b"\r\n", b"\n"))
    assert (
        rescore(spec, root=root)["historical_records_sha256"] == report["historical_records_sha256"]
    )


@pytest.mark.parametrize("changed", ["reference", "video", "historical", "frame0", "duplicate"])
def test_changed_or_misidentified_artifacts_are_refused(saved, changed):
    root, spec, _ = saved
    if changed == "reference":
        (root / "data/one.json").write_text("{}")
    elif changed == "video":
        (root / "videos.npz").write_bytes(b"changed")
    elif changed == "historical":
        (root / "old.json").write_text("changed")
    elif changed == "duplicate":
        spec["videos"].append(spec["videos"][0])
    else:
        with np.load(root / "videos.npz", allow_pickle=False) as archive:
            videos = archive["videos"]
        videos[:, 0] = 0
        np.savez_compressed(root / "videos.npz", videos=videos)
        spec["videos"][0]["file_sha256"] = hashlib.sha256(
            (root / "videos.npz").read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError):
        rescore(spec, root=root)
