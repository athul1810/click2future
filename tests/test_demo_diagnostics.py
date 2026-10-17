"""Verified checkpoint bundles and explicit missing pixel tracks."""

import hashlib
import json

import numpy as np
import pytest

from click2future.demo.bundle import load_bundle
from click2future.demo.diagnostics import pixel_diagnostics
from click2future.evaluation.tracking import TRACKER_VERSION
from click2future.sim import make_scene, run_rollout


def test_pixel_paths_and_missing_observations_remain_separate():
    reference = run_rollout(make_scene(115), frame_count=4, resolution=64, fps=8)
    videos = reference["frames"][None].copy()
    tracks = pixel_diagnostics(videos, reference["instance_masks"][0], reference["object_valid"])
    sample = tracks["samples"][0]
    assert tracks["source"] == "generated pixels" and tracks["extractor_version"] == TRACKER_VERSION
    assert sample["future_tracked"] == sample["future_expected"]
    videos[:, 2] = videos[0, 0, 0, 0]
    tracks = pixel_diagnostics(videos, reference["instance_masks"][0], reference["object_valid"])
    sample = tracks["samples"][0]
    assert sample["future_tracked"] < sample["future_expected"]
    for obj in tracks["object_ids"]:
        assert sample["positions"][2][obj] == [None, None]
        assert not sample["tracked"][2][obj]
        assert sample["tracked"][3][obj]
    json.dumps(tracks, allow_nan=False)


def test_unusable_tracker_inputs_are_visible_failures():
    videos = np.zeros((4, 3, 16, 16, 3), dtype=np.uint8)
    tracks = pixel_diagnostics(videos, np.zeros((1, 16, 16), bool), np.ones(1, bool))
    assert tracks["status"] == "unavailable" and tracks["reason"]
    assert not tracks["samples"]


@pytest.fixture
def bundle(tmp_path):
    payloads = {
        "checkpoints/fixture.pt": b"test-only checkpoint transport",
        "presets/one.npz": b"arrays must not be loaded during bundle validation",
        "presets/one.json": b"{}",
        "presets/dataset_manifest.json": json.dumps(
            {"samples": [{"sample_id": "one", "split": "validation"}]}
        ).encode(),
    }
    hashes = {}
    for name, content in payloads.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "kind": "internal_validation_checkpoint_handoff",
        "files_sha256": hashes,
        "models": [
            {
                "name": "fixture",
                "file": "checkpoints/fixture.pt",
                "checkpoint_sha256": hashes["checkpoints/fixture.pt"],
            }
        ],
        "presets": [{"sample_id": "one", "split": "validation"}],
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_bundle_verifies_every_member_before_loading_models(bundle):
    root, manifest = bundle
    dataset, rows = load_bundle(root, ["fixture"])
    assert dataset == root / "presets" and rows[0]["path"] == root / "checkpoints/fixture.pt"
    (root / "presets/one.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_bundle(root, ["fixture"])


@pytest.mark.parametrize("filename", ["checkpoints/unlisted.pt", "../outside.pt"])
def test_bundle_rejects_unlisted_checkpoint_with_null_digest(bundle, filename):
    root, manifest = bundle
    manifest["models"][0].update(file=filename, checkpoint_sha256=None)
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bundle inventory"):
        load_bundle(root, ["fixture"])


@pytest.mark.parametrize("corruption", ["path", "inventory", "duplicate", "test", "digest"])
def test_bad_bundle_is_rejected_before_model_or_array_loading(bundle, corruption):
    root, manifest = bundle
    if corruption == "path":
        manifest["files_sha256"]["../outside.pt"] = "0" * 64
    elif corruption == "inventory":
        del manifest["files_sha256"]["presets/one.npz"]
    elif corruption == "duplicate":
        manifest["models"].append(manifest["models"][0])
    elif corruption == "test":
        source = root / "presets/dataset_manifest.json"
        source.write_text(json.dumps({"samples": [{"sample_id": "one", "split": "id_test"}]}))
        manifest["files_sha256"]["presets/dataset_manifest.json"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    else:
        manifest["models"][0]["checkpoint_sha256"] = "0" * 64
    (root / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_bundle(root, ["fixture"])
