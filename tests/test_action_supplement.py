"""Training-only action coverage, checked labels and immutable validation."""

import json
from pathlib import Path

import numpy as np
import pytest

from click2future.data import action_supplement as module
from click2future.data import generate_generalization_dataset, load_rollout
from click2future.data.generalization import _digest
from click2future.training.contact_geometry import prepare_contact_geometry
from click2future.training.trajectory_data import (
    add_training_supplement,
    encode_trajectory_rows,
    prepare_trajectories,
    replay_angles,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-actions")
    parent, extra = root / "parent", root / "extra"
    generate_generalization_dataset(parent, seed=879321)
    manifest = module.generate_action_supplement(parent, extra, directions=2, magnitudes=(0.25,))
    return parent, extra, manifest


def test_complete_both_selection_grid_keeps_parent_world_and_replay(corpus):
    parent, extra, manifest = corpus
    assert module.validate_action_supplement(parent, extra) == manifest
    assert manifest["sample_count"] == 18
    for row in manifest["samples"]:
        ref = load_rollout(parent, row["parent_sample_id"])
        clip = load_rollout(extra, row["sample_id"])
        np.testing.assert_array_equal(clip["frames"][0], ref["frames"][0])
        np.testing.assert_array_equal(clip["force_xy"], np.array(row["force_xy"], np.float32))
        assert clip["selected_object_id"] == row["selected_object_id"]
        assert replay_angles(clip).shape == (32, 2)


def test_generation_never_opens_validation_or_test_rollouts(corpus, tmp_path, monkeypatch):
    parent, _, _ = corpus
    _, sources = module.parent_inventory(parent)
    allowed = {source["sample_id"] for source in sources}
    original, calls = module.load_rollout, []

    def checked(root, name):
        if Path(root) == parent:
            assert name in allowed
            calls.append(name)
        return original(root, name)

    monkeypatch.setattr(module, "load_rollout", checked)
    module.generate_action_supplement(parent, tmp_path / "new", directions=2, magnitudes=(0.25,))
    module.validate_action_supplement(parent, tmp_path / "new")
    assert set(calls) == allowed


@pytest.mark.parametrize("change", ["leak", "coverage", "bytes", "binding"])
def test_tampered_supplement_is_rejected(corpus, tmp_path, change):
    import shutil

    parent, extra, _ = corpus
    root = tmp_path / "changed"
    shutil.copytree(extra, root)
    path = root / "supplement_manifest.json"
    manifest = json.loads(path.read_text())
    if change == "leak":
        manifest["samples"][0]["split"] = "validation"
    elif change == "coverage":
        manifest["samples"][-1] = manifest["samples"][0]
    elif change == "binding":
        manifest["parent_manifest_hash"] = "0" * 64
    else:
        target = root / (manifest["samples"][0]["sample_id"] + ".npz")
        target.write_bytes(target.read_bytes() + b"changed")
    manifest["manifest_hash"] = _digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        module.validate_action_supplement(parent, root)


def test_append_changes_training_only_and_geometry_uses_correct_roots(corpus):
    parent, extra, _ = corpus
    data, provenance = prepare_trajectories(parent, augment=False)
    original = data["train"]["motion"].copy()
    combined, identity = add_training_supplement(parent, extra, data, provenance, augment=False)
    assert combined["validation"] is data["validation"]
    np.testing.assert_array_equal(combined["train"]["motion"][: len(original)], original)
    assert len(combined["train"]["motion"]) == len(original) + 18
    assert identity["combined_training_examples"] == len(original) + 18
    geometry = prepare_contact_geometry(parent, combined)
    assert len(geometry["train"]["hulls"]) == len(original) + 18
    with pytest.raises(ValueError, match="training and validation only"):
        encode_trajectory_rows(parent, {"id_test": ["unused"]})
