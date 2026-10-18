"""Workload planning reads metadata only and rejects changed or leaking splits."""

import json

import pytest

from click2future.evaluation.inventory import RESERVED, inventory
from click2future.evaluation.suite import _digest
from click2future.training.train_cvae import sha256


@pytest.fixture
def metadata(tmp_path):
    splits = {name: ["scene-" + name] for name in ("train", "validation", *RESERVED)}
    split_manifest = {"schema_version": "splits-0.1", "splits": splits}
    split_manifest["manifest_hash"] = _digest(split_manifest)
    (tmp_path / "splits.json").write_text(json.dumps(split_manifest))
    rows = [
        {
            "sample_id": scene + "-forward-p0" + str(i),
            "scene_group_id": scene,
            "condition_id": scene + "-forward",
            "split": split,
        }
        for split, scenes in splits.items()
        for scene in scenes
        for i in range(2)
    ]
    index = {
        "schema_version": "dataset-0.3",
        "samples": rows,
        "sample_count": len(rows),
        "split_manifest_sha256": sha256(tmp_path / "splits.json"),
    }
    index["manifest_hash"] = _digest(index)
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(index))
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        "sampling_seeds: [0, 1, 2, 3]\ncore_methods: [E0, E2, E3]\ntraining_seeds: [0, 1, 2]\n"
    )
    return tmp_path, protocol


def test_inventory_works_without_any_rollout_files(metadata):
    root, protocol = metadata
    assert not list(root.glob("*.npz"))
    result = inventory(root, protocol)
    assert result["reference_arrays_opened"] == 0 and result["test_splits_evaluated"] == []
    totals = result["reserved_totals"]
    assert totals["reference_clips"] == 10 and totals["conditions"] == 5
    assert totals["primary_videos_per_run"] == 20
    assert totals["proposed_selection_videos_per_run"] == 40
    assert result["splits"]["force_ood"]["action_controls"] == "omitted_for_force_ood"


@pytest.mark.parametrize("change", ["duplicate", "wrong_split", "missing_scene", "digest"])
def test_changed_or_incomplete_index_is_refused(metadata, change):
    root, protocol = metadata
    path = root / "dataset_manifest.json"
    index = json.loads(path.read_text())
    if change == "duplicate":
        index["samples"][1]["sample_id"] = index["samples"][0]["sample_id"]
    elif change == "wrong_split":
        index["samples"][0]["split"] = "id_test"
    elif change == "missing_scene":
        index["samples"] = [r for r in index["samples"] if r["split"] != "shape_ood"]
        index["sample_count"] = len(index["samples"])
    else:
        index["sample_count"] += 1
    if change != "digest":
        index["manifest_hash"] = _digest({k: v for k, v in index.items() if k != "manifest_hash"})
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        inventory(root, protocol)
