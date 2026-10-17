"""Check frozen inputs and training-only class balancing before any GPU work."""

import json
from pathlib import Path

import numpy as np
import pytest

from click2future.training import multiscene_pilot as pilot


def test_training_class_balance_does_not_open_held_out_labels(monkeypatch):
    opened = []

    def batch(root, ids):
        opened.extend(ids)
        count = len(ids)
        masks = np.zeros((count, 4, 2, 8, 8), bool)
        masks[:, :, 0, :2, :2] = True
        masks[:, :, 1, 5:7, 5:7] = True
        events = np.zeros((count, 3, 2, 2), bool)
        events[:, 0, 0, 1] = events[:, 0, 1, 0] = True
        return dict(
            positions=np.full((count, 4, 2, 2), 0.5, np.float32),
            collision_onset=events,
            object_valid=np.ones((count, 2), bool),
            instance_masks=masks,
        )

    monkeypatch.setattr(pilot, "load_batch", batch)
    manifest = {
        "samples": [
            {"sample_id": name, "split": split}
            for name, split in (("a", "train"), ("b", "validation"), ("c", "id_test"))
        ]
    }
    result = pilot.training_balance(Path("unused"), manifest)
    assert opened == ["a"]
    assert result["positive_count"] == 1
    assert result["negative_count"] == 2
    assert result["positive_weight"] == 2


def test_pilot_freezes_effective_configs_and_detects_changed_sample_bytes(tmp_path):
    spec = json.loads((pilot.PROJECT / "configs/training/multiscene_pilot.json").read_text())
    spec["group_counts"] = dict.fromkeys(spec["group_counts"], 1)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    root = tmp_path / "pilot"
    record = pilot.prepare(root, path)
    assert record["replay"]["replayed_samples"] == 166
    assert record == pilot.read_pilot(root)
    assert record["clips_by_split"]["train"] == 21
    weight = record["training_collision_balance"]["positive_weight"]
    for method in ("E2", "E3", "E4"):
        config = json.loads((root / "configs" / (method + ".json")).read_text())
        assert config["collision_positive_weight"] == weight
    with pytest.raises(ValueError, match="new pilot"):
        pilot.prepare(root, path)
    config_path = root / "configs/E4.json"
    original = config_path.read_bytes()
    config_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="config changed"):
        pilot.read_pilot(root)
    config_path.write_bytes(original)
    manifest = json.loads((root / "data/dataset_manifest.json").read_text())
    row = next(r for r in manifest["samples"] if r["split"] == "train")
    sample = root / "data" / row["arrays"]
    sample.write_bytes(sample.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="sample changed"):
        pilot.read_pilot(root)


def test_pilot_refuses_cpu_training_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    root = tmp_path / "pilot"
    with pytest.raises(ValueError, match="TC2"):
        pilot.main(["--root", str(root)])
    assert not root.exists()
