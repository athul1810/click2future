"""Adjacent ablations and cache transport cannot silently change the experiment."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from test_contact_training import fixture_data

from click2future.models.world_motion import WORLD_CONTRACT, sample_world
from click2future.training.train_cvae import sha256
from click2future.training.world_data import load_cache


def test_adjacent_world_recipes_reject_another_changed_factor(tmp_path):
    paired = runpy.run_path("submission/world_comparison.py")["paired_configs"]
    original = paired(Path.cwd())
    folder = tmp_path / "configs/training"
    folder.mkdir(parents=True)
    for path in original.values():
        (folder / path.name).write_bytes(path.read_bytes())
    paired(tmp_path)
    target = folder / "world_conditioned.json"
    value = json.loads(target.read_text())
    value["steps"] += 1
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="may differ only"):
        paired(tmp_path)


def test_world_cache_roundtrip_preserves_every_array_and_rejects_changed_payload(tmp_path):
    data, geometry, _ = fixture_data()
    arrays, geometry_hashes, world_hashes = {}, {}, {}
    for split, row in data.items():
        for key in ("features", "force", "motion"):
            arrays[split + "_" + key] = row[key]
        arrays[split + "_world"] = sample_world(0).numpy().repeat(6, 0)
        for key, value in geometry[split].items():
            arrays[split + "_" + key] = value
        geometry_hashes[split] = {
            key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in geometry[split].items()
        }
        world_hashes[split] = hashlib.sha256(arrays[split + "_world"].tobytes()).hexdigest()
    np.savez_compressed(tmp_path / "arrays.npz", **arrays)
    record = {
        "format_version": 1,
        "world_contract": WORLD_CONTRACT,
        "examples": {split: row["examples"] for split, row in data.items()},
        "array_sha256": sha256(tmp_path / "arrays.npz"),
        "provenance": {"geometry_sha256": geometry_hashes, "world_label_sha256": world_hashes},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    restored, shapes, worlds, _ = load_cache(tmp_path)
    for split in data:
        for key in ("features", "force", "motion"):
            np.testing.assert_array_equal(restored[split][key], data[split][key])
        np.testing.assert_array_equal(worlds[split], arrays[split + "_world"])
        for key in geometry[split]:
            np.testing.assert_array_equal(shapes[split][key], geometry[split][key])
    with (tmp_path / "arrays.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        load_cache(tmp_path)
