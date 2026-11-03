"""The development subset has its own inventory and preserves source provenance."""

import hashlib
import json
import runpy
from pathlib import Path
from unittest.mock import patch

from click2future.data import generate_dataset
from click2future.demo.bundle import load_bundle


def test_subset_manifest_does_not_inherit_full_dataset_identity(tmp_path):
    data = tmp_path / "dataset"
    generate_dataset(
        data,
        group_count=10,
        variants_per_group=1,
        seed=842000,
        config={"frame_count": 4, "resolution": 16, "fps": 8},
    )
    source = data / "dataset_manifest.json"
    original = json.loads(source.read_text())
    sample = next(r for r in original["samples"] if r["split"] == "train")
    checkpoint = tmp_path / "test-only.pt"
    checkpoint.write_bytes(b"test-only model stub; never used for inference")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "dataset": str(data),
                "samples": [sample["sample_id"]],
                "models": [
                    {
                        "name": "fixture",
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "quality": "test stub",
                        "notes": "manifest-only fixture",
                    }
                ],
            }
        )
    )
    # Real checkpoint loading is exercised by the adapter tests and local bundle rehearsal.
    with patch("click2future.demo.models.CheckpointModel") as model:
        model.return_value.metadata = {"training_steps": 1}
        builder = runpy.run_path(str(Path(__file__).parents[1] / "scripts/build_demo_bundle.py"))
        manifest = builder["build"](recipe, tmp_path / "bundle")
    subset = json.loads((tmp_path / "bundle/presets/dataset_manifest.json").read_text())
    assert subset == {
        "kind": "demo_preset_subset",
        "source_dataset_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "samples": [sample],
    }
    assert len(manifest["presets"]) == 1 < len(original["samples"])
    assert len(manifest["files_sha256"]) == 4
    load_bundle(tmp_path / "bundle", ["fixture"])
