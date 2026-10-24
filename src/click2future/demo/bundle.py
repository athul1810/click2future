"""Validate a local checkpoint handoff before loading models or preset arrays."""

import hashlib
import json
from pathlib import Path


def load_bundle(folder, names):
    root = Path(folder).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest.get("kind") != "internal_validation_checkpoint_handoff":
        raise ValueError("Unsupported checkpoint bundle")
    models = {row["name"]: row for row in manifest["models"]}
    if len(models) != len(manifest["models"]):
        raise ValueError("Duplicate bundled model names")
    if (
        not names
        or len(names) > 4
        or len(set(names)) != len(names)
        or not set(names) <= models.keys()
    ):
        raise ValueError("Choose one to four distinct bundled model names")
    inventory = manifest["files_sha256"]
    for name, expected in inventory.items():
        path = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or "\\" in name
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError("Bundle members must be regular files inside its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Bundle file hash mismatch: " + name)
    if "presets/dataset_manifest.json" not in inventory:
        raise ValueError("Bundle is missing its preset manifest")
    presets = json.loads((root / "presets/dataset_manifest.json").read_text())["samples"]
    if not presets or any(row["split"] not in {"train", "validation"} for row in presets):
        raise ValueError("Only development/validation presets may be bundled")
    if {row["sample_id"] for row in presets} != {row["sample_id"] for row in manifest["presets"]}:
        raise ValueError("Bundled preset inventories disagree")
    for row in presets:
        sample = row["sample_id"]
        if Path(sample).name != sample or any(
            "presets/" + sample + suffix not in inventory for suffix in (".npz", ".json")
        ):
            raise ValueError("Preset files must appear in the verified inventory")
    selected = []
    for name in names:
        row = models[name]
        if row["file"] not in inventory or inventory[row["file"]] != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint digest disagrees with the bundle inventory")
        # The launcher checks the step against the checkpoint and shows quality/notes.
        step = row.get("evaluated_checkpoint_step")
        missing = [
            key for key in ("quality", "notes") if not isinstance(row.get(key), str) or not row[key]
        ]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            missing.insert(0, "evaluated_checkpoint_step")
        if missing:
            raise ValueError(f"Bundled model {name} lacks manifest fields: " + ", ".join(missing))
        selected.append({**row, "path": root / row["file"]})
    return root / "presets", selected
