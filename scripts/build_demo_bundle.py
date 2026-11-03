"""Build a new, hash-verified local demo bundle without modifying checkpoints."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from click2future.demo.bundle import load_bundle
from click2future.demo.models import CheckpointModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(recipe, output):
    if output.exists():
        raise ValueError("Bundle output must be new")
    data = json.loads(recipe.read_text(encoding="utf-8"))
    dataset = Path(data["dataset"])
    original = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    wanted = set(data["samples"])
    rows = [row for row in original["samples"] if row["sample_id"] in wanted]
    if len(rows) != len(wanted) or any(r["split"] not in {"train", "validation"} for r in rows):
        raise ValueError("Choose existing train/validation samples only")
    models = []
    names = [row["name"] for row in data["models"]]
    if not 1 <= len(names) <= 4 or len(names) != len(set(names)):
        raise ValueError("Choose one to four distinct names")
    for index, row in enumerate(data["models"]):
        path = Path(row["checkpoint"])
        expected = row["checkpoint_sha256"]
        if digest(path) != expected:
            raise ValueError("Checkpoint does not match the reviewed recipe")
        model = CheckpointModel(row["name"], path, device="cpu")
        for field in ("quality", "notes"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Record checkpoint quality and supervision notes")
        models.append(
            {
                "name": row["name"],
                "file": f"checkpoints/model-{index}.pt",
                "checkpoint_sha256": expected,
                "evaluated_checkpoint_step": model.metadata["training_steps"],
                "quality": row["quality"],
                "notes": row["notes"],
            }
        )
        del model
    manifest = {
        "kind": "internal_validation_checkpoint_handoff",
        "scope": "presentation development examples; not a new quality evaluation",
        "recipe_sha256": digest(recipe),
        "models": models,
        "presets": rows,
        "source_dataset_manifest_sha256": digest(dataset / "dataset_manifest.json"),
    }
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / "presets").mkdir()
    for source, target in zip(data["models"], models, strict=True):
        shutil.copyfile(source["checkpoint"], output / target["file"])
    for row in rows:
        for suffix in (".json", ".npz"):
            name = row["sample_id"] + suffix
            if Path(name).name != name or "\\" in name:
                raise ValueError("Preset names must be simple filenames")
            shutil.copyfile(dataset / name, output / "presets" / name)
    preset_manifest = {**original, "samples": rows}
    (output / "presets/dataset_manifest.json").write_text(
        json.dumps(preset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    manifest["files_sha256"] = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_bundle(output, names)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build(args.recipe, args.output)
    print(
        json.dumps(
            {
                "models": [m["name"] for m in manifest["models"]],
                "verified_files": len(manifest["files_sha256"]),
            }
        )
    )


if __name__ == "__main__":
    main()
