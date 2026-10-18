"""Paired training actions on existing training worlds; no new evaluation data."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from click2future.sim.rollout import run_rollout

from .export import load_rollout, save_rollout
from .generalization import CONFIG, _digest, _sha
from .splits import load_split_manifest


def parent_inventory(parent):
    """Validate metadata and return four training physics profiles per world.

    No validation or test rollout is opened, including during generation.
    """
    parent = Path(parent)
    path = parent / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != "dataset-0.3"
        or manifest["config"] != CONFIG
        or manifest["split_manifest"] != "splits.json"
        or manifest["manifest_hash"]
        != _digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
        or _sha(parent / "splits.json") != manifest["split_manifest_sha256"]
    ):
        raise ValueError("Frozen parent manifest/splits changed")
    groups = load_split_manifest(parent / "splits.json")["splits"]
    train = set(groups["train"])
    if len(train) != manifest["group_counts"]["train"]:
        raise ValueError("Parent training group count changed")
    sources = []
    for row in manifest["samples"]:
        if (row["scene_group_id"] in train) != (row["split"] == "train"):
            raise ValueError("Parent training membership conflicts with splits")
        if row["split"] == "train" and row["action_variant"] == "forward":
            sources.append(row)
    if (
        len(sources) != 4 * len(train)
        or {(r["scene_group_id"], r["physics_variant"]) for r in sources}
        != {(g, p) for g in train for p in range(4)}
        or len({r["sample_id"] for r in sources}) != len(sources)
    ):
        raise ValueError("Expected exactly four training profiles per world")
    return manifest, sorted(sources, key=lambda row: row["sample_id"])


def recipe_cases(sources, directions, magnitudes):
    if (
        isinstance(directions, bool)
        or not isinstance(directions, int)
        or directions < 2
        or directions > 16
        or not magnitudes
        or len(set(magnitudes)) != len(magnitudes)
        or any(not np.isfinite(v) or not 0 < v <= 1.1 for v in magnitudes)
    ):
        raise ValueError("Invalid paired action grid")
    for source in sources:
        for selected in (0, 1):
            if source["physics_variant"] == 0:
                yield source, selected, -1, -1, [0.0, 0.0]
            for direction in range(directions):
                unit = np.array(
                    [
                        np.cos(direction * 2 * np.pi / directions),
                        np.sin(direction * 2 * np.pi / directions),
                    ]
                )
                unit[np.abs(unit) < 1e-14] = 0
                for strength, magnitude in enumerate(magnitudes):
                    yield source, selected, direction, strength, (unit * magnitude).tolist()


def case_identity(source, selected, direction, strength):
    return (
        f"{source['scene_group_id']}-grid-s{selected}-d{direction + 1}"
        f"-m{strength + 1}-p{source['physics_variant']:02d}"
    )


def source_rollout(parent, source):
    name = source["sample_id"]
    if Path(name).name != name or not name:
        raise ValueError("Invalid parent sample ID")
    for kind, suffix in (("arrays", ".npz"), ("metadata", ".json")):
        if (
            source[kind] != name + suffix
            or _sha(Path(parent) / source[kind]) != source[kind + "_sha256"]
        ):
            raise ValueError("Parent training content changed")
    reference = load_rollout(parent, name)
    if reference["scene_spec"]["scene_group_id"] != source["scene_group_id"]:
        raise ValueError("Parent scene identity changed")
    return reference


def changed_scene(reference, name, selected, force):
    scene = copy.deepcopy(reference["scene_spec"])
    scene["condition_id"] = name
    scene["action_variant"] = "training_action_grid"
    scene["action"] = {"selected_object_id": selected, "force_xy": force, "force_duration": 0.25}
    return scene


def generate_action_supplement(parent, output, *, directions=8, magnitudes=(0.25, 0.5)):
    parent, output = Path(parent), Path(output)
    manifest, sources = parent_inventory(parent)
    cases = list(recipe_cases(sources, directions, magnitudes))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Supplement must use a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    references, rows = {}, []
    for source, selected, direction, strength, force in cases:
        sid = source["sample_id"]
        if sid not in references:
            references[sid] = source_rollout(parent, source)
        ref = references[sid]
        name = case_identity(source, selected, direction, strength)
        scene = changed_scene(ref, name, selected, force)
        rollout = run_rollout(scene, frame_count=32, fps=8, physics_hz=240, resolution=64)
        if not np.array_equal(rollout["frames"][0], ref["frames"][0]) or not np.array_equal(
            rollout["instance_masks"][0], ref["instance_masks"][0]
        ):
            raise ValueError("Supplement changed the parent visible world")
        paths = save_rollout(rollout, output, name)
        rows.append(
            {
                "sample_id": name,
                "parent_sample_id": sid,
                "scene_group_id": source["scene_group_id"],
                "split": "train",
                "selected_object_id": selected,
                "direction_index": direction,
                "magnitude_index": strength,
                "force_xy": force,
                "contact": bool(rollout["collision_onset"].any()),
                **{kind + "_sha256": _sha(path) for kind, path in paths.items()},
            }
        )
    result = {
        "schema_version": "training-action-supplement-0.1",
        "parent_manifest_sha256": _sha(parent / "dataset_manifest.json"),
        "parent_split_sha256": _sha(parent / "splits.json"),
        "parent_manifest_hash": manifest["manifest_hash"],
        "recipe": {
            "directions": directions,
            "magnitudes": list(magnitudes),
            "zero_per_object_world": True,
        },
        "samples": rows,
        "sample_count": len(rows),
        "scope": "training only; original validation and reserved tests unchanged",
    }
    result["manifest_hash"] = _digest(result)
    (output / "supplement_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def validate_action_supplement(parent, root):
    """Reject incomplete grids, changed physics/appearance, test worlds and bytes."""
    parent, root = Path(parent), Path(root)
    _, sources = parent_inventory(parent)
    manifest = json.loads((root / "supplement_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != "training-action-supplement-0.1"
        or manifest["manifest_hash"]
        != _digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
        or manifest["parent_manifest_sha256"] != _sha(parent / "dataset_manifest.json")
        or manifest["parent_split_sha256"] != _sha(parent / "splits.json")
        or manifest["recipe"]["zero_per_object_world"] is not True
    ):
        raise ValueError("Supplement manifest/parent binding changed")
    recipe = manifest["recipe"]
    cases = list(recipe_cases(sources, recipe["directions"], recipe["magnitudes"]))
    rows = manifest["samples"]
    by_id = {row["sample_id"]: row for row in rows}
    if len(cases) != len(rows) or len(by_id) != len(rows) or manifest["sample_count"] != len(rows):
        raise ValueError("Supplement action coverage is incomplete")
    references = {}
    for source, selected, direction, strength, force in cases:
        name = case_identity(source, selected, direction, strength)
        if name not in by_id:
            raise ValueError("Supplement action coverage is incomplete")
        row = by_id[name]
        for field, expected in {
            "parent_sample_id": source["sample_id"],
            "scene_group_id": source["scene_group_id"],
            "split": "train",
            "selected_object_id": selected,
            "direction_index": direction,
            "magnitude_index": strength,
            "force_xy": force,
        }.items():
            if row[field] != expected:
                raise ValueError("Supplement training membership/action changed")
        for kind, suffix in (("arrays", ".npz"), ("metadata", ".json")):
            if row[kind + "_sha256"] != _sha(root / (name + suffix)):
                raise ValueError("Supplement content changed")
        sid = source["sample_id"]
        if sid not in references:
            references[sid] = source_rollout(parent, source)
        ref = references[sid]
        rollout = load_rollout(root, name)
        if (
            rollout["scene_spec"] != changed_scene(ref, name, selected, force)
            or rollout["metadata"] != ref["metadata"]
            or not np.array_equal(rollout["frames"][0], ref["frames"][0])
            or not np.array_equal(rollout["instance_masks"][0], ref["instance_masks"][0])
            or row["contact"] != bool(rollout["collision_onset"].any())
        ):
            raise ValueError("Supplement changed parent physics or initial appearance")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_action_supplement(args.parent, args.output)
    print(json.dumps({"samples": result["sample_count"], "manifest_hash": result["manifest_hash"]}))


if __name__ == "__main__":
    main()
