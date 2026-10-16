"""Versioned training and held-out single-factor generalization datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..sim.controlled import ACTION_VARIANTS
from ..sim.generalization import SPLITS, make_generalization_scene
from ..sim.rollout import run_rollout
from .export import load_rollout, save_rollout
from .splits import load_split_manifest, save_split_manifest

CONFIG = {
    "world_size": [1.0, 1.0],
    "resolution": 64,
    "frame_count": 32,
    "fps": 8,
    "physics_hz": 240,
    "force_duration": 0.25,
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_spec(seed, group_counts):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not isinstance(group_counts, dict)
        or set(group_counts) != set(SPLITS)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in group_counts.values()
        )
    ):
        raise ValueError("Provide a positive group count for each of the seven splits")


def _specs(seed, group_counts):
    """Disjoint visible seeds; all actions and physics variants stay together."""
    offset = 0
    for split in SPLITS:
        for _ in range(group_counts[split]):
            scene_seed = seed + offset
            profiles = 8 if split == "physics_test" else 4
            for action in ACTION_VARIANTS:
                if split == "force_ood" and action == "zero":
                    continue
                for profile in range(1 if action == "zero" else profiles):
                    physics_seed = seed * 1_000_000 + offset * 1_000 + profile + 1
                    yield split, scene_seed, action, profile, physics_seed
            offset += 1


def generate_generalization_dataset(output_dir, *, seed=200000, group_counts=None):
    """Generate dataset-0.3 without modifying dataset-0.1 or dataset-0.2.

    This is a separate matched corpus. Train on its train split, select on its
    validation split, then compare ID and OOD with the same frozen model.
    """
    group_counts = dict.fromkeys(SPLITS, 1) if group_counts is None else dict(group_counts)
    _validate_spec(seed, group_counts)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output folder must be new or empty; frozen data is never overwritten")
    root.mkdir(parents=True, exist_ok=True)
    samples, groups = [], {split: set() for split in SPLITS}
    for split, scene_seed, action, profile, physics_seed in _specs(seed, group_counts):
        scene = make_generalization_scene(
            scene_seed, split=split, action_variant=action, physics_seed=physics_seed
        )
        groups[split].add(scene["scene_group_id"])
        sample_id = f"{scene['condition_id']}-p{profile:02d}"
        rollout = run_rollout(scene, frame_count=32, fps=8, physics_hz=240, resolution=64)
        paths = save_rollout(rollout, root, sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "scene_group_id": scene["scene_group_id"],
                "condition_id": scene["condition_id"],
                "split": split,
                "visible_seed": scene_seed,
                "physics_seed": physics_seed,
                "action_variant": action,
                "physics_variant": profile,
                "arrays": paths["arrays"].name,
                "metadata": paths["metadata"].name,
                "arrays_sha256": _sha(paths["arrays"]),
                "metadata_sha256": _sha(paths["metadata"]),
            }
        )
    total = sum(group_counts.values())
    splits = {
        "schema_version": "splits-0.1",
        "seed": seed,
        "fractions": {split: group_counts[split] / total for split in SPLITS},
        "splits": {split: sorted(groups[split]) for split in SPLITS},
    }
    splits["manifest_hash"] = _digest(splits)
    split_path = save_split_manifest(splits, root / "splits.json")
    manifest = {
        "schema_version": "dataset-0.3",
        "seed": seed,
        "group_counts": group_counts,
        "group_count": total,
        "sample_count": len(samples),
        "object_count": 2,
        "config": dict(CONFIG),
        "split_manifest": split_path.name,
        "split_manifest_sha256": _sha(split_path),
        "samples": samples,
        "scope": "Separate matched corpus; never pool with dataset-0.1/0.2 training",
    }
    manifest["manifest_hash"] = _digest(manifest)
    path = root / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    validation = validate_generalization_dataset(root)
    if not validation["ok"]:
        raise RuntimeError(
            "Generated dataset failed validation: " + "; ".join(validation["errors"])
        )
    return {"manifest_path": path, "manifest": manifest, "validation": validation}


def validate_generalization_dataset(input_dir, *, replay=False):
    """Check hashes, every expected case, factor definitions and native replay."""
    if replay:
        from ..sim.replay import replay_saved_rollout

    root, errors, replayed = Path(input_dir), [], 0
    try:
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        if manifest["schema_version"] != "dataset-0.3":
            raise ValueError("Expected dataset-0.3")
        payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        if manifest["manifest_hash"] != _digest(payload):
            raise ValueError("Dataset manifest hash mismatch")
        _validate_spec(manifest["seed"], manifest["group_counts"])
        if manifest["config"] != CONFIG or manifest["object_count"] != 2:
            raise ValueError("Native dataset-0.3 configuration changed")
        if manifest["split_manifest"] != "splits.json":
            raise ValueError("Expected local splits.json")
        split_path = root / "splits.json"
        splits = load_split_manifest(split_path)["splits"]
        if _sha(split_path) != manifest["split_manifest_sha256"] or set(splits) != set(SPLITS):
            raise ValueError("Split manifest mismatch")
        expected = list(_specs(manifest["seed"], manifest["group_counts"]))
        rows = manifest["samples"]
        if len(rows) != len(expected) or manifest["sample_count"] != len(expected):
            raise ValueError("Sample count differs from complete expected coverage")
        if manifest["group_count"] != sum(manifest["group_counts"].values()):
            raise ValueError("Group count differs from specification")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Sample IDs must be unique")
        by_id = {row["sample_id"]: row for row in rows}
        expected_groups = {split: set() for split in SPLITS}
        initial = {}
        for split, scene_seed, action, profile, physics_seed in expected:
            scene = make_generalization_scene(
                scene_seed, split=split, action_variant=action, physics_seed=physics_seed
            )
            group, condition = scene["scene_group_id"], scene["condition_id"]
            expected_groups[split].add(group)
            name = f"{condition}-p{profile:02d}"
            row = by_id[name]
            for field, value in {
                "scene_group_id": group,
                "condition_id": condition,
                "split": split,
                "visible_seed": scene_seed,
                "physics_seed": physics_seed,
                "action_variant": action,
                "physics_variant": profile,
                "arrays": name + ".npz",
                "metadata": name + ".json",
            }.items():
                if row[field] != value:
                    raise ValueError(f"{name}: {field} differs from the specification")
            for field in ("arrays", "metadata"):
                if _sha(root / row[field]) != row[field + "_sha256"]:
                    raise ValueError(f"{name}: saved {field} hash changed")
            rollout = load_rollout(root, name)
            if rollout["scene_spec"] != scene:
                raise ValueError(f"{name}: scene differs from the single-factor specification")
            expected_metadata = {
                "sim_version": "sim-0.2",
                "render_version": "render-0.2",
                "frame_count": 32,
                "fps": 8,
                "physics_hz": 240,
                "resolution": 64,
            }
            if rollout["metadata"] != expected_metadata or rollout["frames"].shape != (
                32,
                64,
                64,
                3,
            ):
                raise ValueError(f"{name}: native clip shape changed")
            rgb = rollout["frames"][0]
            if group in initial and not np.array_equal(initial[group], rgb):
                raise ValueError(f"{name}: frame 0 changed within a visible scene")
            initial[group] = rgb
            if replay:
                report = replay_saved_rollout(str(root), name, atol=0, rtol=0)
                if not report["ok"]:
                    raise ValueError(f"{name}: exact replay failed")
                replayed += 1
        if any(set(splits[split]) != expected_groups[split] for split in SPLITS):
            raise ValueError("Split groups differ from the disjoint generation specification")
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(str(error))
    return {"ok": not errors, "errors": errors, "replayed_samples": replayed}
