"""Checked train/validation cache; privileged physics never enters inference."""

import hashlib
import json
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.models.object_motion import observe
from click2future.models.world_motion import WORLD_BOUNDS, WORLD_CONTRACT

from .contact_geometry import observed_hulls
from .train_cvae import sha256
from .trajectory_data import add_training_supplement, canonical_ids, prepare_trajectories


def world_geometry(root, data):
    """Bind each augmented spatial slot to its original object's physical labels."""
    references, geometry, worlds = {}, {}, {}
    for split, row in data.items():
        if split not in {"train", "validation"}:
            raise ValueError("World label preparation accepts train/validation only")
        rows, labels = [], []
        for index, example in enumerate(row["examples"]):
            key = (str(Path(example.get("root", root)).resolve()), example["sample_id"])
            if key not in references:
                loaded = load_rollout(*key)
                references[key] = {
                    "frames": loaded["frames"][:1].copy(),
                    "selected_mask": loaded["selected_mask"].copy(),
                    "selected_object_id": loaded["selected_object_id"],
                    "scene_spec": loaded["scene_spec"],
                }
            reference = references[key]
            image = np.rot90(reference["frames"][0], example["quarter_turns"]).copy()
            mask = np.rot90(reference["selected_mask"], example["quarter_turns"]).copy()
            if example["mirror"]:
                image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
            observation = observe(image, mask)
            if not np.array_equal(observation.features, row["features"][index]):
                raise ValueError("World labels and observable slot identities differ")
            order = canonical_ids(observation, reference["selected_object_id"])
            objects = {obj["object_id"]: obj for obj in reference["scene_spec"]["objects"]}
            physical = np.asarray(
                [
                    [
                        objects[obj]["hidden_physics"][name]
                        for name in ("mass", "friction", "restitution")
                    ]
                    for obj in order
                ],
                dtype=np.float32,
            )
            bounds = np.asarray(WORLD_BOUNDS, dtype=np.float32)
            if (
                not np.isfinite(physical).all()
                or ((physical < bounds[:, 0]) | (physical > bounds[:, 1])).any()
            ):
                raise ValueError("Training physical labels lie outside the declared prior")
            rows.append(observed_hulls(observation))
            labels.append(physical)
        geometry[split] = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
        worlds[split] = np.stack(labels)
    return geometry, worlds


def prepare_cache(dataset, supplement, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data, provenance = prepare_trajectories(dataset, augment=True)
    provenance.pop("preparation_seconds", None)
    data, provenance = add_training_supplement(dataset, supplement, data, provenance, augment=True)
    geometry, worlds = world_geometry(dataset, data)
    provenance["geometry_sha256"] = {
        split: {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in row.items()}
        for split, row in geometry.items()
    }
    provenance["world_label_sha256"] = {
        split: hashlib.sha256(value.tobytes()).hexdigest() for split, value in worlds.items()
    }
    arrays = {}
    for split in data:
        for key in ("motion", "force", "features"):
            arrays[split + "_" + key] = data[split][key]
        arrays[split + "_world"] = worlds[split]
        for key, value in geometry[split].items():
            arrays[split + "_" + key] = value
    np.savez_compressed(output / "arrays.npz", **arrays)
    record = {
        "format_version": 1,
        "world_contract": WORLD_CONTRACT,
        "provenance": provenance,
        "examples": {split: row["examples"] for split, row in data.items()},
        "array_sha256": sha256(output / "arrays.npz"),
        "scope": "training/validation only; physical labels supervise training, not inference",
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_cache(root):
    root = Path(root)
    record = json.loads((root / "manifest.json").read_text())
    if (
        record["format_version"] != 1
        or record["world_contract"] != WORLD_CONTRACT
        or set(record["examples"]) != {"train", "validation"}
        or sha256(root / "arrays.npz") != record["array_sha256"]
    ):
        raise ValueError("World cache identity changed")
    data, geometry, worlds = {}, {}, {}
    with np.load(root / "arrays.npz", allow_pickle=False) as arrays:
        for split in ("train", "validation"):
            data[split] = {
                key: arrays[split + "_" + key] for key in ("motion", "features", "force")
            }
            data[split]["examples"] = record["examples"][split]
            geometry[split] = {
                key: arrays[split + "_" + key] for key in ("hulls", "normals", "axis_valid")
            }
            worlds[split] = arrays[split + "_world"]
            for key, value in geometry[split].items():
                if (
                    hashlib.sha256(value.tobytes()).hexdigest()
                    != record["provenance"]["geometry_sha256"][split][key]
                ):
                    raise ValueError("Cached geometry differs")
            if (
                hashlib.sha256(worlds[split].tobytes()).hexdigest()
                != record["provenance"]["world_label_sha256"][split]
            ):
                raise ValueError("Cached world labels differ")
    return data, geometry, worlds, record["provenance"]
