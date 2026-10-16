"""Grouped dataset generation for visible-scene physics variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..sim.rollout import run_rollout
from ..sim.scene import make_scene, visible_scene
from .export import save_rollout
from .splits import assign_split, build_split_manifest, save_split_manifest


def generate_dataset(
    output_dir: str | Path,
    *,
    group_count: int,
    variants_per_group: int,
    seed: int = 0,
    split_seed: int | None = None,
    object_count: int = 2,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate grouped rollout records and write a dataset index.

    Each scene group reuses the same visible scene and action while changing
    only the hidden physics seed. All variants remain in the same split.
    """

    _validate_counts(group_count, variants_per_group)
    resolved_config = dict(config or {})
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    world_size_value = resolved_config.get("world_size", [1.0, 1.0])
    world_size = (float(world_size_value[0]), float(world_size_value[1]))
    scene_seeds = [int(seed) + offset for offset in range(group_count)]
    scenes = [
        make_scene(
            scene_seed,
            object_count=object_count,
            physics_seed=scene_seed,
            action_seed=scene_seed,
            world_size=world_size,
        )
        for scene_seed in scene_seeds
    ]
    for scene in scenes:
        if "force_duration" in resolved_config:
            scene["action"]["force_duration"] = float(resolved_config["force_duration"])

    group_ids = [scene["scene_group_id"] for scene in scenes]
    split_manifest = build_split_manifest(
        group_ids,
        seed=seed if split_seed is None else int(split_seed),
    )
    split_path = save_split_manifest(split_manifest, output_path / "splits.json")

    samples: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes):
        group_id = str(scene["scene_group_id"])
        split_name = assign_split(group_id, split_manifest)
        visible_condition = visible_scene(scene)
        physics_signatures: set[str] = set()
        for variant_index in range(variants_per_group):
            variant_scene = make_scene(
                int(scene["seeds"]["visible"]),
                object_count=object_count,
                physics_seed=_variant_seed(seed, scene_index, variant_index),
                action_seed=int(scene["seeds"]["action"]),
                world_size=world_size,
            )
            if "force_duration" in resolved_config:
                variant_scene["action"]["force_duration"] = float(resolved_config["force_duration"])
            if visible_scene(variant_scene) != visible_condition:
                raise RuntimeError("physics variant changed the visible scene or action")
            physics_signature = json.dumps(
                [obj["hidden_physics"] for obj in variant_scene["objects"]],
                sort_keys=True,
            )
            if physics_signature in physics_signatures:
                raise RuntimeError("physics variants must differ within a scene group")
            physics_signatures.add(physics_signature)

            rollout = run_rollout(
                variant_scene,
                frame_count=int(resolved_config.get("frame_count", 16)),
                fps=int(resolved_config.get("fps", 8)),
                physics_hz=int(resolved_config.get("physics_hz", 240)),
                resolution=int(resolved_config.get("resolution", 64)),
            )
            sample_id = f"{group_id}-variant-{variant_index:03d}"
            paths = save_rollout(rollout, output_path, sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "scene_group_id": group_id,
                    "condition_id": variant_scene["condition_id"],
                    "variant_index": variant_index,
                    "split": split_name,
                    "physics_seed": variant_scene["seeds"]["physics"],
                    "arrays": paths["arrays"].name,
                    "metadata": paths["metadata"].name,
                }
            )

    manifest = {
        "schema_version": "dataset-0.1",
        "seed": int(seed),
        "split_seed": int(seed if split_seed is None else split_seed),
        "group_count": int(group_count),
        "variants_per_group": int(variants_per_group),
        "sample_count": len(samples),
        "object_count": int(object_count),
        "config": resolved_config,
        "split_manifest": split_path.name,
        "samples": samples,
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest_path": manifest_path, "split_path": split_path, "manifest": manifest}


def _variant_seed(seed: int, group_index: int, variant_index: int) -> int:
    """Derive stable seeds for hidden physics."""

    return int(seed) * 1_000_000 + int(group_index) * 1_000 + int(variant_index) + 1


def _validate_counts(group_count: int, variants_per_group: int) -> None:
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    if variants_per_group <= 0:
        raise ValueError("variants_per_group must be positive")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
