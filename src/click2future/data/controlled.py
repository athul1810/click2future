"""Generation and validation for the controlled dataset-0.2 layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.controlled import ACTION_VARIANTS, make_controlled_scene
from ..sim.rollout import run_rollout
from .export import load_rollout, save_rollout
from .splits import (
    assign_split,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
)

PHYSICS_VARIANTS = 4
CLIPS_PER_GROUP = 1 + (len(ACTION_VARIANTS) - 1) * PHYSICS_VARIANTS


def generate_controlled_dataset(
    output_dir: str | Path,
    *,
    group_count: int,
    seed: int = 0,
    split_seed: int | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate dataset-0.2 controlled action and contact cases."""

    if group_count <= 0:
        raise ValueError("group_count must be positive")
    resolved_config = {
        "world_size": [1.0, 1.0],
        "resolution": 64,
        "frame_count": 32,
        "fps": 8,
        "physics_hz": 240,
        "force_duration": 0.25,
        "forward_force": 0.15,
        "collision_force": 0.25,
        "wall_force": 1.1,
    }
    resolved_config.update(config or {})
    _validate_world_size(resolved_config["world_size"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_seeds = [int(seed) + offset for offset in range(group_count)]
    group_ids = [f"scene-{scene_seed:08d}" for scene_seed in scene_seeds]
    resolved_split_seed = int(seed if split_seed is None else split_seed)
    split_manifest = build_split_manifest(group_ids, seed=resolved_split_seed)
    split_path = save_split_manifest(split_manifest, output_path / "splits.json")

    samples: list[dict[str, Any]] = []
    for group_index, scene_seed in enumerate(scene_seeds):
        group_id = group_ids[group_index]
        split_name = assign_split(group_id, split_manifest)
        for action_variant in ACTION_VARIANTS:
            variant_count = 1 if action_variant == "zero" else PHYSICS_VARIANTS
            for physics_variant in range(variant_count):
                physics_seed = _physics_seed(seed, group_index, physics_variant)
                scene = make_controlled_scene(
                    scene_seed,
                    action_variant=action_variant,
                    physics_seed=physics_seed,
                    force_duration=float(resolved_config.get("force_duration", 0.25)),
                    forward_force=float(resolved_config.get("forward_force", 0.15)),
                    collision_force=float(resolved_config.get("collision_force", 0.25)),
                    wall_force=float(resolved_config.get("wall_force", 1.1)),
                )
                rollout = run_rollout(
                    scene,
                    frame_count=int(resolved_config.get("frame_count", 32)),
                    fps=int(resolved_config.get("fps", 8)),
                    physics_hz=int(resolved_config.get("physics_hz", 240)),
                    resolution=int(resolved_config.get("resolution", 64)),
                )
                sample_id = f"{group_id}-{action_variant}-p{physics_variant:02d}"
                paths = save_rollout(rollout, output_path, sample_id)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "scene_group_id": group_id,
                        "condition_id": scene["condition_id"],
                        "action_variant": action_variant,
                        "physics_variant": physics_variant,
                        "split": split_name,
                        "visible_seed": scene_seed,
                        "physics_seed": physics_seed,
                        "has_object_collision": bool(rollout["collision_onset"].any()),
                        "has_wall_contact": bool(rollout["wall_contact_onset"].any()),
                        "arrays": paths["arrays"].name,
                        "metadata": paths["metadata"].name,
                    }
                )

    manifest = {
        "schema_version": "dataset-0.2",
        "seed": int(seed),
        "split_seed": resolved_split_seed,
        "group_count": int(group_count),
        "clips_per_group": CLIPS_PER_GROUP,
        "sample_count": len(samples),
        "object_count": 2,
        "action_variants": list(ACTION_VARIANTS),
        "physics_variants": PHYSICS_VARIANTS,
        "config": resolved_config,
        "split_manifest": split_path.name,
        "samples": samples,
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    manifest_path = output_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_controlled_dataset(output_path)
    if not validation["ok"]:
        message = "; ".join(validation["errors"])
        raise RuntimeError(f"generated dataset failed validation: {message}")
    return {
        "manifest_path": manifest_path,
        "split_path": split_path,
        "manifest": manifest,
        "validation": validation,
    }


def validate_controlled_dataset(
    output_dir: str | Path,
    *,
    replay: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Validate dataset-0.2 structure, action outcomes, and optional replay."""

    output_path = Path(output_dir)
    errors: list[str] = []
    try:
        manifest = json.loads((output_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _validation_report(errors=[f"could not read dataset manifest: {error}"])

    if manifest.get("schema_version") != "dataset-0.2":
        errors.append("manifest schema_version must be dataset-0.2")
    saved_hash = manifest.get("manifest_hash")
    hash_input = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if saved_hash != _manifest_hash(hash_input):
        errors.append("manifest_hash does not match manifest contents")

    records = manifest.get("samples")
    if not isinstance(records, list):
        return _validation_report(errors=errors + ["manifest samples must be a list"])
    expected_count = int(manifest.get("group_count", 0)) * CLIPS_PER_GROUP
    if len(records) != expected_count or manifest.get("sample_count") != expected_count:
        errors.append(f"manifest must contain {expected_count} samples")
    sample_ids = [record.get("sample_id") for record in records if isinstance(record, dict)]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("sample IDs must be unique")

    try:
        split_manifest = load_split_manifest(output_path / str(manifest["split_manifest"]))
    except (OSError, KeyError, TypeError, ValueError) as error:
        return _validation_report(errors=errors + [f"invalid split manifest: {error}"])

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("scene_group_id"), str):
            errors.append("every sample record must have a scene_group_id")
            continue
        groups.setdefault(record["scene_group_id"], []).append(record)
    if len(groups) != manifest.get("group_count"):
        errors.append("scene group count does not match the manifest")

    object_collision_clips = 0
    wall_contact_clips = 0
    replayed_samples = 0
    if replay:
        from ..sim.replay import replay_saved_rollout

    for group_id, group_records in groups.items():
        try:
            expected_split = assign_split(group_id, split_manifest)
        except (KeyError, ValueError) as error:
            errors.append(f"{group_id}: invalid split assignment: {error}")
            continue
        if any(record.get("split") != expected_split for record in group_records):
            errors.append(f"{group_id}: all samples must use the group split")

        rollouts: dict[tuple[str, int], dict[str, Any]] = {}
        for record in group_records:
            action = record.get("action_variant")
            physics_variant = record.get("physics_variant")
            sample_id = record.get("sample_id")
            if (
                action not in ACTION_VARIANTS
                or isinstance(physics_variant, bool)
                or not isinstance(physics_variant, int)
            ):
                errors.append(f"{group_id}: invalid action or physics variant")
                continue
            try:
                rollout = load_rollout(output_path, str(sample_id))
            except (OSError, KeyError, ValueError) as error:
                errors.append(f"{sample_id}: could not load rollout: {error}")
                continue
            scene = rollout["scene_spec"]
            if (
                scene.get("scene_group_id") != group_id
                or scene.get("condition_id") != record.get("condition_id")
                or scene.get("action_variant") != action
                or scene.get("seeds", {}).get("visible") != record.get("visible_seed")
                or scene.get("seeds", {}).get("physics") != record.get("physics_seed")
            ):
                errors.append(f"{sample_id}: manifest labels do not match the saved scene")
            rollouts[(action, physics_variant)] = rollout
            has_object_collision = bool(rollout["collision_onset"].any())
            has_wall_contact = bool(rollout["wall_contact_onset"].any())
            object_collision_clips += int(has_object_collision)
            wall_contact_clips += int(has_wall_contact)
            if record.get("has_object_collision") != has_object_collision:
                errors.append(f"{sample_id}: object-collision summary is incorrect")
            if record.get("has_wall_contact") != has_wall_contact:
                errors.append(f"{sample_id}: wall-contact summary is incorrect")
            if replay:
                replay_report = replay_saved_rollout(
                    str(output_path), str(sample_id), atol=atol, rtol=rtol
                )
                replayed_samples += 1
                if not replay_report["ok"]:
                    errors.append(f"{sample_id}: replay does not match")

        errors.extend(_validate_group(group_id, group_records, rollouts))

    return _validation_report(
        errors=errors,
        group_count=len(groups),
        sample_count=len(records),
        object_collision_clips=object_collision_clips,
        wall_contact_clips=wall_contact_clips,
        replayed_samples=replayed_samples,
    )


def _validate_group(
    group_id: str,
    records: list[dict[str, Any]],
    rollouts: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    expected_keys = {("zero", 0)} | {
        (action, variant)
        for action in ACTION_VARIANTS
        if action != "zero"
        for variant in range(PHYSICS_VARIANTS)
    }
    if set(rollouts) != expected_keys or len(records) != len(expected_keys):
        return [f"{group_id}: expected one zero and four variants per nonzero action"]

    first = rollouts[("zero", 0)]
    for rollout in rollouts.values():
        if not np.array_equal(rollout["frames"][0], first["frames"][0]):
            errors.append(f"{group_id}: frame 0 must match across all actions and physics")
            break

    stationary_positions = np.broadcast_to(first["positions"][0], first["positions"].shape)
    if not np.array_equal(first["positions"], stationary_positions):
        errors.append(f"{group_id}: zero action must not move")
    if first["collision_onset"].any() or first["wall_contact_onset"].any():
        errors.append(f"{group_id}: zero action must not contact anything")

    zero_physics = _hidden_physics(first)
    physics_profiles = []
    for variant in range(PHYSICS_VARIANTS):
        forward = rollouts[("forward", variant)]
        reverse = rollouts[("reverse", variant)]
        collision_0_to_1 = rollouts[("collision_0_to_1", variant)]
        collision_1_to_0 = rollouts[("collision_1_to_0", variant)]
        wall = rollouts[("wall", variant)]
        if not np.array_equal(forward["force_xy"], -reverse["force_xy"]):
            errors.append(f"{group_id}: forward and reverse forces must be exact opposites")
        for action, rollout in (("forward", forward), ("reverse", reverse)):
            if rollout["collision_onset"].any() or rollout["wall_contact_onset"].any():
                errors.append(f"{group_id}: {action} variant {variant} must be contact-free")
            displacement = (
                rollout["positions"][-1, rollout["selected_object_id"]]
                - rollout["positions"][0, rollout["selected_object_id"]]
            )
            if float(np.dot(displacement, rollout["force_xy"])) <= 0:
                errors.append(f"{group_id}: {action} variant {variant} moved against its force")
        for action, rollout, selected_object_id in (
            ("collision_0_to_1", collision_0_to_1, 0),
            ("collision_1_to_0", collision_1_to_0, 1),
        ):
            if rollout["selected_object_id"] != selected_object_id:
                errors.append(f"{group_id}: {action} selected the wrong object")
            if not rollout["collision_onset"].any():
                errors.append(f"{group_id}: {action} variant {variant} did not collide")
            if rollout["wall_contact_onset"].any():
                errors.append(f"{group_id}: {action} variant {variant} reached a wall")
        if not wall["wall_contact_onset"].any():
            errors.append(f"{group_id}: wall variant {variant} did not reach a wall")
        if wall["collision_onset"].any():
            errors.append(f"{group_id}: wall variant {variant} hit another object")
        profile = _hidden_physics(forward)
        physics_profiles.append(profile)
        matching_profiles = all(
            profile == _hidden_physics(rollout)
            for rollout in (reverse, collision_0_to_1, collision_1_to_0, wall)
        )
        if not matching_profiles:
            errors.append(f"{group_id}: physics variant {variant} changed across actions")
        if variant == 0 and profile != zero_physics:
            errors.append(f"{group_id}: zero action must use physics variant 0")

    unique_profiles = {json.dumps(profile, sort_keys=True) for profile in physics_profiles}
    if len(unique_profiles) != PHYSICS_VARIANTS:
        errors.append(f"{group_id}: hidden-physics profiles must differ")

    for action in ACTION_VARIANTS:
        condition_ids = {
            record.get("condition_id")
            for record in records
            if record.get("action_variant") == action
        }
        if condition_ids != {f"{group_id}-{action}"}:
            errors.append(f"{group_id}: {action} condition_id is inconsistent")
    return errors


def _hidden_physics(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return [obj["hidden_physics"] for obj in rollout["scene_spec"]["objects"]]


def _validation_report(
    *,
    errors: list[str],
    group_count: int = 0,
    sample_count: int = 0,
    object_collision_clips: int = 0,
    wall_contact_clips: int = 0,
    replayed_samples: int = 0,
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "group_count": group_count,
        "sample_count": sample_count,
        "object_collision_clips": object_collision_clips,
        "wall_contact_clips": wall_contact_clips,
        "replayed_samples": replayed_samples,
        "errors": errors,
    }


def _physics_seed(seed: int, group_index: int, physics_variant: int) -> int:
    return int(seed) * 1_000_000 + int(group_index) * 1_000 + int(physics_variant) + 1


def _validate_world_size(world_size: Any) -> None:
    try:
        is_unit_world = len(world_size) == 2 and all(float(size) == 1.0 for size in world_size)
    except (TypeError, ValueError):
        is_unit_world = False
    if not is_unit_world:
        raise ValueError("world_size must be [1.0, 1.0] for dataset-0.2")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
