"""Deterministic scene-group split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def build_split_manifest(
    scene_group_ids: list[str],
    *,
    seed: int = 0,
    fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assign each unique scene group to exactly one split."""

    fractions = dict(DEFAULT_FRACTIONS if fractions is None else fractions)
    _validate_fractions(fractions)
    unique_ids = sorted(set(scene_group_ids))
    rng = np.random.default_rng(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)

    counts = _split_counts(len(shuffled), fractions)
    splits: dict[str, list[str]] = {}
    start = 0
    for split_name in fractions:
        end = start + counts[split_name]
        splits[split_name] = sorted(shuffled[start:end])
        start = end

    manifest = {
        "schema_version": "splits-0.1",
        "seed": int(seed),
        "fractions": fractions,
        "splits": splits,
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    return manifest


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Validate and write a split manifest as canonical JSON."""

    validate_split_manifest(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("manifest must contain a non-empty splits mapping")

    groups: list[str] = []
    for split_name, split_groups in splits.items():
        if not isinstance(split_name, str) or not isinstance(split_groups, list):
            raise ValueError("split names and group lists must be strings/lists")
        if any(not isinstance(group, str) for group in split_groups):
            raise ValueError("scene group IDs must be strings")
        groups.extend(split_groups)

    if len(groups) != len(set(groups)):
        raise ValueError("scene groups must not appear in multiple splits")
    if manifest.get("manifest_hash") != _manifest_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise ValueError("manifest_hash does not match manifest contents")


def assign_split(scene_group_id: str, manifest: dict[str, Any]) -> str:
    """Return the split containing a scene group."""

    validate_split_manifest(manifest)
    for split_name, groups in manifest["splits"].items():
        if scene_group_id in groups:
            return split_name
    raise KeyError(f"scene group is not present in manifest: {scene_group_id}")


def _split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    nonzero = [name for name, fraction in fractions.items() if fraction > 0]
    if total < len(nonzero):
        counts = {name: 0 for name in fractions}
        for name in sorted(nonzero, key=lambda value: (-fractions[value], list(fractions).index(value)))[:total]:
            counts[name] = 1
        return counts

    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: (1 if fractions[name] > 0 else 0) for name in fractions}
    for _ in range(total - sum(counts.values())):
        target = max(
            fractions,
            key=lambda name: (raw[name] - counts[name], -list(fractions).index(name)),
        )
        counts[target] += 1
    return counts


def _validate_fractions(fractions: dict[str, float]) -> None:
    if not fractions or any(value < 0 for value in fractions.values()):
        raise ValueError("split fractions must be non-negative")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("split fractions must sum to 1")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
