"""JSON plus compressed NumPy storage for rollout records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .validate import validate_rollout


ARRAY_FIELDS = (
    "frames",
    "instance_masks",
    "object_valid",
    "selected_mask",
    "force_xy",
    "action_map",
    "timestamps",
    "positions",
    "collision_onset",
    "wall_contact_onset",
)


def save_rollout(
    rollout: dict[str, Any],
    output_dir: str | Path,
    sample_id: str,
) -> dict[str, Path]:
    """Save one validated rollout as ``sample_id.json`` and ``sample_id.npz``."""

    validate_rollout(rollout)
    _validate_sample_id(sample_id)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    arrays = {field: rollout[field] for field in ARRAY_FIELDS}
    metadata = {
        "schema_version": rollout["schema_version"],
        "sample_id": sample_id,
        "selected_object_id": int(rollout["selected_object_id"]),
        "force_duration": float(rollout["force_duration"]),
        "scene_spec": rollout["scene_spec"],
        "metadata": rollout["metadata"],
    }

    array_path = output_path / f"{sample_id}.npz"
    metadata_path = output_path / f"{sample_id}.json"
    np.savez_compressed(array_path, **arrays)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"arrays": array_path, "metadata": metadata_path}


def load_rollout(output_dir: str | Path, sample_id: str) -> dict[str, Any]:
    """Load and validate a rollout written by :func:`save_rollout`."""

    _validate_sample_id(sample_id)
    output_path = Path(output_dir)
    array_path = output_path / f"{sample_id}.npz"
    metadata_path = output_path / f"{sample_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    with np.load(array_path, allow_pickle=False) as arrays:
        rollout = {field: arrays[field] for field in ARRAY_FIELDS}
    rollout.update(
        {
            "schema_version": metadata["schema_version"],
            "selected_object_id": metadata["selected_object_id"],
            "force_duration": metadata["force_duration"],
            "scene_spec": metadata["scene_spec"],
            "metadata": metadata["metadata"],
        }
    )
    validate_rollout(rollout)
    return rollout


def _validate_sample_id(sample_id: str) -> None:
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError("sample_id must be a non-empty filename component")
