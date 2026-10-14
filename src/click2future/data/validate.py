"""Validation for the stable rollout record contract."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_rollout(rollout: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a rollout is missing or violates its contract."""

    required = {
        "schema_version",
        "frames",
        "instance_masks",
        "object_valid",
        "selected_object_id",
        "selected_mask",
        "force_xy",
        "force_duration",
        "action_map",
        "timestamps",
        "positions",
        "collision_onset",
        "wall_contact_onset",
        "scene_spec",
        "metadata",
    }
    missing = sorted(required.difference(rollout))
    if missing:
        raise ValueError(f"rollout is missing fields: {', '.join(missing)}")

    frames = _array(rollout, "frames", np.uint8, ndim=4)
    masks = _array(rollout, "instance_masks", np.bool_, ndim=4)
    object_valid = _array(rollout, "object_valid", np.bool_, ndim=1)
    selected_mask = _array(rollout, "selected_mask", np.bool_, ndim=2)
    force_xy = _array(rollout, "force_xy", np.float32, ndim=1)
    action_map = _array(rollout, "action_map", np.float32, ndim=3)
    timestamps = _array(rollout, "timestamps", np.float64, ndim=1)
    positions = _array(rollout, "positions", np.float32, ndim=3)
    collision_onset = _array(rollout, "collision_onset", np.bool_, ndim=3)
    wall_contact_onset = _array(rollout, "wall_contact_onset", np.bool_, ndim=3)

    frame_count, height, width, channels = frames.shape
    object_count = object_valid.shape[0]
    if channels != 3:
        raise ValueError("frames must have three color channels")
    if masks.shape != (frame_count, object_count, height, width):
        raise ValueError("instance_masks shape does not match frames/object_valid")
    if selected_mask.shape != (height, width):
        raise ValueError("selected_mask shape does not match frame size")
    if force_xy.shape != (2,):
        raise ValueError("force_xy must have shape (2,)")
    if action_map.shape != (2, height, width):
        raise ValueError("action_map must have shape (2,H,W)")
    if timestamps.shape != (frame_count,):
        raise ValueError("timestamps length must match frame count")
    if positions.shape != (frame_count, object_count, 2):
        raise ValueError("positions must have shape (T,N,2)")
    if collision_onset.shape != (max(frame_count - 1, 0), object_count, object_count):
        raise ValueError("collision_onset has an invalid shape")
    if wall_contact_onset.shape != (max(frame_count - 1, 0), object_count, 4):
        raise ValueError("wall_contact_onset has an invalid shape")
    if not 0 <= int(rollout["selected_object_id"]) < object_count:
        raise ValueError("selected_object_id is outside object_valid")
    if not np.array_equal(selected_mask, masks[0, int(rollout["selected_object_id"])]):
        raise ValueError("selected_mask must equal the selected frame-0 instance mask")
    if np.any(np.diff(timestamps) <= 0) and frame_count > 1:
        raise ValueError("timestamps must be strictly increasing")


def _array(
    rollout: dict[str, Any],
    field: str,
    dtype: np.dtype,
    *,
    ndim: int,
) -> np.ndarray:
    value = rollout[field]
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{field} must be a NumPy array")
    if value.dtype != dtype:
        raise ValueError(f"{field} must have dtype {dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{field} must have {ndim} dimensions")
    return value
