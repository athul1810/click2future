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
    if frame_count == 0:
        raise ValueError("frames must contain at least one frame")
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
    selected_object_id = int(rollout["selected_object_id"])
    if not 0 <= selected_object_id < object_count:
        raise ValueError("selected_object_id is outside object_valid")
    if not object_valid[selected_object_id]:
        raise ValueError("selected_object_id must refer to a valid object")
    if not np.array_equal(selected_mask, masks[0, selected_object_id]):
        raise ValueError("selected_mask must equal the selected frame-0 instance mask")
    for field, values in (
        ("force_xy", force_xy),
        ("action_map", action_map),
        ("timestamps", timestamps),
        ("positions", positions),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field} must contain only finite values")
    force_duration = float(rollout["force_duration"])
    if not np.isfinite(force_duration) or force_duration < 0:
        raise ValueError("force_duration must be finite and non-negative")
    if timestamps[0] != 0.0:
        raise ValueError("timestamps must start at zero")
    if frame_count > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if np.any((positions < 0.0) | (positions > 1.0)):
        raise ValueError("positions must be normalized to [0,1]")
    if not np.array_equal(collision_onset, np.swapaxes(collision_onset, 1, 2)):
        raise ValueError("collision_onset must be symmetric")
    if np.any(np.diagonal(collision_onset, axis1=1, axis2=2)):
        raise ValueError("collision_onset diagonal must be false")
    expected_action_map = selected_mask[None].astype(np.float32) * force_xy[:, None, None]
    if not np.array_equal(action_map, expected_action_map):
        raise ValueError("action_map must equal selected_mask multiplied by force_xy")
    _validate_unit_world(rollout["scene_spec"])


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


def _validate_unit_world(scene_spec: Any) -> None:
    if not isinstance(scene_spec, dict):
        raise ValueError("scene_spec must be a mapping")
    world_size = scene_spec.get("world_size")
    try:
        is_unit_world = len(world_size) == 2 and all(float(size) == 1.0 for size in world_size)
    except (TypeError, ValueError):
        is_unit_world = False
    if not is_unit_world:
        raise ValueError("scene world_size must be [1.0, 1.0] for rollout-0.1")
