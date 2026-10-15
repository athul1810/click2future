"""Small NumPy loaders for downstream model code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .export import load_rollout


def load_sample(
    output_dir: str | Path,
    sample_id: str,
    *,
    normalize_frames: bool = True,
) -> dict[str, Any]:
    """Load one sample with model-friendly channel-first frames."""

    rollout = load_rollout(output_dir, sample_id)
    result = dict(rollout)
    frames = rollout["frames"].astype(np.float32)
    if normalize_frames:
        frames /= 255.0
    result["frames"] = np.transpose(frames, (0, 3, 1, 2))
    result["selected_mask"] = rollout["selected_mask"][None, ...]
    return result


def load_batch(
    output_dir: str | Path,
    sample_ids: list[str] | tuple[str, ...],
    *,
    normalize_frames: bool = True,
) -> dict[str, Any]:
    """Load samples into padded model tensors.

    Frames are returned as ``float32[B,T,3,H,W]``. Object-dependent arrays are
    padded to the largest object count in the batch and tracked by
    ``object_valid``.
    """

    ids = list(sample_ids)
    if not ids:
        raise ValueError("sample_ids must not be empty")
    rollouts = [load_rollout(output_dir, sample_id) for sample_id in ids]
    frame_shapes = {rollout["frames"].shape for rollout in rollouts}
    if len(frame_shapes) != 1:
        raise ValueError("all samples in a batch must have the same frame shape")
    action_shapes = {rollout["action_map"].shape for rollout in rollouts}
    if len(action_shapes) != 1:
        raise ValueError("all samples in a batch must have the same action-map shape")

    batch_size = len(rollouts)
    frame_count, height, width, _ = rollouts[0]["frames"].shape
    max_objects = max(rollout["object_valid"].shape[0] for rollout in rollouts)
    frames = np.stack([rollout["frames"] for rollout in rollouts]).astype(np.float32)
    if normalize_frames:
        frames /= 255.0

    instance_masks = np.zeros(
        (batch_size, frame_count, max_objects, height, width),
        dtype=bool,
    )
    object_valid = np.zeros((batch_size, max_objects), dtype=bool)
    positions = np.zeros((batch_size, frame_count, max_objects, 2), dtype=np.float32)
    collision_onset = np.zeros(
        (batch_size, max(frame_count - 1, 0), max_objects, max_objects),
        dtype=bool,
    )
    wall_contact_onset = np.zeros(
        (batch_size, max(frame_count - 1, 0), max_objects, 4),
        dtype=bool,
    )
    for batch_index, rollout in enumerate(rollouts):
        object_count = rollout["object_valid"].shape[0]
        instance_masks[batch_index, :, :object_count] = rollout["instance_masks"]
        object_valid[batch_index, :object_count] = rollout["object_valid"]
        positions[batch_index, :, :object_count] = rollout["positions"]
        collision_onset[batch_index, :, :object_count, :object_count] = rollout[
            "collision_onset"
        ]
        wall_contact_onset[batch_index, :, :object_count] = rollout[
            "wall_contact_onset"
        ]

    return {
        "sample_ids": ids,
        "frames": np.transpose(frames, (0, 1, 4, 2, 3)),
        "instance_masks": instance_masks,
        "object_valid": object_valid,
        "selected_object_id": np.asarray(
            [rollout["selected_object_id"] for rollout in rollouts],
            dtype=np.int64,
        ),
        "selected_mask": np.stack(
            [rollout["selected_mask"] for rollout in rollouts],
            axis=0,
        )[:, None],
        "force_xy": np.stack([rollout["force_xy"] for rollout in rollouts]),
        "force_duration": np.asarray(
            [rollout["force_duration"] for rollout in rollouts],
            dtype=np.float32,
        ),
        "action_map": np.stack([rollout["action_map"] for rollout in rollouts]),
        "timestamps": np.stack([rollout["timestamps"] for rollout in rollouts]),
        "positions": positions,
        "collision_onset": collision_onset,
        "wall_contact_onset": wall_contact_onset,
        "scene_specs": [rollout["scene_spec"] for rollout in rollouts],
        "metadata": [rollout["metadata"] for rollout in rollouts],
    }
