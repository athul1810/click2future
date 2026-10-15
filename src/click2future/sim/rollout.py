"""Assemble simulator steps, rendered frames, and oracle labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import WALL_ORDER, apply_force, build_space, consume_step_contacts, snapshot, step
from .render import render_frame


def run_rollout(
    scene: dict[str, Any],
    *,
    frame_count: int = 16,
    fps: int = 8,
    physics_hz: int = 240,
    resolution: int = 64,
) -> dict[str, Any]:
    """Generate one complete in-memory clip record from a scene description."""

    _validate_inputs(scene, frame_count, fps, physics_hz)
    steps_per_frame = physics_hz // fps
    object_count = len(scene["objects"])
    selected_object_id = int(scene["action"]["selected_object_id"])
    force_xy = np.asarray(scene["action"]["force_xy"], dtype=np.float32)
    force_duration = float(scene["action"]["force_duration"])

    state = build_space(scene, physics_hz=physics_hz)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    collision_onset = np.zeros((max(frame_count - 1, 0), object_count, object_count), dtype=bool)
    wall_contact_onset = np.zeros(
        (max(frame_count - 1, 0), object_count, len(WALL_ORDER)),
        dtype=bool,
    )

    current_snapshot = snapshot(state)
    frame, instance_masks = render_frame(scene, current_snapshot, resolution=resolution)
    frames.append(frame)
    masks.append(instance_masks)
    positions.append(_positions(current_snapshot, object_count))

    for interval_index in range(frame_count - 1):
        for _ in range(steps_per_frame):
            if state["time"] < force_duration:
                apply_force(state, selected_object_id, tuple(force_xy.tolist()))
            step(state)
            events = consume_step_contacts(state)
            _record_events(
                events,
                collision_onset[interval_index],
                wall_contact_onset[interval_index],
            )

        current_snapshot = snapshot(state)
        frame, instance_masks = render_frame(scene, current_snapshot, resolution=resolution)
        frames.append(frame)
        masks.append(instance_masks)
        positions.append(_positions(current_snapshot, object_count))

    selected_mask = masks[0][selected_object_id]
    action_map = np.stack(
        [
            selected_mask.astype(np.float32) * force_xy[0],
            selected_mask.astype(np.float32) * force_xy[1],
        ],
        axis=0,
    )

    return {
        "schema_version": "rollout-0.1",
        "frames": np.stack(frames, axis=0),
        "instance_masks": np.stack(masks, axis=0),
        "object_valid": np.ones(object_count, dtype=bool),
        "selected_object_id": selected_object_id,
        "selected_mask": selected_mask.copy(),
        "force_xy": force_xy,
        "force_duration": force_duration,
        "action_map": action_map,
        "timestamps": np.arange(frame_count, dtype=np.float64) / float(fps),
        "positions": np.stack(positions, axis=0),
        "collision_onset": collision_onset,
        "wall_contact_onset": wall_contact_onset,
        "scene_spec": scene,
        "metadata": {
            "sim_version": "sim-0.1",
            "render_version": "render-0.1",
            "frame_count": frame_count,
            "fps": fps,
            "physics_hz": physics_hz,
            "resolution": resolution,
        },
    }


def _positions(current_snapshot: dict[str, Any], object_count: int) -> np.ndarray:
    positions = np.zeros((object_count, 2), dtype=np.float32)
    for obj in current_snapshot["objects"]:
        positions[int(obj["object_id"])] = np.asarray(obj["position"], dtype=np.float32)
    return positions


def _record_events(
    events: dict[str, list[tuple[Any, ...]]],
    collision_target: np.ndarray,
    wall_target: np.ndarray,
) -> None:
    for first, second in events["object_object"]:
        collision_target[int(first), int(second)] = True
        collision_target[int(second), int(first)] = True
    for object_id, wall_name in events["object_wall"]:
        wall_target[int(object_id), WALL_ORDER.index(wall_name)] = True


def _validate_inputs(
    scene: dict[str, Any],
    frame_count: int,
    fps: int,
    physics_hz: int,
) -> None:
    world_size = scene.get("world_size")
    try:
        is_unit_world = len(world_size) == 2 and all(float(size) == 1.0 for size in world_size)
    except (TypeError, ValueError):
        is_unit_world = False
    if not is_unit_world:
        raise ValueError("scene world_size must be [1.0, 1.0] for rollout-0.1")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if fps <= 0 or physics_hz <= 0:
        raise ValueError("fps and physics_hz must be positive")
    if physics_hz % fps != 0:
        raise ValueError("physics_hz must be divisible by fps")
