"""Deterministic scenes for dataset-0.2 action comparisons."""

from __future__ import annotations

from typing import Any

from .scene import make_scene

ACTION_VARIANTS = (
    "zero",
    "forward",
    "reverse",
    "collision_0_to_1",
    "collision_1_to_0",
    "wall",
)


def make_controlled_scene(
    seed: int,
    *,
    action_variant: str,
    physics_seed: int | None = None,
    force_duration: float = 0.25,
    forward_force: float = 0.15,
    collision_force: float = 0.25,
    wall_force: float = 1.1,
) -> dict[str, Any]:
    """Create one two-object scene for a controlled dataset-0.2 action."""

    if action_variant not in ACTION_VARIANTS:
        raise ValueError(f"action_variant must be one of {ACTION_VARIANTS}")
    if force_duration < 0:
        raise ValueError("force_duration must be non-negative")
    if forward_force <= 0 or collision_force <= 0 or wall_force <= 0:
        raise ValueError("force magnitudes must be positive")

    scene = make_scene(
        seed,
        object_count=2,
        physics_seed=seed if physics_seed is None else physics_seed,
        action_seed=seed,
    )
    selected, target = scene["objects"]
    selected["angle"] = 0.0
    target["angle"] = 0.0

    selected_x, selected_y = _half_extents(selected)
    target_x, target_y = _half_extents(target)
    lateral_side = 1.0 if int(seed) % 2 == 0 else -1.0
    lateral_offset = lateral_side * 0.35 * (selected_x + target_x)
    selected["position"] = [0.5, 0.45]
    target["position"] = [
        0.5 + lateral_offset,
        0.45 + selected_y + target_y + 0.04,
    ]

    forces = {
        "zero": [0.0, 0.0],
        "forward": [forward_force, 0.0],
        "reverse": [-forward_force, 0.0],
        "collision_0_to_1": [0.0, collision_force],
        "collision_1_to_0": [0.0, -collision_force],
        "wall": [-wall_force, 0.0],
    }
    selected_object_id = 1 if action_variant == "collision_1_to_0" else 0
    scene.update(
        {
            "schema_version": "scene-0.2",
            "condition_id": f"{scene['scene_group_id']}-{action_variant}",
            "action_variant": action_variant,
            "action": {
                "selected_object_id": selected_object_id,
                "force_xy": forces[action_variant],
                "force_duration": float(force_duration),
            },
        }
    )
    return scene


def _half_extents(obj: dict[str, Any]) -> tuple[float, float]:
    geometry = obj["geometry"]
    if obj["shape"] == "circle":
        radius = float(geometry["radius"])
        return radius, radius
    return float(geometry["width"]) / 2.0, float(geometry["height"]) / 2.0
