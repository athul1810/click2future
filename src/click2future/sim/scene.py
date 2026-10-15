"""Deterministic scene sampling.

This module only creates serializable scene descriptions. Physics, rendering,
and dataset writing are kept in separate modules so they can evolve without
changing the scene contract.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


MIN_OBJECTS = 2
MAX_OBJECTS = 4
SHAPES = ("circle", "rectangle")


def make_scene(
    seed: int,
    *,
    object_count: int = 2,
    physics_seed: int | None = None,
    action_seed: int | None = None,
    world_size: tuple[float, float] = (1.0, 1.0),
) -> dict[str, Any]:
    """Create one JSON-serializable scene description.

    ``seed`` controls visible geometry, poses, and appearance. Physics variants
    can therefore reuse the same seed while changing only ``physics_seed``.
    ``action_seed`` is independent so action controls remain fixed when hidden
    physics changes.
    """

    _validate_inputs(object_count, world_size)

    visible_rng = np.random.default_rng(seed)
    physics_seed_value = seed if physics_seed is None else physics_seed
    action_seed_value = seed if action_seed is None else action_seed
    hidden_rng = np.random.default_rng(physics_seed_value)
    action_rng = np.random.default_rng(action_seed_value)

    objects: list[dict[str, Any]] = []
    for object_id in range(object_count):
        shape = str(visible_rng.choice(SHAPES))
        if shape == "circle":
            radius = float(visible_rng.uniform(0.05, 0.09))
            geometry = {"radius": radius}
            clearance = radius
        else:
            width = float(visible_rng.uniform(0.08, 0.16))
            height = float(visible_rng.uniform(0.08, 0.16))
            geometry = {"width": width, "height": height}
            clearance = math.hypot(width, height) / 2.0

        angle = float(visible_rng.uniform(-math.pi, math.pi))
        position = _sample_position(
            visible_rng,
            objects,
            clearance,
            world_size,
        )

        objects.append(
            {
                "object_id": object_id,
                "shape": shape,
                "geometry": geometry,
                "position": position,
                "angle": angle,
                "initial_velocity": [0.0, 0.0],
                "initial_angular_velocity": 0.0,
                "appearance": {
                    "color": [
                        int(visible_rng.integers(40, 220)),
                        int(visible_rng.integers(40, 220)),
                        int(visible_rng.integers(40, 220)),
                    ],
                    "texture": "solid",
                },
                "hidden_physics": {
                    "mass": float(hidden_rng.uniform(0.5, 2.0)),
                    "friction": float(hidden_rng.uniform(0.1, 0.8)),
                    "restitution": float(hidden_rng.uniform(0.0, 0.4)),
                },
            }
        )

    force_angle = float(action_rng.uniform(-math.pi, math.pi))
    force_magnitude = float(action_rng.uniform(0.15, 0.5))
    action = {
        "selected_object_id": int(action_rng.integers(0, object_count)),
        "force_xy": [
            force_magnitude * math.cos(force_angle),
            force_magnitude * math.sin(force_angle),
        ],
        "force_duration": 0.25,
    }

    return {
        "schema_version": "scene-0.1",
        "scene_group_id": f"scene-{int(seed):08d}",
        "condition_id": f"scene-{int(seed):08d}-action-{int(action_seed_value):08d}",
        "world_size": [float(world_size[0]), float(world_size[1])],
        "objects": objects,
        "action": action,
        "seeds": {
            "visible": int(seed),
            "physics": int(physics_seed_value),
            "action": int(action_seed_value),
        },
    }


def visible_scene(scene: dict[str, Any]) -> dict[str, Any]:
    """Return the scene fields visible to the model and renderer.

    Hidden mass, friction, and restitution are intentionally omitted. The
    returned value is suitable for comparing identical visible conditions.
    """

    return {
        "schema_version": scene["schema_version"],
        "world_size": list(scene["world_size"]),
        "objects": [
            {
                "object_id": obj["object_id"],
                "shape": obj["shape"],
                "geometry": dict(obj["geometry"]),
                "position": list(obj["position"]),
                "angle": obj["angle"],
                "initial_velocity": list(obj["initial_velocity"]),
                "initial_angular_velocity": obj["initial_angular_velocity"],
                "appearance": {
                    "color": list(obj["appearance"]["color"]),
                    "texture": obj["appearance"]["texture"],
                },
            }
            for obj in scene["objects"]
        ],
        "action": {
            "selected_object_id": scene["action"]["selected_object_id"],
            "force_xy": list(scene["action"]["force_xy"]),
            "force_duration": scene["action"]["force_duration"],
        },
    }


def _validate_inputs(object_count: int, world_size: tuple[float, float]) -> None:
    if not MIN_OBJECTS <= object_count <= MAX_OBJECTS:
        raise ValueError(f"object_count must be between {MIN_OBJECTS} and {MAX_OBJECTS}")
    if len(world_size) != 2 or any(size <= 0 for size in world_size):
        raise ValueError("world_size must contain two positive values")
    if tuple(float(size) for size in world_size) != (1.0, 1.0):
        raise ValueError("world_size must be (1.0, 1.0) for scene-0.1")


def _sample_position(
    rng: np.random.Generator,
    objects: list[dict[str, Any]],
    clearance: float,
    world_size: tuple[float, float],
) -> list[float]:
    """Sample a non-overlapping position using conservative bounding circles."""

    for _ in range(1_000):
        x = float(rng.uniform(clearance, world_size[0] - clearance))
        y = float(rng.uniform(clearance, world_size[1] - clearance))
        valid = True
        for other in objects:
            other_clearance = _bounding_radius(other)
            minimum_distance = clearance + other_clearance + 0.02
            if math.hypot(x - other["position"][0], y - other["position"][1]) < minimum_distance:
                valid = False
                break
        if valid:
            return [x, y]

    raise RuntimeError("could not place non-overlapping objects; adjust scene ranges")


def _bounding_radius(obj: dict[str, Any]) -> float:
    geometry = obj["geometry"]
    if obj["shape"] == "circle":
        return float(geometry["radius"])
    return math.hypot(float(geometry["width"]), float(geometry["height"])) / 2.0
