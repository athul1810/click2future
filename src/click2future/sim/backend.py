"""Small Pymunk adapter used by the tabletop simulator."""

from __future__ import annotations

from typing import Any

import pymunk

from .geometry import polygon_vertices

OBJECT_COLLISION_TYPE = 1
WALL_COLLISION_TYPE = 2
WALL_ORDER = ("left", "right", "top", "bottom")


def build_space(
    scene: dict[str, Any],
    *,
    physics_hz: int = 240,
    solver_iterations: int = 10,
    gravity: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Build a Pymunk space and return a plain state dictionary."""

    if physics_hz <= 0:
        raise ValueError("physics_hz must be positive")
    if solver_iterations <= 0:
        raise ValueError("solver_iterations must be positive")

    space = pymunk.Space()
    space.gravity = gravity
    space.iterations = solver_iterations

    state: dict[str, Any] = {
        "space": space,
        "physics_hz": physics_hz,
        "time": 0.0,
        "bodies": {},
        "shape_to_object": {},
        "shape_to_wall": {},
        "step_object_contacts": set(),
        "step_wall_contacts": set(),
    }

    _add_walls(state, scene["world_size"])
    for obj in scene["objects"]:
        _add_object(state, obj)

    space.on_collision(
        OBJECT_COLLISION_TYPE,
        OBJECT_COLLISION_TYPE,
        begin=_object_contact_begin,
        data=state,
    )
    space.on_collision(
        OBJECT_COLLISION_TYPE,
        WALL_COLLISION_TYPE,
        begin=_wall_contact_begin,
        data=state,
    )
    return state


def apply_force(state: dict[str, Any], object_id: int, force_xy: tuple[float, float]) -> None:
    """Apply a force at the centre of one object."""

    try:
        body = state["bodies"][object_id]
    except KeyError as exc:
        raise KeyError(f"unknown object_id: {object_id}") from exc
    body.apply_force_at_world_point(force_xy, body.position)


def step(state: dict[str, Any], dt: float | None = None) -> None:
    """Advance the world by one fixed physics step."""

    if dt is None:
        dt = 1.0 / state["physics_hz"]
    if dt <= 0:
        raise ValueError("dt must be positive")

    state["step_object_contacts"].clear()
    state["step_wall_contacts"].clear()
    state["space"].step(dt)
    state["time"] += dt


def snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Return serializable object poses for the current step."""

    objects = []
    for object_id in sorted(state["bodies"]):
        body = state["bodies"][object_id]
        objects.append(
            {
                "object_id": object_id,
                "position": [float(body.position.x), float(body.position.y)],
                "angle": float(body.angle),
            }
        )
    return {"objects": objects}


def consume_step_contacts(state: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    """Return and clear contact onsets observed during the most recent step."""

    events = {
        "object_object": sorted(state["step_object_contacts"]),
        "object_wall": sorted(state["step_wall_contacts"]),
    }
    state["step_object_contacts"].clear()
    state["step_wall_contacts"].clear()
    return events


def _add_walls(state: dict[str, Any], world_size: list[float]) -> None:
    width, height = float(world_size[0]), float(world_size[1])
    static_body = state["space"].static_body
    endpoints = {
        "left": ((0.0, 0.0), (0.0, height)),
        "right": ((width, 0.0), (width, height)),
        "top": ((0.0, 0.0), (width, 0.0)),
        "bottom": ((0.0, height), (width, height)),
    }
    for wall_name in WALL_ORDER:
        shape = pymunk.Segment(static_body, *endpoints[wall_name], radius=0.0)
        shape.collision_type = WALL_COLLISION_TYPE
        state["shape_to_wall"][id(shape)] = wall_name
        state["space"].add(shape)


def _add_object(state: dict[str, Any], obj: dict[str, Any]) -> None:
    object_id = int(obj["object_id"])
    mass = float(obj["hidden_physics"]["mass"])
    shape_name = obj["shape"]
    geometry = obj["geometry"]

    if shape_name == "circle":
        radius = float(geometry["radius"])
        moment = pymunk.moment_for_circle(mass, 0.0, radius)
        body = pymunk.Body(mass, moment)
        shape = pymunk.Circle(body, radius)
    elif shape_name in {"rectangle", "triangle"}:
        vertices = polygon_vertices(shape_name, geometry)
        moment = pymunk.moment_for_poly(mass, vertices)
        body = pymunk.Body(mass, moment)
        shape = pymunk.Poly(body, vertices)
    else:
        raise ValueError(f"unsupported shape: {shape_name}")

    body.position = tuple(obj["position"])
    body.angle = float(obj["angle"])
    body.velocity = tuple(obj["initial_velocity"])
    body.angular_velocity = float(obj["initial_angular_velocity"])

    shape.friction = float(obj["hidden_physics"]["friction"])
    shape.elasticity = float(obj["hidden_physics"]["restitution"])
    shape.collision_type = OBJECT_COLLISION_TYPE

    state["bodies"][object_id] = body
    state["shape_to_object"][id(shape)] = object_id
    state["space"].add(body, shape)


def _object_contact_begin(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, state: dict[str, Any]
) -> bool:
    object_ids = tuple(
        sorted(
            (
                _object_id(arbiter.shapes[0], state),
                _object_id(arbiter.shapes[1], state),
            )
        )
    )
    state["step_object_contacts"].add(object_ids)
    return True


def _wall_contact_begin(
    arbiter: pymunk.Arbiter, _space: pymunk.Space, state: dict[str, Any]
) -> bool:
    object_shape = (
        arbiter.shapes[0]
        if id(arbiter.shapes[0]) in state["shape_to_object"]
        else arbiter.shapes[1]
    )
    wall_shape = (
        arbiter.shapes[0] if id(arbiter.shapes[0]) in state["shape_to_wall"] else arbiter.shapes[1]
    )
    object_id = _object_id(object_shape, state)
    wall_contact = (object_id, state["shape_to_wall"][id(wall_shape)])
    state["step_wall_contacts"].add(wall_contact)
    return True


def _object_id(shape: pymunk.Shape, state: dict[str, Any]) -> int:
    return state["shape_to_object"][id(shape)]
