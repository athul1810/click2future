"""Observable-geometry, sampled-prior rigid-body baseline inspired by PhysGen.

This explicitly uses a physics solver at inference. It is not a learned video
model, a reproduction of the complete PhysGen pipeline, or simulator truth.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pymunk

from click2future.evaluation.rigid_geometry import _hull, _penetration

from .object_motion import encode_motion, observe, render_motion
from .world_motion import WORLD_BOUNDS, WORLD_CONTRACT, sample_world

PHYSICS_BASELINE = "observed-hull-sampled-world-pymunk-1"


def rollout_observation(
    observation, force, world, *, future_frames=31, physics_hz=240, circle_radii=None
):
    """Integrate inferred convex sprites; no scene recipe or hidden labels enter."""
    force, world = np.asarray(force, dtype=np.float64), np.asarray(world, dtype=np.float64)
    bounds = np.asarray(WORLD_BOUNDS)
    if (
        force.shape != (2,)
        or world.shape != (2, 3)
        or not np.isfinite(force).all()
        or not np.isfinite(world).all()
        or ((world < bounds[:, 0]) | (world > bounds[:, 1])).any()
        or not 1 <= future_frames <= 31
        or physics_hz not in (240, 480)
    ):
        raise ValueError("Use finite force, in-prior worlds and the native bounded clip")
    resolution = observation.first.shape[0]
    if resolution != 64:
        raise ValueError("The physics-assisted baseline uses native 64px observations")
    radii = np.zeros(2) if circle_radii is None else np.asarray(circle_radii)
    if radii.shape != (2,) or not np.isfinite(radii).all() or ((radii < 0) | (radii > 0.5)).any():
        raise ValueError("Provide two finite observed circle radii, or zero for hulls")
    hulls = []
    for mask in observation.masks:
        y, x = np.nonzero(mask)
        hulls.append(_hull(np.stack((x + 0.5, y + 0.5), -1)) / resolution)
    if _penetration(*hulls) > 0:
        raise ValueError("Inferred convex object hulls must be initially separated")
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.iterations = 10
    # Solver tolerance is in world units. Keep it below one tenth of a
    # native pixel, instead of using the engine's 0.1-world-unit default.
    space.collision_slop = 0.1 / resolution
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    for a, b in zip(corners, (*corners[1:], corners[0]), strict=True):
        space.add(pymunk.Segment(space.static_body, a, b, 0))
    bodies = []
    for index, (mass, friction, restitution) in enumerate(world):
        vertices = (hulls[index] - observation.centers[index]).tolist()
        # Pixel masks describe occupied cells. A half-pixel bevel approximates
        # the cell extent around hulls of pixel centres, including at walls.
        radius = 0.5 / resolution
        moment = (
            pymunk.moment_for_circle(float(mass), 0, float(radii[index]))
            if radii[index] > 0
            else pymunk.moment_for_poly(float(mass), vertices, radius=radius)
        )
        body = pymunk.Body(float(mass), moment)
        body.position = tuple(observation.centers[index].astype(float))
        shape = (
            pymunk.Circle(body, float(radii[index]))
            if radii[index] > 0
            else pymunk.Poly(body, vertices, radius=radius)
        )
        shape.friction, shape.elasticity = float(friction), float(restitution)
        space.add(body, shape)
        bodies.append(body)
    positions, angles, velocities, kinetic = [], [], [], []
    frame_steps, force_steps = physics_hz // 8, physics_hz // 4
    for frame in range(future_frames + 1):
        if frame:
            for step in range((frame - 1) * frame_steps, frame * frame_steps):
                if step < force_steps:
                    body = bodies[observation.selected_index]
                    body.apply_force_at_world_point(tuple(force), body.position)
                space.step(1 / physics_hz)
        positions.append([tuple(body.position) for body in bodies])
        angles.append([body.angle for body in bodies])
        velocities.append([tuple(body.velocity) for body in bodies])
        kinetic.append(
            sum(
                0.5 * body.mass * body.velocity.length_squared
                + 0.5 * body.moment * body.angular_velocity**2
                for body in bodies
            )
        )
    positions, angles = np.asarray(positions), np.asarray(angles)
    return {
        "motion": encode_motion(positions, angles),
        "positions": positions,
        "angles": angles,
        "velocities": np.asarray(velocities),
        "kinetic_energy": np.asarray(kinetic),
        "world": world.copy(),
    }


class PhysicsBaseline:
    force_limit = 1.1

    def __init__(self, name="Physics-assisted"):
        folder = Path(__file__).parents[1]
        sources = {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in (
                "models/physics_baseline.py",
                "models/object_motion.py",
                "models/world_motion.py",
                "evaluation/rigid_geometry.py",
            )
        }
        source = hashlib.sha256(
            json.dumps({"sources": sources, "pymunk": pymunk.version}, sort_keys=True).encode()
        ).hexdigest()
        self.metadata = {
            "id": name,
            "kind": "physics_assisted_baseline",
            "detail": "inferred convex sprites, sampled physical prior, explicit rigid-body solver",
            "checkpoint_sha256": "source:" + source,
            "sampling_source_sha256": sources,
            "generation_source": "live physics-assisted baseline",
            "physics_contract": PHYSICS_BASELINE,
            "world_contract": WORLD_CONTRACT,
            "world_condition": True,
            "quality": (
                "physics-assisted baseline; appearance geometry and hidden-world inference "
                "remain approximate"
            ),
        }

    def predict(self, first, mask, force, *, seeds, future_frames, cancelled):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Use distinct sampling seeds")
        force = np.asarray(force)
        if not np.isfinite(force).all() or np.linalg.norm(force) > self.force_limit + 1e-6:
            raise ValueError("Force exceeds the baseline's declared range")
        observation = observe(first, mask)
        result = []
        for seed in seeds:
            if cancelled.is_set():
                raise InterruptedError("Physics-assisted generation cancelled")
            state = rollout_observation(
                observation, force, sample_world(seed)[0].numpy(), future_frames=future_frames
            )
            result.append(render_motion(observation, state["motion"]))
        return np.stack(result)
