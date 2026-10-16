"""Guard the frozen contact-only physics against accidental reinterpretation."""

import numpy as np
import pytest

from click2future.sim.backend import apply_force, build_space, consume_step_contacts, step
from click2future.sim.scene import make_scene


def _free_scene(friction):
    scene = make_scene(7)
    for index, obj in enumerate(scene["objects"]):
        obj.update(shape="circle", geometry={"radius": 0.04}, angle=0.0)
        obj["position"] = [0.25, 0.25] if index == 0 else [0.75, 0.75]
        obj["hidden_physics"].update(mass=1.0, friction=friction, restitution=0.3)
    return scene


def test_contact_friction_does_not_slow_contact_free_motion():
    trajectories = []
    for friction in (0.1, 0.8):
        state = build_space(_free_scene(friction))
        assert state["space"].damping == 1.0
        positions = []
        velocities = []
        for tick in range(480):
            if tick < 60:
                apply_force(state, 0, (0.2, 0.0))
            step(state)
            contacts = consume_step_contacts(state)
            assert not contacts["object_object"] and not contacts["object_wall"]
            positions.append(tuple(state["bodies"][0].position))
            velocities.append(tuple(state["bodies"][0].velocity))
        # F * duration / m = 0.05; after force-off there is no surface drag.
        np.testing.assert_allclose(velocities[59:], [(0.05, 0.0)] * 421, atol=1e-12)
        trajectories.append(positions)
    np.testing.assert_array_equal(*trajectories)


def test_frozen_walls_are_frictionless_and_inelastic():
    state = build_space(_free_scene(0.8))
    walls = [s for s in state["space"].shapes if id(s) in state["shape_to_wall"]]
    assert len(walls) == 4
    for wall in walls:
        assert wall.friction == pytest.approx(0.0)
        assert wall.elasticity == pytest.approx(0.0)
