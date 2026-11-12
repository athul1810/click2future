import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from click2future.models.momentum_block import apply_momentum_block
from click2future.models.object_motion import observe
from click2future.sim.rollout import run_rollout


def observation(centers):
    image = Image.new("RGB", (64, 64), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(centers, [(170, 60, 50), (50, 100, 190)], strict=True):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    first = np.asarray(image)
    return observe(first, np.all(first == [170, 60, 50], axis=-1))


def test_enforces_post_force_momentum_without_moving_the_passive_object():
    obs = observation([(16, 24), (44, 40)])
    motion = np.zeros((6, 2, 4), dtype=np.float32)
    motion[:, 0, 0] = [0.01, 0.03, 0.07, 0.12, 0.18, 0.25]
    motion[:, 1, 0] = [0.00, 0.01, 0.01, 0.00, -0.02, -0.05]
    motion[..., 2] = 0.2
    masses = np.array([1.0, 3.0])
    force = np.array([0.2, 0.0])

    corrected = apply_momentum_block(obs, motion, masses, force)
    raw_positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * motion[..., :2]))
    positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * corrected[..., :2]))
    raw_velocity = np.diff(raw_positions, axis=0) * 8
    velocity = np.diff(positions, axis=0) * 8

    np.testing.assert_array_equal(corrected[:2], motion[:2])
    np.testing.assert_allclose(
        (masses[:, None] * velocity[2:]).sum(1),
        np.broadcast_to(force * 0.25, (4, 2)),
        atol=1e-7,
    )
    passive = 1 - obs.selected_index
    np.testing.assert_allclose(velocity[2:, passive], raw_velocity[2:, passive], atol=1e-7)
    np.testing.assert_array_equal(corrected[..., 2:], motion[..., 2:])


def test_wall_trajectory_is_unchanged():
    obs = observation([(8, 32), (48, 32)])
    motion = np.zeros((4, 2, 4), dtype=np.float32)
    motion[:, 0, 0] = -0.12
    assert np.array_equal(apply_momentum_block(obs, motion, [1, 1], [0.2, 0]), motion)


def test_contact_correction_preserves_relative_collision_motion():
    obs = observation([(20, 32), (32, 32)])
    motion = np.zeros((5, 2, 4), dtype=np.float32)
    motion[:, 0, 0] = [0.02, 0.08, 0.15, 0.20, 0.24]
    motion[:, 1, 0] = [0.00, 0.00, 0.01, 0.04, 0.08]

    corrected = apply_momentum_block(obs, motion, [1, 2], [0.2, 0])
    raw_positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * motion[..., :2]))
    positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * corrected[..., :2]))
    raw_velocity = np.diff(raw_positions, axis=0) * 8
    velocity = np.diff(positions, axis=0) * 8
    np.testing.assert_allclose(
        velocity[2:, 1] - velocity[2:, 0],
        raw_velocity[2:, 1] - raw_velocity[2:, 0],
        atol=1e-7,
    )


def test_simulator_wall_free_motion_already_satisfies_the_block():
    scene = json.loads((Path(__file__).parent / "fixtures" / "tiny_scene.json").read_text())
    rollout = run_rollout(scene, frame_count=32)
    obs = observe(rollout["frames"][0], rollout["selected_mask"])
    motion = np.zeros((31, 2, 4), dtype=np.float32)
    motion[..., :2] = (rollout["positions"][1:] - rollout["positions"][:1]) / 0.5

    corrected = apply_momentum_block(
        obs,
        motion,
        [obj["hidden_physics"]["mass"] for obj in scene["objects"]],
        scene["action"]["force_xy"],
    )
    np.testing.assert_allclose(corrected, motion, atol=1e-7, rtol=0)


def test_rejects_invalid_physical_inputs():
    obs = observation([(16, 16), (48, 48)])
    motion = np.zeros((4, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_momentum_block(obs, motion, [0, 1], [0.2, 0])
    with pytest.raises(ValueError):
        apply_momentum_block(obs, motion, [1, 1], [np.nan, 0])
