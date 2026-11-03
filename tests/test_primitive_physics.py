"""Observable primitive recognition, mechanics and exact image-rest contract."""

import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw

from click2future.models.object_motion import observe
from click2future.models.primitive_physics import (
    PRIMITIVE_PHYSICS,
    PrimitivePhysicsBaseline,
    fit_circles,
    primitive_prediction,
)


def observation(rectangle=False):
    image = Image.new("RGB", (64, 64), (210, 214, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 63, 63), outline=(70, 75, 82))
    draw.ellipse((12, 28, 20, 36), fill=(170, 60, 50))
    if rectangle:
        draw.rectangle((36, 28, 44, 36), fill=(50, 100, 190))
    else:
        draw.ellipse((28, 28, 36, 36), fill=(50, 100, 190))
    image = np.asarray(image)
    return observe(image, np.all(image == [170, 60, 50], -1))


def test_exact_mask_recognition_has_hull_fallback_for_rectangles():
    obs = observation(rectangle=True)
    fits = fit_circles(obs)
    assert [fit["circle"] for fit in fits] == [True, False]
    assert fits[0]["radius"] == 4 / 64
    assert fits[1]["radius"] == 0
    world = np.array([[1, 0.2, 0.1], [1, 0.2, 0.1]])
    result = primitive_prediction(obs, [0, 0], world)
    np.testing.assert_array_equal(result["videos"], np.broadcast_to(obs.first, (32, 64, 64, 3)))


def test_circle_contact_preserves_impulse_and_dissipates_energy():
    obs = observation()
    world = np.array([[1, 0.2, 0.1], [1, 0.2, 0.1]])
    result = primitive_prediction(obs, [0.5, 0], world)
    np.testing.assert_allclose(result["velocities"][-1].sum(0), [0.125, 0], atol=1e-10)
    assert result["velocities"][-1, 1, 0] > 0
    assert np.diff(result["kinetic_energy"][2:]).max() <= 1e-12
    np.testing.assert_array_equal(result["motion"][..., 2:], 0)


def test_primitive_service_adapter_keeps_all_seeds_exact_rest_and_source_kind():
    obs = observation()
    model = PrimitivePhysicsBaseline()
    args = dict(seeds=[0, 1, 2, 3], future_frames=31, cancelled=threading.Event())
    video = model.predict(obs.first, obs.masks[0], [0, 0], **args)
    np.testing.assert_array_equal(video, np.broadcast_to(obs.first, (4, 32, 64, 64, 3)))
    assert model.metadata["physics_contract"] == PRIMITIVE_PHYSICS
    assert model.metadata["kind"] == "physics_assisted_baseline"
    assert "models/primitive_physics.py" in model.metadata["sampling_source_sha256"]
    with pytest.raises(ValueError):
        model.predict(obs.first, obs.masks[0], [2, 0], **args)
    args["cancelled"].set()
    with pytest.raises(InterruptedError):
        model.predict(obs.first, obs.masks[0], [0.1, 0], **args)
