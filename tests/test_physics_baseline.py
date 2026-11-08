"""Analytical mechanics and observable-only sampling for the physics baseline."""

import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw

from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.models.object_motion import observe
from click2future.models.physics_baseline import PhysicsBaseline, rollout_observation


def observation(contact=False):
    image = Image.new("RGB", (64, 64), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 63, 63), outline=(40, 45, 50))
    centers = [(16, 32), (32, 32)] if contact else [(16, 16), (48, 48)]
    for (x, y), color in zip(centers, [(170, 60, 50), (50, 100, 190)], strict=True):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    first = np.asarray(image)
    return observe(first, np.all(first == [170, 60, 50], axis=-1))


@pytest.mark.parametrize("mass", [0.5, 1.0, 2.0])
def test_free_motion_matches_newton_and_force_impulse_exactly(mass):
    obs = observation()
    world = np.array([[mass, 0.2, 0.1], [1, 0.2, 0.1]])
    force = np.array([0.15, 0.0])
    values = rollout_observation(obs, force, world)
    time, duration, step = np.arange(32) / 8, 0.25, 1 / 240
    active = np.minimum(time, duration)
    expected = force[0] / mass * (active * time - active**2 / 2 - active * step / 2)
    np.testing.assert_allclose(
        values["positions"][:, 0, 0] - obs.centers[0, 0], expected, atol=1e-12
    )
    np.testing.assert_allclose(values["velocities"][-1, 0], force * duration / mass, atol=1e-12)
    np.testing.assert_array_equal(
        values["positions"][:, 1], np.repeat(obs.centers[None, 1], 32, axis=0)
    )
    # Smaller integration steps approach the continuous solution.
    finer = rollout_observation(obs, force, world, physics_hz=480)
    exact = force[0] / mass * (duration * time[-1] - duration**2 / 2)
    coarse_error = abs(values["positions"][-1, 0, 0] - obs.centers[0, 0] - exact)
    fine_error = abs(finer["positions"][-1, 0, 0] - obs.centers[0, 0] - exact)
    assert fine_error == pytest.approx(coarse_error / 2, abs=1e-12)


def test_contact_conserves_linear_momentum_and_dissipates_unforced_energy():
    obs = observation(contact=True)
    world = np.array([[1, 0.2, 0.1], [1, 0.2, 0.1]])
    values = rollout_observation(obs, [0.5, 0], world)
    momentum = (world[:, 0, None] * values["velocities"][-1]).sum(0)
    np.testing.assert_allclose(momentum, [0.5 * 0.25, 0], atol=1e-10)
    assert values["velocities"][-1, 1, 0] > 0
    assert np.diff(values["kinetic_energy"][2:]).max() <= 1e-12
    geometry = geometry_diagnostics(obs, values["motion"][None])
    assert geometry["clips_with_penetration_over_one_pixel"] == 0


def test_zero_action_is_static_and_all_four_worlds_remain_paired():
    obs = observation()
    baseline = PhysicsBaseline()
    args = dict(seeds=[0, 1, 2, 3], future_frames=31, cancelled=threading.Event())
    videos = baseline.predict(obs.first, obs.masks[0], np.zeros(2), **args)
    assert videos.shape == (4, 32, 64, 64, 3)
    assert np.array_equal(videos, np.broadcast_to(obs.first, videos.shape))
    force = np.array([0.15, 0])
    a = baseline.predict(obs.first, obs.masks[0], force, **args)
    b = baseline.predict(obs.first, obs.masks[0], force, **args)
    assert np.array_equal(a, b) and np.all(a[:, 0] == obs.first)
    assert len({clip.tobytes() for clip in a}) > 1
    assert baseline.metadata["kind"] == "physics_assisted_baseline"
    args["cancelled"].set()
    with pytest.raises(InterruptedError):
        baseline.predict(obs.first, obs.masks[0], force, **args)


def test_demo_service_preserves_explicit_solver_source_label(tmp_path):
    import time

    from click2future.data import generate_dataset
    from click2future.demo.service import DemoService

    dataset = tmp_path / "data"
    generated = generate_dataset(
        dataset,
        group_count=1,
        variants_per_group=1,
        seed=850000,
        config={"frame_count": 4, "resolution": 64, "fps": 8},
    )
    sample = generated["manifest"]["samples"][0]["sample_id"]
    service = DemoService(
        dataset, [sample], {"Physics-assisted": PhysicsBaseline()}, cache=tmp_path / "cache"
    )
    try:
        identifier = service.submit(
            {
                "scene": sample,
                "object": 0,
                "force": [0.15, 0],
                "seeds": [0, 1, 2, 3],
                "models": ["Physics-assisted"],
            }
        )
        deadline = time.monotonic() + 10
        while (
            service.job(identifier)["status"] in {"queued", "running"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        job = service.job(identifier)
        assert job["status"] == "complete"
        assert job["result"]["models"][0]["source"] == "live physics-assisted baseline"
        assert job["result"]["models"][0]["details"]["kind"] == "physics_assisted_baseline"
    finally:
        service.close()


def test_fresh_screen_rejects_an_unregistered_corpus_before_sampling(tmp_path):
    import json

    from click2future.evaluation.physics_screen import screen

    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "dataset_manifest.json").write_text(json.dumps({"seed": 1, "group_counts": {}}))
    with pytest.raises(ValueError, match="registered fresh corpus"):
        screen(dataset, tmp_path / "result", fresh_id=True, training_data=tmp_path / "original")
    assert not (tmp_path / "result").exists()


def test_default_bevel_is_unchanged_and_calibrated_bevel_is_bounded():
    from click2future.models.physics_baseline import CALIBRATED_BEVEL_PIXELS

    obs = observation(contact=True)
    world = np.array([[1, 0.2, 0.1], [1, 0.2, 0.1]])
    default = rollout_observation(obs, [0.5, 0], world)["positions"]
    explicit = rollout_observation(obs, [0.5, 0], world, bevel_pixels=0.5)["positions"]
    assert np.array_equal(default, explicit)
    calibrated = rollout_observation(obs, [0.5, 0], world, bevel_pixels=CALIBRATED_BEVEL_PIXELS)
    assert np.isfinite(calibrated["positions"]).all()
    for bad in (-0.1, 0.6):
        with pytest.raises(ValueError, match="bevel"):
            rollout_observation(obs, [0.5, 0], world, bevel_pixels=bad)
