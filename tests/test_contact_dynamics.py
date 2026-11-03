"""Mechanics and experiment recovery checks for the local dynamics candidate."""

import copy

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from click2future.models.contact_dynamics import ContactDynamics, motion_tokens
from click2future.models.object_motion import observe
from click2future.models.world_motion import sample_world
from click2future.training.contact_geometry import observed_hulls
from click2future.training.train_contact_dynamics import Trainer, evaluate


def inputs(contact=False):
    image = Image.new("RGB", (64, 64), (210, 214, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 63, 63), outline=(70, 75, 82))
    for x, color in zip(
        (20, 29) if contact else (16, 48), ((180, 50, 60), (50, 90, 180)), strict=True
    ):
        draw.ellipse((x - 4, 28, x + 4, 36), fill=color)
    image = np.asarray(image)
    obs = observe(image, np.all(image == [180, 50, 60], -1))
    return (
        obs,
        torch.from_numpy(obs.features[None]),
        {key: torch.from_numpy(value[None]) for key, value in observed_hulls(obs).items()},
    )


def test_force_only_integration_matches_native_discrete_newton_motion():
    _, features, geometry = inputs()
    model = ContactDynamics(32)
    world = sample_world(19)
    force = torch.tensor([[0.05, 0.0]])
    values = model.rollout(features, force, world, geometry).detach().numpy()
    time = np.arange(1, 32) / 8
    active = np.minimum(time, 0.25)
    expected = 0.05 / float(world[0, 0, 0]) * (active * time - active**2 / 2 - active / 480)
    np.testing.assert_allclose(values[0, :, 0, 0], expected, atol=2e-7)
    np.testing.assert_array_equal(values[0, :, 1], 0)


def test_nonzero_network_has_exact_rest_and_balanced_internal_impulses():
    _, features, geometry = inputs(contact=True)
    model = ContactDynamics(32)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    world = sample_world(19)
    zero = torch.zeros(1, 2)
    torch.testing.assert_close(
        model.rollout(features, zero, world, geometry), torch.zeros(1, 31, 2, 3), rtol=0, atol=0
    )
    state = torch.zeros(1, 2, 3)
    delta = torch.tensor([[[0.002, 0, 0], [0, 0, 0.001]]])
    next_state, change = model.advance(
        state, delta, features, zero, world, geometry, torch.tensor([3])
    )
    impulse = (change[..., :2] - delta[..., :2]) * world[..., :1]
    torch.testing.assert_close(impulse.sum(1), torch.zeros(1, 2), rtol=0, atol=2e-10)
    assert (change - delta).abs().max() > 0
    assert torch.isfinite(next_state).all()


def test_locality_and_seed_pairing_without_reference_labels():
    _, features, geometry = inputs()
    model = ContactDynamics(32)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.1, 0.1)
    zero = torch.zeros(1, 2)
    state = torch.zeros(1, 2, 3)
    delta = torch.tensor([[[0.0001, 0, 0], [0, 0, 0]]])
    _, change = model.advance(
        state, delta, features, zero, sample_world(19), geometry, torch.tensor([3])
    )
    torch.testing.assert_close(change, delta, rtol=0, atol=0)
    force = torch.tensor([[0.02, 0]])
    a = model.sample(features, force, geometry=geometry, seeds=[0, 1, 2, 3])
    b = model.sample(features, force, geometry=geometry, seeds=[0, 1, 2, 3])
    assert a.shape == (4, 31, 2, 4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a[0], a[1])


def test_training_recovers_model_ema_optimizer_and_private_rng(tmp_path):
    _, features, geometry = inputs(contact=True)
    motion = np.zeros((4, 31, 2, 4), np.float32)
    motion[:, :, 0, 0] = np.arange(1, 32) * 0.0001
    row = {
        "features": features.numpy().repeat(4, 0),
        "force": np.array([[0.1, 0]] * 4, np.float32),
        "motion": motion,
        "examples": [{"sample_id": "fixture-collision"}] * 4,
    }
    data = {"train": row, "validation": row}
    geom = {s: {k: v.numpy().repeat(4, 0) for k, v in geometry.items()} for s in data}
    worlds = {s: sample_world(19).numpy().repeat(4, 0) for s in data}
    trainer = Trainer(data, geom, worlds, device="cpu")
    trainer.train_step()
    trainer.save(tmp_path / "state.pt", {"fixture": True})
    expected_loss = trainer.train_step()
    expected = copy.deepcopy(trainer.model.state_dict())
    expected_ema = copy.deepcopy(trainer.ema.state_dict())
    trainer.resume(tmp_path / "state.pt", {"fixture": True})
    assert trainer.train_step() == expected_loss
    for k, v in trainer.model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
        torch.testing.assert_close(trainer.ema.state_dict()[k], expected_ema[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identical source"):
        trainer.resume(tmp_path / "state.pt", {"fixture": False})


def test_validation_driver_scores_real_rgb_and_native_geometry(tmp_path):
    from click2future.data import generate_generalization_dataset

    dataset = tmp_path / "data"
    generate_generalization_dataset(
        dataset,
        group_counts={
            "train": 1,
            "validation": 1,
            "id_test": 1,
            "shape_ood": 1,
            "texture_ood": 1,
            "force_ood": 1,
            "physics_test": 1,
        },
        seed=990000,
    )
    output = tmp_path / "result"
    output.mkdir()
    report = evaluate(ContactDynamics(32), dataset, output)
    assert report["quality"]["expected_selected_clips"] == 24
    assert report["zero_force"]["drifting_samples"] == 0
    assert report["zero_force"]["total_samples"] == 4
    assert report["test_splits_evaluated"] == []
    # An untrained force-only model must expose its physical failures, rather
    # than treating a successful software run as a successful dynamics model.
    assert report["penetrating_clips"] > 0
    assert len(list((output / "validation").glob("*.npz"))) == 6
    assert motion_tokens(torch.zeros(1, 2, 3)).shape == (1, 2, 4)
