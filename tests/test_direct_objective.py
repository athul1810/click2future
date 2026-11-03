"""Expected loss keeps the original objective and removes independent noise weights."""

import copy
from dataclasses import replace

import pytest
import torch
from test_contact_training import fixture_data

from click2future.demo.models import CheckpointModel
from click2future.models.world_motion import sample_world
from click2future.training.direct_objective import expected_coefficients, expected_direct_objective
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.train_expected_prior import (
    ExpectedPriorConfig,
    ExpectedPriorTrainer,
    sources,
)


def fixture(expected=True, anchored=False):
    data, geometry, obs = fixture_data()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    config = ExpectedPriorConfig(
        steps=4,
        width=16,
        layers=1,
        batch_size=3,
        expected_objective=expected,
        rest_anchored=anchored,
        physics_sigma_max=1.0,
    )
    return ExpectedPriorTrainer(data, geometry, worlds, config), obs


def test_analytic_coefficients_include_clipping_and_match_independent_quadrature():
    config = ExpectedPriorConfig()
    z = torch.linspace(-10, 10, 200001, dtype=torch.float64)
    density = torch.exp(-z.square() / 2) / (2 * torch.pi) ** 0.5
    sigma = (
        (z * config.log_sigma_std + config.log_sigma_mean)
        .exp()
        .clamp(config.sigma_min, config.sigma_max)
    )
    numerical = torch.trapezoid((config.sigma_data**-2 + sigma.reciprocal().square()) * density, z)
    coefficient = expected_coefficients(config)
    assert coefficient["edm_weight"] == pytest.approx(float(numerical), rel=1e-8)
    assert config.sigma_min < coefficient["equivalent_sigma"] < config.sigma_max
    assert 0 < coefficient["physics_probability"] < 1
    assert (
        expected_coefficients(replace(config, physics_sigma_max=config.sigma_max))[
            "physics_probability"
        ]
        == 1
    )
    with pytest.raises(ValueError, match="[Nn]onlinear tail risk"):
        expected_coefficients(replace(config, risk_penetration_weight=1))


def test_expected_loss_matches_weighted_deterministic_terms_and_has_real_gradients():
    from click2future.models.world_motion import BoundWorld
    from click2future.training.contact_geometry import contact_objective

    trainer, _ = fixture()
    with torch.no_grad():
        trainer.model.output.weight.normal_(
            std=0.02, generator=torch.Generator().manual_seed(31019)
        )
    row = trainer.data["train"]
    clean, features, force = (row[key] for key in ("motion", "features", "force"))
    bound = BoundWorld(trainer.model, trainer.worlds["train"])
    geometry = trainer.geometry["train"]
    coefficient = expected_coefficients(trainer.config)
    loss, _ = expected_direct_objective(bound, clean, features, force, geometry, trainer.config)
    # Separate evaluations derive the two deterministic terms independently.
    base = replace(trainer.config, penetration_weight=0, contact_weight=0, boundary_weight=0)
    unit_sigma = clean.new_ones(len(clean))
    original, _ = contact_objective(
        bound, clean, features, force, unit_sigma, torch.zeros_like(clean), geometry, trainer.config
    )
    regression, _ = contact_objective(
        bound, clean, features, force, unit_sigma, torch.zeros_like(clean), geometry, base
    )
    unit_weight = trainer.config.sigma_data**-2 + 1
    expected = regression * coefficient["edm_weight"] / unit_weight
    expected += (original - regression) * coefficient["physics_probability"]
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert trainer.model.condition[0].weight.grad.abs().sum() > 0
    assert trainer.model.motion.weight.grad.count_nonzero() == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.model.parameters())


@pytest.mark.parametrize("expected,anchored", [(True, False), (False, False), (True, True)])
def test_exact_restart_and_public_sampler_preserve_new_objective_identity(
    tmp_path, expected, anchored
):
    trainer, obs = fixture(expected, anchored)
    trainer.train_step()
    checkpoint = tmp_path / "checkpoint.pt"
    provenance = {"source_sha256": sources()}
    trainer.save(checkpoint, provenance)
    restored, _ = fixture(expected, anchored)
    restored.resume(checkpoint, provenance)
    assert trainer.train_step() == restored.train_step()
    for kind in ("model", "ema"):
        assert all(
            torch.equal(value, getattr(restored, kind).state_dict()[name])
            for name, value in getattr(trainer, kind).state_dict().items()
        )
    assert torch.equal(trainer.generator.get_state(), restored.generator.get_state())
    model, state = load_trajectory(checkpoint)
    assert state["config"]["expected_objective"] is expected
    features = torch.from_numpy(obs.features[None])
    force = torch.tensor([[0.15, 0]])
    rng = torch.get_rng_state().clone()
    result = model.sample(features, force, seeds=[0, 1, 2, 3], steps=1)
    assert torch.equal(result[0], model.decode_world(features, force, sample_world(0))[0])
    assert torch.equal(rng, torch.get_rng_state())
    adapter = CheckpointModel("candidate", checkpoint, trajectory_steps=1)
    assert adapter.metadata["expected_objective"] is expected
    assert adapter.metadata["study_id"] == ("D020" if anchored else "D019")
    assert adapter.metadata["network_evaluations_per_sample"] == (2 if anchored else 1)
    wrong, _ = fixture(not expected)
    with pytest.raises(ValueError, match="identical objective"):
        wrong.resume(checkpoint, provenance)
    corrupted = copy.deepcopy(torch.load(checkpoint, weights_only=True))
    corrupted["expected_coefficients"]["edm_weight"] += 1
    torch.save(corrupted, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="Unsupported"):
        load_trajectory(tmp_path / "bad.pt")


def test_rest_anchor_is_exact_for_every_world_without_suppressing_action_gradients():
    trainer, obs = fixture(True, True)
    with torch.no_grad():
        trainer.model.output.weight.normal_(std=0.1, generator=torch.Generator().manual_seed(23020))
    features = torch.from_numpy(obs.features[None])
    zeros, sigma = torch.zeros(1, 31, 2, 4), torch.ones(1)
    for seed in (0, 1, 2, 3):
        world = sample_world(seed)
        rest = trainer.model(zeros, sigma, features, torch.zeros(1, 2), world)
        assert torch.equal(rest, torch.zeros_like(rest))
        response = trainer.model(zeros, sigma, features, torch.tensor([[0.25, 0.0]]), world)
        assert response.abs().sum() > 0
        response.square().mean().backward()
    assert trainer.model.condition[0].weight.grad.abs().sum() > 0


def test_paired_arms_start_identically_and_keep_exact_batch_rng_draws():
    original, _ = fixture(False)
    expected, _ = fixture(True)
    assert all(
        torch.equal(value, expected.model.state_dict()[name])
        for name, value in original.model.state_dict().items()
    )
    for _ in range(3):
        original.train_step()
        expected.train_step()
        assert torch.equal(original.generator.get_state(), expected.generator.get_state())
