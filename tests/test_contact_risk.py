"""Worst-case gradients, neutral feasible labels and explicit EMA warm starts."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_contact_training import fixture_data

from click2future.training.contact_geometry import contact_objective
from click2future.training.contact_risk import penetration_risk, upper_tail
from click2future.training.train_contact import ContactConfig, ContactTrainer, initialize_from_ema
from click2future.training.train_cvae import sha256


def test_upper_tail_has_explicit_rounded_mass_and_permutation_invariance():
    values = torch.arange(31, dtype=torch.float32)
    assert upper_tail(values, 0.25) == values[-8:].mean()
    assert upper_tail(values.flip(0), 0.25) == upper_tail(values, 0.25)
    assert upper_tail(values, 1.0) == values.mean()
    with pytest.raises(ValueError):
        upper_tail(values, 0)


def test_feasible_labels_and_high_noise_get_no_risk_or_gradient():
    depth = torch.full((4, 31), 0.9 / 64, requires_grad=True)
    risk = penetration_risk(depth, torch.ones(4, dtype=torch.bool))
    assert risk == 0
    risk.backward()
    assert torch.count_nonzero(depth.grad) == 0
    deep = torch.full((4, 31), 4.0 / 64, requires_grad=True)
    risk = penetration_risk(deep, torch.zeros(4, dtype=torch.bool))
    risk.backward()
    assert risk == 0 and torch.count_nonzero(deep.grad) == 0


def test_rare_deep_overlap_gets_more_gradient_than_averaged_geometry():
    depth = torch.zeros((4, 32), requires_grad=True)
    with torch.no_grad():
        depth[2, 19] = 3 / 64
    risk = penetration_risk(depth, torch.ones(4, dtype=torch.bool))
    mean = (depth - 1 / 64).relu().square().mean()
    assert float(risk.detach()) == pytest.approx(float(mean.detach()) * 16)
    gradient = torch.autograd.grad(risk, depth)[0]
    average_gradient = torch.autograd.grad(mean, depth)[0]
    assert gradient[2, 19] == 16 * average_gradient[2, 19]
    assert torch.count_nonzero(gradient) == 1


def test_disabled_risk_preserves_objective_and_enabled_risk_is_finite():
    data, geometry, _ = fixture_data()
    config = ContactConfig(steps=4, width=16, layers=1, batch_size=3)
    trainer = ContactTrainer(data, geometry, config)
    row = trainer.data["train"]
    clean, features, force = (row[key][:3] for key in ("motion", "features", "force"))
    args = (
        trainer.model,
        clean,
        features,
        force,
        torch.full((3,), 0.1),
        torch.zeros_like(clean),
        {key: value[:3] for key, value in trainer.geometry["train"].items()},
    )
    original, original_parts = contact_objective(*args, config)
    disabled, parts = contact_objective(*args, replace(config, risk_tail_fraction=1.0))
    assert torch.equal(original, disabled) and original_parts.keys() == parts.keys()
    enabled, parts = contact_objective(*args, replace(config, risk_penetration_weight=256))
    assert enabled >= original and torch.isfinite(parts["tail_penetration"])
    for bad in (0, -0.1, 1.1, True):
        with pytest.raises(ValueError):
            replace(config, risk_tail_fraction=bad).validate()


def test_warm_start_binds_data_and_copies_ema_with_fresh_optimizer_rng(tmp_path):
    data, geometry, _ = fixture_data()
    config = ContactConfig(steps=4, width=16, layers=1, batch_size=3)
    old = ContactTrainer(data, geometry, config)
    old.train_step()
    provenance = {
        key: "unit-data-id"
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
        )
    }
    provenance["source_sha256"] = {
        "models/object_motion.py": sha256(Path("src/click2future/models/object_motion.py"))
    }
    path = tmp_path / "source.pt"
    old.save(path, provenance)
    new = ContactTrainer(data, geometry, config)
    rng = new.generator.get_state().clone()
    identity = initialize_from_ema(new, path, provenance)
    assert identity["checkpoint_sha256"] == sha256(path)
    assert new.step == 0 and not new.optimizer.state
    assert torch.equal(rng, new.generator.get_state())
    for key, value in old.ema.state_dict().items():
        assert torch.equal(value, new.model.state_dict()[key])
        assert torch.equal(value, new.ema.state_dict()[key])
    wrong = {**provenance, "geometry_sha256": "changed"}
    with pytest.raises(ValueError, match="training data differ"):
        initialize_from_ema(new, path, wrong)
    new.train_step()
    with pytest.raises(ValueError, match="fresh trainer"):
        initialize_from_ema(new, path, provenance)
