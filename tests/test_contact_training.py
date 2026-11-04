"""SAT gradients/calibration, balanced training and exact continuation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.models.object_motion import observe
from click2future.sim import run_rollout
from click2future.training.contact_geometry import (
    contact_objective,
    observed_hulls,
    signed_penetration,
)
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_contact import ContactConfig, ContactTrainer, main, resume_probe
from click2future.training.train_cvae import sha256


def observation():
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    ref = run_rollout(scene, frame_count=32, resolution=64, fps=8)
    return observe(ref["frames"][0], ref["selected_mask"])


def tensor_geometry(obs, count=1):
    return {
        key: torch.from_numpy(value)[None].repeat(count, 1, 1, *([1] if value.ndim == 3 else []))
        for key, value in observed_hulls(obs).items()
    }


def test_differentiable_sat_matches_independent_audit_and_has_finite_gradients():
    obs = observation()
    generator = np.random.default_rng(47211)
    motion = generator.normal(0, 0.08, (3, 7, 2, 4)).astype(np.float32)
    motion[..., 2] *= 2
    motion[0, 0, 1, :2] = 2 * (obs.centers[0] - obs.centers[1]) + [0.03, 0.01]
    direct = geometry_diagnostics(obs, motion)
    tensor = torch.tensor(motion, requires_grad=True)
    features = torch.from_numpy(obs.features)[None].repeat(3, 1, 1)
    signed = signed_penetration(tensor, features, tensor_geometry(obs, 3))
    expected = [row["hull_penetration_pixels_per_frame"] for row in direct["clips"]]
    np.testing.assert_allclose(signed.detach().clamp_min(0).numpy() * 64, expected, atol=5e-6)
    signed.relu().square().mean().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    # Rotation cannot collapse hulls when the sine/cosine pair degenerates.
    degenerate = torch.zeros((1, 3, 2, 4), requires_grad=True)
    with torch.no_grad():
        degenerate[..., 3] = -0.5
    value = signed_penetration(degenerate, features[:1], tensor_geometry(obs))
    identity = signed_penetration(torch.zeros_like(degenerate), features[:1], tensor_geometry(obs))
    torch.testing.assert_close(value, identity)
    value.sum().backward()
    assert torch.isfinite(degenerate.grad).all()


def test_sat_is_symmetric_under_identity_swap_and_world_translation():
    obs = observation()
    geometry = tensor_geometry(obs)
    features = torch.from_numpy(obs.features)[None]
    motion = torch.randn((1, 4, 2, 4), generator=torch.Generator().manual_seed(11)) * 0.02
    expected = signed_penetration(motion, features, geometry)
    swapped = signed_penetration(
        motion.flip(2), features.flip(1), {key: value.flip(1) for key, value in geometry.items()}
    )
    torch.testing.assert_close(expected, swapped)
    translated = features.clone()
    translated[..., :2] += torch.tensor([0.1, -0.2])
    torch.testing.assert_close(
        expected, signed_penetration(motion, translated, geometry), atol=1e-7, rtol=1e-6
    )


def fixture_data():
    obs = observation()
    features = np.repeat(obs.features[None], 6, 0)
    force = np.array([[0, 0], [0, 0], [0.25, 0], [0.25, 0], [-0.25, 0], [0, 0.25]], np.float32)
    motion = np.zeros((6, 31, 2, 4), np.float32)
    motion[2:, ..., :2] = 0.01
    examples = [
        {"sample_id": name}
        for name in (
            "unit-zero-0",
            "unit-zero-1",
            "unit-collision_0",
            "unit-collision_1",
            "unit-forward",
            "unit-reverse",
        )
    ]
    row = {"features": features, "force": force, "motion": motion, "examples": examples}
    data = {split: copy.deepcopy(row) for split in ("train", "validation")}
    geometry = {
        split: {key: value.numpy() for key, value in tensor_geometry(obs, 6).items()}
        for split in data
    }
    return data, geometry, obs


def test_balanced_mass_resume_and_real_sampler_contract(tmp_path):
    data, geometry, obs = fixture_data()
    config = ContactConfig(steps=4, width=16, layers=1, batch_size=3)
    trainer = ContactTrainer(data, geometry, config)
    assert float(trainer.weights[:2].sum()) == pytest.approx(config.zero_probability)
    assert float(trainer.weights[2:4].sum()) == pytest.approx(config.collision_probability)
    record = trainer.train_step()
    assert np.isfinite(list(record.values())).all()
    assert resume_probe(data, geometry, config, tmp_path / "resume.pt", device="cpu")[
        "losses_equal"
    ]
    checkpoint = tmp_path / "fixture.pt"
    trainer.save(
        checkpoint,
        {
            "source_sha256": {
                "models/object_motion.py": sha256(Path("src/click2future/models/object_motion.py"))
            }
        },
    )
    model, state = load_trajectory(checkpoint)
    assert state["trajectory_format_version"] == 2
    original = torch.get_rng_state().clone()
    generated = sample_trajectory_videos(
        model,
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.25, 0], np.float32),
        seeds=[3],
        sampling_steps=3,
    )
    assert generated.shape == (1, 32, 64, 64, 3)
    assert np.array_equal(generated[0, 0], obs.first)
    assert torch.equal(original, torch.get_rng_state())
    with pytest.raises(ValueError, match="identical"):
        trainer.resume(checkpoint, {"changed": True})


def test_invalid_stationary_labels_and_config_are_rejected():
    data, geometry, _ = fixture_data()
    data["train"]["motion"][0] = 0.1
    with pytest.raises(ValueError, match="stationary"):
        ContactTrainer(data, geometry, ContactConfig())
    with pytest.raises(ValueError, match="configuration"):
        ContactConfig(zero_probability=0.8, collision_probability=0.4).validate()


def test_stationary_token_supervision_covers_unpushed_rest_without_freezing_moving_targets():
    data, geometry, _ = fixture_data()
    features = torch.from_numpy(data["train"]["features"][:1])
    geometry = {key: torch.from_numpy(value[:1]) for key, value in geometry["train"].items()}
    clean = torch.zeros((1, 31, 2, 4))
    clean[:, :, 0, 0] = 0.1

    class Prediction:
        sigma_data = 0.25

        def __init__(self, object_id):
            self.value = clean.clone()
            self.value[:, :, object_id, 0] += 0.01

        def __call__(self, *args):
            return self.value

    config = ContactConfig(stationary_tokens=True)
    arguments = (
        clean,
        features,
        torch.tensor([[0.25, 0]]),
        torch.tensor([0.1]),
        torch.zeros_like(clean),
        geometry,
        config,
    )
    _, passive = contact_objective(Prediction(1), *arguments)
    _, active = contact_objective(Prediction(0), *arguments)
    assert passive["stationary"] > 0 and active["stationary"] == 0


def test_cli_training_requires_actual_allocation(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="scheduled TC2"):
        main(
            [
                "--dataset",
                str(tmp_path),
                "--config",
                str(tmp_path / "config.json"),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )
    assert not (tmp_path / "run").exists()


def test_canvas_loss_matches_independent_geometry_and_pushes_motion_inward():
    from click2future.training.contact_geometry import canvas_excess

    obs = observation()
    motion = np.zeros((1, 4, 2, 4), np.float32)
    motion[:, :, 0, :2] = [1.5, -0.8]
    motion[:, :, 1, :2] = [-1.4, 0.9]
    features = torch.from_numpy(obs.features)[None]
    geometry = tensor_geometry(obs)
    independent = geometry_diagnostics(obs, motion)["clips"][0]
    # The old diagnostic measures from a one-pixel interior; the training loss
    # measures the native canvas and keeps that raster bias separate.
    expected = np.maximum(np.array(independent["boundary_depth_pixels_per_object_frame"]) - 1, 0)
    value = torch.tensor(motion, requires_grad=True)
    excess = canvas_excess(value, features, geometry)
    np.testing.assert_allclose(excess.detach()[0].numpy() * 64, expected, atol=5e-6)
    loss = excess.square().mean()
    loss.backward()
    assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    improved = canvas_excess(value.detach() - value.grad * 0.2, features, geometry)
    assert improved.square().mean() < loss


def test_wall_regularization_preserves_legacy_objective_when_disabled():
    data, geometry, _ = fixture_data()
    features = torch.from_numpy(data["train"]["features"][:1])
    geometry = {key: torch.from_numpy(value[:1]) for key, value in geometry["train"].items()}
    clean = torch.zeros((1, 31, 2, 4))
    prediction = torch.ones_like(clean, requires_grad=True)

    class FixedPrediction:
        sigma_data = 0.25

        def __call__(self, *args):
            return prediction

    arguments = (
        FixedPrediction(),
        clean,
        features,
        torch.tensor([[0.25, 0]]),
        torch.tensor([0.1]),
        torch.zeros_like(clean),
        geometry,
    )
    original, original_parts = contact_objective(*arguments, ContactConfig())
    enabled, enabled_parts = contact_objective(*arguments, ContactConfig(boundary_weight=256))
    assert "canvas_excess" not in original_parts
    assert enabled_parts["canvas_excess"] > 0
    torch.testing.assert_close(
        enabled, original + 256 * enabled_parts["canvas_excess"], rtol=0, atol=0
    )
    for key in original_parts:
        torch.testing.assert_close(enabled_parts[key], original_parts[key], rtol=0, atol=0)
    enabled.backward()
    assert torch.isfinite(prediction.grad).all()


def test_old_contact_checkpoints_resume_with_default_wall_loss_only(tmp_path):
    data, geometry, _ = fixture_data()
    config = ContactConfig(steps=4, width=16, layers=1, batch_size=3)
    trainer = ContactTrainer(data, geometry, config)
    trainer.train_step()
    checkpoint = tmp_path / "legacy.pt"
    identity = {
        "source_sha256": {
            "models/object_motion.py": sha256(Path("src/click2future/models/object_motion.py"))
        }
    }
    trainer.save(checkpoint, identity)
    state = torch.load(checkpoint, weights_only=True)
    state["config"].pop("boundary_weight")
    state["config"].pop("boundary_margin_pixels")
    torch.save(state, checkpoint)
    restored = ContactTrainer(data, geometry, config)
    restored.resume(checkpoint, identity)
    assert trainer.train_step() == restored.train_step()
    with pytest.raises(ValueError, match="identical"):
        ContactTrainer(
            data, geometry, ContactConfig(**{**config.__dict__, "boundary_weight": 256})
        ).resume(checkpoint, identity)
    assert load_trajectory(checkpoint)[1]["config"]["stationary_tokens"] is False
