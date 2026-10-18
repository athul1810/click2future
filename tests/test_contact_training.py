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
from click2future.training.contact_geometry import observed_hulls, signed_penetration
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
