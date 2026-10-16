"""Numerical sampling, inference boundary, and interrupted-training regression checks."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from click2future.data import generate_dataset, load_batch
from click2future.models.pilot_denoiser import DenoiserOutput
from click2future.training.conditioning import observable_condition, training_inputs
from click2future.training.diffusion import Diffusion
from click2future.training.train_diffusion import TrainConfig, Trainer


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data"
    generate_dataset(
        path,
        group_count=8,
        variants_per_group=2,
        seed=909,
        config={"frame_count": 4, "resolution": 16, "fps": 8, "physics_hz": 240},
    )
    return path


def test_ddim_recovers_known_clean_signal_and_preserves_rng():
    diffusion = Diffusion(20)
    condition = observable_condition(
        torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 8, 8), torch.zeros(1, 2)
    )

    class Oracle(nn.Module):
        def forward(self, noisy, timesteps, **_):
            alpha = diffusion.alpha_bar[timesteps][:, None, None, None, None]
            return DenoiserOutput((noisy - alpha.sqrt() * 0.25) / (1 - alpha).sqrt())

    oracle = Oracle().train()
    state = torch.get_rng_state().clone()
    sampled = diffusion.sample(oracle, condition, future_frames=3, sampling_steps=7, seed=42)
    torch.testing.assert_close(sampled, torch.full_like(sampled, 0.25), atol=1e-6, rtol=0)
    assert torch.equal(state, torch.get_rng_state())
    assert oracle.training

    # A non-oracle predictor retains seed dependence, without global RNG changes.
    class Zero(nn.Module):
        def forward(self, noisy, timesteps, **_):
            return DenoiserOutput(torch.zeros_like(noisy))

    zero = Zero()
    first = diffusion.sample(zero, condition, future_frames=2, sampling_steps=4, seed=8)
    repeated = diffusion.sample(zero, condition, future_frames=2, sampling_steps=4, seed=8)
    changed = diffusion.sample(zero, condition, future_frames=2, sampling_steps=4, seed=9)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.equal(state, torch.get_rng_state())


def test_noise_equation_matches_analytic_endpoints():
    diffusion = Diffusion(100)
    clean = torch.tensor([[[1.0]], [[-0.5]]])
    noise = torch.tensor([[[0.5]], [[2.0]]])
    t = torch.tensor([0, 99])
    actual = diffusion.add_noise(clean, t, noise)
    for index in range(2):
        a = diffusion.alpha_bar[t[index]]
        torch.testing.assert_close(
            actual[index], a.sqrt() * clean[index] + (1 - a).sqrt() * noise[index]
        )
    assert (diffusion.alpha_bar[1:] < diffusion.alpha_bar[:-1]).all()


def test_observable_boundary_ignores_oracle_fields_and_rejects_action_drift(dataset):
    trainer = Trainer(dataset, TrainConfig(base_channels=8))
    batch = load_batch(dataset, trainer.sample_ids[:1])
    future, condition = training_inputs(batch, "cpu")
    batch.update(
        scene_specs=object(), positions=object(), instance_masks=object(), metadata=object()
    )
    other_future, other = training_inputs(batch, "cpu")
    assert torch.equal(future, other_future)
    for key, value in condition.kwargs().items():
        assert torch.equal(value, other.kwargs()[key])
    np.testing.assert_allclose(condition.force_xy, batch["force_xy"] / 0.5)
    batch["force_duration"][:] = 0.5
    with pytest.raises(ValueError, match="0.25"):
        training_inputs(batch, "cpu")
    with pytest.raises(ValueError, match="Force magnitude"):
        observable_condition(torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 8, 8), torch.ones(1, 2))


@pytest.mark.parametrize("auxiliary", [False, True])
def test_full_training_resume_matches_uninterrupted(dataset, tmp_path, auxiliary):
    config = TrainConfig(steps=6, base_channels=8, diffusion_steps=20, auxiliary=auxiliary)
    uninterrupted = Trainer(dataset, config)
    records = [uninterrupted.train_step() for _ in range(6)]
    interrupted = Trainer(dataset, config)
    for _ in range(3):
        interrupted.train_step()
    # Probe must leave the next training sample, timestep and noise unchanged.
    interrupted.probe_loss()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save(checkpoint)
    resumed = Trainer(dataset, config)
    resumed.resume(checkpoint)
    continuation = [resumed.train_step() for _ in range(3)]
    for expected, actual in zip(records[3:], continuation, strict=True):
        assert actual["sample_ids"] == expected["sample_ids"]
        assert actual["loss"] == expected["loss"]
        assert actual["learning_rate_next_step"] == expected["learning_rate_next_step"]
    for key, value in uninterrupted.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[key]), key
    for key, value in uninterrupted.ema.state_dict().items():
        assert torch.equal(value, resumed.ema.state_dict()[key]), key
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    changed_config = Trainer(dataset, replace(config, learning_rate=0.002))
    with pytest.raises(ValueError, match="config differs"):
        changed_config.resume(checkpoint)
    # Hash actual sample bytes, not merely a self-reported manifest identifier.
    sample = dataset / f"{resumed.sample_ids[0]}.npz"
    sample.write_bytes(sample.read_bytes() + b"changed")
    changed_data = Trainer(dataset, config)
    with pytest.raises(ValueError, match="data changed"):
        changed_data.resume(checkpoint)
