"""Numerical sampling, inference boundary, and interrupted-training regression checks."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from click2future.data import generate_dataset, load_batch
from click2future.models.pilot_denoiser import DenoiserOutput
from click2future.training.conditioning import observable_condition, training_inputs
from click2future.training.diffusion import Diffusion
from click2future.training.sample_diffusion import load_generator, sample_videos
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


@pytest.mark.parametrize("prediction_type", ["epsilon", "velocity"])
def test_ddim_recovers_known_clean_signal_and_preserves_rng(prediction_type):
    diffusion = Diffusion(20, prediction_type=prediction_type)
    condition = observable_condition(
        torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 8, 8), torch.zeros(1, 2)
    )

    class Oracle(nn.Module):
        def forward(self, noisy, timesteps, **_):
            alpha = diffusion.alpha_bar[timesteps][:, None, None, None, None]
            noise = (noisy - alpha.sqrt() * 0.25) / (1 - alpha).sqrt()
            prediction = noise
            if prediction_type == "velocity":
                prediction = alpha.sqrt() * noise - (1 - alpha).sqrt() * 0.25
            return DenoiserOutput(prediction)

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


def test_velocity_round_trip_stays_well_conditioned_at_low_signal():
    velocity = Diffusion(1000, prediction_type="velocity")
    epsilon = Diffusion(1000)
    clean = torch.full((3, 2, 2), 0.25)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(8))
    times = torch.tensor([0, 500, 999])
    noisy = velocity.add_noise(clean, times, noise)
    target = velocity.training_target(clean, times, noise)
    torch.testing.assert_close(
        velocity.predict_clean(noisy, times, target), clean, atol=1e-6, rtol=0
    )
    velocity_error = (velocity.predict_clean(noisy, times, target + 0.001) - clean)[-1].abs().mean()
    epsilon_error = (epsilon.predict_clean(noisy, times, noise + 0.001) - clean)[-1].abs().mean()
    assert velocity_error <= 0.00101
    assert epsilon_error > 1000 * velocity_error


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


def test_checkpoint_to_video_inference_preserves_frame_zero(dataset, tmp_path):
    trainer = Trainer(dataset, TrainConfig(steps=2, base_channels=8, diffusion_steps=10))
    trainer.train_step()
    checkpoint = tmp_path / "inference.pt"
    trainer.save(checkpoint)
    model, diffusion, state = load_generator(checkpoint)
    assert state["step"] == 1
    batch = load_batch(dataset, trainer.sample_ids[:1])
    first = (batch["frames"][0, 0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    kwargs = dict(seeds=(17, 18), future_frames=3, sampling_steps=4)
    videos = sample_videos(
        model, diffusion, first, batch["selected_mask"][0, 0], batch["force_xy"][0], **kwargs
    )
    repeated = sample_videos(
        model, diffusion, first, batch["selected_mask"][0, 0], batch["force_xy"][0], **kwargs
    )
    assert videos.dtype == np.uint8
    assert videos.shape == (2, 4, 16, 16, 3)
    np.testing.assert_array_equal(videos[:, 0], np.broadcast_to(first, (2, *first.shape)))
    np.testing.assert_array_equal(videos, repeated)


def test_recorded_source_change_cannot_be_called_exact_resume(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("C2F_SOURCE_REVISION", "a" * 40)
    config = TrainConfig(steps=2, base_channels=8)
    trainer = Trainer(dataset, config)
    checkpoint = tmp_path / "source.pt"
    trainer.save(checkpoint)
    monkeypatch.setenv("C2F_SOURCE_REVISION", "b" * 40)
    changed = Trainer(dataset, config)
    with pytest.raises(ValueError, match="Source revision changed"):
        changed.resume(checkpoint)


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.parametrize("prediction_type", ["epsilon", "velocity"])
def test_full_training_resume_matches_uninterrupted(dataset, tmp_path, auxiliary, prediction_type):
    config = TrainConfig(
        steps=6,
        base_channels=8,
        diffusion_steps=20,
        auxiliary=auxiliary,
        prediction_type=prediction_type,
    )
    uninterrupted = Trainer(dataset, config)
    records = []
    for index in range(6):
        if index == 3:
            random.random()
            np.random.random(7)
            torch.randn(9)
        records.append(uninterrupted.train_step())
    expected_python = random.random()
    expected_numpy = np.random.random(7)
    expected_torch = torch.randn(9)
    interrupted = Trainer(dataset, config)
    for _ in range(3):
        interrupted.train_step()
    random.random()
    np.random.random(7)
    torch.randn(9)
    # Probe must leave the next training sample, timestep and noise unchanged.
    interrupted.probe_loss()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save(checkpoint)
    resumed = Trainer(dataset, config)
    resumed.resume(checkpoint)
    continuation = [resumed.train_step() for _ in range(3)]
    assert random.random() == expected_python
    np.testing.assert_array_equal(np.random.random(7), expected_numpy)
    torch.testing.assert_close(torch.randn(9), expected_torch, rtol=0, atol=0)
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
