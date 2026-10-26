"""Latent coupling, parameter-matched arms, exact migration and observable sampling."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_contact_training import fixture_data

from click2future.demo.models import CheckpointModel
from click2future.models.world_motion import WorldTrajectoryDiffusion, sample_world
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_contact import ContactConfig, ContactTrainer
from click2future.training.train_world import WorldConfig, WorldTrainer, source_identity


def setup_trainer(**flags):
    data, geometry, obs = fixture_data()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    config = WorldConfig(steps=4, width=16, layers=1, batch_size=3, **flags)
    return WorldTrainer(data, geometry, worlds, config), data, geometry, obs


def test_world_prior_is_action_independent_reproducible_and_does_not_touch_rng():
    original = torch.get_rng_state().clone()
    a = sample_world(0)
    assert torch.equal(a, sample_world(0)) and not torch.equal(a, sample_world(1))
    assert torch.equal(original, torch.get_rng_state())
    assert a.shape == (1, 2, 3)
    assert (a[..., 0] >= 0.5).all() and (a[..., 0] <= 2).all()
    for bad in (True, -1, 2**63):
        with pytest.raises(ValueError):
            sample_world(bad)


def test_warm_start_preserves_teacher_and_initializes_bitwise_matched_arms(tmp_path):
    _, data, geometry, _ = setup_trainer()
    original = ContactTrainer(
        data, geometry, ContactConfig(steps=4, width=16, layers=1, batch_size=3)
    )
    original.train_step()
    identity = {
        key: "unit-data"
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
        )
    }
    identity["source_sha256"] = source_identity()
    original.save(tmp_path / "wall.pt", identity)
    row = original.data["train"]
    noisy = torch.randn((2, 31, 2, 4), generator=torch.Generator().manual_seed(6))
    features, force, sigma = row["features"][:2], row["force"][:2], torch.ones(2)
    expected = original.ema(noisy, sigma, features, force)
    outputs, parameters = [], []
    for relational, world_condition in ((False, False), (True, False), (True, True)):
        trainer, _, _, _ = setup_trainer(relational=relational, world_condition=world_condition)
        rng = trainer.generator.get_state().clone()
        trainer.initialize(tmp_path / "wall.pt", identity)
        actual = trainer.model(noisy, sigma, features, force, sample_world(5).expand(2, -1, -1))
        # Tensor layout can change kernel summation order on arm64 (observed 2.9e-11).
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-9)
        outputs.append(actual)
        assert trainer.step == 0 and not trainer.optimizer.state
        assert torch.equal(rng, trainer.generator.get_state())
        parameters.append(copy.deepcopy(trainer.model.state_dict()))
        with pytest.raises(ValueError, match="data differ"):
            trainer.initialize(tmp_path / "wall.pt", {**identity, "geometry_sha256": "wrong"})
    assert all(torch.equal(output, outputs[0]) for output in outputs[1:])
    for state in parameters[1:]:
        assert all(torch.equal(value, state[key]) for key, value in parameters[0].items())


def test_world_condition_changes_output_only_in_enabled_arm():
    trainer, _, _, _ = setup_trainer()
    model = trainer.model
    with torch.no_grad():
        model.output.weight.normal_(std=0.1, generator=torch.Generator().manual_seed(4))
        model.world_context[-1].weight.normal_(generator=torch.Generator().manual_seed(5))
    row = trainer.data["train"]
    args = (torch.ones((1, 31, 2, 4)) * 0.1, torch.ones(1), row["features"][:1], row["force"][:1])
    a, b = model(*args, sample_world(0)), model(*args, sample_world(1))
    assert (a - b).abs().max() > 1e-4
    model.world_condition = False
    assert torch.equal(model(*args, sample_world(0)), model(*args, sample_world(1)))
    bad = sample_world(0)
    bad[..., 0] = 0
    with pytest.raises(ValueError, match="latents"):
        model(*args, bad)


def test_new_adapters_receive_gradients_and_checkpoint_continuation_is_exact(tmp_path):
    trainer, _, _, _ = setup_trainer()
    trainer.train_step()
    trainer.train_step()
    assert trainer.model.messages[0][-1].weight.grad.abs().sum() > 0
    assert trainer.model.world_context[-1].weight.grad.abs().sum() > 0
    provenance = {"source_sha256": source_identity()}
    checkpoint = tmp_path / "world.pt"
    trainer.save(checkpoint, provenance)
    restored, _, _, _ = setup_trainer()
    restored.resume(checkpoint, provenance)
    assert trainer.train_step() == restored.train_step()
    assert all(
        torch.equal(value, restored.model.state_dict()[key])
        for key, value in trainer.model.state_dict().items()
    )
    model, state = load_trajectory(checkpoint)
    assert isinstance(model, WorldTrajectoryDiffusion) and state["trajectory_format_version"] == 3
    adapter = CheckpointModel("E7-unit", checkpoint, trajectory_steps=3)
    assert adapter.metadata["training_variant"] == "shared_world_relational"
    _, _, _, obs = setup_trainer()
    force = np.array([0.25, 0], np.float32)
    expected_video = sample_trajectory_videos(
        model, obs.first, obs.masks[obs.selected_index], force, seeds=[0], sampling_steps=3
    )
    actual_video = adapter.predict(
        obs.first, obs.masks[obs.selected_index], force, seeds=[0], future_frames=31, cancelled=None
    )
    np.testing.assert_array_equal(actual_video, expected_video)
    with pytest.raises(ValueError, match="identical"):
        restored.resume(checkpoint, {"changed": True})
    state["provenance"]["source_sha256"]["models/world_motion.py"] = "tampered"
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="source differs"):
        load_trajectory(checkpoint)


def test_sampler_binds_one_world_for_every_heun_step_and_accepts_no_hidden_scene(monkeypatch):
    trainer, _, _, obs = setup_trainer()
    model = trainer.model
    model.force_limit, model.sigma_min, model.sigma_max = 1.1, 0.002, 20
    seen = []
    original_forward = model.forward

    def capture(noisy, sigma, features, force, world):
        seen.append(world.clone())
        return original_forward(noisy, sigma, features, force, world)

    monkeypatch.setattr(model, "forward", capture)
    original = torch.get_rng_state().clone()
    for selected in (0, 1):
        generated = sample_trajectory_videos(
            model,
            obs.first,
            obs.masks[selected],
            np.array([0.25, 0], np.float32),
            seeds=[2],
            sampling_steps=3,
        )
        assert generated.shape == (1, 32, 64, 64, 3)
        assert np.array_equal(generated[0, 0], obs.first)
    assert len(seen) == 10 and all(torch.equal(value, sample_world(2)) for value in seen)
    assert torch.equal(original, torch.get_rng_state())
    with pytest.raises(ValueError):
        model.sample(torch.from_numpy(obs.features[None]), torch.zeros(1, 2), seeds=[1, 1])


def test_world_config_rejects_nonboolean_switches():
    with pytest.raises(ValueError):
        replace(WorldConfig(), relational=1).validate()
    with pytest.raises(ValueError):
        replace(WorldConfig(), world_condition="yes").validate()
