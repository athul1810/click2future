"""Noise removal, E7 control parity, continuation and observable public sampling."""

import copy
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
from test_contact_training import fixture_data

from click2future.demo.models import CheckpointModel
from click2future.models.contact_projection import PROJECTION_CONTRACT, project_motion
from click2future.models.momentum_block import MOMENTUM_BLOCK_CONTRACT, apply_momentum_block
from click2future.models.world_motion import WorldTrajectoryDiffusion, sample_world
from click2future.training.sample_trajectory import (
    load_trajectory,
    sample_trajectory_motions,
    sample_trajectory_videos,
)
from click2future.training.sample_trajectory import (
    main as sample_main,
)
from click2future.training.train_expected_prior import ExpectedPriorConfig
from click2future.training.train_prior import PriorConfig, PriorTrainer, prior_sources
from click2future.training.train_world import WorldConfig, WorldTrainer, source_identity


def fixture(direct=True):
    data, geometry, obs = fixture_data()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    config = PriorConfig(steps=4, width=16, layers=1, batch_size=3, world_prior_only=direct)
    return PriorTrainer(data, geometry, worlds, config), data, geometry, obs


def test_warm_start_arms_share_teacher_rng_weights_and_update_draws(tmp_path):
    _, data, geometry, _ = fixture()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    teacher = WorldTrainer(
        data, geometry, worlds, WorldConfig(steps=4, width=16, layers=1, batch_size=3)
    )
    teacher.train_step()
    identity = {
        key: "unit-data"
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
            "world_label_sha256",
            "training_supplement",
            "combined_training_examples",
        )
    }
    teacher.save(tmp_path / "teacher.pt", {**identity, "source_sha256": source_identity()})
    arms = [fixture(flag)[0] for flag in (True, False)]
    records = [arm.initialize(tmp_path / "teacher.pt", identity) for arm in arms]
    assert records[0] == records[1]
    for arm in arms:
        assert all(
            torch.equal(v, arm.model.state_dict()[k]) for k, v in teacher.ema.state_dict().items()
        )
        arm.train_step()
    assert torch.equal(arms[0].generator.get_state(), arms[1].generator.get_state())
    with pytest.raises(ValueError, match="data differ"):
        fresh = fixture()[0]
        fresh.initialize(
            tmp_path / "teacher.pt", {**identity, "combined_training_examples": "wrong"}
        )


def test_control_is_exact_e7_and_direct_removes_only_corruption_inputs():
    direct, data, *_ = fixture()
    control, *_ = fixture(False)
    parent = WorldTrajectoryDiffusion(width=16, layers=1, future_frames=31)
    parent.load_state_dict(control.model.state_dict())
    assert all(
        torch.equal(v, control.model.state_dict()[k]) for k, v in direct.model.state_dict().items()
    )
    row = control.data["train"]
    noisy = torch.randn((1, 31, 2, 4), generator=torch.Generator().manual_seed(9))
    sigma = torch.tensor([0.2])
    features, force, world = row["features"][:1], row["force"][:1], sample_world(10)
    assert torch.equal(
        parent(noisy, sigma, features, force, world),
        control.model(noisy, sigma, features, force, world),
    )
    assert torch.equal(
        parent.sample(features, force, seeds=[0, 1], steps=2),
        control.model.sample(features, force, seeds=[0, 1], steps=2),
    )
    a = direct.model(noisy, sigma, features, force, world)
    b = direct.model(-3 * noisy, sigma * 7, features, force, world)
    assert torch.equal(a, b)
    for bad in (noisy * float("nan"), noisy[..., :3]):
        with pytest.raises(ValueError):
            direct.model(bad, sigma, features, force, world)
    for bad in (torch.tensor([0.0]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError):
            direct.model(noisy, bad, features, force, world)


def test_contact_dynamics_loss_is_an_optional_prior_training_term():
    control, data, geometry, _ = fixture()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    config = PriorConfig(
        steps=4,
        width=16,
        layers=1,
        batch_size=3,
        contact_transition_weight=8,
        normal_velocity_weight=4,
    )
    candidate = PriorTrainer(data, geometry, worlds, config)
    control_record = control.train_step()
    candidate_record = candidate.train_step()
    assert "contact_transition" not in control_record
    assert "normal_velocity" not in control_record
    assert candidate_record["contact_transition"] >= 0
    assert candidate_record["normal_velocity"] >= 0
    with pytest.raises(ValueError):
        PriorConfig(contact_transition_weight=-1).validate()


def test_dynamics_residual_is_an_optional_prior_training_term():
    control, data, geometry, _ = fixture()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    candidate = PriorTrainer(
        data,
        geometry,
        worlds,
        PriorConfig(
            steps=4,
            width=16,
            layers=1,
            batch_size=3,
            dynamics_residual_weight=1,
        ),
    )
    assert "dynamics_residual" not in control.train_step()
    assert candidate.train_step()["dynamics_residual"] >= 0
    with pytest.raises(ValueError):
        PriorConfig(dynamics_residual_weight=-1).validate()


def test_collision_impulse_is_an_optional_prior_training_term():
    _, data, geometry, _ = fixture()
    worlds = {split: sample_world(1).numpy().repeat(6, axis=0) for split in data}
    candidate = PriorTrainer(
        data,
        geometry,
        worlds,
        PriorConfig(
            steps=4,
            width=16,
            layers=1,
            batch_size=3,
            collision_impulse_weight=1,
        ),
    )
    assert candidate.train_step()["collision_impulse"] >= 0
    with pytest.raises(ValueError):
        PriorConfig(collision_impulse_weight=-1).validate()


def test_rest_anchored_residual_recipe_changes_only_the_new_weight():
    root = Path(__file__).parents[1] / "configs/training"
    control = json.loads((root / "prior_rest_anchored.json").read_text())
    candidate = json.loads((root / "prior_rest_anchored_dynamics_residual.json").read_text())
    control["dynamics_residual_weight"] = 0.0
    assert {key for key in control if control[key] != candidate[key]} == {
        "dynamics_residual_weight"
    }
    ExpectedPriorConfig(**candidate).validate()


def test_seed20_residual_comparison_changes_only_seed_and_residual_weight():
    root = Path(__file__).parents[1] / "configs/training"
    control = json.loads((root / "prior_rest_anchored_seed20.json").read_text())
    candidate = json.loads((root / "prior_rest_anchored_dynamics_residual_seed20.json").read_text())
    assert control["seed"] == candidate["seed"] == 20
    assert {key for key in control if control[key] != candidate[key]} == {
        "dynamics_residual_weight"
    }
    ExpectedPriorConfig(**control).validate()
    ExpectedPriorConfig(**candidate).validate()


def test_collision_impulse_screen_changes_only_the_registered_weight():
    root = Path(__file__).parents[1] / "configs/training"
    control = json.loads((root / "collision_impulse_screen_control.json").read_text())
    candidate = json.loads((root / "collision_impulse_screen_x1024.json").read_text())
    assert control["seed"] == candidate["seed"] == 22
    assert control["steps"] == candidate["steps"] == 2000
    assert {key for key in control if control[key] != candidate[key]} == {
        "collision_impulse_weight"
    }
    ExpectedPriorConfig(**control).validate()
    ExpectedPriorConfig(**candidate).validate()


@pytest.mark.parametrize("direct", [True, False])
def test_checkpoint_resume_sampler_and_demo_keep_all_seeded_samples(tmp_path, direct):
    trainer, _, _, obs = fixture(direct)
    trainer.train_step()
    trainer.train_step()
    if direct:
        assert trainer.model.motion.weight.grad.count_nonzero() == 0
        assert trainer.model.condition[0].weight.grad.abs().sum() > 0
    checkpoint = tmp_path / "fixture.pt"
    provenance = {"source_sha256": prior_sources()}
    trainer.save(checkpoint, provenance)
    if direct:
        legacy = torch.load(checkpoint, weights_only=True)
        legacy["config"].pop("contact_transition_weight")
        legacy["config"].pop("normal_velocity_weight")
        torch.save(legacy, checkpoint)
    restored, *_ = fixture(direct)
    restored.resume(checkpoint, provenance)
    assert trainer.train_step() == restored.train_step()
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )
    assert all(
        torch.equal(v, restored.ema.state_dict()[k]) for k, v in trainer.ema.state_dict().items()
    )
    model, _ = load_trajectory(checkpoint)
    steps = 1 if direct else 2
    rng = torch.get_rng_state().clone()
    features = torch.from_numpy(obs.features[None])
    force = torch.tensor([[0.15, 0.0]])
    samples = model.sample(features, force, seeds=[11, 12, 13, 14], steps=steps)
    assert samples.shape == (4, 31, 2, 4) and torch.isfinite(samples).all()
    assert torch.equal(rng, torch.get_rng_state())
    if direct:
        assert torch.equal(samples[0], model.decode_world(features, force, sample_world(11))[0])
        with pytest.raises(ValueError):
            model.sample(features, force, seeds=[0], steps=16)
        with pytest.raises(ValueError):
            model.sample(features, force, seeds=[0, 0], steps=1)
    videos = sample_trajectory_videos(
        model,
        obs.first,
        obs.masks[obs.selected_index],
        force[0].numpy(),
        seeds=[11, 12, 13, 14],
        sampling_steps=steps,
    )
    if direct:
        _, raw = sample_trajectory_motions(
            model,
            obs.first,
            obs.masks[obs.selected_index],
            force[0].numpy(),
            seeds=[11, 12],
            sampling_steps=steps,
        )
        _, projected = sample_trajectory_motions(
            model,
            obs.first,
            obs.masks[obs.selected_index],
            force[0].numpy(),
            seeds=[11, 12],
            sampling_steps=steps,
            contact_project=True,
        )
        expected = np.stack(
            [
                project_motion(obs, motion, sample_world(seed)[0, :, 0].numpy())
                for seed, motion in zip((11, 12), raw, strict=True)
            ]
        )
        np.testing.assert_array_equal(projected, expected)
        _, momentum = sample_trajectory_motions(
            model,
            obs.first,
            obs.masks[obs.selected_index],
            force[0].numpy(),
            seeds=[11, 12],
            sampling_steps=steps,
            momentum_block=True,
        )
        expected_momentum = np.stack(
            [
                apply_momentum_block(
                    obs,
                    motion,
                    sample_world(seed)[0, :, 0].numpy(),
                    force[0].numpy(),
                )
                for seed, motion in zip((11, 12), raw, strict=True)
            ]
        )
        np.testing.assert_array_equal(momentum, expected_momentum)
        condition = tmp_path / "condition.npz"
        output = tmp_path / "projected.npz"
        np.savez_compressed(
            condition,
            first_frame=obs.first,
            selected_mask=obs.masks[obs.selected_index],
            force_xy=force[0].numpy(),
        )
        sample_main(
            [
                "--checkpoint",
                str(checkpoint),
                "--condition",
                str(condition),
                "--output",
                str(output),
                "--sampling-steps",
                "1",
                "--seeds",
                "11",
                "12",
                "--contact-project",
                "--momentum-block",
            ]
        )
        with np.load(output, allow_pickle=False) as arrays:
            assert arrays["videos"].shape == (2, 32, 64, 64, 3)
        metadata = json.loads(output.with_suffix(".json").read_text())
        assert metadata["contact_projection"] == PROJECTION_CONTRACT
        assert metadata["momentum_block"] == MOMENTUM_BLOCK_CONTRACT
    adapter = CheckpointModel("fixture", checkpoint, trajectory_steps=steps)
    actual = adapter.predict(
        obs.first,
        obs.masks[obs.selected_index],
        force[0].numpy(),
        seeds=[11, 12, 13, 14],
        future_frames=31,
        cancelled=threading.Event(),
    )
    assert np.array_equal(videos, actual)
    assert adapter.metadata["network_evaluations_per_sample"] == 2 * steps - 1
    corrupt = copy.deepcopy(torch.load(checkpoint, weights_only=True))
    corrupt["provenance"]["source_sha256"]["models/prior_motion.py"] = "tampered"
    torch.save(corrupt, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="source differs"):
        load_trajectory(tmp_path / "bad.pt")
