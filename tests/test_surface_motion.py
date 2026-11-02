"""Observed geometry, matched arms, full-gradient training and exact continuation."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_contact_training import fixture_data, tensor_geometry
from test_world_motion import setup_trainer

from click2future.demo.models import CheckpointModel
from click2future.models.surface_motion import SurfaceTrajectoryDiffusion, surface_features
from click2future.models.world_motion import sample_world
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.surface_checkpoint import load_surface, sample_surface_videos
from click2future.training.train_surface import SurfaceConfig, SurfaceTrainer, surface_sources


def fixture(local=True):
    data, geometry, obs = fixture_data()
    worlds = {key: sample_world(1).numpy().repeat(6, axis=0) for key in data}
    config = SurfaceConfig(
        steps=4, width=16, layers=1, batch_size=2, surface_width=16, local_surfaces=local
    )
    trainer = SurfaceTrainer(data, geometry, worlds, config)
    return trainer, data, geometry, worlds, obs


def test_surface_descriptors_preserve_identity_symmetry_and_padding():
    trainer, _, _, _, obs = fixture()
    features = trainer.data["train"]["features"][:1]
    geometry = tensor_geometry(obs)
    motion = torch.randn((1, 3, 2, 4), generator=torch.Generator().manual_seed(15)) * 0.02
    original = motion.clone()
    result, valid = surface_features(motion, features, geometry)
    assert result.shape == (1, 3, 2, 32, 17)
    assert torch.equal(original, motion)
    swapped, _ = surface_features(
        motion.flip(2), features.flip(1), {k: v.flip(1) for k, v in geometry.items()}
    )
    torch.testing.assert_close(result.flip(2), swapped)
    changed = {k: v.clone() for k, v in geometry.items()}
    changed["hulls"][~valid] = 100
    changed["normals"][~valid] = -100
    padded, _ = surface_features(motion, features, changed)
    torch.testing.assert_close(result, padded, rtol=0, atol=0)
    assert (result[~valid[:, None].expand(1, 3, 2, 32)] == 0).all()


def test_local_correspondences_differ_from_pooled_but_all_parameters_match():
    a, _, _, _, obs = fixture(True)
    b, *_ = fixture(False)
    assert all(torch.equal(v, b.model.state_dict()[k]) for k, v in a.model.state_dict().items())
    assert sum(p.numel() for p in a.model.parameters()) == sum(
        p.numel() for p in b.model.parameters()
    )
    features = a.data["train"]["features"][:1]
    geometry = tensor_geometry(obs)
    motion = torch.zeros((1, 3, 2, 4))
    local, valid = surface_features(motion, features, geometry, local=True)
    pooled, _ = surface_features(motion, features, geometry, local=False)
    assert not torch.allclose(local[..., 4:9], pooled[..., 4:9])
    # Correspondence pooling is over valid surfaces only, before the shared MLP.
    for obj in (0, 1):
        values = pooled[0, 0, obj, valid[0, obj], 4:9]
        torch.testing.assert_close(values, values[:1].expand_as(values))


def test_degenerate_rotation_and_tangent_distance_have_finite_gradients():
    trainer, _, _, _, obs = fixture()
    features = trainer.data["train"]["features"][:1]
    motion = torch.zeros((1, 3, 2, 4), requires_grad=True)
    with torch.no_grad():
        motion[..., 3] = -0.5
    result, _ = surface_features(motion, features, tensor_geometry(obs))
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(motion.grad).all()


def test_nearest_segment_matches_analytic_square_gap_and_motion_gradient():
    hull = torch.tensor([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    normals = torch.tensor([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    geometry = {
        "hulls": torch.cat((hull, torch.zeros((28, 2))))[None, None].repeat(1, 2, 1, 1),
        "normals": torch.cat((normals, torch.zeros((28, 2))))[None, None].repeat(1, 2, 1, 1),
        "axis_valid": (torch.arange(32) < 4)[None, None].repeat(1, 2, 1),
    }
    features = torch.zeros((1, 2, 11))
    features[0, :, :2] = torch.tensor([[0.3, 0.5], [0.5, 0.5]])
    features[0, 0, -1] = 1
    motion = torch.zeros((1, 1, 2, 4), requires_grad=True)
    values, _ = surface_features(motion, features, geometry)
    # Object 0's lower-right vertex is x=.35; object 1's left segment is x=.45.
    torch.testing.assert_close(values[0, 0, 0, 1, 4:6], torch.tensor([0.1, 0.0]), atol=1e-7, rtol=0)
    torch.testing.assert_close(values[0, 0, 0, 1, 8], torch.tensor(0.1), atol=1e-7, rtol=0)
    values[0, 0, 0, 1, 8].backward()
    torch.testing.assert_close(motion.grad[0, 0, 0, 0], torch.tensor(-0.5), atol=1e-6, rtol=0)
    torch.testing.assert_close(motion.grad[0, 0, 1, 0], torch.tensor(0.5), atol=1e-6, rtol=0)


def test_teacher_migration_is_exact_with_fresh_optimizer_and_data_binding(tmp_path):
    teacher, _, _, _ = setup_trainer()
    teacher.train_step()
    trainer, _, _, _, _ = fixture()
    parent = {
        key: "unit-data"
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "training_supplement",
            "combined_training_examples",
        )
    }
    parent.update(
        geometry_sha256={"train": "a", "validation": "b"},
        world_label_sha256={"train": "c", "validation": "d"},
        source_sha256=surface_sources(),
    )
    path = tmp_path / "teacher.pt"
    teacher.save(path, parent)
    rng = trainer.generator.get_state().clone()
    trainer.initialize(path, parent)
    assert torch.equal(rng, trainer.generator.get_state()) and not trainer.optimizer.state
    assert all(
        torch.equal(v, trainer.model.backbone.state_dict()[k])
        for k, v in teacher.ema.state_dict().items()
    )
    row = trainer.data["train"]
    args = (
        torch.zeros((2, 31, 2, 4)),
        torch.ones(2),
        row["features"][:2],
        row["force"][:2],
        trainer.worlds["train"][:2],
    )
    trainer.model.eval()
    with torch.inference_mode():
        actual = trainer.model(*args, {k: v[:2] for k, v in trainer.geometry["train"].items()})
        assert torch.equal(actual, teacher.ema(*args))
    wrong = copy.deepcopy(parent)
    wrong["angular_labels_sha256"] = "different"
    with pytest.raises(ValueError, match="teacher data"):
        trainer.initialize(path, wrong)


def test_full_training_gradients_and_exact_resume(tmp_path):
    trainer, _, _, _, _ = fixture()
    trainer.train_step()
    trainer.train_step()
    assert trainer.model.points[0].weight.grad.abs().sum() > 0
    assert trainer.model.backbone.output.weight.grad.abs().sum() > 0
    assert all(p.requires_grad for p in trainer.model.parameters())
    identity = {"source_sha256": surface_sources()}
    path = tmp_path / "surface.pt"
    trainer.save(path, identity)
    expected = trainer.train_step()
    restored, *_ = fixture()
    restored.resume(path, identity)
    assert restored.train_step() == expected
    assert all(
        torch.equal(v, restored.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )
    assert all(
        torch.equal(v, restored.ema.state_dict()[k]) for k, v in trainer.ema.state_dict().items()
    )
    restored.config = replace(restored.config, local_surfaces=False)
    with pytest.raises(ValueError, match="identical surface"):
        restored.resume(path, identity)


def test_strict_loading_and_observable_sampler_preserve_global_rng(tmp_path):
    trainer, _, _, _, obs = fixture()
    trainer.train_step()
    path = tmp_path / "surface.pt"
    trainer.save(path, {"source_sha256": surface_sources()})
    original = torch.get_rng_state().clone()
    model, state = load_surface(path)
    assert torch.equal(original, torch.get_rng_state())
    videos = sample_surface_videos(
        model,
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.2, 0], np.float32),
        seeds=(0, 1),
        sampling_steps=2,
    )
    assert videos.shape == (2, 32, 64, 64, 3) and videos.dtype == np.uint8
    assert np.array_equal(videos[:, 0], np.stack([obs.first] * 2))
    assert torch.equal(original, torch.get_rng_state())
    public, public_state = load_trajectory(path)
    assert public_state["trajectory_format_version"] == 5
    public_videos = sample_trajectory_videos(
        public,
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.2, 0], np.float32),
        seeds=(0, 1),
        sampling_steps=2,
    )
    assert np.array_equal(public_videos, videos)
    adapter = CheckpointModel("E9-fixture", path, trajectory_steps=2)
    assert adapter.metadata["training_variant"] == "surface_messages"
    assert adapter.metadata["local_surfaces"] is True
    actual = adapter.predict(
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.2, 0], np.float32),
        seeds=[0, 1],
        future_frames=31,
        cancelled=None,
    )
    assert np.array_equal(actual, videos)
    assert torch.equal(original, torch.get_rng_state())
    state["surface_contract"] = "other"
    torch.save(state, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_surface(path)


def test_surface_requires_observed_valid_geometry_and_conditioned_backbone():
    trainer, _, _, _, obs = fixture()
    geometry = tensor_geometry(obs)
    features = trainer.data["train"]["features"][:1]
    broken = copy.deepcopy(geometry)
    broken["axis_valid"][0, 0, 0] = False
    with pytest.raises(ValueError, match="contiguous"):
        surface_features(torch.zeros((1, 3, 2, 4)), features, broken)
    with pytest.raises(ValueError, match="conditioned"):
        SurfaceTrajectoryDiffusion(world_condition=False, relational=True, width=16, layers=1)
