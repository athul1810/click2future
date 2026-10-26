"""Temporal history, immutable backbone and public E8 inference contracts."""

import copy
import json
import runpy
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_contact_training import fixture_data

from click2future.demo.models import CheckpointModel
from click2future.models.world_motion import sample_world
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_memory import (
    MemoryConfig,
    MemoryTrainer,
    frozen_digest,
    source_identity,
)
from click2future.training.train_world import WorldConfig, WorldTrainer


def trainer(temporal_memory=True):
    data, geometry, obs = fixture_data()
    worlds = {key: sample_world(1).numpy().repeat(6, 0) for key in data}
    config = MemoryConfig(
        steps=4, width=16, layers=1, batch_size=2, memory_width=8, temporal_memory=temporal_memory
    )
    return MemoryTrainer(data, geometry, worlds, config), (data, geometry, worlds, obs)


def test_zero_residual_controls_share_every_tensor_and_reproduce_frozen_backbone():
    a, _ = trainer(False)
    b, _ = trainer(True)
    row = a.data["train"]
    noisy = torch.ones((2, 31, 2, 4)) * 0.1
    args = (
        noisy,
        torch.ones(2),
        row["features"][:2],
        row["force"][:2],
        sample_world(5).expand(2, -1, -1),
    )
    expected = a.model.backbone(*args)
    assert torch.equal(a.model(*args), expected)
    assert torch.equal(b.model(*args), expected)
    assert all(
        torch.equal(value, b.model.state_dict()[key]) for key, value in a.model.state_dict().items()
    )
    assert sum(p.numel() for p in a.model.parameters() if p.requires_grad) == sum(
        p.numel() for p in b.model.parameters() if p.requires_grad
    )


def test_history_changes_only_carried_residual_and_never_leaks_between_calls():
    carried, _ = trainer(True)
    model = carried.model
    with torch.no_grad():
        model.output.weight.normal_(generator=torch.Generator().manual_seed(13))
    reset = copy.deepcopy(model)
    reset.temporal_memory = False
    row = carried.data["train"]
    motion = torch.zeros((1, 31, 2, 4))
    args = (torch.ones(1), row["features"][:1], row["force"][:1], sample_world(4))
    before = model.residual(motion, *args)
    altered = motion.clone()
    altered[:, 7, :, 0] = 1
    after = model.residual(altered, *args)
    assert torch.equal(before[:, :7], after[:, :7])
    assert not torch.equal(before[:, 10], after[:, 10])
    assert torch.equal(reset.residual(motion, *args)[:, 10], reset.residual(altered, *args)[:, 10])
    assert torch.equal(before, model.residual(motion, *args))


def test_updates_and_resume_preserve_frozen_weights_and_exact_next_step(tmp_path):
    a, _ = trainer()
    identity = {"source_sha256": source_identity()}
    original = frozen_digest(a.model)
    a.train_step()
    a.train_step()
    assert all(p.grad is None for p in a.model.backbone.parameters())
    assert a.model.output.weight.grad.abs().sum() > 0
    assert a.model.memory.weight_ih_l0.grad.abs().sum() > 0
    a.save(tmp_path / "memory.pt", identity)
    b, _ = trainer()
    b.resume(tmp_path / "memory.pt", identity)
    assert a.train_step() == b.train_step()
    assert frozen_digest(a.model) == frozen_digest(a.ema) == original
    assert all(
        torch.equal(value, b.model.state_dict()[key]) for key, value in a.model.state_dict().items()
    )
    with torch.no_grad():
        next(a.model.backbone.parameters()).add_(1)
    with pytest.raises(ValueError, match="Frozen backbone"):
        a.save(tmp_path / "changed.pt", identity)


def test_e7_warm_start_checks_data_and_preserves_sampling(tmp_path):
    target, (data, geometry, worlds, obs) = trainer()
    teacher = WorldTrainer(
        data, geometry, worlds, WorldConfig(steps=4, width=16, layers=1, batch_size=2)
    )
    teacher.train_step()
    identity = {
        key: "fixture"
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
            "world_label_sha256",
        )
    }
    identity["source_sha256"] = source_identity()
    teacher.save(tmp_path / "teacher.pt", identity)
    target.initialize(tmp_path / "teacher.pt", identity)
    teacher_model, _ = load_trajectory(tmp_path / "teacher.pt")
    for key in ("force_limit", "sigma_min", "sigma_max"):
        setattr(target.model, key, getattr(target.config, key))
    expected = sample_trajectory_videos(
        teacher_model,
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.1, 0], np.float32),
        seeds=[0, 1],
        sampling_steps=2,
    )
    actual = sample_trajectory_videos(
        target.model,
        obs.first,
        obs.masks[obs.selected_index],
        np.array([0.1, 0], np.float32),
        seeds=[0, 1],
        sampling_steps=2,
    )
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="data differ"):
        target.initialize(tmp_path / "teacher.pt", {**identity, "world_label_sha256": "changed"})


def test_checkpoint_public_loader_demo_parity_rng_and_tamper_rejection(tmp_path):
    trained, (_, _, _, obs) = trainer()
    trained.train_step()
    identity = {"source_sha256": source_identity()}
    path = tmp_path / "memory.pt"
    trained.save(path, identity)
    original_rng = torch.get_rng_state().clone()
    model, state = load_trajectory(path)
    assert torch.equal(original_rng, torch.get_rng_state())
    mask, force = obs.masks[obs.selected_index], np.array([0.1, 0], np.float32)
    offline = sample_trajectory_videos(
        model, obs.first, mask, force, seeds=[0, 1], sampling_steps=2
    )
    adapter = CheckpointModel("fixture", path, trajectory_steps=2)
    online = adapter.predict(
        obs.first, mask, force, seeds=[0, 1], future_frames=31, cancelled=threading.Event()
    )
    np.testing.assert_array_equal(online, offline)
    assert adapter.metadata["training_variant"] == "temporal_residual"
    assert state["trajectory_format_version"] == 4
    state["frozen_backbone_sha256"] = "changed"
    torch.save(state, path)
    with pytest.raises(ValueError, match="Frozen backbone"):
        load_trajectory(path)


def test_invalid_memory_configuration_is_rejected():
    for changed in ({"memory_width": True}, {"temporal_memory": 1}, {"world_condition": False}):
        with pytest.raises(ValueError):
            replace(MemoryConfig(), **changed).validate()


def test_memory_recipes_reject_a_second_changed_factor(tmp_path):
    matched = runpy.run_path("submission/memory_comparison.py")["matched_configs"]
    paths = matched(Path.cwd())
    folder = tmp_path / "configs/training"
    folder.mkdir(parents=True)
    for path in paths.values():
        (folder / path.name).write_bytes(path.read_bytes())
    matched(tmp_path)
    path = folder / "memory_carried.json"
    config = json.loads(path.read_text())
    config["learning_rate"] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="differ only"):
        matched(tmp_path)
