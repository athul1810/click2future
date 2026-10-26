"""Fresh refinement must retain teacher identity and paired sampling contracts."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from test_world_motion import setup_trainer

from click2future.training.train_coverage import CoverageTrainer
from click2future.training.train_world import source_identity


def test_refinement_preserves_initial_teacher_and_rejects_other_parent(tmp_path):
    teacher, data, geometry, _ = setup_trainer()
    teacher.train_step()
    parent = {
        key: "unit-source"
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
        source_sha256=source_identity(),
    )
    path = tmp_path / "teacher.pt"
    teacher.save(path, parent)
    provenance = {**parent, "coverage_addition": {"parent_provenance": copy.deepcopy(parent)}}
    trainer = CoverageTrainer(
        data, geometry, {k: v.cpu().numpy() for k, v in teacher.worlds.items()}, teacher.config
    )
    rng = trainer.generator.get_state().clone()
    trainer.initialize(path, provenance)
    assert not trainer.optimizer.state and trainer.step == 0
    assert torch.equal(rng, trainer.generator.get_state())
    assert all(
        torch.equal(v, trainer.model.state_dict()[k])
        and torch.equal(v, trainer.ema.state_dict()[k])
        for k, v in teacher.ema.state_dict().items()
    )
    assert all(p.requires_grad for p in trainer.model.parameters())
    changed = copy.deepcopy(provenance)
    changed["coverage_addition"]["parent_provenance"]["angular_labels_sha256"] = "different"
    with pytest.raises(ValueError, match="teacher data"):
        trainer.initialize(path, changed)
    changed = copy.deepcopy(provenance)
    changed["geometry_sha256"]["validation"] = "different"
    with pytest.raises(ValueError, match="Validation labels"):
        trainer.initialize(path, changed)


def test_comparison_rejects_unequal_parameters_rng_or_sampling():
    spec = importlib.util.spec_from_file_location(
        "coverage_comparison", Path("submission/coverage_comparison.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {"steps": 8000}
    addition = {
        key: "same"
        for key in (
            "manifest_sha256",
            "base_cache_manifest_sha256",
            "base_cache_arrays_sha256",
            "parent_provenance",
            "examples_added",
        )
    }
    row = {
        "status": "completed",
        "steps_completed": 8000,
        "config": config,
        "provenance": {"coverage_addition": addition},
        **{
            key: "same"
            for key in (
                "parameter_count",
                "trainable_parameter_count",
                "initial_model_tensors_sha256",
                "initial_private_rng_sha256",
                "sampling_weights_sha256",
            )
        },
    }
    models = {arm: {"training": copy.deepcopy(row)} for arm in ("narrow", "broad")}
    module.check_training(models, config)
    for key in (
        "parameter_count",
        "initial_model_tensors_sha256",
        "initial_private_rng_sha256",
        "sampling_weights_sha256",
    ):
        broken = copy.deepcopy(models)
        broken["broad"]["training"][key] = "changed"
        with pytest.raises(ValueError, match="Unmatched"):
            module.check_training(broken, config)
