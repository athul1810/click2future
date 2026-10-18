"""The sampler study fixes its subset without reading generated model outputs."""

import json
import runpy

import pytest

from click2future.data.generalization import generate_generalization_dataset
from click2future.evaluation.suite import prepare_plan, read_plan
from click2future.training.multiscene_pilot import PROJECT


def test_sampler_subset_is_fixed_and_keeps_scene_group_bindings(tmp_path):
    prepare_study = runpy.run_path(str(PROJECT / "submission/sampler_study.py"))["prepare_study"]
    counts = dict.fromkeys(
        ("train", "validation", "id_test", "shape_ood", "texture_ood", "force_ood", "physics_test"),
        1,
    )
    counts["validation"] = 5
    dataset = tmp_path / "data"
    generate_generalization_dataset(dataset, seed=940000, group_counts=counts)
    parent = prepare_plan(
        dataset,
        PROJECT / "configs/evaluation/pixel.yaml",
        split="validation",
        metrics=("ssim",),
        protocol=PROJECT / "configs/evaluation/comparison.yaml",
        actions_config=PROJECT / "configs/evaluation/actions.yaml",
    )
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(parent))
    plan = prepare_study(path)
    assert len(plan["cases"]) == 16
    assert len(plan["sampler_study"]["excluded_cases"]) == 14
    groups = sorted({c["scene_group_id"] for c in parent["cases"]})[:4]
    assert sorted({c["scene_group_id"] for c in plan["cases"]}) == groups
    assert sorted(g["scene_group_id"] for g in plan["action_groups"]) == groups
    assert plan["files"] == parent["files"]
    assert plan["sampler_study"]["parent_plan_sha256"] == parent["plan_sha256"]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(plan))
    assert read_plan(path) == plan
    held_out = prepare_plan(
        dataset,
        PROJECT / "configs/evaluation/pixel.yaml",
        split="id_test",
        metrics=("ssim",),
        protocol=PROJECT / "configs/evaluation/comparison.yaml",
    )
    path.write_text(json.dumps(held_out))
    with pytest.raises(ValueError, match="validation"):
        prepare_study(path)
