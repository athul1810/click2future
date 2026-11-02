"""Format-5 evaluation identity, native replay and matched training recipes."""

import json
import runpy
from pathlib import Path

import pytest
from test_surface_motion import fixture

from click2future.data import generate_dataset
from click2future.evaluation.suite import _write, prepare_plan, run_suite
from click2future.evaluation.trajectory_audit import audit
from click2future.evaluation.trajectory_pilot import sample_plan
from click2future.training.train_surface import surface_sources

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("changed", [None, "steps", "local_surfaces"])
def test_surface_recipes_cannot_silently_change_the_control(tmp_path, changed):
    folder = tmp_path / "configs/training"
    folder.mkdir(parents=True)
    for name in ("local", "pooled"):
        values = json.loads((ROOT / f"configs/training/surface_{name}.json").read_text())
        if name == "pooled" and changed:
            values[changed] = values[changed] + 1 if changed == "steps" else True
        (folder / f"surface_{name}.json").write_text(json.dumps(values))
    namespace = runpy.run_path(str(ROOT / "submission/surface_comparison.py"))
    if changed:
        with pytest.raises(ValueError, match="only in local_surfaces"):
            namespace["matched_configs"](tmp_path)
    else:
        assert set(namespace["matched_configs"](tmp_path)) == {"local", "pooled"}


def test_surface_evaluation_replays_all_pixels_and_refuses_e7_alias(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(
        dataset,
        group_count=10,
        variants_per_group=1,
        seed=992100,
        config={"frame_count": 32, "resolution": 64, "fps": 8},
    )
    plan = prepare_plan(
        dataset, ROOT / "configs/evaluation/pixel.yaml", split="validation", metrics=("ssim",)
    )
    plan_path = tmp_path / "plan.json"
    _write(plan_path, plan)
    trainer, *_ = fixture()
    trainer.train_step()
    checkpoint = tmp_path / "fixture.pt"
    trainer.save(
        checkpoint,
        {
            "source_sha256": surface_sources(),
            "dataset_manifest_sha256": plan["input_sha256"]["dataset_manifest"],
            "split_manifest_sha256": plan["input_sha256"]["splits"],
        },
    )
    with pytest.raises(ValueError, match="variant and requested method"):
        sample_plan(plan_path, checkpoint, tmp_path / "alias", method="E7")
    assert not (tmp_path / "alias").exists()
    predictions = tmp_path / "predictions"
    sampled = sample_plan(plan_path, checkpoint, predictions, method="E9")
    assert sampled["runs"][0]["method"] == "E9"
    metrics = run_suite(
        plan_path, predictions / "predictions.json", tmp_path / "metrics", bootstrap=2
    )
    assert metrics["methods"][0]["method"] == "E9"
    replay = audit(
        plan_path, predictions / "predictions.json", dataset, checkpoint, tmp_path / "geometry"
    )
    assert replay["method"] == "E9"
    assert all(row["pixel_exact_pose_replay"] for row in replay["cases"])
