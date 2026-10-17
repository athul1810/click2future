"""Checkpoint-to-metrics wiring using explicit untrained fixtures, never training."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from click2future.data import generate_dataset, load_rollout
from click2future.evaluation.sample_suite import generate_samples, main
from click2future.evaluation.suite import prepare_plan, run_suite
from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.train_cvae import dataset_index, save_checkpoint
from click2future.training.train_diffusion import TrainConfig, Trainer


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("sampling-fixtures")
    dataset = root / "data"
    generate_dataset(
        dataset,
        group_count=10,
        variants_per_group=2,
        seed=211000,
        config={"frame_count": 4, "resolution": 64, "fps": 8},
    )
    paths = {}
    for method, condition, auxiliary in (
        ("E1", "first_frame", False),
        ("E2", "numeric", False),
        ("E3", "spatial", False),
        ("E4", "spatial", True),
    ):
        trainer = Trainer(
            dataset,
            TrainConfig(
                base_channels=8,
                diffusion_steps=10,
                conditioning=condition,
                auxiliary=auxiliary,
                seed=3,
                limit_train=2,
            ),
        )
        paths[method] = root / (method + ".pt")
        trainer.save(paths[method])
    config = yaml.safe_load(Path("configs/training/cvae.yaml").read_text())
    config["data"]["total_frames"] = 4
    config["model"].update(future_frames=3, base_channels=8, latent_dim=4)
    config["training"]["seed"] = 3
    torch.manual_seed(3)
    model = ConditionalVideoVAE(**config["model"])
    _, _, provenance = dataset_index(dataset)
    paths["E0"] = root / "E0.pt"
    save_checkpoint(
        paths["E0"],
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        state={"step": 0},
        config=config,
        provenance=provenance,
        train_generator=torch.Generator().manual_seed(4),
        shuffle_generator=torch.Generator().manual_seed(5),
    )
    return dataset, paths


@pytest.fixture
def prepared(checkpoints, tmp_path):
    dataset, paths = checkpoints
    config = Path("configs/evaluation/pixel.yaml")
    plan = prepare_plan(dataset, config, metrics=("ssim",))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    spec = {
        "sampling_steps": 4,
        "runs": [
            {"method": method, "run_id": "untrained-" + method, "checkpoint": str(path)}
            for method, path in sorted(paths.items())
        ],
    }
    spec_path = tmp_path / "runs.json"
    spec_path.write_text(json.dumps(spec))
    return plan_path, spec_path, plan, spec


def test_all_five_methods_reach_all_metric_families_without_rng_drift(prepared, tmp_path):
    plan_path, spec_path, plan, _ = prepared
    output = tmp_path / "samples"
    rng = torch.get_rng_state().clone()
    record = generate_samples(plan_path, spec_path, output, allow_untrained=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert record["status"] == "passed" and len(record["runs"]) == 5
    for run in record["runs"]:
        assert run["training_steps"] == 0 and run["untrained_fixture"]
        assert run["training_seed"] == 3
        assert set(run["videos"]) == {case["sample_id"] for case in plan["cases"]}
        for case in plan["cases"]:
            with np.load(output / run["videos"][case["sample_id"]], allow_pickle=False) as archive:
                videos = archive["videos"]
            assert videos.shape == (4, 4, 64, 64, 3) and videos.dtype == np.uint8
            reference = load_rollout(plan["dataset"], case["sample_id"])
            np.testing.assert_array_equal(
                videos[:, 0], np.broadcast_to(reference["frames"][0], (4, 64, 64, 3))
            )
    report = run_suite(plan_path, output / "predictions.json", tmp_path / "results", bootstrap=20)
    assert {row["method"] for row in report["methods"]} == {"E0", "E1", "E2", "E3", "E4"}
    assert len(report["provenance"]) == 15 * len(plan["cases"])
    assert len(report["untrained_fixture_runs"]) == 5


def test_untrained_checkpoint_requires_explicit_fixture_opt_in(prepared, tmp_path):
    plan_path, spec_path, _, _ = prepared
    with pytest.raises(ValueError, match="allow-untrained-fixtures"):
        generate_samples(plan_path, spec_path, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize(
    "mutation",
    ["method", "dataset", "force_limit", "target", "normalization", "duplicate", "loss_weight"],
)
def test_incompatible_runs_fail_before_creating_outputs(prepared, tmp_path, mutation):
    plan_path, spec_path, _, spec = prepared
    changed = copy.deepcopy(spec)
    if mutation == "method":
        changed["runs"][2]["method"] = "E3"
    elif mutation == "duplicate":
        extra = copy.deepcopy(changed["runs"][2])
        extra["run_id"] = "duplicate-E2"
        changed["runs"].append(extra)
    else:
        index = 0 if mutation == "normalization" else 3
        state = torch.load(changed["runs"][index]["checkpoint"], weights_only=True)
        if mutation == "dataset":
            state["provenance"]["file_hashes"]["dataset_manifest.json"] = "0" * 64
        elif mutation == "force_limit":
            state["config"]["force_limit"] = 0.01
        elif mutation == "target":
            state["config"]["prediction_type"] = "velocity"
        elif mutation == "loss_weight":
            state["config"]["foreground_weight"] = 32.0
        else:
            state["config"]["data"]["force_scale"] = 1.0
        checkpoint = tmp_path / "changed.pt"
        torch.save(state, checkpoint)
        changed["runs"][index]["checkpoint"] = str(checkpoint)
    spec_path.write_text(json.dumps(changed))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        generate_samples(plan_path, spec_path, output, allow_untrained=True)
    assert not output.exists()


def test_mid_run_failure_never_publishes_a_prediction_manifest(prepared, tmp_path, monkeypatch):
    from click2future.evaluation import sample_suite

    plan_path, spec_path, _, _ = prepared
    output = tmp_path / "failed"

    def failed(*_, **__):
        raise RuntimeError("injected sampler failure")

    monkeypatch.setattr(sample_suite, "load_generator", failed)
    with pytest.raises(RuntimeError, match="injected"):
        generate_samples(plan_path, spec_path, output, allow_untrained=True)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    assert not (output / "predictions.json").exists()


@pytest.mark.parametrize("changed", ["checkpoint", "video"])
def test_evaluator_refuses_artifacts_overwritten_after_sampling(prepared, tmp_path, changed):
    plan_path, spec_path, _, spec = prepared
    state = torch.load(spec["runs"][0]["checkpoint"], weights_only=True)
    checkpoint = tmp_path / "local.pt"
    torch.save(state, checkpoint)
    spec["runs"] = [{**spec["runs"][0], "checkpoint": str(checkpoint)}]
    spec_path.write_text(json.dumps(spec))
    output = tmp_path / "samples"
    record = generate_samples(plan_path, spec_path, output, allow_untrained=True)
    path = (
        checkpoint
        if changed == "checkpoint"
        else output / next(iter(record["runs"][0]["videos"].values()))
    )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after sampling"):
        run_suite(plan_path, output / "predictions.json", tmp_path / "invalid-results")
    assert not (tmp_path / "invalid-results").exists()


def test_cuda_requires_tc2_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit):
        main(
            [
                "--plan",
                str(tmp_path / "missing.json"),
                "--runs",
                str(tmp_path / "runs.json"),
                "--output-dir",
                str(tmp_path / "output"),
                "--device",
                "cuda",
            ]
        )
    assert not (tmp_path / "output").exists()
