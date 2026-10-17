"""Run the paired diagnostic through real training, saved weights and RGB scoring."""

import json
from dataclasses import asdict

import pytest

from click2future.data import generate_dataset
from click2future.training.compare_representation import main
from click2future.training.train_diffusion import TrainConfig


def test_paired_diagnostic_keeps_controls_and_reports_copy_baseline(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(
        dataset,
        group_count=8,
        variants_per_group=2,
        seed=226000,
        config={"resolution": 16, "frame_count": 4, "fps": 8},
    )
    config = TrainConfig(steps=2, base_channels=8, diffusion_steps=32, limit_train=1)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(asdict(config)))
    output = tmp_path / "comparison"
    main(["--dataset", str(dataset), "--output-dir", str(output), "--config", str(cfg)])
    report = json.loads((output / "comparison.json").read_text())
    assert report["status"] == "measured_training_only_diagnostic"
    assert report["copy_frame0_baseline"]["all_objects"]["complete_samples"] == 4
    first, second = report["runs"]
    assert first["sample_id"] == second["sample_id"]
    assert first["training"]["provenance"] == second["training"]["provenance"]
    for run in (first, second):
        assert run["training"]["step"] == 2
        assert (output / run["representation"] / "videos.npz").is_file()
        assert run["scores"]["all_objects"]["total_samples"] == 4
    a, b = (run["training"]["config"] for run in report["runs"])
    assert a.pop("representation") == "rgb"
    assert b.pop("representation") == "frame0_residual"
    assert a == b
    with pytest.raises(SystemExit):
        main(["--dataset", str(dataset), "--output-dir", str(output)])
