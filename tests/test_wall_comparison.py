"""A single loss switch isolates the declared paired training experiment."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("wall_comparison", "submission/wall_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pair_rejects_unmatched_training_budget_or_sampler_inputs(tmp_path):
    configs = module.paired_configs(Path.cwd())
    target = tmp_path / "configs/training"
    target.mkdir(parents=True)
    for path in configs.values():
        (target / path.name).write_bytes(path.read_bytes())
    wall = target / configs["wall"].name
    config = json.loads(wall.read_text())
    config["steps"] += 1
    wall.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="only"):
        module.paired_configs(tmp_path)


def test_comparison_requires_tc2_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="scheduled TC2"):
        module.main(["--dataset", str(tmp_path), "--output", str(tmp_path / "results")])
    assert not (tmp_path / "results").exists()
