"""Verify profiler configuration, allocation checks, and failure reporting."""

import json

import pytest
import torch
import yaml

from click2future import profile_pilot as profiler
from click2future.models.pilot_denoiser import PilotDenoiser
from click2future.profile_pilot import main


def test_required_slurm_allocation_is_enforced(monkeypatch, capsys):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert main(["--require-slurm"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "allocated Slurm job" in report["error"]


@pytest.fixture
def small_profile():
    return {
        "device": "cpu",
        "base_channels": 8,
        "batch_size": 1,
        "future_frames": 2,
        "resolution": 8,
        "warmup": 0,
        "steps": 1,
    }


def test_profile_uses_config_and_records_resolved_settings(tmp_path, capsys):
    config = {
        "data": {"resolution": 8, "future_frames": 2, "max_objects": 3},
        "model": {"base_channels": 8, "conditioning": "numeric", "auxiliary": True},
        "profile": {
            "batch_size": 1,
            "warmup_steps": 0,
            "measured_steps": 1,
            "precision": "float32",
        },
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config))
    destination = tmp_path / "reports" / "profile.json"
    assert main(["--config", str(path), "--device", "cpu", "--output", str(destination)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(destination.read_text())
    assert report["conditioning"] == "numeric"
    assert report["auxiliary"] is True
    assert report["max_objects"] == 3
    assert report["precision"] == "torch.float32"
    assert report["timing_stable"] is False
    model = PilotDenoiser(8, conditioning="numeric", auxiliary=True, max_objects=3)
    assert report["parameters"] == sum(parameter.numel() for parameter in model.parameters())


def test_cli_overrides_config_including_disabling_auxiliary(tmp_path, monkeypatch, capsys):
    config = {
        "data": {"resolution": 8, "future_frames": 2, "max_objects": 4},
        "model": {"base_channels": 8, "conditioning": "spatial", "auxiliary": True},
        "profile": {"batch_size": 1, "warmup_steps": 0, "measured_steps": 1, "precision": "auto"},
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(config))
    captured = {}

    def capture(**settings):
        captured.update(settings)
        return {"status": "passed"}

    monkeypatch.setattr(profiler, "profile", capture)
    assert (
        main(
            [
                "--config",
                str(path),
                "--device",
                "cpu",
                "--no-auxiliary",
                "--conditioning",
                "first_frame",
                "--max-objects",
                "2",
                "--precision",
                "float32",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert captured["auxiliary"] is False
    assert captured["conditioning"] == "first_frame"
    assert captured["max_objects"] == 2
    assert captured["precision"] == "float32"


def test_nonfinite_gradients_fail_instead_of_reporting_success(small_profile, monkeypatch):
    def model_with_bad_gradient(*args, **kwargs):
        model = PilotDenoiser(*args, **kwargs)
        model.stem.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        return model

    monkeypatch.setattr(profiler, "PilotDenoiser", model_with_bad_gradient)
    with pytest.raises(RuntimeError, match="Non-finite gradients"):
        profiler.profile(**small_profile)


def test_unsupported_cpu_precision_is_rejected(small_profile):
    with pytest.raises(ValueError, match="CPU profiling"):
        profiler.profile(**small_profile, precision="float16")
