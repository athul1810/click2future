"""Dependency, artifact-I/O and CPU/CUDA checks for the environment diagnostic."""

import json

import torch

from click2future.env_check import main


def test_cpu_check_writes_a_readable_report(tmp_path, capsys):
    destination = tmp_path / "reports" / "cpu.json"
    assert main(["--device", "cpu", "--output", str(destination)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(destination.read_text(encoding="utf-8"))
    assert report["device"] == "cpu"
    assert report["status"] == "passed"
    assert report["checks"]["cpu_forward_backward_checkpoint"] == "passed"


def test_explicit_cuda_request_fails_when_cuda_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["--device", "cuda"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "failed"
    assert "CUDA requested but unavailable" in report["error"]
