"""Verify that the cluster-only profiling option refuses a login-node execution."""

import json

from click2future.profile_pilot import main


def test_required_slurm_allocation_is_enforced(monkeypatch, capsys):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert main(["--require-slurm"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "allocated Slurm job" in report["error"]
