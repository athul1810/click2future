"""VPN retry cannot submit a second job or discard immutable experiment identity."""

import importlib.util
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def submitter(monkeypatch):
    pytest.importorskip("fcntl", reason="The submission helper runs on the Linux TC2 host")
    spec = importlib.util.spec_from_file_location("submit_once", ROOT / "cluster/submit_once.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("C2F_SOURCE_REVISION", "unit-committed-source")
    return module


def test_retry_after_lost_response_adopts_existing_job_without_resubmission(tmp_path, monkeypatch):
    module = submitter(monkeypatch)
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\nexit 0\n")
    queue, calls = [], []

    def command(args, **kwargs):
        calls.append(args[0])
        if args[0] == "squeue":
            return "\n".join(queue)
        queue.append("99123")
        # Simulate an accepted job whose acknowledgement is lost.
        raise subprocess.CalledProcessError(255, args)

    monkeypatch.setattr(module.subprocess, "check_output", command)
    with pytest.raises(subprocess.CalledProcessError):
        module.submit_once(tmp_path / "run", script, "c2f-unit")
    receipt = module.submit_once(tmp_path / "run", script, "c2f-unit")
    assert receipt["job_id"] == "99123" and receipt["recovered_from_queue"]
    assert calls.count("sbatch") == 1
    assert module.submit_once(tmp_path / "run", script, "c2f-unit") == receipt
    assert calls.count("sbatch") == 1
    monkeypatch.setenv("C2F_SOURCE_REVISION", "different-source")
    with pytest.raises(ValueError, match="different immutable experiment"):
        module.submit_once(tmp_path / "run", script, "c2f-unit")
    assert json.loads((tmp_path / "run/submission.json").read_text()) == receipt


def test_successful_receipt_is_atomic_and_retry_does_not_contact_slurm(tmp_path, monkeypatch):
    module = submitter(monkeypatch)
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\nexit 0\n")
    answers = iter(["", "99124;slurm\n"])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(answers))
    receipt = module.submit_once(tmp_path / "run", script, "c2f-unit")
    assert receipt["job_id"] == "99124" and not receipt["recovered_from_queue"]
    assert not (tmp_path / "run/submission.tmp").exists()
    assert module.submit_once(tmp_path / "run", script, "c2f-unit") == receipt


@pytest.mark.parametrize("change", [None, "steps", "risk_penetration_weight", "world_prior_only"])
def test_expected_recipes_isolate_loss_variance_without_other_changes(tmp_path, change):
    shutil.copytree(ROOT / "configs/training", tmp_path / "configs/training")
    path = tmp_path / "configs/training/prior_expected_expected.json"
    if change:
        value = json.loads(path.read_text())
        value[change] = False if change == "world_prior_only" else value[change] + 1
        path.write_text(json.dumps(value))
    recipes = runpy.run_path(str(ROOT / "submission/expected_prior_comparison.py"))[
        "matched_configs"
    ]
    if change:
        with pytest.raises(ValueError):
            recipes(tmp_path)
    else:
        assert set(recipes(tmp_path)) == {"control", "expected"}
