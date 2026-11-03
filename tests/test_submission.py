"""Submission restrictions must hold even when a working tree contains artifacts."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from click2future.submission import build_archive, inspect_archive


@pytest.fixture
def source_copy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    target = tmp_path / "source"
    target.mkdir()
    for name in ("src", "configs", "tests", "submission", "cluster"):
        shutil.copytree(root / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in (".python-version", "pyproject.toml"):
        shutil.copyfile(root / name, target / name)
    (target / "scripts").mkdir()
    shutil.copyfile(root / "scripts/build_demo_bundle.py", target / "scripts/build_demo_bundle.py")
    if (root / "uv.lock").is_file():
        shutil.copyfile(root / "uv.lock", target / "uv.lock")
    return target


def test_deterministic_source_archive_extracts_and_replays_without_working_data(
    source_copy, tmp_path
):
    for name in ("AGENTS.md", ".env", "checkpoints/model.pth", "data/saved.npz", ".git/config"):
        path = source_copy / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private working artifact")
    archive = tmp_path / "group_07_code.zip"
    first = build_archive(source_copy, archive)
    assert build_archive(source_copy, archive)["archive_sha256"] == first["archive_sha256"]
    folder = tmp_path / "extracted"
    with ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert not names & {"AGENTS.md", ".env", "uv.lock", "data/saved.npz", ".git/config"}
        assert "src/click2future/models/pilot_denoiser.py" in names
        assert "src/click2future/training/auxiliary.py" in names
        assert "src/click2future/training/train_diffusion.py" in names
        assert "configs/training/pilot_diffusion.json" in names
        assert "submission/train_diffusion.sbatch" in names
        assert "src/click2future/evaluation/suite.py" in names
        assert "src/click2future/evaluation/sample_suite.py" in names
        assert "submission/sample_evaluation.sbatch" in names
        assert "submission/train_pilot.sbatch" in names
        assert "submission/evaluate_pilot.sbatch" in names
        assert "submission/train_trajectory.sbatch" in names
        assert "submission/train_contact.sbatch" in names
        assert "submission/wall_comparison.py" in names
        assert "submission/wall_comparison.sbatch" in names
        assert "submission/contact_risk_comparison.py" in names
        assert "submission/contact_risk_comparison.sbatch" in names
        assert "submission/world_comparison.py" in names
        assert "submission/world_comparison.sbatch" in names
        assert "submission/memory_comparison.py" in names
        assert "submission/memory_comparison.sbatch" in names
        assert "submission/coverage_comparison.py" in names
        assert "submission/coverage_comparison.sbatch" in names
        assert "submission/trajectory_response.py" in names
        assert "submission/routing_comparison.py" in names
        assert "submission/routing_comparison.sbatch" in names
        assert "cluster/replication_evaluation.py" in names
        assert "cluster/replication_evaluation.sbatch" in names
        assert "src/click2future/models/routed_motion.py" in names
        assert "configs/training/routing/numeric-seed19.json" in names
        assert "src/click2future/models/contact_memory.py" in names
        assert "src/click2future/training/contact_geometry.py" in names
        assert "src/click2future/training/train_contact.py" in names
        assert "configs/training/contact_pilot.json" in names
        assert "submission/evaluate_trajectory.sbatch" in names
        assert "submission/audit_trajectory.sbatch" in names
        assert "src/click2future/models/object_motion.py" in names
        assert "src/click2future/training/sample_trajectory.py" in names
        assert "src/click2future/evaluation/trajectory_pilot.py" in names
        assert "configs/training/trajectory_pilot.json" in names
        assert "submission/sampler_study.py" in names
        assert "submission/study_sampler.sbatch" in names
        assert "submission/flow_evaluation.py" in names
        assert "submission/cvae_evaluation.py" in names
        assert "submission/cvae_force_probe.py" in names
        assert "submission/cvae_motion_timing.py" in names
        assert "submission/cvae_selection_drift.py" in names
        assert "submission/check_e4_collision.py" in names
        assert "submission/check_e4_collision.sbatch" in names
        assert "submission/pilot_image_metrics.py" in names
        assert "cluster/final_evaluation.py" in names
        assert "cluster/diffusion_long.sbatch" in names
        assert "cluster/README.md" not in names
        assert "src/click2future/models/cvae_flow.py" in names
        assert "src/click2future/evaluation/pilot.py" in names
        assert "src/click2future/training/multiscene_pilot.py" in names
        assert "src/click2future/demo/static/index.html" in names
        assert "src/click2future/demo/static/app.js" in names
        assert "src/click2future/demo/static/style.css" in names
        bundle.extractall(folder)
    child = subprocess.run(
        [sys.executable, "-I", str(folder / "submission/smoke.py")],
        cwd=folder,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(child.stdout)["generated_rollouts"] == 2
    assert json.loads(child.stdout)["source_root"] == str(folder)
    # The URL-free dependency recipe also permits rebuilding the extracted source.
    assert build_archive(folder, archive)["archive_sha256"] == first["archive_sha256"]


@pytest.mark.parametrize("bad", ["link", "weight", "symlink", "secret"])
def test_unexpected_source_content_is_rejected_before_overwrite(source_copy, tmp_path, bad):
    output = tmp_path / "group_07_code.zip"
    output.write_bytes(b"keep previous archive")
    if bad == "link":
        (source_copy / "src/click2future/bad.py").write_text(
            "# source: " + "https" + ":" + "/" * 2 + "example.invalid/file\n"
        )
    elif bad == "secret":
        (source_copy / "src/click2future/bad.py").write_text("# " + "ghp_" + "a" * 36 + "\n")
    elif bad == "symlink":
        (source_copy / "src/click2future/bad.py").symlink_to(tmp_path / "outside.py")
        (tmp_path / "outside.py").write_text("secret")
    else:
        path = source_copy / "submission/allowlist.json"
        recipe = json.loads(path.read_text())
        recipe["include"].append("checkpoints/*.pth")
        path.write_text(json.dumps(recipe))
        (source_copy / "checkpoints").mkdir()
        (source_copy / "checkpoints/model.pth").write_bytes(b"model")
    with pytest.raises(ValueError):
        build_archive(source_copy, output)
    assert output.read_bytes() == b"keep previous archive"


@pytest.mark.parametrize("corruption", ["changed", "extra", "traversal", "duplicate"])
def test_inspector_rejects_tampered_archives(source_copy, tmp_path, corruption):
    output = tmp_path / "group_07_code.zip"
    build_archive(source_copy, output)
    with ZipFile(output) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    if corruption == "changed":
        files["README.md"] += b"changed"
    elif corruption == "extra":
        files["extras.txt"] = b"unlisted source"
    elif corruption == "traversal":
        files["../outside.py"] = b"outside"
    with ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        if corruption == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("README.md", b"duplicate")
    with pytest.raises(ValueError):
        inspect_archive(output)
