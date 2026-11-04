"""Strict low-memory checkpoint assignment, separate from model quality."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / "cluster/streamforce_runtime.py"
spec = importlib.util.spec_from_file_location("streamforce_runtime", path)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_meta_assignment_preserves_the_published_tensor_cast_and_forward():
    original = torch.nn.Linear(3, 2)
    state = original.state_dict()
    with torch.device("meta"):
        target = torch.nn.Linear(3, 2)
    assert runtime.assign_state(target, state, device="cpu", dtype=torch.bfloat16) == 2
    assert not any(p.is_meta for p in target.parameters())
    reference = original.to(torch.bfloat16)
    inputs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    assert torch.equal(reference(inputs), target(inputs))


def test_loading_rejects_partial_wrong_shape_and_ambiguous_wrappers():
    module = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        runtime.assign_state(module, {}, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Invalid"):
        runtime.assign_state(
            module,
            {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)},
            device="cpu",
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="collide"):
        runtime.clean_state({"weight": torch.zeros(1), "_orig_mod.weight": torch.zeros(1)})


def test_worker_rechecks_verified_artifacts_before_loading_external_code(tmp_path):
    from click2future.demo.photo_service import sha256, write_json

    spec = importlib.util.spec_from_file_location(
        "photo_worker", Path(__file__).resolve().parents[1] / "cluster/photo_worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    files = {}
    for name in [
        "weights/CogVideoX-5b-I2V/model_index.json",
        "checkpoints/step-5000-checkpoint-point-force.pt",
        "src/force-prompting/example.py",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"small integrity fixture, not model weights")
        files[name] = {"sha256": sha256(path)}
    revision = "ef3d0f847f787552b5ce004906c4944292e58996"
    source = {"revision": revision, "files_sha256": {name: files[name]["sha256"]}}
    write_json(tmp_path / "SOURCE.json", source)
    manifest = {
        "status": "passed",
        "source_revision": revision,
        "revisions": {
            "THUDM/CogVideoX-5b-I2V": "a6f0f4858a8395e7429d82493864ce92bf73af11",
            "brown-palm/force-prompting": "bc1a3faa381a8c935ebce980a18afb597d4a15c4",
        },
        "files": {k: v for k, v in files.items() if k.startswith(("weights/", "checkpoints/"))},
    }
    write_json(tmp_path / "ARTIFACTS.json", manifest)
    assert worker.verify_worker_artifacts(tmp_path) == manifest
    path.write_bytes(b"changed external code")
    with pytest.raises(ValueError, match="source changed"):
        worker.verify_worker_artifacts(tmp_path)
    write_json(tmp_path / "ARTIFACTS.json", {**manifest, "status": "failed"})
    with pytest.raises(ValueError, match="Verify"):
        worker.verify_worker_artifacts(tmp_path)
    assert json.loads((tmp_path / "ARTIFACTS.json").read_text())["status"] == "failed"
