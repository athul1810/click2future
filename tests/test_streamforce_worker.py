"""Native gating and real-process cancellation, using explicitly fake media."""

import base64
import importlib.util
import io
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from click2future.demo.photo_service import PhotoService, read_json, sha256, write_json

path = Path(__file__).resolve().parents[1] / "cluster/streamforce_worker.py"
spec = importlib.util.spec_from_file_location("streamforce_worker", path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("Worker fixture did not reach its declared state")
        time.sleep(0.02)


def test_gate_rejects_incomplete_native_probe_before_serving(tmp_path):
    write_json(tmp_path / "ARTIFACTS.json", {"status": "passed"})
    write_json(tmp_path / "probe.json", {"status": "running", "seeds": [0, 1, 2, 3]})
    with pytest.raises(ValueError, match="passed native"):
        worker.verify_native_probe(tmp_path, tmp_path / "probe.json")


@pytest.mark.parametrize("changed", [None, "seed", "shape", "finite", "file"])
def test_gate_binds_complete_native_contract_and_artifact_bytes(tmp_path, changed):
    names = [
        "checkpoints/streamforce_v1_ckpt.pt",
        "wan_models/Wan2.2-TI2V-5B/config.json",
        "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json",
    ]
    files = {}
    for name in names:
        item = tmp_path / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(b"artifact identity fixture; no model tensors")
        files[name] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
    manifest = {
        "status": "passed",
        "source_revision": "57a15e95eca25bbbb32d6ac6322ba3e12fd45083",
        "files": files,
    }
    write_json(tmp_path / "ARTIFACTS.json", manifest)
    probe = {
        "status": "passed_functional_probe",
        "seeds": [0, 1, 2, 3],
        "artifacts_manifest_sha256": sha256(tmp_path / "ARTIFACTS.json"),
        "loader_sha256": sha256(path.with_name("streamforce_runtime.py")),
        "runs": [
            {"seed": seed, "shape": [81, 480, 832, 3], "native_float_output_finite": True}
            for seed in [0, 1, 2, 3, 0]
        ],
    }
    if changed == "seed":
        probe["runs"][-1]["seed"] = 9
    elif changed == "shape":
        probe["runs"][0]["shape"][0] = 49
    elif changed == "finite":
        probe["runs"][0]["native_float_output_finite"] = False
    elif changed == "file":
        (tmp_path / names[0]).write_bytes(b"changed")
    write_json(tmp_path / "probe.json", probe)
    if changed:
        with pytest.raises(ValueError):
            worker.verify_native_probe(tmp_path, tmp_path / "probe.json")
    else:
        assert worker.verify_native_probe(tmp_path, tmp_path / "probe.json") == probe


@pytest.mark.parametrize("cancel", [True, False, "complete"])
def test_parent_retains_a_finished_sample_and_reaps_failed_or_cancelled_child(
    tmp_path, monkeypatch, cancel
):
    binary = tmp_path / "attention-fixture.so"
    binary.write_bytes(b"native gate fixture, not an actual CUDA binary")
    probe = {
        "model": {"backend": "queue fixture only"},
        "versions": {},
        "flash_attention_binary_sha256": sha256(binary),
    }
    monkeypatch.setattr(worker, "verify_native_probe", lambda *_: probe)
    original_find = worker.importlib.util.find_spec
    monkeypatch.setattr(
        worker.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(origin=str(binary))
            if name == "flash_attn_2_cuda"
            else original_find(name)
        ),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    (tmp_path / "repo").mkdir()
    write_json(tmp_path / "repo/ARTIFACTS.json", {})
    write_json(tmp_path / "probe.json", probe)
    current = worker.StreamForceWorker(
        tmp_path / "repo",
        tmp_path / "spool",
        tmp_path / "probe.json",
        raw_root=tmp_path / "raw",
        idle_seconds=60,
    )
    original_popen = subprocess.Popen
    children = []

    def fixture_child(command, **kwargs):
        request_path = command[command.index("--child-request") + 1]
        program = """
import hashlib,json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);root=p.parent;request=json.loads(p.read_text())
data=b'queue fixture bytes; not generated RGB'
(root/'0.mp4').write_bytes(data)
row={'file':'0.mp4','seed':request['seeds'][0],'sha256':hashlib.sha256(data).hexdigest()}
attempt={'complete':False,'request':request,'videos':[row],'scope':'queue fixture only'}
(root/'attempt.json').write_text(json.dumps(attempt))
if sys.argv[2]=='cancel':time.sleep(60)
elif sys.argv[2]=='complete':
 rows=[]
 for i,seed in enumerate(request['seeds']):
  content=data+str(seed).encode();(root/f'{i}.mp4').write_bytes(content)
  rows.append({'file':f'{i}.mp4','seed':seed,'sha256':hashlib.sha256(content).hexdigest()})
 result={'request':request,'videos':rows,**request['layout'],'scope':'queue fixture only'}
 (root/'result.json').write_text(json.dumps(result))
else:sys.exit(7)
"""
        child = original_popen(
            [
                sys.executable,
                "-c",
                program,
                request_path,
                "complete" if cancel == "complete" else "cancel" if cancel else "error",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(worker.subprocess, "Popen", fixture_child)
    thread = threading.Thread(target=current.serve)
    thread.start()
    service = PhotoService(tmp_path / "spool", backend="streamforce")
    try:
        wait_for(lambda: service.worker()["ready"])
        stream = io.BytesIO()
        Image.new("RGB", (128, 96), "grey").save(stream, format="PNG")
        photo = service.add_image(
            {"image": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}
        )
        receipt = service.submit(
            {
                "photo": photo["id"],
                "prompt": "queue fixture",
                "anchor": [0.5, 0.5],
                "force": [0.5, 0],
                "seeds": [0, 1, 2, 3],
            }
        )
        directory = service.directory(receipt["id"])
        wait_for(lambda: (directory / "attempt.json").is_file())
        if cancel is True:
            service.cancel(receipt["id"])
        expected = "complete" if cancel == "complete" else "cancelled" if cancel else "error"
        wait_for(lambda: service.job(receipt["id"])["status"] == expected)
        assert children[0].poll() is not None
        assert (directory / "0.mp4").read_bytes().startswith(b"queue fixture")
        assert (directory / "result.json").exists() == (cancel == "complete")
        assert len(read_json(directory / "attempt.json")["videos"]) == 1
    finally:
        current.stop.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
