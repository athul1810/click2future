"""Raw-artifact transfer preserves bytes and stays inside the allocated job."""

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/photo_remote.py"
spec = importlib.util.spec_from_file_location("photo_remote", path)
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class Result(io.BytesIO):
    @property
    def channel(self):
        return self

    def recv_exit_status(self):
        return 0


class Client:
    def __init__(self, raw):
        self.raw = raw
        self.command = None

    def exec_command(self, command, timeout):
        self.command = command
        return None, Result(self.raw), io.BytesIO()


def test_raw_copy_uses_the_existing_allocation_and_checks_its_hash(tmp_path):
    raw = b"raw numpy artifact fixture"
    client = Client(raw)
    job_id = "a" * 32
    root = "/tmp/user/c2f-photo/raw"
    row = {
        "raw_file": "0.npy",
        "raw_location": f"{root}/{job_id}/0.npy",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }
    target = tmp_path / "0.npy"
    remote.archive_raw_file(
        client, {"allocation_job_id": "50999", "raw_root": root}, job_id, row, target
    )
    assert target.read_bytes() == raw
    assert "--jobid=50999" in client.command and "--gres=none" in client.command
    row["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        remote.archive_raw_file(
            client,
            {"allocation_job_id": "50999", "raw_root": root},
            job_id,
            row,
            tmp_path / "1.npy",
        )


def test_raw_copy_rejects_path_and_allocation_injection_before_execution(tmp_path):
    job_id = "a" * 32
    client = Client(b"")
    row = {"raw_file": "0.npy", "raw_location": "/elsewhere/0.npy", "raw_sha256": "0" * 64}
    with pytest.raises(ValueError, match="outside"):
        remote.archive_raw_file(
            client,
            {"allocation_job_id": "50999", "raw_root": "/tmp/raw"},
            job_id,
            row,
            tmp_path / "0.npy",
        )
    with pytest.raises(ValueError, match="allocated"):
        remote.archive_raw_file(
            client,
            {"allocation_job_id": "1; echo bad", "raw_root": "/tmp/raw"},
            job_id,
            row,
            tmp_path / "0.npy",
        )
    assert client.command is None
