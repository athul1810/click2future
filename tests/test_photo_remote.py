"""Raw-artifact transfer preserves bytes and stays inside the allocated job."""

import hashlib
import importlib.util
import io
import shutil
from pathlib import Path, PurePosixPath

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


def test_cancelled_request_backs_up_its_finished_sample_without_claiming_completion(tmp_path):
    from click2future.demo.photo_service import PhotoService, read_json, write_json

    service = PhotoService(tmp_path / "local", backend="force-prompting")
    remote_root = tmp_path / "remote"
    job_id = "b" * 32
    folder = service.root / "jobs" / job_id
    folder.mkdir()
    remote_folder = remote_root / "spool/jobs" / job_id
    remote_folder.mkdir(parents=True)
    for directory in [folder, remote_folder]:
        write_json(directory / "request.json", {"seeds": [0, 1, 2, 3]})
    raw = b"partial raw transport fixture, not a generated array"
    video = b"partial mp4 transport fixture, not a generated video"
    (remote_folder / "0.mp4").write_bytes(video)
    write_json(remote_folder / "status.json", {"status": "cancelled", "stage": "cancelled"})
    write_json(
        remote_folder / "attempt.json",
        {
            "complete": False,
            "model": {"allocation_job_id": "50999", "raw_root": "/tmp/fixture/raw"},
            "videos": [
                {
                    "file": "0.mp4",
                    "seed": 0,
                    "sha256": hashlib.sha256(video).hexdigest(),
                    "raw_file": "0.npy",
                    "raw_location": f"/tmp/fixture/raw/{job_id}/0.npy",
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        },
    )

    class SFTP:
        def path(self, value):
            return remote_root.joinpath(*PurePosixPath(value).parts[1:])

        def stat(self, value):
            return self.path(value).stat()

        def mkdir(self, value, mode):
            self.path(value).mkdir(mode=mode)

        def get(self, source, target):
            shutil.copyfile(self.path(source), target)

    remote.sync(SFTP(), service.root, PurePosixPath("/spool"), archive_raw=True, client=Client(raw))
    assert (folder / "0.mp4").read_bytes() == video
    assert (folder / "0.npy").read_bytes() == raw
    assert not read_json(folder / "attempt.json")["complete"]
    assert service.job(job_id)["status"] == "cancelled"
    assert service.job(job_id)["result"] is None
