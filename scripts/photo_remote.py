"""Mirror a photo spool over verified SSH; passwords are prompted and never saved."""

import argparse
import getpass
import json
import logging
import re
import shlex
import time
from pathlib import Path, PurePosixPath

import paramiko


def exists(sftp, path):
    try:
        return sftp.stat(str(path))
    except FileNotFoundError:
        return None


def mkdir(sftp, path):
    if not exists(sftp, path):
        sftp.mkdir(str(path), mode=0o700)


def put(sftp, source, target):
    temporary = str(target) + ".part"
    sftp.put(str(source), temporary)
    sftp.posix_rename(temporary, str(target))


def get(sftp, source, target):
    temporary = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            if str(source).endswith(".json"):
                # Read mutable metadata through one handle. stat/prefetch can race
                # the worker's atomic rename on the shared cluster filesystem.
                with sftp.open(str(source), "rb") as handle:
                    payload = handle.read()
                json.loads(payload)
                temporary.write_bytes(payload)
            else:
                sftp.get(str(source), str(temporary))
            temporary.replace(target)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))


def archive_raw_file(client, model, job_id, row, target):
    from click2future.demo.photo_service import JOB_ID, sha256

    allocation = str(model.get("allocation_job_id", ""))
    if (
        not allocation.isdecimal()
        or not JOB_ID.fullmatch(job_id)
        or not re.fullmatch(r"[0-3]\.npy", row["raw_file"])
    ):
        raise ValueError("Invalid allocated raw-artifact request")
    root = PurePosixPath(model["raw_root"])
    source = PurePosixPath(row["raw_location"])
    if not root.is_absolute() or ".." in root.parts or source != root / job_id / row["raw_file"]:
        raise ValueError("Raw artifact is outside the worker's recorded directory")
    command = [
        "srun",
        "--jobid=" + allocation,
        "--overlap",
        "--exact",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=256M",
        "cat",
        "--",
        str(source),
    ]
    _, stdout, stderr = client.exec_command(shlex.join(command), timeout=120)
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as output:
        for block in iter(lambda: stdout.read(1024 * 1024), b""):
            output.write(block)
    if stdout.channel.recv_exit_status() != 0:
        raise RuntimeError("Raw backup step failed: " + stderr.read(2048).decode(errors="replace"))
    if sha256(temporary) != row["raw_sha256"]:
        raise ValueError("Raw artifact hash mismatch")
    temporary.replace(target)


def sync(sftp, local, remote, *, archive_raw=False, client=None):
    from click2future.demo.photo_service import JOB_ID, PHOTO_ID, read_json, sha256

    for name in ["MODEL.json", "WORKER.json"]:
        if exists(sftp, remote / name):
            get(sftp, remote / name, local / name)
    for name in ["images", "jobs"]:
        mkdir(sftp, remote / name)
    for path in (local / "images").glob("*.png"):
        if not PHOTO_ID.fullmatch(path.stem):
            continue
        if not exists(sftp, remote / "images" / path.name):
            put(sftp, path, remote / "images" / path.name)
            put(sftp, path.with_suffix(".json"), remote / "images" / path.with_suffix(".json").name)
    raw_pending = []
    for request in (local / "jobs").glob("*/request.json"):
        job = request.parent.name
        if not JOB_ID.fullmatch(job):
            continue
        folder = remote / "jobs" / job
        mkdir(sftp, folder)
        if not exists(sftp, folder / "request.json"):
            put(sftp, request, folder / "request.json")
        if (request.parent / "cancel.json").is_file() and not exists(sftp, folder / "cancel.json"):
            put(sftp, request.parent / "cancel.json", folder / "cancel.json")
        for name in ["status.json", "failure.json", "attempt.json", "result.json"]:
            if exists(sftp, folder / name):
                get(sftp, folder / name, request.parent / name)
        result = read_json(request.parent / "result.json") or read_json(
            request.parent / "attempt.json"
        )
        if result:
            for row in result["videos"]:
                copies = [(row["file"], row["sha256"])]
                if archive_raw:
                    raw_path = request.parent / row["raw_file"]
                    if not re.fullmatch(r"[0-3]\.npy", row["raw_file"]):
                        raise ValueError("Unexpected raw media name")
                    if not raw_path.is_file() or sha256(raw_path) != row["raw_sha256"]:
                        raw_pending.append((result["model"], job, row, raw_path))
                for name, expected in copies:
                    if not re.fullmatch(r"[0-3]\.(?:mp4|npy)", name):
                        raise ValueError("Unexpected remote media name")
                    path = request.parent / name
                    if not path.is_file() or sha256(path) != expected:
                        get(sftp, folder / name, path)
                        if sha256(path) != expected:
                            raise ValueError("Remote media hash mismatch")
    # Keep heartbeat and cancellation synchronization responsive during large backups.
    if raw_pending:
        if client is None:
            raise ValueError("Raw backup requires the authenticated SSH transport")
        archive_raw_file(client, *raw_pending[0])


def main():
    from click2future.demo.photo_service import PhotoService

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--archive-raw", action="store_true")
    parser.add_argument("--backend", default="force-prompting")
    args = parser.parse_args()
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)
    service = PhotoService(args.spool, backend=args.backend)
    remote = PurePosixPath(args.remote)
    if not remote.is_absolute() or ".." in remote.parts:
        raise ValueError("Use the allocated worker's absolute spool directory")
    client = paramiko.SSHClient()
    client.load_system_host_keys(str(Path.home() / ".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    try:
        password = getpass.getpass("NTU network password (not saved): ")
        try:
            client.connect(
                args.host,
                username=args.username,
                password=password,
                look_for_keys=False,
                allow_agent=False,
                timeout=15,
            )
        finally:
            password = None
        client.get_transport().set_keepalive(20)
        print("photo transport connected; leave this process running", flush=True)
        failures = 0
        while True:
            try:
                with client.open_sftp() as sftp:
                    if not exists(sftp, remote):
                        raise ValueError("Start the allocated worker before connecting its spool")
                    while True:
                        sync(
                            sftp, service.root, remote, archive_raw=args.archive_raw, client=client
                        )
                        failures = 0
                        time.sleep(1)
            except (OSError, EOFError, paramiko.SSHException):
                failures += 1
                transport = client.get_transport()
                if not transport or not transport.is_active() or failures >= 5:
                    raise
                print(
                    "file transfer interrupted; reopening the channel; remote jobs continue",
                    flush=True,
                )
                time.sleep(min(2 ** (failures - 1), 8))
    finally:
        client.close()


if __name__ == "__main__":
    main()
