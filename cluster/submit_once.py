"""Run detached on TC2: persist one Slurm receipt despite a lost SSH response."""

import argparse
import fcntl
import getpass
import json
import os
import re
import subprocess
from pathlib import Path


def submit_once(run_dir, script, name):
    run_dir, script = Path(run_dir).resolve(), Path(script).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or not script.is_file():
        raise ValueError("Provide a safe unique job name and an existing batch script")
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = run_dir / "submission.json"
    identity = {
        "source_revision": os.environ["C2F_SOURCE_REVISION"],
        "script": str(script),
        "job_name": name,
    }
    with (run_dir / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if any(record.get(key) != value for key, value in identity.items()):
                raise ValueError("Existing receipt belongs to a different immutable experiment")
            return record
        # Also recover a submission whose SSH client (or submit helper) lost
        # the response before the receipt was written. The account/name pair
        # must be unique; ambiguous state never triggers another allocation.
        queued = subprocess.check_output(
            ["squeue", "-h", "-u", getpass.getuser(), "-n", name, "-o", "%A"], text=True
        ).split()
        if len(set(queued)) > 1:
            raise ValueError("Multiple matching allocations exist; inspect them before retrying")
        if queued:
            identifier, recovered = queued[0], True
        else:
            output = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--job-name",
                    name,
                    "--output",
                    str(run_dir / "slurm-%j.out"),
                    "--error",
                    str(run_dir / "slurm-%j.err"),
                    str(script),
                ],
                text=True,
            ).strip()
            identifier, recovered = output.split(";", 1)[0], False
        if not identifier.isdigit():
            raise ValueError("Slurm did not return an unambiguous numeric job ID")
        record = {**identity, "job_id": identifier, "recovered_from_queue": recovered}
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(receipt)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(json.dumps(submit_once(args.run_dir, args.script, args.name)), flush=True)


if __name__ == "__main__":
    main()
