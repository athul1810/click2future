"""D018: two further refinement seeds, conditional on the same frozen E7 teacher."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_prior import PriorConfig, prior_sources

SEEDS = (20, 21)
ARMS = ("direct", "diffusion")


def recipes(project):
    project = Path(project)
    result = {}
    for seed in SEEDS:
        for arm in ARMS:
            base = json.loads((project / f"configs/training/prior_{arm}.json").read_text())
            path = project / f"configs/training/prior_repeats/{arm}-seed{seed}.json"
            value = json.loads(path.read_text())
            PriorConfig(**value).validate()
            if value != {**base, "seed": seed}:
                raise ValueError("Repeat recipes may change only the refinement seed")
            result[f"{arm}-seed{seed}"] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "teacher", "screen-summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = recipes(project)
    original = json.loads(args.screen_summary.read_text())
    if (
        original["status"] != "completed"
        or original["steps_completed"] != 8000
        or original["config"]["seed"] != 19
    ):
        raise ValueError("Require the completed seed-19 screen")
    for name, path in configs.items():
        expected = {
            **original["config"],
            "seed": int(name.rsplit("seed", 1)[1]),
            "world_prior_only": name.startswith("direct-"),
        }
        if json.loads(path.read_text()) != expected:
            raise ValueError("Repeat hyperparameters differ from the completed screen")
    if original["provenance"]["source_sha256"] != prior_sources():
        raise ValueError("Repeat training/model code differs from the seed-19 screen")
    if original["provenance"]["initialization"]["checkpoint_sha256"] != sha256(args.teacher):
        raise ValueError("Repeat teacher differs from the frozen screen")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {
        str(p): sha256(p)
        for p in (
            args.teacher,
            args.screen_summary,
            args.cache / "manifest.json",
            args.cache / "arrays.npz",
            *configs.values(),
        )
    }
    record = {
        "decision": "D018",
        "status": "running",
        "source_revision": os.environ["C2F_SOURCE_REVISION"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "input_sha256": inputs,
        "scope": "independent refinement RNG seeds conditional on one fixed teacher; no evaluation",
        "runs": {},
        "test_splits_evaluated": [],
    }

    def save():
        (args.output / "repeats.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        for name, path in configs.items():
            print(json.dumps({"training": name}), flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "click2future.training.train_prior",
                    "--cache",
                    str(args.cache),
                    "--warm-start",
                    str(args.teacher),
                    "--config",
                    str(path),
                    "--run-dir",
                    str(args.output / name),
                ],
                cwd=project,
                check=True,
            )
            summary = json.loads((args.output / name / "summary.json").read_text())
            if summary["status"] != "completed" or summary["steps_completed"] != 8000:
                raise ValueError("Incomplete repeat: " + name)
            record["runs"][name] = {
                "summary": summary,
                "checkpoint_sha256": sha256(args.output / name / "final.pt"),
            }
            save()
        for seed in SEEDS:
            a, b = (record["runs"][f"{arm}-seed{seed}"]["summary"] for arm in ARMS)
            for key in ("provenance", "parameter_count", "trainable_parameter_count"):
                if a[key] != b[key]:
                    raise ValueError("Unmatched repeat: " + key)
        for path, digest in inputs.items():
            if sha256(path) != digest:
                raise ValueError("Repeat input changed")
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
