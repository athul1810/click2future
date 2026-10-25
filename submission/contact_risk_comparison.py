"""One bounded, matched warm-start contact-risk experiment on validation only."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.train_contact import ContactConfig
from click2future.training.train_cvae import require_tc2, sha256


def paired_configs(project):
    paths = {
        name: Path(project) / "configs/training" / f"contact_risk_{name}.json"
        for name in ("control", "tail")
    }
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for value in values.values():
        ContactConfig(**value).validate()
    a, b = values["control"], values["tail"]
    if (
        a.keys() != b.keys()
        or {key for key in a if a[key] != b[key]} != {"risk_penetration_weight"}
        or a["risk_penetration_weight"] != 0
        or b["risk_penetration_weight"] <= 0
    ):
        raise ValueError("Matched recipes may differ only in enabled penetration risk")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "supplement", "warm-start", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = paired_configs(project)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    teacher_sha = sha256(args.warm_start)
    report = {
        "status": "running",
        "scope": "matched EMA warm-start; one seed; validation only",
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "warm_start_sha256": teacher_sha,
        "config_sha256": {key: sha256(value) for key, value in configs.items()},
        "test_splits_evaluated": [],
        "phases": [],
        "models": {},
    }

    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, *arguments):
        phase = {"name": name, "started_seconds": time.perf_counter() - started}
        report["phases"].append(phase)
        save()
        print(json.dumps({"phase": name, "state": "started"}), flush=True)
        subprocess.run([sys.executable, *map(str, arguments)], cwd=project, check=True)
        phase["completed_seconds"] = time.perf_counter() - started
        save()

    try:
        for name, config in configs.items():
            run(
                "train_" + name,
                "-m",
                "click2future.training.train_contact",
                "--dataset",
                args.dataset,
                "--supplement",
                args.supplement,
                "--warm-start",
                args.warm_start,
                "--config",
                config,
                "--run-dir",
                args.output / name,
            )
            training = json.loads((args.output / name / "summary.json").read_text())
            if training["status"] != "completed":
                raise ValueError("Bounded training did not finish: " + name)
            report["models"][name] = {"training": training}
        identities = [value["training"]["provenance"] for value in report["models"].values()]
        if identities[0] != identities[1] or sha256(args.warm_start) != teacher_sha:
            raise ValueError("Warm-start source/data/provenance changed between recipes")
        checkpoints = {
            "anchor": args.warm_start,
            **{name: args.output / name / "best.pt" for name in configs},
        }
        for name, checkpoint in checkpoints.items():
            report["models"].setdefault(name, {})
            output = args.output / (name + "-validation")
            run(
                "evaluate_" + name,
                "-m",
                "click2future.evaluation.trajectory_pilot",
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--output",
                output,
                "--device",
                "cuda",
                "--steps",
                "16",
                "--method",
                "E6",
            )
            common = (
                "--plan",
                output / "plan.json",
                "--predictions",
                output / "validation-samples/predictions.json",
            )
            run(
                "passive_" + name,
                "-m",
                "click2future.evaluation.passive_stationarity",
                *common,
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--output",
                output / "passive-stationarity.json",
            )
            run(
                "clipping_" + name,
                "-m",
                "click2future.evaluation.wall_clipping",
                *common,
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--poses-dir",
                output / "geometry-audit",
                "--output",
                output / "wall-clipping.json",
            )
            run(
                "horizons_" + name,
                "-m",
                "click2future.evaluation.action_horizon",
                *common,
                "--output",
                output / "action-horizons.json",
            )
            report["models"][name]["quality"] = json.loads(
                (output / "trajectory-results.json").read_text()
            )["quality"]
            report["models"][name]["checkpoint_sha256"] = sha256(checkpoint)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
