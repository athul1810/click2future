"""Run the predeclared E9 numeric-versus-spatial routing comparison (D016) on TC2."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_routing import RoutingConfig

SEEDS = (19, 20, 21)
ARMS = ("spatial", "numeric")


def routing_configs(project):
    """Arms differ only in force routing, seeds only in seed, and both only
    in force_limit from the E6 canvas recipe (an input contract, not a loss)."""
    project = Path(project)
    base = json.loads((project / "configs/training/contact_wall_pilot.json").read_text())
    paths = {
        f"{arm}-seed{seed}": project / f"configs/training/routing/{arm}-seed{seed}.json"
        for seed in SEEDS
        for arm in ARMS
    }
    for name, path in paths.items():
        values = json.loads(path.read_text())
        RoutingConfig(**values).validate()
        arm, seed = name.split("-seed")
        expected = {
            **base,
            "seed": int(seed),
            "force_limit": 1.5,
            "broadcast_force": arm == "numeric",
        }
        if values != expected:
            raise ValueError("Routing configs may differ from E6 only as declared: " + name)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = routing_configs(project)
    if args.output.exists():
        parser.error("Preserve attempted comparisons; provide a fresh output directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "status": "running",
        "decision": "D016",
        "scope": "E9 numeric versus spatial force routing; three training seeds; validation",
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "dataset_manifest_sha256": sha256(args.dataset / "dataset_manifest.json"),
        "config_sha256": {name: sha256(path) for name, path in configs.items()},
        "test_splits_evaluated": [],
        "phases": [],
        "models": {},
    }

    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, *arguments):
        # Explicit subprocess boundaries release CUDA allocations between phases.
        phase = {"name": name, "started_seconds": time.perf_counter() - started}
        report["phases"].append(phase)
        save()
        print(json.dumps({"phase": name, "state": "started"}), flush=True)
        subprocess.run([sys.executable, *map(str, arguments)], check=True, cwd=project)
        phase["completed_seconds"] = time.perf_counter() - started
        save()

    try:
        supplement = args.output / "training-actions"
        run(
            "training_actions",
            "-m",
            "click2future.data.action_supplement",
            "--parent",
            args.dataset,
            "--output",
            supplement,
        )
        report["supplement_manifest_sha256"] = sha256(supplement / "supplement_manifest.json")
        for name, config in configs.items():
            run(
                "train_" + name,
                "-m",
                "click2future.training.train_routing",
                "--dataset",
                args.dataset,
                "--supplement",
                supplement,
                "--config",
                config,
                "--run-dir",
                args.output / name,
            )
            training = json.loads((args.output / name / "summary.json").read_text())
            if training["status"] != "completed":
                raise ValueError("Bounded training did not finish: " + name)
            report["models"][name] = {"training": training}
        for name in configs:
            checkpoint = args.output / name / "best.pt"
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
                "E9",
            )
            run(
                "passive_" + name,
                "-m",
                "click2future.evaluation.passive_stationarity",
                "--plan",
                output / "plan.json",
                "--predictions",
                output / "validation-samples/predictions.json",
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--output",
                output / "passive-stationarity.json",
            )
            run(
                "wall_clipping_" + name,
                "-m",
                "click2future.evaluation.wall_clipping",
                "--plan",
                output / "plan.json",
                "--predictions",
                output / "validation-samples/predictions.json",
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--poses-dir",
                output / "geometry-audit",
                "--output",
                output / "wall-clipping.json",
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
