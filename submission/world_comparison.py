"""Three matched E7 arms: local adapters, relational messages, sampled-world conditioning."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_world import WorldConfig
from click2future.training.world_data import prepare_cache


def paired_configs(project):
    paths = {
        name: Path(project) / "configs/training" / f"world_{name}.json"
        for name in ("local", "relational", "conditioned")
    }
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for value in values.values():
        WorldConfig(**value).validate()
    for left, right, switch in (
        ("local", "relational", "relational"),
        ("relational", "conditioned", "world_condition"),
    ):
        a, b = values[left], values[right]
        if (
            a.keys() != b.keys()
            or {key for key in a if a[key] != b[key]} != {switch}
            or a[switch] is not False
            or b[switch] is not True
        ):
            raise ValueError("Adjacent world recipes may differ only in " + switch)
    return paths


def retained_training(root, configs, teacher_sha):
    """Reuse completed arms after an evaluation failure, without training again."""
    root = Path(root)
    previous = json.loads((root / "comparison.json").read_text())
    if previous["warm_start_sha256"] != teacher_sha or previous["test_splits_evaluated"]:
        raise ValueError("Retained experiment has a different teacher or test scope")
    for key, path in (
        ("cache_manifest_sha256", root / "cache/manifest.json"),
        ("cache_array_sha256", root / "cache/arrays.npz"),
    ):
        if sha256(path) != previous[key]:
            raise ValueError("Retained training cache changed")
    models = {}
    for name, config in configs.items():
        training = json.loads((root / name / "summary.json").read_text())
        if (
            training["status"] != "completed"
            or training["steps_completed"] != 8000
            or training["config"] != json.loads(config.read_text())
            or previous["config_sha256"][name] != sha256(config)
            or training != previous["models"][name]["training"]
        ):
            raise ValueError("Retained arm is incomplete or changed: " + name)
        from click2future.training.world_checkpoint import load_world

        _, state = load_world(root / name / "best.pt", "cpu")
        if (
            state["step"] != training["steps_completed"]
            or state["config"] != training["config"]
            or state["provenance"] != training["provenance"]
        ):
            raise ValueError("Retained checkpoint does not match its training record")
        models[name] = {
            "training": training,
            "retained_checkpoint_sha256": sha256(root / name / "best.pt"),
        }
    return previous, models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "supplement", "warm-start", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--trained-run", type=Path, help="Completed training; fresh evaluation only"
    )
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = paired_configs(project)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    teacher_sha = sha256(args.warm_start)
    report = {
        "status": "running",
        "scope": "post-D014 exploratory; matched backbone; extra physical supervision",
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
        if args.trained_run:
            previous, report["models"] = retained_training(args.trained_run, configs, teacher_sha)
            for key in ("cache_manifest_sha256", "cache_array_sha256"):
                report[key] = previous[key]
            report.update(
                training_source_revision=previous["source_revision"],
                training_slurm_job_id=previous["slurm_job_id"],
                retained_comparison_sha256=sha256(args.trained_run / "comparison.json"),
                scope="evaluation recovery; no retraining; post-D014 exploratory validation",
            )
            save()
        else:
            report["phases"].append(
                {"name": "prepare_cache", "started_seconds": time.perf_counter() - started}
            )
            save()
            print(json.dumps({"phase": "prepare_cache", "state": "started"}), flush=True)
            prepare_cache(args.dataset, args.supplement, args.output / "cache")
            report["phases"][-1]["completed_seconds"] = time.perf_counter() - started
            report["cache_manifest_sha256"] = sha256(args.output / "cache/manifest.json")
            report["cache_array_sha256"] = sha256(args.output / "cache/arrays.npz")
            save()
            for name, config in configs.items():
                run(
                    "train_" + name,
                    "-m",
                    "click2future.training.train_world",
                    "--cache",
                    args.output / "cache",
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
        if (
            any(value != identities[0] for value in identities[1:])
            or sha256(args.warm_start) != teacher_sha
        ):
            raise ValueError("Warm-start source/data/provenance changed between recipes")
        checkpoints = {
            "anchor": args.warm_start,
            **{name: (args.trained_run or args.output) / name / "best.pt" for name in configs},
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
                "E6" if name == "anchor" else "E7",
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
            run(
                "collisions_" + name,
                "-m",
                "click2future.evaluation.collision_audit",
                *common,
                "--dataset",
                args.dataset,
                "--checkpoint",
                checkpoint,
                "--poses-dir",
                output / "geometry-audit",
                "--config",
                project / "configs/evaluation/pixel.yaml",
                "--output",
                output / "collision-audit.json",
            )
            if name != "anchor":
                run(
                    "mass_interventions_" + name,
                    "-m",
                    "click2future.evaluation.world_interventions",
                    *common,
                    "--dataset",
                    args.dataset,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    output / "world-interventions",
                    "--device",
                    "cuda",
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
