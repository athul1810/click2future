"""Matched E7 refinement: extra old scenes versus paired new visible scenes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.coverage_data import ARMS, append_cache, read, write
from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_world import WorldConfig


def check_training(models, config):
    rows = [models[name]["training"] for name in ARMS]
    for row in rows:
        if (
            row["status"] != "completed"
            or row["steps_completed"] != config["steps"]
            or row["config"] != config
        ):
            raise ValueError("Incomplete or changed paired training")
    for key in (
        "parameter_count",
        "trainable_parameter_count",
        "initial_model_tensors_sha256",
        "initial_private_rng_sha256",
        "sampling_weights_sha256",
    ):
        if rows[0][key] != rows[1][key]:
            raise ValueError("Unmatched coverage training: " + key)
    additions = [row["provenance"]["coverage_addition"] for row in rows]
    for key in (
        "manifest_sha256",
        "base_cache_manifest_sha256",
        "base_cache_arrays_sha256",
        "parent_provenance",
        "examples_added",
    ):
        if additions[0][key] != additions[1][key]:
            raise ValueError("Different paired data sources: " + key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "cache", "warm-start", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trained-run", type=Path)
    parser.add_argument("--coverage", type=Path, help="Retained, checked training-only generation")
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    config_path = project / "configs/training/coverage_world.json"
    recipe_path = project / "configs/training/coverage_recipe.json"
    config = read(config_path)
    WorldConfig(**config).validate()
    revision = os.environ["C2F_SOURCE_REVISION"]
    if not revision or revision == "unrecorded":
        raise ValueError("Require immutable source revision")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "running",
        "source_revision": revision,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "scope": (
            "post-D014 exploratory data intervention; reused validation; "
            "no external SOTA comparison"
        ),
        "teacher_sha256": sha256(args.warm_start),
        "base_cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "base_cache_arrays_sha256": sha256(args.cache / "arrays.npz"),
        "config_sha256": sha256(config_path),
        "recipe_sha256": sha256(recipe_path),
        "models": {},
        "phases": [],
        "test_splits_evaluated": [],
    }

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        write(args.output / "comparison.json", report)

    def run(name, *arguments):
        phase = {"name": name, "started_seconds": time.perf_counter() - started}
        report["phases"].append(phase)
        save()
        print(json.dumps({"phase": name, "state": "started"}), flush=True)
        subprocess.run([sys.executable, *map(str, arguments)], cwd=project, check=True)
        phase["completed_seconds"] = time.perf_counter() - started
        save()

    try:
        training_root = args.trained_run or args.output
        if args.trained_run:
            previous = read(args.trained_run / "comparison.json")
            if previous["test_splits_evaluated"] or any(
                previous[k] != report[k]
                for k in (
                    "teacher_sha256",
                    "base_cache_manifest_sha256",
                    "base_cache_arrays_sha256",
                    "config_sha256",
                    "recipe_sha256",
                )
            ):
                raise ValueError("Retained comparison inputs changed")
            report.update(
                retained_comparison_sha256=sha256(args.trained_run / "comparison.json"),
                training_source_revision=previous["source_revision"],
                training_slurm_job_id=previous["slurm_job_id"],
            )
        elif not args.coverage:
            run(
                "generate_paired_training",
                "-m",
                "click2future.training.coverage_data",
                "--parent",
                args.dataset,
                "--output",
                args.output / "coverage",
                "--recipe",
                recipe_path,
                "--workers",
                min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))),
            )
        coverage = args.coverage or training_root / "coverage"
        report["coverage_manifest_sha256"] = sha256(coverage / "coverage.json")
        if read(coverage / "recipe.json") != read(recipe_path):
            raise ValueError("Prepared coverage recipe differs")
        report["coverage_path"] = str(coverage.resolve())
        for arm in ARMS:
            cache = training_root / (arm + "-cache")
            if not args.trained_run:
                phase = {"name": "prepare_" + arm, "started_seconds": time.perf_counter() - started}
                report["phases"].append(phase)
                save()
                print(json.dumps({"phase": "prepare_" + arm, "state": "started"}), flush=True)
                append_cache(args.dataset, args.cache, coverage, cache, arm)
                phase["completed_seconds"] = time.perf_counter() - started
                save()
                run(
                    "train_" + arm,
                    "-m",
                    "click2future.training.train_coverage",
                    "--cache",
                    cache,
                    "--warm-start",
                    args.warm_start,
                    "--config",
                    config_path,
                    "--run-dir",
                    args.output / arm,
                )
            report["models"][arm] = {
                "training": read(training_root / arm / "summary.json"),
                "checkpoint_sha256": sha256(training_root / arm / "final.pt"),
                "cache_manifest_sha256": sha256(cache / "manifest.json"),
                "cache_arrays_sha256": sha256(cache / "arrays.npz"),
            }
            if (
                report["models"][arm]["training"]["provenance"]["coverage_addition"][
                    "manifest_sha256"
                ]
                != report["coverage_manifest_sha256"]
            ):
                raise ValueError("Evaluated training used different paired generation")
            if args.trained_run and report["models"][arm] != previous["models"][arm]:
                raise ValueError("Retained full state or cache changed")
            save()
        check_training(report["models"], config)
        checkpoints = {
            "anchor": args.warm_start,
            **{arm: training_root / arm / "final.pt" for arm in ARMS},
        }
        for name, checkpoint in checkpoints.items():
            report["models"].setdefault(name, {"checkpoint_sha256": sha256(checkpoint)})
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
                16,
                "--method",
                "E7",
            )
            common = (
                "--plan",
                output / "plan.json",
                "--predictions",
                output / "validation-samples/predictions.json",
            )
            diagnostic = (*common, "--dataset", args.dataset, "--checkpoint", checkpoint)
            for module, filename, extra in (
                ("passive_stationarity", "passive-stationarity.json", diagnostic),
                (
                    "wall_clipping",
                    "wall-clipping.json",
                    (*diagnostic, "--poses-dir", output / "geometry-audit"),
                ),
                ("action_horizon", "action-horizons.json", common),
                (
                    "collision_audit",
                    "collision-audit.json",
                    (
                        *diagnostic,
                        "--poses-dir",
                        output / "geometry-audit",
                        "--config",
                        project / "configs/evaluation/pixel.yaml",
                    ),
                ),
                ("world_interventions", "world-interventions", (*diagnostic, "--device", "cuda")),
            ):
                run(
                    module + "_" + name,
                    "-m",
                    "click2future.evaluation." + module,
                    *extra,
                    "--output",
                    output / filename,
                )
        for path, digest in (
            (args.warm_start, report["teacher_sha256"]),
            (args.cache / "manifest.json", report["base_cache_manifest_sha256"]),
            (args.cache / "arrays.npz", report["base_cache_arrays_sha256"]),
        ):
            if sha256(path) != digest:
                raise ValueError("Original input changed during comparison")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
