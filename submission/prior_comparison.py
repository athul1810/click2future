"""Matched prior-only/conditional diffusion, refining the same E7 teacher."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_prior import PriorConfig


def matched_configs(project):
    paths = {
        name: Path(project) / "configs/training" / f"prior_{name}.json"
        for name in ("direct", "diffusion")
    }
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for value in values.values():
        PriorConfig(**value).validate()
    a, b = values.values()
    if (
        a.keys() != b.keys()
        or {key for key in a if a[key] != b[key]} != {"world_prior_only"}
        or a["world_prior_only"] is not True
        or b["world_prior_only"] is not False
    ):
        raise ValueError("Recipes may differ only in world_prior_only")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "dataset", "warm-start", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trained-run", type=Path, help="Fresh evaluation of completed training")
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = matched_configs(project)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "status": "running",
        "scope": ("post-D014 exploratory; full E7 refinement; original cache; reused validation"),
        "source_revision": os.environ["C2F_SOURCE_REVISION"],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "teacher_sha256": sha256(args.warm_start),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "cache_arrays_sha256": sha256(args.cache / "arrays.npz"),
        "config_sha256": {name: sha256(path) for name, path in configs.items()},
        "models": {},
        "phases": [],
        "test_splits_evaluated": [],
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
        training_root = args.trained_run or args.output
        if args.trained_run:
            previous = json.loads((args.trained_run / "comparison.json").read_text())
            for key in (
                "teacher_sha256",
                "cache_manifest_sha256",
                "cache_arrays_sha256",
                "config_sha256",
            ):
                if previous[key] != report[key]:
                    raise ValueError("Retained experiment identity differs: " + key)
            report["retained_comparison_sha256"] = sha256(args.trained_run / "comparison.json")
            report["training_source_revision"] = previous["source_revision"]
            report["training_slurm_job_id"] = previous["slurm_job_id"]
        for name, path in configs.items():
            if not args.trained_run:
                run(
                    "train_" + name,
                    "-m",
                    "click2future.training.train_prior",
                    "--cache",
                    args.cache,
                    "--warm-start",
                    args.warm_start,
                    "--config",
                    path,
                    "--run-dir",
                    args.output / name,
                )
            training = json.loads((training_root / name / "summary.json").read_text())
            if (
                training["status"] != "completed"
                or training["steps_completed"] != 8000
                or training["config"] != json.loads(path.read_text())
            ):
                raise ValueError("Incomplete or changed retained training: " + name)
            checkpoint = training_root / name / "final.pt"
            report["models"][name] = {"training": training, "checkpoint_sha256": sha256(checkpoint)}
            if args.trained_run and report["models"][name] != previous["models"][name]:
                raise ValueError("Retained training record or weights changed")
        a, b = (report["models"][name]["training"] for name in configs)
        for key in ("provenance", "parameter_count", "trainable_parameter_count"):
            if a[key] != b[key]:
                raise ValueError("Unmatched training: " + key)
        checkpoints = {
            "anchor": args.warm_start,
            **{name: training_root / name / "final.pt" for name in configs},
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
                1 if name == "direct" else 16,
                "--method",
                "E7" if name == "anchor" else "E11",
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
            run(
                "response_" + name,
                "submission/trajectory_response.py",
                *diagnostic,
                "--device",
                "cuda",
                "--output",
                output / "response-diagnostic",
            )
        for path, digest in (
            (args.warm_start, report["teacher_sha256"]),
            (args.cache / "manifest.json", report["cache_manifest_sha256"]),
            (args.cache / "arrays.npz", report["cache_arrays_sha256"]),
        ):
            if sha256(path) != digest:
                raise ValueError("Training input changed")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
