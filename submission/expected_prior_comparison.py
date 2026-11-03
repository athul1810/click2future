"""D019: original versus expected direct loss; resumable, validation only."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_expected_prior import ExpectedPriorConfig
from click2future.training.train_expected_prior import sources as training_sources


def matched_configs(project):
    paths = {
        name: Path(project) / "configs/training" / f"prior_expected_{name}.json"
        for name in ("control", "expected")
    }
    values = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for value in values.values():
        ExpectedPriorConfig(**value).validate()
    a, b = values.values()
    if (
        a.keys() != b.keys()
        or {key for key in a if a[key] != b[key]} != {"expected_objective"}
        or a["expected_objective"] is not False
        or b["expected_objective"] is not True
    ):
        raise ValueError("D019 recipes may differ only in expected_objective")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "dataset", "warm-start", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rest-anchored", action="store_true", help="Run the single D020 repair candidate"
    )
    parser.add_argument(
        "--trained-run", type=Path, help="Evaluate retained final weights without training"
    )
    args = parser.parse_args(argv)
    require_tc2()
    project = Path(__file__).resolve().parents[1]
    configs = matched_configs(project)
    if args.rest_anchored:
        configs = {"rest-anchored": project / "configs/training/prior_rest_anchored.json"}
        candidate = ExpectedPriorConfig(**json.loads(configs["rest-anchored"].read_text()))
        candidate.validate()
        if (
            not candidate.rest_anchored
            or not candidate.expected_objective
            or candidate.steps != 8000
        ):
            raise ValueError(
                "D020 requires its declared anchored, expected-loss final-update recipe"
            )
    identity = {
        "source_revision": os.environ["C2F_SOURCE_REVISION"],
        "teacher_sha256": sha256(args.warm_start),
        "cache_manifest_sha256": sha256(args.cache / "manifest.json"),
        "cache_arrays_sha256": sha256(args.cache / "arrays.npz"),
        "config_sha256": {name: sha256(path) for name, path in configs.items()},
    }
    record_path = args.output / "comparison.json"
    training_root = args.trained_run or args.output
    retained = None
    if args.trained_run:
        retained = json.loads((args.trained_run / "comparison.json").read_text())
        if (
            args.resume
            or args.output.exists()
            or any(
                retained.get(key) != value
                for key, value in identity.items()
                if key != "source_revision"
            )
        ):
            raise ValueError("Recovery requires fresh output and identical training inputs/recipes")
    if args.output.exists():
        if not args.resume or not record_path.is_file():
            parser.error("Use a fresh directory, or --resume with the original experiment record")
        report = json.loads(record_path.read_text())
        if any(report.get(key) != value for key, value in identity.items()):
            raise ValueError("Refusing to resume changed source, data, teacher or recipes")
        if report["status"] == "completed":
            print("D019 already completed; nothing submitted or overwritten")
            return
    else:
        if args.resume:
            parser.error("Cannot resume an absent experiment")
        args.output.mkdir(parents=True)
        report = {
            **identity,
            "study_id": "D020" if args.rest_anchored else "D019",
            "scope": "single rest-anchored repair candidate; validation only"
            if args.rest_anchored
            else "paired seed-19 training-loss screen; original training/validation only",
            "models": {},
            "attempts": [],
            "test_splits_evaluated": [],
        }
        if retained is not None:
            report.update(
                retained_comparison_sha256=sha256(args.trained_run / "comparison.json"),
                training_source_revision=retained["source_revision"],
            )
    attempt = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "phases": []}
    report["attempts"].append(attempt)
    report["status"] = "running"
    started = time.perf_counter()
    child = None
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGUSR1)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    def save():
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(record_path)

    def run(label, *arguments):
        nonlocal child
        if interrupted:
            raise RuntimeError("Allocation interrupted; retained for exact checkpoint resume")
        phase = {"name": label, "started_seconds": time.perf_counter() - started}
        attempt["phases"].append(phase)
        save()
        print(json.dumps({"phase": label}), flush=True)
        child = subprocess.Popen([sys.executable, *map(str, arguments)], cwd=project)
        returncode = child.wait()
        child = None
        phase["exit_code"] = returncode
        phase["elapsed_seconds"] = time.perf_counter() - started - phase["started_seconds"]
        save()
        if returncode:
            raise RuntimeError(f"Phase {label} stopped with code {returncode}; retained for resume")

    save()
    try:
        for name, recipe in configs.items():
            folder = training_root / name
            summary = folder / "summary.json"
            completed = (
                summary.is_file() and json.loads(summary.read_text())["status"] == "completed"
            )
            if completed and name in report["models"]:
                if sha256(folder / "final.pt") != report["models"][name]["checkpoint_sha256"]:
                    raise ValueError("Retained completed checkpoint changed: " + name)
            if not completed:
                if retained is not None:
                    raise ValueError("Evaluation recovery cannot train an incomplete arm")
                run(
                    "train_" + name,
                    "-m",
                    "click2future.training.train_expected_prior",
                    "--cache",
                    args.cache,
                    "--warm-start",
                    args.warm_start,
                    "--config",
                    recipe,
                    "--run-dir",
                    folder,
                    *(["--resume"] if (folder / "latest.pt").is_file() else []),
                )
            training = json.loads(summary.read_text())
            if retained is not None and (
                training["provenance"]["source_sha256"] != training_sources()
                or training != retained["models"][name]["training"]
                or sha256(folder / "final.pt") != retained["models"][name]["checkpoint_sha256"]
            ):
                raise ValueError("Retained training source, summary or checkpoint changed")
            if (
                training["status"] != "completed"
                or training["steps_completed"] != 8000
                or training["config"] != json.loads(recipe.read_text())
            ):
                raise ValueError("Incomplete or changed D019 training: " + name)
            report["models"][name] = {
                "training": training,
                "checkpoint_sha256": sha256(folder / "final.pt"),
            }
            save()
        if not args.rest_anchored:
            a, b = (report["models"][name]["training"] for name in configs)
            for key in ("provenance", "parameter_count", "trainable_parameter_count"):
                if a[key] != b[key]:
                    raise ValueError("Unmatched paired refinement: " + key)
        # Every evaluation attempt uses a fresh directory. Completed earlier
        # reports stay available even if a later metric fails or a node stops.
        evaluation = args.output / f"evaluation-{len(report['attempts']):03d}"
        evaluation.mkdir()
        report["evaluation_directory"] = str(evaluation)
        for name in configs:
            checkpoint = training_root / name / "final.pt"
            output = evaluation / name
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
                "1",
                "--method",
                "E11",
            )
            common = (
                "--plan",
                output / "plan.json",
                "--predictions",
                output / "validation-samples/predictions.json",
            )
            diagnostic = (*common, "--dataset", args.dataset, "--checkpoint", checkpoint)
            for module, filename, extra in (
                ("passive_stationarity", "passive.json", diagnostic),
                ("action_horizon", "action-horizons.json", common),
                (
                    "collision_audit",
                    "collision.json",
                    (
                        *diagnostic,
                        "--poses-dir",
                        output / "geometry-audit",
                        "--config",
                        project / "configs/evaluation/pixel.yaml",
                    ),
                ),
                (
                    "wall_clipping",
                    "clipping.json",
                    (*diagnostic, "--poses-dir", output / "geometry-audit"),
                ),
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
            if sha256(checkpoint) != report["models"][name]["checkpoint_sha256"]:
                raise ValueError("A D019 checkpoint changed during evaluation")
        for path, key in (
            (args.warm_start, "teacher_sha256"),
            (args.cache / "manifest.json", "cache_manifest_sha256"),
            (args.cache / "arrays.npz", "cache_arrays_sha256"),
        ):
            if sha256(path) != identity[key]:
                raise ValueError("A frozen D019 input changed")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        attempt["elapsed_seconds"] = time.perf_counter() - started
        save()


if __name__ == "__main__":
    main()
