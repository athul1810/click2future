"""Evaluate frozen checkpoints on the separately generated D017 corpus.

Use ten new scene groups per split and reject visible-scene overlap with the
training corpus. Prepare all plans before sampling. E0/E2/E3 use the D014
sampler; E10 uses its 16-step Heun sampler. No checkpoint is trained or selected."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from click2future.data.generalization import (
    generate_generalization_dataset,
    validate_generalization_dataset,
)
from click2future.evaluation import action_horizon
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _sha, _write, prepare_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan
from click2future.evaluation.trajectory_pilot import sample_plan

SPLITS = ("id_test", "shape_ood", "texture_ood", "force_ood", "physics_test")
SEED = 790000
GROUPS_PER_SPLIT = 10


def held_out_corpus(path):
    """Generate the declared corpus once; later jobs must find exactly that corpus."""
    counts = {"train": 1, "validation": 1, **dict.fromkeys(SPLITS, GROUPS_PER_SPLIT)}
    if not path.exists():
        generate_generalization_dataset(path, seed=SEED, group_counts=counts)
    manifest = json.loads((path / "dataset_manifest.json").read_text())
    if manifest["seed"] != SEED or manifest["group_counts"] != counts:
        raise ValueError("The held-out corpus differs from the D017 declaration")
    validate_generalization_dataset(path)
    return {
        "path": str(path),
        "seed": SEED,
        "group_counts": counts,
        "dataset_manifest_sha256": _sha(path / "dataset_manifest.json"),
        "splits_sha256": _sha(path / "splits.json"),
        "sample_count": manifest["sample_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, required=True)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--pose-runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--without-actions", nargs="*", default=["force_ood"])
    args = parser.parse_args()
    if args.runs is None and args.pose_runs is None:
        parser.error("Provide --runs and/or --pose-runs")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True)
    actions = args.project / "configs/evaluation/actions.yaml"
    pose_runs = json.loads(args.pose_runs.read_text())["runs"] if args.pose_runs else []
    record = {
        "scope": "D017 held-out replication on a separate corpus; no training or tuning",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "runs_spec_sha256": _sha(args.runs) if args.runs else None,
        "pose_runs_spec_sha256": _sha(args.pose_runs) if args.pose_runs else None,
        "training_corpus_manifest_sha256": _sha(args.training_data / "dataset_manifest.json"),
        "plans": {},
        "suites": {},
    }
    started = time.perf_counter()

    def save():
        record["seconds"] = round(time.perf_counter() - started, 2)
        _write(out / "replication.json", record)

    record["held_out_corpus"] = held_out_corpus(args.held_out)
    save()
    suites = {}
    for split in args.splits:
        folder = out / split
        plan = prepare_plan(
            args.held_out,
            args.project / "configs/evaluation/pixel.yaml",
            split=split,
            metrics=("ssim", "lpips"),
            protocol=args.project / "configs/evaluation/comparison.yaml",
            actions_config=None if split in args.without_actions else actions,
            training_corpus=args.training_data,
        )
        folder.mkdir()
        _write(folder / "plan.json", plan)
        record["plans"][split] = {"plan_sha256": plan["plan_sha256"], "cases": len(plan["cases"])}
        suites[split] = folder / "plan.json"
        try:
            prepare_target_plan(
                folder / "plan.json",
                folder / "target-controls",
                config=actions,
                force_magnitude=0.4,
            )
            target = json.loads((folder / "target-controls/plan.json").read_text())
            record["plans"][split + "-selection"] = {
                "plan_sha256": target["plan_sha256"],
                "cases": len(target["cases"]),
            }
            suites[split + "-selection"] = folder / "target-controls/plan.json"
        except (ValueError, KeyError) as error:
            record["plans"][split + "-selection"] = {"status": "not_prepared", "reason": str(error)}
        save()
    print(json.dumps({"plans_frozen": list(suites)}), flush=True)

    def evaluate(name, plan_path, samples, metrics, sample):
        suite = record["suites"][name] = {
            "plan_sha256": record["plans"][name.split("/")[0]]["plan_sha256"]
        }
        try:
            for phase, function, arguments in (
                ("sample", sample, ()),
                ("score", run_suite, (plan_path, samples / "predictions.json", metrics)),
            ):
                print(json.dumps({"suite": name, "phase": phase}), flush=True)
                begin = time.perf_counter()
                result = function(*arguments)
                suite[phase + "_seconds"] = round(time.perf_counter() - begin, 2)
            suite["status"] = result["status"]
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            suite.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            return
        try:
            horizon = action_horizon.evaluate(plan_path, samples / "predictions.json")
            _write(metrics.parent / (metrics.name + "-action-horizons.json"), horizon)
        except (ValueError, KeyError) as error:
            suite["horizons"] = f"not scored: {error}"
        save()

    for name, plan_path in suites.items():
        folder = plan_path.parent
        if args.runs:
            samples = folder / "samples"
            evaluate(
                name,
                plan_path,
                samples,
                folder / "metrics",
                lambda plan_path=plan_path, samples=samples: generate_samples(
                    plan_path, args.runs, samples, device=args.device
                ),
            )
        for run in pose_runs:
            samples = folder / (run["run_id"] + "-samples")
            evaluate(
                name + "/" + run["run_id"],
                plan_path,
                samples,
                folder / (run["run_id"] + "-metrics"),
                lambda plan_path=plan_path, samples=samples, run=run: sample_plan(
                    plan_path,
                    run["checkpoint"],
                    samples,
                    device=args.device,
                    steps=16,
                    method="E10",
                    run_id=run["run_id"],
                ),
            )
    failed = [name for name, suite in record["suites"].items() if suite["status"] != "passed"]
    record["status"] = "completed_with_failures" if failed else "completed"
    record["failed_suites"] = failed
    save()


if __name__ == "__main__":
    main()
