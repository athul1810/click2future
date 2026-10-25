"""Final held-out E0/E2/E3 evaluation from one account and one set of frozen plans (D014, D015).

All plans are prepared and recorded before any video is generated. For every reserved
split the primary plan keeps the action controls, except on splits without zero force,
and the force-0.4 selected-object plan is derived from it. Every run is sampled against
the same plans and scored with the suite, then both action horizons are added. A failed
suite is recorded and the remaining suites still run. Nothing is tuned on these results.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from click2future.evaluation import action_horizon
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _sha, _write, prepare_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan

SPLITS = ("id_test", "shape_ood", "texture_ood", "force_ood", "physics_test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--without-actions", nargs="*", default=["force_ood"])
    parser.add_argument("--decision", default="D014")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = args.output
    out.mkdir(parents=True)
    actions = args.project / "configs/evaluation/actions.yaml"
    record = {
        "scope": f"final held-out evaluation under {args.decision}; no tuning on these results",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "runs_spec_sha256": _sha(args.runs),
        "plans": {},
        "suites": {},
    }
    started = time.perf_counter()

    def save():
        record["seconds"] = round(time.perf_counter() - started, 2)
        _write(out / "final-evaluation.json", record)

    suites = {}
    for split in args.splits:
        folder = out / split
        plan = prepare_plan(
            args.data,
            args.project / "configs/evaluation/pixel.yaml",
            split=split,
            metrics=("ssim", "lpips"),
            protocol=args.project / "configs/evaluation/comparison.yaml",
            actions_config=None if split in args.without_actions else actions,
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
    for name, plan_path in suites.items():
        folder = plan_path.parent
        samples, metrics = folder / "samples", folder / "metrics"
        suite = record["suites"][name] = {"plan_sha256": record["plans"][name]["plan_sha256"]}
        try:
            for phase, function, arguments in (
                ("sample", generate_samples, (plan_path, args.runs, samples)),
                ("score", run_suite, (plan_path, samples / "predictions.json", metrics)),
            ):
                print(json.dumps({"suite": name, "phase": phase}), flush=True)
                begin = time.perf_counter()
                kwargs = {"device": args.device} if phase == "sample" else {}
                result = function(*arguments, **kwargs)
                suite[phase + "_seconds"] = round(time.perf_counter() - begin, 2)
            suite["status"] = result["status"]
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            suite.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            continue
        try:
            begin = time.perf_counter()
            horizon = action_horizon.evaluate(plan_path, samples / "predictions.json")
            _write(folder / "action-horizons.json", horizon)
            suite["horizon_seconds"] = round(time.perf_counter() - begin, 2)
        except (ValueError, KeyError) as error:
            suite["horizons"] = f"not scored: {error}"
        save()
    failed = [name for name, suite in record["suites"].items() if suite["status"] != "passed"]
    record["status"] = "completed_with_failures" if failed else "completed"
    record["failed_suites"] = failed
    save()


if __name__ == "__main__":
    main()
