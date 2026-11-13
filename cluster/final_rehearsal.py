"""Run the E0/E2/E3 validation comparison on one shared plan.

Includes force-0.4 selection controls, SSIM/LPIPS and both action horizons.
Reserved test splits are not read."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from click2future.evaluation import action_horizon
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _write, prepare_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--metrics", nargs="+", default=["ssim", "lpips"])
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
        "scope": "validation-only rehearsal; no test split opened",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "phases": {},
    }
    started = time.perf_counter()

    def phase(name, function, *a, **k):
        t = time.perf_counter()
        print(json.dumps({"phase": name, "state": "started"}), flush=True)
        value = function(*a, **k)
        record["phases"][name] = round(time.perf_counter() - t, 2)
        _write(out / "rehearsal.json", record)
        return value

    try:
        plan = phase(
            "prepare_validation",
            prepare_plan,
            args.data,
            args.project / "configs/evaluation/pixel.yaml",
            split="validation",
            metrics=tuple(args.metrics),
            protocol=args.project / "configs/evaluation/comparison.yaml",
            actions_config=actions,
        )
        _write(out / "plan.json", plan)
        phase(
            "prepare_selection",
            prepare_target_plan,
            out / "plan.json",
            out / "target-controls",
            config=actions,
            force_magnitude=0.4,
        )
        suites = {"validation": out / "plan.json", "selection": out / "target-controls/plan.json"}
        for name, path in suites.items():
            samples = out / (name + "-samples")
            phase("sample_" + name, generate_samples, path, args.runs, samples, device=args.device)
            report = phase(
                "score_" + name,
                run_suite,
                path,
                samples / "predictions.json",
                out / (name + "-metrics"),
            )
            record[name + "_status"] = report["status"]
            horizon = phase(
                "horizons_" + name, action_horizon.evaluate, path, samples / "predictions.json"
            )
            _write(out / f"action-horizon-{name}.json", horizon)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
        _write(out / "rehearsal.json", record)


if __name__ == "__main__":
    main()
