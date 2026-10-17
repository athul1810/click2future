"""Fixed 16/32/64-step validation study on the same trained E3 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _digest, _write, read_plan, run_suite
from click2future.training.multiscene_pilot import read_pilot
from click2future.training.train_cvae import require_tc2, sha256

ACTIONS = ("zero", "forward", "reverse", "collision_0_to_1")


def prepare_study(parent_path):
    """Select fixed families and the first four groups before reading model outputs."""
    plan = read_plan(parent_path)
    if plan["split"] != "validation" or plan.get("intervention"):
        raise ValueError("The study requires an ordinary held-out validation plan")
    groups = sorted({c["scene_group_id"] for c in plan["cases"]})[:4]
    selected, excluded = [], []
    for case in plan["cases"]:
        action = case["condition_id"].removeprefix(case["scene_group_id"] + "-")
        (selected if case["scene_group_id"] in groups and action in ACTIONS else excluded).append(
            case
        )
    if len(selected) != len(groups) * len(ACTIONS) or not groups:
        raise ValueError("Each selected validation scene needs all four fixed action families")
    study = plan["comparison_protocol"]["validation_study"]
    if len(selected) > study["max_conditions"] or study["sampling_steps"] != [16, 32, 64]:
        raise ValueError("Study exceeds the reserved protocol")
    parent_hash = plan.pop("plan_sha256")
    plan["cases"] = selected
    plan["action_groups"] = [g for g in plan["action_groups"] if g["scene_group_id"] in groups]
    plan["sampler_study"] = {
        "parent_plan_sha256": parent_hash,
        "selection": "first four sorted validation scene groups; fixed action families",
        "actions": list(ACTIONS),
        "excluded_cases": excluded,
        "scope": "Declared validation subset, not a full held-out comparison or test selection",
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def run(root, parent, output, *, device):
    root, output = Path(root).resolve(), Path(output).resolve()
    if device == "cuda":
        require_tc2()
    pilot = read_pilot(root)
    plan = prepare_study(parent)
    if output.exists():
        raise ValueError("Use a new sampler-study directory")
    output.mkdir(parents=True)
    _write(output / "plan.json", plan)
    records = []
    try:
        for steps in (16, 32, 64):
            folder = output / str(steps)
            folder.mkdir()
            _write(
                folder / "runs.json",
                {
                    "sampling_steps": steps,
                    "runs": [
                        {
                            "method": "E3",
                            "run_id": "pilot-E3",
                            "checkpoint": str(root / "E3/checkpoint.pt"),
                        }
                    ],
                },
            )
            print(json.dumps({"sampler_study_steps": steps}), flush=True)
            sampling = generate_samples(
                output / "plan.json", folder / "runs.json", folder / "samples", device=device
            )
            summary = run_suite(
                output / "plan.json", folder / "samples/predictions.json", folder / "metrics"
            )
            manifest = json.loads((folder / "metrics/comparison.json").read_text())
            reports = [
                json.loads((folder / "metrics" / r["trajectory"]).read_text())
                for r in manifest["runs"][0]["reports"]
            ]
            records.append(
                {
                    "sampling_steps": steps,
                    "checkpoint_sha256": sampling["runs"][0]["checkpoint_sha256"],
                    "seconds_including_load_and_export": sampling["runs"][0][
                        "seconds_including_load_and_export"
                    ],
                    "quality": summarize_quality(reports),
                    "metrics": summary["methods"][0]["metrics"],
                    "actions": json.loads((folder / "metrics/actions.json").read_text())["summary"],
                }
            )
            _write(output / "progress.json", {"completed_settings": records})
        record = {
            "status": "measured_validation_study",
            "settings": records,
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "runner_sha256": sha256(Path(__file__)),
            "dataset_manifest_sha256": pilot["dataset_manifest_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "sampling_seeds": plan["sampling_seeds"],
            "scope": "Same E3 checkpoint and fixed validation cases. No training or test tuning.",
        }
        _write(output / "study.json", record)
        return record
    except BaseException as error:
        _write(output / "failure.json", {"error": str(error), "completed_settings": records})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    run(args.root, args.parent_plan, args.output, device=args.device)


if __name__ == "__main__":
    main()
