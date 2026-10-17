"""Evaluate the separate flow-decoder CVAE follow-up on the fixed validation corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from click2future.evaluation.pilot import cvae_diagnostics, summarize_quality
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _write, prepare_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan
from click2future.training.multiscene_pilot import PROJECT, read_pilot
from click2future.training.train_cvae import require_tc2, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        require_tc2()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    pilot = read_pilot(args.root)
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["model_config"].get("decoder_kind") != "flow" or payload["state"]["step"] <= 0:
        raise ValueError("Use an actually trained flow-decoder CVAE checkpoint")
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    args.output.mkdir(parents=True)
    try:
        plan = prepare_plan(
            args.root / "data",
            PROJECT / "configs/evaluation/pixel.yaml",
            split="validation",
            metrics=("ssim",),
            protocol=PROJECT / "configs/evaluation/comparison.yaml",
            actions_config=PROJECT / "configs/evaluation/actions.yaml",
        )
        _write(args.output / "plan.json", plan)
        _write(
            args.output / "runs.json",
            {
                "sampling_steps": 32,
                "runs": [{"method": "E0", "run_id": "flow-E0", "checkpoint": str(checkpoint)}],
            },
        )
        for name, path in (("validation", args.output / "plan.json"), ("selected-object", None)):
            if path is None:
                prepare_target_plan(
                    args.output / "plan.json",
                    args.output / "target-controls",
                    config=PROJECT / "configs/evaluation/actions.yaml",
                    force_magnitude=0.4,
                )
                path = args.output / "target-controls/plan.json"
            print(json.dumps({"flow_evaluation": name}), flush=True)
            generate_samples(
                path,
                args.output / "runs.json",
                args.output / (name + "-samples"),
                device=args.device,
            )
            run_suite(
                path,
                args.output / (name + "-samples/predictions.json"),
                args.output / (name + "-metrics"),
            )
        cvae_diagnostics(
            args.root / "data",
            checkpoint,
            args.output / "cvae-diagnostics.json",
            device=args.device,
        )
        manifest = json.loads((args.output / "validation-metrics/comparison.json").read_text())
        reports = [
            json.loads((args.output / "validation-metrics" / r["trajectory"]).read_text())
            for r in manifest["runs"][0]["reports"]
        ]
        _write(
            args.output / "flow-results.json",
            {
                "status": "measured_validation_followup",
                "quality": summarize_quality(reports),
                "checkpoint_sha256": sha256(checkpoint),
                "training_steps": payload["state"]["step"],
                "model_config": payload["model_config"],
                "dataset_manifest_sha256": pilot["dataset_manifest_sha256"],
                "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "test_splits_evaluated": [],
                "scope": (
                    "Separate decoder follow-up chosen after validation appearance failures. "
                    "Not an independent final test or a pure latent-variable ablation."
                ),
            },
        )
    except BaseException as error:
        _write(args.output / "failure.json", {"error": str(error)})
        raise


if __name__ == "__main__":
    main()
