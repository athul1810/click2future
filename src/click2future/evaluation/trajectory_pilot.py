"""Frozen validation/action checks for the separate pose-supervised E5 variant."""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_cvae import require_tc2, sha256

from .pilot import summarize_quality
from .suite import _write, prepare_plan, read_plan, run_suite
from .targets import prepare_target_plan
from .tracking import TRACKER_VERSION

PROJECT = Path(__file__).parents[3]


def sample_plan(plan_path, checkpoint, output, *, device="cpu", steps=16):
    plan = read_plan(plan_path)
    checkpoint, output = Path(checkpoint).resolve(), Path(output)
    if plan["split"] not in {"validation", "validation-selected-object"}:
        raise ValueError("This bounded pilot may not evaluate final-test splits")
    model, state = load_trajectory(checkpoint, device)
    provenance = state["provenance"]
    if (
        provenance["dataset_manifest_sha256"] != plan["input_sha256"]["dataset_manifest"]
        or provenance["split_manifest_sha256"] != plan["input_sha256"]["splits"]
    ):
        raise ValueError("Checkpoint and plan identify different frozen data")
    checksum = sha256(checkpoint)
    if output.exists():
        raise ValueError("Use a fresh prediction directory")
    output.mkdir(parents=True)
    videos, hashes = {}, {}
    started = time.perf_counter()
    try:
        for case in plan["cases"]:
            ref = load_rollout(plan["dataset"], case["sample_id"])
            generated = sample_trajectory_videos(
                model,
                ref["frames"][0],
                ref["selected_mask"],
                ref["force_xy"],
                seeds=plan["sampling_seeds"],
                sampling_steps=steps,
            )
            if generated.shape != (4, *ref["frames"].shape) or not np.all(
                generated[:, 0] == ref["frames"][0]
            ):
                raise ValueError("Generated RGB violates the native clip contract")
            name = case["sample_id"] + ".npz"
            np.savez_compressed(output / name, videos=generated)
            videos[case["sample_id"]], hashes[case["sample_id"]] = name, sha256(output / name)
        read_plan(plan_path)
        if sha256(checkpoint) != checksum:
            raise ValueError("Checkpoint changed during sampling")
        run = {
            "method": "E5",
            "run_id": "trajectory-E5",
            "training_seed": state["config"]["seed"],
            "training_steps": state["step"],
            "untrained_fixture": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checksum,
            "videos": videos,
            "video_sha256": hashes,
            "supervision": "future positions and checked replay angles; training only",
        }
        predictions = {
            "source": "generated_video",
            "evaluation_plan_sha256": plan["plan_sha256"],
            "sampling_seeds": plan["sampling_seeds"],
            "runs": [run],
        }
        _write(output / "predictions.json", predictions)
        _write(
            output / "sampling.json",
            {
                **predictions,
                "device": device,
                "sampling_steps": steps,
                "network_evaluations_per_sample": 2 * steps - 1,
                "sampler": "EDM Heun rho=7, no churn, float32",
                "seconds": time.perf_counter() - started,
                "source_sha256": {
                    "sample_trajectory.py": sha256(
                        PROJECT / "src/click2future/training/sample_trajectory.py"
                    ),
                    "object_motion.py": sha256(
                        PROJECT / "src/click2future/models/object_motion.py"
                    ),
                    "trajectory_pilot.py": sha256(Path(__file__)),
                },
            },
        )
        return predictions
    except Exception as error:
        _write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args(argv)
    if args.device == "cuda":
        require_tc2()
    if args.output.exists():
        parser.error("Keep attempted evaluations; use a fresh output directory")
    try:
        importlib.import_module("skimage.metrics")
    except ImportError as error:
        raise RuntimeError("Install the image-metrics extra before this SSIM pilot") from error
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    try:
        plan = prepare_plan(
            args.dataset,
            PROJECT / "configs/evaluation/pixel.yaml",
            split="validation",
            metrics=("ssim",),
            protocol=PROJECT / "configs/evaluation/comparison.yaml",
            actions_config=PROJECT / "configs/evaluation/actions.yaml",
        )
        _write(args.output / "plan.json", plan)
        prepare_target_plan(
            args.output / "plan.json",
            args.output / "target-controls",
            config=PROJECT / "configs/evaluation/actions.yaml",
            force_magnitude=0.4,
        )
        for name, plan_path in (
            ("validation", args.output / "plan.json"),
            ("selected-object", args.output / "target-controls/plan.json"),
        ):
            print(json.dumps({"sampling": name}), flush=True)
            sample_plan(
                plan_path,
                args.checkpoint,
                args.output / (name + "-samples"),
                device=args.device,
                steps=args.steps,
            )
            print(json.dumps({"scoring": name}), flush=True)
            run_suite(
                plan_path,
                args.output / (name + "-samples/predictions.json"),
                args.output / (name + "-metrics"),
            )
        comparison = json.loads((args.output / "validation-metrics/comparison.json").read_text())
        reports = [
            json.loads((args.output / "validation-metrics" / item["trajectory"]).read_text())
            for item in comparison["runs"][0]["reports"]
        ]
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        result = {
            "status": "measured_validation_followup",
            "method": "E5",
            "quality": summarize_quality(reports),
            "checkpoint_sha256": sha256(args.checkpoint),
            "training_steps": state["step"],
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "extractor_version": TRACKER_VERSION,
            "test_splits_evaluated": [],
            "sampling_steps": args.steps,
            "scope": (
                "one supervised trajectory-diffusion seed; "
                "not a pure E2/E3 ablation or SOTA comparison"
            ),
        }
        _write(args.output / "trajectory-results.json", result)
        print(json.dumps(result["quality"]), flush=True)
    except Exception as error:
        _write(args.output / "failure.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
