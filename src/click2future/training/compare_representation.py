"""Matched RGB/residual training-clip diagnostic; no held-out quality claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.metrics import trajectory_metrics
from click2future.evaluation.tracking import TRACKER_VERSION, TrackerConfig, track_video

from .sample_diffusion import load_generator, sample_videos, save_preview
from .train_diffusion import TrainConfig
from .train_diffusion import main as train_main


def score_samples(reference, videos):
    """Score RGB tracks, keeping missing trajectories undefined, plus a copy baseline."""
    tracker = TrackerConfig()
    tracks = [
        track_video(clip, reference["instance_masks"][0], reference["object_valid"], tracker)
        for clip in videos
    ]
    report = trajectory_metrics(
        np.stack([t.positions for t in tracks]),
        reference["positions"],
        reference["object_valid"],
        np.stack([t.tracked for t in tracks]),
        selected_object=reference["selected_object_id"],
    )
    report["extractor_version"] = TRACKER_VERSION
    report["tracker_settings"] = asdict(tracker)
    report["rgb_mae_future_0_1"] = float(
        np.abs(videos[:, 1:].astype(np.float32) - reference["frames"][None, 1:]).mean() / 255
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/diffusion_one_clip_boundary.json")
    )
    parser.add_argument(
        "--until", type=int, help="Same stopping step in both runs; LR horizon unchanged"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("TC2 GPU diagnostics require an allocated Slurm job")
    if args.output_dir.exists():
        parser.error("Use a new output directory; never overwrite an attempted experiment")
    config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    config.validate()
    if config.limit_train != 1:
        parser.error("This diagnostic requires limit_train=1")
    if config.diffusion_steps < 32:
        parser.error("The fixed 32-step sampler needs diffusion_steps >= 32")
    stop = config.steps if args.until is None else args.until
    if not 0 < stop <= config.steps:
        parser.error("--until must be in 1..configured steps")
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True)
    records, reference = [], None
    try:
        for representation in ("rgb", "frame0_residual"):
            run = args.output_dir / representation
            run.mkdir()
            cfg = run / "config.json"
            cfg.write_text(
                json.dumps(asdict(replace(config, representation=representation)), indent=2)
            )
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            train_main(
                [
                    "--dataset",
                    str(args.dataset),
                    "--run-dir",
                    str(run),
                    "--config",
                    str(cfg),
                    "--device",
                    args.device,
                    "--until",
                    str(stop),
                ]
            )
            checkpoint = run / "checkpoint.pt"
            model, diffusion, state = load_generator(checkpoint, args.device)
            sample_id = state["provenance"]["training_sample_ids"][0]
            reference = load_rollout(args.dataset, sample_id)
            videos = sample_videos(
                model,
                diffusion,
                reference["frames"][0],
                reference["selected_mask"],
                reference["force_xy"],
                seeds=(0, 1, 2, 3),
                future_frames=len(reference["frames"]) - 1,
                sampling_steps=32,
                force_limit=config.force_limit,
            )
            np.savez_compressed(run / "videos.npz", videos=videos)
            save_preview(videos, run / "videos.gif")
            records.append(
                {
                    "representation": representation,
                    "sample_id": sample_id,
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "training": json.loads((run / f"summary-{stop}.json").read_text()),
                    "scores": score_samples(reference, videos),
                }
            )
            del model, diffusion, state
            if args.device == "cuda":
                torch.cuda.empty_cache()
        copied = np.broadcast_to(reference["frames"][0], (4, *reference["frames"].shape)).copy()
        record = {
            "status": "measured_training_only_diagnostic",
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "controls": (
                "Only representation changes; same architecture/data/loss/seed/"
                "LR horizon/steps/sampler"
            ),
            "sampling_steps": 32,
            "sampling_seeds": [0, 1, 2, 3],
            "copy_frame0_baseline": score_samples(reference, copied),
            "runs": records,
            "scope": (
                "One training clip only. Copying static pixels is not learned motion "
                "or action adherence."
            ),
        }
        (args.output_dir / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "output": str(args.output_dir)}), flush=True)
    except Exception as error:
        (args.output_dir / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error)})
        )
        raise


if __name__ == "__main__":
    main()
