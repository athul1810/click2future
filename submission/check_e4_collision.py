"""Actual E4 overfit/sampling check on the first frozen training collision clip."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.evaluation.pilot import score_heads, summarize_quality
from click2future.evaluation.suite import _write
from click2future.evaluation.tracking import TrackerConfig, track_video
from click2future.evaluation.video import evaluate_videos
from click2future.training.multiscene_pilot import PROJECT, read_pilot
from click2future.training.sample_diffusion import load_generator, sample_videos
from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.train_diffusion import main as train_diffusion
from click2future.training.train_diffusion import training_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_tc2()
    if args.output.exists():
        raise ValueError("Use a new overfit-check directory")
    pilot = read_pilot(args.root)
    dataset = args.root / "data"
    ids, _ = training_index(dataset, 1)
    ref = load_rollout(dataset, ids[0])
    known = track_video(ref["frames"], ref["instance_masks"][0], ref["object_valid"])
    positives = int(ref["collision_onset"][:, 0, 1].sum())
    if (
        ref["scene_spec"]["action_variant"] != "collision_0_to_1"
        or not positives
        or not known.tracked.all()
    ):
        raise ValueError(
            "The declared first training clip must have contacts and complete reference tracks"
        )
    args.output.mkdir(parents=True)
    config = PROJECT / "configs/training/diffusion_collision_overfit.json"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    train_diffusion(
        [
            "--dataset",
            str(dataset),
            "--run-dir",
            str(args.output / "run"),
            "--config",
            str(config),
            "--device",
            "cuda",
            "--require-slurm",
        ]
    )
    checkpoint = args.output / "run/checkpoint.pt"
    model, diffusion, _ = load_generator(checkpoint, "cuda")
    videos = sample_videos(
        model,
        diffusion,
        ref["frames"][0],
        ref["selected_mask"],
        ref["force_xy"],
        seeds=[0, 1, 2, 3],
        future_frames=31,
        sampling_steps=32,
        force_limit=1.5,
    )
    folder = args.output / "samples/pilot-E4"
    folder.mkdir(parents=True)
    np.savez_compressed(folder / (ids[0] + ".npz"), videos=videos)
    plan = dict(
        dataset=str(dataset),
        cases=[dict(sample_id=ids[0])],
        sampling_seeds=[0, 1, 2, 3],
        config=str(PROJECT / "configs/evaluation/pixel.yaml"),
    )
    metrics = yaml.safe_load(Path(plan["config"]).read_text())
    report, _ = evaluate_videos(
        ref,
        videos,
        tracker=TrackerConfig(**metrics["tracker"]),
        max_gap_pixels=metrics["max_gap_pixels"],
        event_tolerance=1,
        motion_window=metrics["motion_window"],
        motion_threshold=metrics["motion_threshold"],
        motion_rule=metrics["motion_rule"],
        run_id="E4-collision-overfit",
    )
    _write(args.output / "rgb.json", report)
    score_heads(
        plan,
        dict(checkpoint=str(checkpoint), config=str(config)),
        args.output / "samples",
        args.output / "heads",
        device="cuda",
    )
    _write(
        args.output / "results.json",
        dict(
            status="measured_training_only_overfit_check",
            sample_id=ids[0],
            reference_contact_onsets=positives,
            quality=summarize_quality([report]),
            dataset_manifest_sha256=pilot["dataset_manifest_sha256"],
            checkpoint_sha256=sha256(checkpoint),
            source_revision=os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
            scope=(
                "One fixed training clip, four prior-noise seeds. "
                "Not generalization or physical uncertainty evidence."
            ),
        ),
    )


if __name__ == "__main__":
    main()
