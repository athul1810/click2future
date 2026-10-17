"""Training-scene sampler diagnostics; never a final held-out comparison."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.evaluation.tracking import TrackerConfig, track_video

from .sample_diffusion import load_generator, sample_videos, save_preview
from .train_diffusion import TrainConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[32, 128, 1000])
    args = parser.parse_args(argv)
    if args.require_slurm and not os.environ.get("SLURM_JOB_ID"):
        parser.error("TC2 application execution requires a Slurm allocation")
    model, diffusion, state = load_generator(args.checkpoint, args.device)
    sample_id = state["provenance"]["training_sample_ids"][0]
    reference = load_rollout(args.dataset, sample_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for weights in ("ema", "model"):
        model.load_state_dict(state[weights])
        for steps in args.sampling_steps:
            started = time.perf_counter()
            videos = sample_videos(
                model,
                diffusion,
                reference["frames"][0],
                reference["selected_mask"],
                reference["force_xy"],
                seeds=(100,),
                future_frames=len(reference["frames"]) - 1,
                sampling_steps=steps,
                force_limit=TrainConfig(**state["config"]).force_limit,
            )
            seconds = time.perf_counter() - started
            tracked = track_video(
                videos[0],
                reference["instance_masks"][0],
                reference["object_valid"],
                TrackerConfig(),
            )
            name = f"{weights}-{steps}"
            np.savez_compressed(args.output_dir / f"{name}.npz", videos=videos)
            save_preview(videos, args.output_dir / f"{name}.gif")
            records.append(
                {
                    "weights": weights,
                    "steps": steps,
                    "seconds": seconds,
                    "tracked_future_observations": int(tracked.tracked[1:].sum()),
                    "expected_future_observations": int(reference["object_valid"].sum())
                    * (len(videos[0]) - 1),
                }
            )
    record = {
        "training_step": state["step"],
        "prediction_type": diffusion.prediction_type,
        "representation": diffusion.representation,
        "sample_id": sample_id,
        "seed": 100,
        "scope": "Training-scene sampler diagnostic; not model selection or held-out evidence",
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "records": records,
    }
    (args.output_dir / "diagnostic.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
