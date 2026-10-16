"""Sample an EMA checkpoint from observable-only NPZ inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from click2future.models.pilot_denoiser import PilotDenoiser

from .conditioning import observable_condition
from .diffusion import Diffusion
from .train_diffusion import TrainConfig


def load_generator(checkpoint: Path, device="cpu"):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if state["checkpoint_version"] != 1:
        raise ValueError("Unknown checkpoint version")
    config = TrainConfig(**state["config"])
    config.validate()
    model = PilotDenoiser(
        base_channels=config.base_channels,
        conditioning=config.conditioning,
        auxiliary=config.auxiliary,
    ).to(device)
    model.load_state_dict(state["ema"])
    model.eval()
    return (
        model,
        Diffusion(config.diffusion_steps, device, prediction_type=config.prediction_type),
        state,
    )


def sample_videos(
    model,
    diffusion,
    first_frame,
    selected_mask,
    force_xy,
    *,
    seeds=(100, 101, 102, 103),
    future_frames=31,
    sampling_steps=32,
):
    """Single observable request -> uint8 [K,T,H,W,3], without simulator data."""
    first_frame = np.asarray(first_frame)
    if first_frame.dtype != np.uint8 or first_frame.ndim != 3 or first_frame.shape[-1] != 3:
        raise ValueError("first_frame must be uint8 [H,W,3]")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct explicit sampling seeds")
    device = diffusion.alpha_bar.device
    frame = torch.from_numpy(first_frame.copy()).to(device).permute(2, 0, 1)[None].float() / 255
    mask = torch.as_tensor(np.array(selected_mask, copy=True), device=device)
    if mask.shape != first_frame.shape[:2]:
        raise ValueError("selected_mask must be [H,W]")
    force = torch.as_tensor(np.array(force_xy, copy=True), device=device, dtype=torch.float32)
    condition = observable_condition(frame, mask[None, None], force[None])
    videos = []
    for seed in seeds:
        generated = diffusion.sample(
            model,
            condition,
            future_frames=future_frames,
            sampling_steps=sampling_steps,
            seed=seed,
        )
        pixels = ((generated[0].permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255)
        videos.append(np.concatenate((first_frame[None], pixels.byte().cpu().numpy()), axis=0))
    return np.stack(videos)


def save_preview(videos, path: Path):
    """Native-resolution rows of model samples; NPZ remains the evaluation artifact."""
    images = [Image.fromarray(np.concatenate(videos[:, t], axis=1)) for t in range(videos.shape[1])]
    images[0].save(path, save_all=True, append_images=images[1:], duration=125, loop=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--condition", type=Path, required=True, help="first_frame, selected_mask, force_xy"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    parser.add_argument("--future-frames", type=int, default=31)
    parser.add_argument("--sampling-steps", type=int, default=32)
    args = parser.parse_args(argv)
    if args.require_slurm and not os.environ.get("SLURM_JOB_ID"):
        parser.error("TC2 application execution requires an allocated Slurm job")
    if args.output.resolve() in {args.condition.resolve(), args.checkpoint.resolve()}:
        parser.error("Output must not overwrite an input")
    model, diffusion, state = load_generator(args.checkpoint, args.device)
    with np.load(args.condition, allow_pickle=False) as archive:
        # Deliberately read three named fields, never pass the entire archive to a model.
        inputs = {key: archive[key] for key in ("first_frame", "selected_mask", "force_xy")}
    started = time.perf_counter()
    videos = sample_videos(
        model,
        diffusion,
        **inputs,
        seeds=args.seeds,
        future_frames=args.future_frames,
        sampling_steps=args.sampling_steps,
    )
    seconds = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, videos=videos)
    save_preview(videos, args.output.with_suffix(".gif"))
    record = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "condition_sha256": hashlib.sha256(args.condition.read_bytes()).hexdigest(),
        "training_step": state["step"],
        "prediction_type": diffusion.prediction_type,
        "weights": "ema",
        "seed_policy": "private generator per sample; paired seeds across changed actions",
        "seeds": args.seeds,
        "sampler": "DDIM eta=0, clipped x0 with recomputed epsilon",
        "sampling_steps": args.sampling_steps,
        "seconds_including_uint8_transfer": seconds,
        "shape": list(videos.shape),
        "scope": "Generated model video; quality and action response require evaluation",
    }
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
