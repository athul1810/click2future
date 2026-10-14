"""Measure the pilot denoiser on synthetic tensors, not dataset training or video quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from click2future.models.pilot_denoiser import PilotDenoiser


def profile(
    *,
    device: str,
    base_channels: int,
    batch_size: int,
    future_frames: int,
    resolution: int,
    warmup: int,
    steps: int,
    auxiliary: bool = False,
) -> dict:
    if min(batch_size, future_frames, resolution, steps) < 1 or warmup < 0:
        raise ValueError("Dimensions/steps must be positive and warmup nonnegative")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    model = PilotDenoiser(base_channels, auxiliary=auxiliary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_cuda = device == "cuda"
    precision = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and precision == torch.float16)
    future = torch.randn(batch_size, 3, future_frames, resolution, resolution, device=device)
    target = torch.randn_like(future)
    first_frame = torch.randn(batch_size, 3, resolution, resolution, device=device)
    mask = torch.zeros(batch_size, 1, resolution, resolution, device=device)
    mask[:, :, resolution // 4 : resolution // 2, resolution // 4 : resolution // 2] = 1
    force = torch.tensor([0.5, -0.25], device=device).expand(batch_size, -1)
    timesteps = torch.full((batch_size,), 500, device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=precision) if use_cuda else nullcontext()
        with context:
            result = model(future, timesteps, first_frame, mask, force)
            loss = (result.predicted_noise.float() - target).square().mean()
            if auxiliary:
                loss = loss + result.positions.float().square().mean()
                pair_mask = ~torch.eye(model.max_objects, dtype=torch.bool, device=device)
                loss = loss + result.collision_logits[..., pair_mask].float().square().mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite synthetic profile loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(steps):
        start = time.perf_counter()
        step()
        if use_cuda:
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    report = {
        "status": "passed",
        "kind": "synthetic_denoiser_forward_backward_profile",
        "limitation": "Not dataset training, a sampler, or a model-quality result",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name() if use_cuda else None,
        "precision": str(precision) if use_cuda else "torch.float32",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_channels": base_channels,
        "batch_size": batch_size,
        "future_frames": future_frames,
        "resolution": resolution,
        "auxiliary": auxiliary,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds_per_step_mean": sum(durations) / len(durations),
        "seconds_per_step_median": statistics.median(durations),
        "seconds_per_step_min": min(durations),
        "seconds_per_step_max": max(durations),
        "step_durations_seconds": durations,
        "timing_stable": max(durations) <= 2 * min(durations),
        "timing_note": "Short synthetic profile; do not use unstable timings as a training budget",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if use_cuda else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--future-frames", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.require_slurm and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("This command requires an allocated Slurm job")
        config_bytes = args.config.read_bytes()
        config = yaml.safe_load(config_bytes)
        settings = {
            "device": args.device,
            "base_channels": args.base_channels
            if args.base_channels is not None
            else config["model"]["base_channels"],
            "batch_size": args.batch_size
            if args.batch_size is not None
            else config["profile"]["batch_size"],
            "future_frames": args.future_frames
            if args.future_frames is not None
            else config["data"]["future_frames"],
            "resolution": config["data"]["resolution"],
            "warmup": args.warmup if args.warmup is not None else config["profile"]["warmup_steps"],
            "steps": args.steps if args.steps is not None else config["profile"]["measured_steps"],
            "auxiliary": args.auxiliary,
        }
        report = profile(**settings)
        report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
