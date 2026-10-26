"""Observable-only object-trajectory diffusion sampling and rigid RGB decoding."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from click2future.models.object_motion import (
    MOTION_CONTRACT,
    TrajectoryDiffusion,
    observe,
    render_motion,
)

from .conditioning import observable_condition
from .train_cvae import sha256
from .train_trajectory import TrajectoryConfig, model_config


def load_trajectory(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    checksum = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("trajectory_format_version") == 4:
        from .memory_checkpoint import load_memory

        return load_memory(checkpoint, device)
    if state.get("trajectory_format_version") == 3:
        from .world_checkpoint import load_world

        return load_world(checkpoint, device)
    if state.get("trajectory_format_version") == 2:
        from .contact_checkpoint import load_contact

        return load_contact(checkpoint, device)
    config = TrajectoryConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != 1
        or state["motion_contract"] != MOTION_CONTRACT
        or state["model_config"] != model_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained trajectory checkpoint")
    source = Path(__file__).parents[1] / "models/object_motion.py"
    if state["provenance"]["source_sha256"]["object_motion.py"] != sha256(source):
        raise ValueError("Trajectory model source differs from its checkpoint")
    with torch.random.fork_rng(devices=[]):
        model = TrajectoryDiffusion(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit = config.force_limit
    model.sigma_min, model.sigma_max = config.sigma_min, config.sigma_max
    if sha256(checkpoint) != checksum:
        raise ValueError("Checkpoint changed while loading")
    return model, state


@torch.inference_mode()
def sample_trajectory_videos(
    model,
    first_frame,
    selected_mask,
    force_xy,
    *,
    seeds=(0, 1, 2, 3),
    sampling_steps=16,
    force_limit=None,
    cancelled=None,
):
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or isinstance(sampling_steps, bool)
        or not isinstance(sampling_steps, int)
        or not 2 <= sampling_steps <= 128
    ):
        raise ValueError("Use distinct nonnegative integer seeds and 2-128 sampling steps")
    first_frame = np.asarray(first_frame)
    if first_frame.shape != (64, 64, 3) or first_frame.dtype != np.uint8:
        raise ValueError("This trajectory checkpoint requires native 64px uint8 RGB")
    obs = observe(first_frame, selected_mask)
    device = next(model.parameters()).device
    force = torch.as_tensor(np.array(force_xy, dtype=np.float32, copy=True), device=device)[None]
    frame = torch.from_numpy(first_frame.copy()).to(device).permute(2, 0, 1)[None].float() / 255
    mask = torch.as_tensor(np.array(selected_mask, copy=True), device=device)[None, None]
    force_limit = model.force_limit if force_limit is None else force_limit
    if not 0 < force_limit <= model.force_limit:
        raise ValueError("The requested force bound exceeds the checkpoint's range")
    observable_condition(frame, mask, force, force_limit=force_limit)
    features = torch.from_numpy(obs.features[None]).to(device)
    videos = []
    for seed in seeds:
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("Generation cancelled")
        motion = model.sample(
            features,
            force,
            seeds=[seed],
            steps=sampling_steps,
            sigma_min=model.sigma_min,
            sigma_max=model.sigma_max,
        )[0]
        videos.append(render_motion(obs, motion.cpu().numpy()))
    return np.stack(videos)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output file")
    model, state = load_trajectory(args.checkpoint, args.device)
    with np.load(args.condition, allow_pickle=False) as archive:
        condition = {key: archive[key] for key in ("first_frame", "selected_mask", "force_xy")}
    videos = sample_trajectory_videos(
        model, **condition, seeds=args.seeds, sampling_steps=args.sampling_steps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, videos=videos)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": sha256(args.checkpoint),
                "step": state["step"],
                "seeds": args.seeds,
                "sampling_steps": args.sampling_steps,
                "sampler": "EDM rho-7 Heun, no churn, float32 integration",
                "network_evaluations_per_sample": 2 * args.sampling_steps - 1,
                "scope": (
                    "supervised trajectory diffusion and rigid RGB decoding; quality unverified"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
