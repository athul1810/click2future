"""Isolated prototype loading; historical E5-E8 loaders are untouched."""

from pathlib import Path

import numpy as np
import torch

from click2future.models.object_motion import MOTION_CONTRACT, observe, render_motion
from click2future.models.surface_motion import SURFACE_CONTRACT, SurfaceTrajectoryDiffusion
from click2future.models.world_motion import WORLD_CONTRACT

from .conditioning import observable_condition
from .contact_geometry import observed_hulls
from .train_cvae import sha256
from .train_surface import SurfaceConfig, surface_config


def load_surface(path, device="cpu"):
    checksum = sha256(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = SurfaceConfig(**state["config"])
    config.validate()
    expected = {
        "trajectory_format_version": 5,
        "training_variant": "surface_messages",
        "motion_contract": MOTION_CONTRACT,
        "world_contract": WORLD_CONTRACT,
        "surface_contract": SURFACE_CONTRACT,
        "model_config": surface_config(config),
    }
    if (
        any(state[k] != v for k, v in expected.items())
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or not 0 < state["step"] <= config.steps
    ):
        raise ValueError("Unsupported or untrained surface checkpoint")
    root = Path(__file__).parents[1]
    for name in ("object_motion.py", "world_motion.py", "surface_motion.py"):
        key = "models/" + name
        if sha256(root / key) != state["provenance"]["source_sha256"][key]:
            raise ValueError("Surface model source differs")
    with torch.random.fork_rng(devices=[]):
        model = SurfaceTrajectoryDiffusion(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit, model.sigma_min, model.sigma_max = (
        config.force_limit,
        config.sigma_min,
        config.sigma_max,
    )
    if sha256(path) != checksum:
        raise ValueError("Surface checkpoint changed while loading")
    return model, state


@torch.inference_mode()
def sample_surface_videos(
    model, first_frame, selected_mask, force_xy, *, seeds=(0, 1, 2, 3), sampling_steps=16
):
    first = np.asarray(first_frame)
    if first.dtype != np.uint8 or first.shape != (64, 64, 3):
        raise ValueError("Require native 64px RGB")
    obs = observe(first, selected_mask)
    device = next(model.parameters()).device
    force = torch.as_tensor(np.array(force_xy, dtype=np.float32, copy=True), device=device)[None]
    frame = torch.from_numpy(first.copy()).to(device).permute(2, 0, 1)[None].float() / 255
    mask = torch.as_tensor(np.array(selected_mask, copy=True), device=device)[None, None]
    observable_condition(frame, mask, force, force_limit=model.force_limit)
    features = torch.from_numpy(obs.features[None]).to(device)
    geometry = {k: torch.from_numpy(v[None]).to(device) for k, v in observed_hulls(obs).items()}
    motion = model.sample(
        features,
        force,
        geometry,
        seeds=list(seeds),
        steps=sampling_steps,
        sigma_min=model.sigma_min,
        sigma_max=model.sigma_max,
    )
    return np.stack([render_motion(obs, m.cpu().numpy()) for m in motion])
