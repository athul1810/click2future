"""Strict loading for the E10 force-routing checkpoints."""

from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.routed_motion import RoutedTrajectoryDiffusion

from .train_cvae import sha256
from .train_routing import FORMAT_VERSION, VARIANT, RoutingConfig, routing_model_config


def load_routing(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    checksum = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = RoutingConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != FORMAT_VERSION
        or state["training_variant"] != VARIANT
        or state["motion_contract"] != MOTION_CONTRACT
        or state["model_config"] != routing_model_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained force-routing checkpoint")
    root = Path(__file__).parents[1]
    recorded = state["provenance"]["source_sha256"]
    for name in ("models/object_motion.py", "models/routed_motion.py"):
        if recorded[name] != sha256(root / name):
            raise ValueError("Trajectory model source differs from its checkpoint: " + name)
    with torch.random.fork_rng(devices=[]):
        model = RoutedTrajectoryDiffusion(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit = config.force_limit
    model.sigma_min, model.sigma_max = config.sigma_min, config.sigma_max
    if sha256(checkpoint) != checksum:
        raise ValueError("Checkpoint changed while loading")
    return model, state
