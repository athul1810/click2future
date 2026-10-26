"""Strict E7 loading, distinct from the existing E5/E6 checkpoints."""

from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.world_motion import WORLD_CONTRACT, WorldTrajectoryDiffusion

from .train_cvae import sha256
from .train_trajectory import model_config
from .train_world import WorldConfig


def load_world(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = WorldConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != 3
        or state["training_variant"] != "shared_world_relational"
        or state["motion_contract"] != MOTION_CONTRACT
        or state["world_contract"] != WORLD_CONTRACT
        or state["model_config"]
        != {
            **model_config(config),
            "world_condition": config.world_condition,
            "relational": config.relational,
        }
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained world checkpoint")
    for name in ("object_motion.py", "world_motion.py"):
        key = "models/" + name
        if state["provenance"]["source_sha256"][key] != sha256(Path(__file__).parents[1] / key):
            raise ValueError("World model source differs: " + name)
    with torch.random.fork_rng(devices=[]):
        model = WorldTrajectoryDiffusion(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit = config.force_limit
    model.sigma_min, model.sigma_max = config.sigma_min, config.sigma_max
    if sha256(checkpoint) != digest:
        raise ValueError("World checkpoint changed while loading")
    return model, state
