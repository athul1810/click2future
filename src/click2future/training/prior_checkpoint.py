"""Strict format-7 loading; format 6 is reserved for the team's E10 study."""

from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.prior_motion import PRIOR_CONTRACT, PriorTrajectoryModel
from click2future.models.world_motion import WORLD_CONTRACT

from .train_cvae import sha256
from .train_prior import PriorConfig, prior_config, prior_sources


def load_prior(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = PriorConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != 7
        or state["training_variant"] != "prior_only_comparison"
        or state["motion_contract"] != MOTION_CONTRACT
        or state["world_contract"] != WORLD_CONTRACT
        or state["prior_contract"] != PRIOR_CONTRACT
        or state["model_config"] != prior_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained prior comparison checkpoint")
    for name in ("models/object_motion.py", "models/world_motion.py", "models/prior_motion.py"):
        if state["provenance"]["source_sha256"][name] != prior_sources()[name]:
            raise ValueError("Prior model source differs: " + name)
    with torch.random.fork_rng(devices=[]):
        model = PriorTrajectoryModel(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit = config.force_limit
    model.sigma_min, model.sigma_max = config.sigma_min, config.sigma_max
    if sha256(checkpoint) != digest:
        raise ValueError("Prior checkpoint changed while loading")
    return model, state
