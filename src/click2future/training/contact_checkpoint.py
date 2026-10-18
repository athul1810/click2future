"""Strict loading for the separately trained contact-regularized checkpoint."""

from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT, TrajectoryDiffusion

from .train_contact import ContactConfig
from .train_cvae import sha256
from .train_trajectory import model_config


def load_contact(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    checksum = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ContactConfig(**state["config"])
    config.validate()
    source = Path(__file__).parents[1] / "models/object_motion.py"
    if (
        state["trajectory_format_version"] != 2
        or state["training_variant"] != "contact_regularized"
        or state["motion_contract"] != MOTION_CONTRACT
        or state["model_config"] != model_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained contact-regularized checkpoint")
    if state["provenance"]["source_sha256"]["models/object_motion.py"] != sha256(source):
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
