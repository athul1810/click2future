"""Strict E8 loading; E5/E6/E7 checkpoints keep their original model code."""

from pathlib import Path

import torch

from click2future.models.contact_memory import MEMORY_CONTRACT, ContactMemoryDiffusion
from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.world_motion import WORLD_CONTRACT

from .train_cvae import sha256
from .train_memory import MemoryConfig, frozen_digest, memory_model_config


def load_memory(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    checksum = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = MemoryConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != 4
        or state["training_variant"] != "temporal_residual"
        or state["motion_contract"] != MOTION_CONTRACT
        or state["world_contract"] != WORLD_CONTRACT
        or state["memory_contract"] != MEMORY_CONTRACT
        or state["model_config"] != memory_model_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or not 0 < state["step"] <= config.steps
    ):
        raise ValueError("Unsupported or untrained temporal residual checkpoint")
    for name in ("object_motion.py", "world_motion.py", "contact_memory.py"):
        key = "models/" + name
        if state["provenance"]["source_sha256"][key] != sha256(Path(__file__).parents[1] / key):
            raise ValueError("Memory model source differs: " + name)
    with torch.random.fork_rng(devices=[]):
        model = ContactMemoryDiffusion(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    if frozen_digest(model) != state["frozen_backbone_sha256"]:
        raise ValueError("Frozen backbone identity differs")
    model.to(device).eval()
    model.force_limit, model.sigma_min, model.sigma_max = (
        config.force_limit,
        config.sigma_min,
        config.sigma_max,
    )
    if sha256(checkpoint) != checksum:
        raise ValueError("Checkpoint changed while loading")
    return model, state
