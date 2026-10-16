"""Auxiliary target/loss components; the dataset diffusion trainer is still pending."""

from .auxiliary import (
    AuxiliaryLoss,
    AuxiliaryLossConfig,
    auxiliary_loss,
    collision_balance,
)
from .targets import (
    AuxiliaryTargets,
    build_auxiliary_targets,
    canonical_object_order,
    to_scene_order,
)

__all__ = [
    "AuxiliaryLoss",
    "AuxiliaryLossConfig",
    "AuxiliaryTargets",
    "auxiliary_loss",
    "build_auxiliary_targets",
    "canonical_object_order",
    "collision_balance",
    "to_scene_order",
]
