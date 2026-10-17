"""Controlled single-factor shifts, separate from the frozen scene-0.1/0.2 paths."""

import math

import numpy as np

from .controlled import make_controlled_scene
from .scene import make_scene

SPLITS = ("train", "validation", "id_test", "shape_ood", "texture_ood", "force_ood", "physics_test")
ID_TEXTURES = ("solid", "stripes")
OOD_TEXTURES = ("checker",)
OOD_FORCES = (1.3, 1.5)


def texture_scene(scene, *, textures=ID_TEXTURES):
    """Assign patterns using visible seeds only; never perturb physics or legacy RNG."""
    if not textures or not set(textures) <= {"solid", "stripes", "checker"}:
        raise ValueError("Choose supported nonempty textures")
    rng = np.random.default_rng(np.random.SeedSequence([scene["seeds"]["visible"], 0x544558]))
    for obj in scene["objects"]:
        obj["appearance"]["texture"] = str(rng.choice(textures))
    scene["schema_version"] = "scene-0.3"
    return scene


def make_textured_scene(seed, *, object_count=2, physics_seed=None):
    return texture_scene(make_scene(seed, object_count=object_count, physics_seed=physics_seed))


def make_generalization_scene(seed, *, split, action_variant, physics_seed):
    if split not in SPLITS:
        raise ValueError("Unknown generalization split")
    if split == "force_ood" and action_variant == "zero":
        raise ValueError("Zero force is ID, so it is excluded from force_ood")
    scene = make_controlled_scene(seed, action_variant=action_variant, physics_seed=physics_seed)
    texture_scene(scene, textures=OOD_TEXTURES if split == "texture_ood" else ID_TEXTURES)
    if split == "shape_ood":
        for obj in scene["objects"]:
            g = obj["geometry"]
            radius = g["radius"] if obj["shape"] == "circle" else min(g["width"], g["height"]) / 2
            obj.update(shape="triangle", geometry={"radius": radius})
    if split == "force_ood":
        force = scene["action"]["force_xy"]
        magnitude = OOD_FORCES[int(seed) % len(OOD_FORCES)]
        norm = math.hypot(*force)
        scene["action"]["force_xy"] = [component * magnitude / norm for component in force]
    # New namespaces prevent accidental mixing with the legacy dataset family.
    scene["scene_group_id"] = f"g03-scene-{int(seed):08d}"
    scene["condition_id"] = f"{scene['scene_group_id']}-{action_variant}"
    return scene
