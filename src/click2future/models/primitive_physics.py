"""PhysGen-inspired image-derived circles with conservative hull fallback.

Circle recognition requires an exact solid-color ellipse-mask reconstruction.
Its inferred size/centre retain subpixel uncertainty. No scene recipe enters.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
import pymunk
from PIL import Image, ImageDraw

from .object_motion import observe
from .physics_baseline import PhysicsBaseline, rollout_observation
from .world_motion import sample_world

PRIMITIVE_PHYSICS = "observed-circle-or-convex-hull-physics-1"


def fit_circles(observation):
    fits = []
    for mask in observation.masks:
        y, x = np.nonzero(mask)
        box = tuple(map(int, (x.min(), y.min(), x.max(), y.max())))
        reconstruction = Image.new("1", (64, 64))
        ImageDraw.Draw(reconstruction).ellipse(box, fill=1)
        circle = bool(
            abs((box[2] - box[0]) - (box[3] - box[1])) <= 1
            and np.array_equal(np.asarray(reconstruction), mask)
            and np.all(observation.first[mask] == observation.first[mask][0])
        )
        fits.append(
            {
                "circle": circle,
                "radius": ((box[2] - box[0]) + (box[3] - box[1])) / (4 * 64) if circle else 0,
                "color": tuple(int(v) for v in observation.first[mask][0]),
            }
        )
    return fits


def primitive_prediction(observation, force, world, *, future_frames=31):
    fits = fit_circles(observation)
    result = rollout_observation(
        observation,
        force,
        world,
        future_frames=future_frames,
        circle_radii=[fit["radius"] for fit in fits],
    )
    motion = result["motion"].copy()
    # The silhouette of a uniform circle has no observable orientation. Keep
    # its true angular state inside the solver, but do not rotate a pixel disk.
    for i, fit in enumerate(fits):
        if fit["circle"]:
            motion[:, i, 2:] = 0
    sprites = [
        Image.fromarray(np.dstack((observation.first, mask.astype(np.uint8) * 255)))
        for mask in observation.masks
    ]
    frames = [observation.first.copy()]
    for frame, state in enumerate(motion, 1):
        canvas = Image.fromarray(observation.background).convert("RGBA")
        for i, value in enumerate(state):
            if fits[i]["circle"]:
                x, y = result["positions"][frame, i] * 64
                radius = fits[i]["radius"] * 64
                ImageDraw.Draw(canvas).ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=(*fits[i]["color"], 255)
                )
            else:
                dx, dy = value[:2] * 0.5 * 64
                angle = np.arctan2(value[2], 1 + 2 * value[3])
                c, s = np.cos(angle), np.sin(angle)
                cx, cy = observation.centers[i] * 64
                affine = (
                    c,
                    s,
                    cx - c * (cx + dx) - s * (cy + dy),
                    -s,
                    c,
                    cy + s * (cx + dx) - c * (cy + dy),
                )
                canvas.alpha_composite(
                    sprites[i].transform(
                        (64, 64), Image.Transform.AFFINE, affine, resample=Image.Resampling.NEAREST
                    )
                )
        frames.append(np.asarray(canvas.convert("RGB")))
    return {**result, "motion": motion, "videos": np.stack(frames), "fits": fits}


class PrimitivePhysicsBaseline(PhysicsBaseline):
    def __init__(self, name="Physics-assisted"):
        super().__init__(name)
        sources = self.metadata["sampling_source_sha256"]
        sources["models/primitive_physics.py"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        digest = hashlib.sha256(
            json.dumps(
                {
                    "sources": sources,
                    "pymunk": pymunk.version,
                    "pillow": PIL.__version__,
                    "contract": PRIMITIVE_PHYSICS,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.metadata.update(
            checkpoint_sha256="source:" + digest,
            physics_contract=PRIMITIVE_PHYSICS,
            detail="image-fitted solid circles, hull fallback, sampled prior, explicit solver",
            quality="physics-assisted; circle recognition and subpixel geometry remain approximate",
        )

    def predict(self, first, mask, force, *, seeds, future_frames, cancelled):
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Use distinct sampling seeds")
        force = np.asarray(force)
        if not np.isfinite(force).all() or np.linalg.norm(force) > self.force_limit + 1e-6:
            raise ValueError("Force exceeds the baseline's declared range")
        obs = observe(first, mask)
        outputs = []
        for seed in seeds:
            if cancelled.is_set():
                raise InterruptedError("Physics-assisted generation cancelled")
            outputs.append(
                primitive_prediction(
                    obs, force, sample_world(seed)[0].numpy(), future_frames=future_frames
                )["videos"]
            )
        return np.stack(outputs)
