"""Contact-onset estimates from visible pixel masks, with missing-observation flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Contacts:
    state: np.ndarray  # [T,N,N]
    onset: np.ndarray  # [T-1,N,N]
    observed: np.ndarray  # [T-1,N,N]


def estimate_contacts(masks, tracked, object_valid, *, max_gap_pixels: float) -> Contacts:
    """Estimate object-object contact from the minimum gap between occupied pixel squares.

    A transition is observed only when both endpoints track both objects. Missing
    frames cannot create a new onset on reappearance. Between-frame collisions can
    still be missed; this is a video proxy, not a substitute for simulator labels.
    """
    masks, tracked, valid = map(np.asarray, (masks, tracked, object_valid))
    if masks.ndim != 4 or masks.dtype != np.bool_ or min(masks.shape) < 1:
        raise ValueError("masks must be nonempty boolean [T,N,H,W]")
    frames, objects, _, _ = masks.shape
    if tracked.shape != (frames, objects) or tracked.dtype != np.bool_:
        raise ValueError("tracked must be boolean [T,N]")
    if valid.shape != (objects,) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("object_valid must be boolean [N] with a valid object")
    if not np.isfinite(max_gap_pixels) or max_gap_pixels < 0:
        raise ValueError("max_gap_pixels must be finite and nonnegative")
    known = tracked & valid[None]
    if np.any(known & ~masks.any(axis=(2, 3))):
        raise ValueError("Tracked objects must have nonempty masks")
    state = np.zeros((frames, objects, objects), bool)
    for t in range(frames):
        pixels = {obj: np.argwhere(masks[t, obj]) for obj in np.flatnonzero(known[t])}
        for first in pixels:
            for second in pixels:
                if first >= second:
                    continue
                # Bound memory for malformed masks covering most of a frame.
                minimum = np.inf
                for start in range(0, len(pixels[first]), 64):
                    delta = np.abs(pixels[first][start : start + 64, None] - pixels[second][None])
                    gap = np.maximum(delta - 1, 0)
                    minimum = min(minimum, float(np.min(np.sum(gap.astype(float) ** 2, axis=-1))))
                    if minimum <= max_gap_pixels**2:
                        break
                contact = minimum <= max_gap_pixels**2
                state[t, first, second] = state[t, second, first] = contact
    endpoints = known[:-1] & known[1:]
    observed = endpoints[:, :, None] & endpoints[:, None, :]
    observed[:, np.arange(objects), np.arange(objects)] = False
    onset = state[1:] & ~state[:-1] & observed
    return Contacts(state, onset, observed)
