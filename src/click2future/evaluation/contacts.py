"""Contact-onset estimates from visible pixel masks, with missing-observation flags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Contacts:
    state: np.ndarray  # [T,N,N], pixel proximity, not confirmed physical contact
    onset: np.ndarray  # [T-1,N,N]
    observed: np.ndarray  # [T-1,N,N]


def estimate_contacts(
    masks,
    tracked,
    object_valid,
    *,
    max_gap_pixels: float,
    motion_threshold: float | None = None,
    motion_window: int = 3,
) -> Contacts:
    """Estimate object-object contact from the minimum gap between occupied pixel squares.

    Without motion_threshold, use transitions in proximity at the frame endpoints.
    With it, find opposing velocity-change peaks near proximity and require tracked
    observations across each motion window. Missing frames cannot create an onset
    on reappearance. This is a video proxy, not a substitute for simulator labels.
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
    if motion_threshold is not None:
        if not np.isfinite(motion_threshold) or motion_threshold <= 0:
            raise ValueError("motion_threshold must be finite and positive")
        if (
            isinstance(motion_window, bool)
            or not isinstance(motion_window, int)
            or motion_window < 2
        ):
            raise ValueError("motion_window must be an integer >= 2")
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
    if motion_threshold is not None:
        onset, observed = _motion_onsets(masks, known, state, motion_threshold, motion_window)
    return Contacts(state, onset, observed)


def _motion_onsets(masks, known, proximity, threshold, window):
    """Use opposing velocity changes near contact, without a physics-state input.

    Slopes are least-squares estimates in pixel/frame units. At the first interval
    the incoming velocity is zero under the initially stationary scene contract.
    At the last interval the outgoing slope uses the final two frames. This is an
    offline estimator; it uses future video context, never future oracle labels.
    """
    frames, objects = known.shape
    centres = np.full((frames, objects, 2), np.nan)
    for t, obj in zip(*np.nonzero(known), strict=True):
        centres[t, obj] = np.argwhere(masks[t, obj]).mean(axis=0)
    changes = np.zeros((frames - 1, objects, 2))
    available = np.zeros((frames - 1, objects), bool)

    def slope(values):
        if len(values) == 1:
            return np.zeros(values.shape[1:])
        times = np.arange(len(values), dtype=float)
        times -= times.mean()
        return np.sum(values * times[:, None, None], axis=0) / np.sum(times**2)

    for i in range(frames - 1):
        left = max(0, i - window + 1)
        right = min(frames, i + 1 + window)
        after_start = i + 1 if right - i - 1 >= 2 else i
        available[i] = known[left:right].all(axis=0)
        changes[i] = slope(centres[after_start:right]) - slope(centres[left : i + 1])
    observed = available[:, :, None] & available[:, None, :]
    observed[:, np.arange(objects), np.arange(objects)] = False
    onset = np.zeros_like(observed)
    for first in range(objects):
        for second in range(first + 1, objects):
            score = np.sqrt(np.maximum(0, -np.sum(changes[:, first] * changes[:, second], axis=-1)))
            eligible = observed[:, first, second] & (
                proximity[:-1, first, second] | proximity[1:, first, second]
            )
            score = np.where(eligible, score, 0)
            # Keep local peaks, then suppress neighbouring peaks from the same impulse.
            padded = np.pad(score, 1, constant_values=-np.inf)
            peaks = np.flatnonzero(
                (score >= threshold) & (score > padded[:-2]) & (score >= padded[2:])
            )
            chosen = []
            for index in sorted(peaks, key=lambda i: (-score[i], i)):
                if all(abs(index - previous) > window for previous in chosen):
                    chosen.append(index)
                    onset[index, first, second] = onset[index, second, first] = True
    return onset, observed
