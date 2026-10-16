"""Frame-0 palette tracking for the simulator's solid-colour objects.

Only RGB frames and frame-0 identity masks enter extraction. No future masks,
positions, contact labels, or physical parameters are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRACKER_VERSION = "palette-components-1"


@dataclass(frozen=True)
class TrackerConfig:
    color_tolerance: float = 24.0
    color_margin: float = 4.0
    min_area_ratio: float = 0.55
    max_area_ratio: float = 1.8
    dominant_fraction: float = 0.85
    min_pixels: int = 4

    def validate(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Tracker settings must be finite")
        if self.color_tolerance < 0 or self.color_margin <= 0:
            raise ValueError("Color tolerance must be nonnegative and margin positive")
        if not 0 < self.min_area_ratio <= 1 <= self.max_area_ratio:
            raise ValueError("Area ratios must bracket 1")
        if not 0.5 < self.dominant_fraction <= 1:
            raise ValueError("dominant_fraction must be in (0.5,1]")
        if (
            isinstance(self.min_pixels, bool)
            or not isinstance(self.min_pixels, int)
            or self.min_pixels < 1
        ):
            raise ValueError("min_pixels must be a positive integer")


@dataclass
class Tracks:
    positions: np.ndarray  # [T,N,2], NaN where unreliable
    tracked: np.ndarray  # [T,N]
    masks: np.ndarray  # [T,N,H,W], only accepted components
    reasons: np.ndarray  # [T,N], explicit failure categories


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Find 8-connected components; return the largest, its area, and total area."""
    height, width = mask.shape
    remaining = mask.copy()
    largest = []
    total = int(mask.sum())
    for y, x in zip(*np.nonzero(mask), strict=True):
        if not remaining[y, x]:
            continue
        component = []
        stack = [(y, x)]
        remaining[y, x] = False
        while stack:
            row, column = stack.pop()
            component.append((row, column))
            for ny in range(max(0, row - 1), min(height, row + 2)):
                for nx in range(max(0, column - 1), min(width, column + 2)):
                    if remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    result = np.zeros_like(mask)
    if largest:
        rows, columns = zip(*largest, strict=True)
        result[rows, columns] = True
    return result, len(largest), total


def track_video(frames, initial_masks, object_valid, config: TrackerConfig | None = None) -> Tracks:
    """Track fixed identities using a shared frame-0 palette, without future oracle labels.

    This baseline is limited to solid colours. Similar colours, fragmentation,
    disappearing objects, and large area changes cause abstention, not fabricated tracks.
    Frame-0 masks align evaluation identities only; they are not model conditioning.
    """
    config = config or TrackerConfig()
    config.validate()
    frames = np.asarray(frames)
    initial_masks = np.asarray(initial_masks)
    valid = np.asarray(object_valid)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("frames must be uint8 [T,H,W,3]")
    count, height, width, _ = frames.shape
    if min(count, height, width) < 1:
        raise ValueError("Video dimensions must be positive")
    if valid.ndim != 1 or valid.dtype != np.bool_ or not valid.any():
        raise ValueError("object_valid must be a boolean vector with a valid object")
    objects = len(valid)
    if initial_masks.shape != (objects, height, width) or initial_masks.dtype != np.bool_:
        raise ValueError("initial_masks must be boolean [N,H,W]")
    if np.any(initial_masks[valid].sum(axis=0) > 1):
        raise ValueError("Frame-0 identity masks must not overlap")
    if np.any(initial_masks[~valid]):
        raise ValueError("Padded initial masks must be empty")
    areas = initial_masks.sum(axis=(1, 2))
    if np.any(areas[valid] < config.min_pixels):
        raise ValueError("Each valid initial mask must contain at least min_pixels pixels")
    palette = np.zeros((objects, 3), dtype=np.float64)
    for obj in np.flatnonzero(valid):
        palette[obj] = np.median(frames[0][initial_masks[obj]], axis=0)
    background = frames[0][~initial_masks.any(axis=0)]
    if not len(background):
        raise ValueError("Frame 0 must expose some background")
    colors, counts = np.unique(background, axis=0, return_counts=True)
    # Solid tabletop and border normally occupy two colours; tolerate small incidental regions.
    background_palette = colors[np.argsort(counts, kind="stable")[-4:]].astype(np.float64)
    palette = np.concatenate([palette, background_palette])
    positions = np.full((count, objects, 2), np.nan, dtype=np.float64)
    tracked = np.zeros((count, objects), bool)
    masks = np.zeros((count, objects, height, width), bool)
    reasons = np.full((count, objects), "padding", dtype="U24")
    for time, frame in enumerate(frames):
        distances = np.linalg.norm(frame.astype(np.float64)[:, :, None] - palette, axis=-1)
        distances[..., np.flatnonzero(~valid)] = np.inf
        order = np.argsort(distances, axis=-1, kind="stable")
        best = order[..., 0]
        sorted_distances = np.take_along_axis(distances, order[..., :2], axis=-1)
        confident = (sorted_distances[..., 0] <= config.color_tolerance) & (
            sorted_distances[..., 1] - sorted_distances[..., 0] >= config.color_margin
        )
        for obj in np.flatnonzero(valid):
            candidates = (best == obj) & confident
            mask, area, total = _largest_component(candidates)
            reason = "ok"
            if area < config.min_pixels:
                reason = "missing_or_ambiguous"
            elif area / total < config.dominant_fraction:
                reason = "fragmented_or_duplicate"
            elif not config.min_area_ratio <= area / areas[obj] <= config.max_area_ratio:
                reason = "area_change"
            reasons[time, obj] = reason
            if reason != "ok":
                continue
            y, x = np.nonzero(mask)
            positions[time, obj] = [(x.mean() + 0.5) / width, (y.mean() + 0.5) / height]
            tracked[time, obj] = True
            masks[time, obj] = mask
    return Tracks(positions, tracked, masks, reasons)
