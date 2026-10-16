"""Shared validation for extracted future sequences."""

from __future__ import annotations

import numpy as np

from .metrics import _boolean, _positions


def tracks_input(positions, tracked, object_valid):
    positions = _positions(positions, "positions")
    if positions.ndim != 4 or positions.shape[-1] != 2 or min(positions.shape[:3]) < 1:
        raise ValueError("positions must be nonempty [K,T,N,2]")
    if positions.shape[1] < 2:
        raise ValueError("At least one future frame is required")
    valid = _boolean(object_valid, (positions.shape[2],), "object_valid")
    known = _boolean(tracked, positions.shape[:3], "tracked") & valid[None, None]
    if not valid.any() or not np.isfinite(positions[known]).all():
        raise ValueError("At least one valid object and finite tracked positions are required")
    return positions, known, valid


def timestamps_input(timestamps, frames):
    times = _positions(timestamps, "timestamps")
    if (
        times.shape != (frames,)
        or not np.isfinite(times).all()
        or times[0] != 0
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("timestamps must start at zero and increase strictly, one per frame")
    return times


def finite_norm(values, *, axis=-1):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.linalg.norm(values, axis=axis)
    if not np.isfinite(result).all():
        raise ValueError("Sequence distance exceeds the supported numeric range")
    return result
