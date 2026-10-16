"""Replay and compare saved rollout records."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.export import load_rollout
from .rollout import run_rollout

COMPARE_FIELDS = (
    "frames",
    "instance_masks",
    "object_valid",
    "selected_mask",
    "force_xy",
    "action_map",
    "timestamps",
    "positions",
    "collision_onset",
    "wall_contact_onset",
)


def replay_saved_rollout(
    input_dir: str,
    sample_id: str,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Load a saved sample, regenerate it, and return a comparison report."""

    saved = load_rollout(input_dir, sample_id)
    regenerated = replay_rollout(saved)
    return compare_rollouts(saved, regenerated, atol=atol, rtol=rtol)


def replay_rollout(saved: dict[str, Any]) -> dict[str, Any]:
    """Regenerate one rollout using its saved scene and resolved metadata."""

    metadata = saved["metadata"]
    return run_rollout(
        saved["scene_spec"],
        frame_count=int(metadata["frame_count"]),
        fps=int(metadata["fps"]),
        physics_hz=int(metadata["physics_hz"]),
        resolution=int(metadata["resolution"]),
    )


def compare_rollouts(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare the saved and regenerated fields without hiding mismatches."""

    fields: dict[str, dict[str, Any]] = {}
    for field in COMPARE_FIELDS:
        fields[field] = _compare_array(expected[field], actual[field], atol=atol, rtol=rtol)

    scalar_fields = (
        "schema_version",
        "selected_object_id",
        "force_duration",
        "scene_spec",
        "metadata",
    )
    for field in scalar_fields:
        fields[field] = {"ok": expected[field] == actual[field], "max_abs_error": 0.0}

    return {
        "ok": all(result["ok"] for result in fields.values()),
        "fields": fields,
    }


def _compare_array(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if expected.shape != actual.shape:
        return {"ok": False, "max_abs_error": None, "reason": "shape mismatch"}

    if expected.size:
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        max_abs_error = float(np.max(difference))
    else:
        max_abs_error = 0.0

    if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
        matches = np.array_equal(expected, actual)
    else:
        matches = bool(np.allclose(expected, actual, atol=atol, rtol=rtol))
    return {"ok": matches, "max_abs_error": max_abs_error}
