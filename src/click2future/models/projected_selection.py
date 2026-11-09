"""Repair-aware selection for geometrically projected trajectories."""

import numpy as np

from click2future.models.physics_selection import score_physics

SELECTION_CONTRACT = "projected-physics-candidate-selection-1"


def _distance(first, second, width, height):
    scale = 0.5 * np.array([width, height])
    delta = (first[..., :2] - second[..., :2]) * scale
    return float(np.linalg.norm(delta, axis=-1).mean())


def select_projected(observation, raw, projected, force_xy, *, output_count=4):
    """Select diverse projected futures while preferring smaller geometric repairs."""
    raw, projected = np.asarray(raw), np.asarray(projected)
    if (
        raw.shape != projected.shape
        or raw.ndim != 4
        or raw.shape[2:] != (2, 4)
        or not np.isfinite(raw).all()
        or not np.isfinite(projected).all()
        or isinstance(output_count, bool)
        or not isinstance(output_count, int)
        or not 1 <= output_count <= len(raw)
    ):
        raise ValueError("Provide matching finite [K,F,2,4] pools and a valid output count")

    height, width = observation.first.shape[:2]
    scale = 0.5 * np.array([width, height])
    repair = np.linalg.norm((projected[..., :2] - raw[..., :2]) * scale, axis=-1)
    scores = score_physics(observation, projected, force_xy)
    for index, score in enumerate(scores):
        score["mean_projection_pixels"] = float(repair[index].mean())
        score["total"] += score["mean_projection_pixels"]

    ranked = sorted(
        range(len(projected)),
        key=lambda index: (
            scores[index]["hard_violation_pixels"] > 0,
            scores[index]["hard_violation_pixels"],
            scores[index]["total"],
            index,
        ),
    )
    shortlist = ranked[: min(len(projected), 2 * output_count)]
    selected = [shortlist[0]]
    while len(selected) < output_count:
        remaining = [index for index in shortlist if index not in selected]
        selected.append(
            max(
                remaining,
                key=lambda index: (
                    min(
                        _distance(projected[index], projected[chosen], width, height)
                        for chosen in selected
                    ),
                    -scores[index]["total"],
                    -index,
                ),
            )
        )
    return {
        "contract": SELECTION_CONTRACT,
        "selected_indices": selected,
        "quality_ranking": ranked,
        "shortlist_size": len(shortlist),
        "scores": scores,
    }
