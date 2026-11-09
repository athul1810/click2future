"""Reference-free physics screening for sampled object trajectories."""

import numpy as np

from click2future.evaluation.rigid_geometry import geometry_diagnostics

SELECTION_CONTRACT = "visible-physics-candidate-selection-1"


def score_physics(observation, motions, force_xy, *, action_frames=4):
    """Score unavoidable physical violations without using a future reference."""
    motions = np.asarray(motions)
    force = np.asarray(force_xy, dtype=np.float64)
    if (
        motions.ndim != 4
        or motions.shape[2:] != (2, 4)
        or len(motions) < 1
        or not np.isfinite(motions).all()
        or force.shape != (2,)
        or not np.isfinite(force).all()
        or isinstance(action_frames, bool)
        or not isinstance(action_frames, int)
        or not 1 <= action_frames <= motions.shape[1]
    ):
        raise ValueError("Provide finite [K,F,2,4] motions, a force vector and action frames")

    height, width = observation.first.shape[:2]
    displacement = motions[..., :2] * (0.5 * np.array([width, height]))
    geometry = geometry_diagnostics(observation, motions)
    force_norm = np.linalg.norm(force)
    force_direction = force / force_norm if force_norm else force
    scores = []
    for index, clip in enumerate(geometry["clips"]):
        penetration = np.maximum(
            np.asarray(clip["hull_penetration_pixels_per_frame"]) - 1.0, 0
        ).sum()
        boundary = np.maximum(
            np.asarray(clip["boundary_depth_pixels_per_object_frame"]) - 1.0, 0
        ).sum()
        zero_force_drift = 0.0
        force_opposition = 0.0
        if force_norm == 0:
            zero_force_drift = np.linalg.norm(displacement[index], axis=-1).max()
        else:
            selected = displacement[index, :action_frames, observation.selected_index]
            force_opposition = max(0.0, float(-(selected @ force_direction).min()))
        scores.append(
            {
                "candidate_index": index,
                "penetration_excess_pixels": float(penetration),
                "boundary_excess_pixels": float(boundary),
                "hard_violation_pixels": float(penetration + boundary),
                "zero_force_drift_pixels": float(zero_force_drift),
                "force_opposition_pixels": float(force_opposition),
                "total": float(penetration + boundary + zero_force_drift + force_opposition),
            }
        )
    return scores


def _trajectory_distance(first, second, width, height):
    scale = 0.5 * np.array([width, height])
    difference = (first[..., :2] - second[..., :2]) * scale
    return float(np.linalg.norm(difference, axis=-1).mean())


def select_physics_guided(
    observation,
    motions,
    force_xy,
    *,
    output_count=4,
    shortlist_size=None,
    action_frames=4,
):
    """Choose diverse trajectories from the best reference-free physics shortlist."""
    motions = np.asarray(motions)
    if (
        motions.ndim != 4
        or isinstance(output_count, bool)
        or not isinstance(output_count, int)
        or not 1 <= output_count <= len(motions)
    ):
        raise ValueError("output_count must fit a nonempty [K,F,2,4] candidate pool")
    if shortlist_size is None:
        shortlist_size = min(len(motions), 2 * output_count)
    if (
        isinstance(shortlist_size, bool)
        or not isinstance(shortlist_size, int)
        or not output_count <= shortlist_size <= len(motions)
    ):
        raise ValueError("shortlist_size must be between output_count and candidate count")

    scores = score_physics(observation, motions, force_xy, action_frames=action_frames)
    ranked = sorted(
        range(len(motions)),
        key=lambda index: (
            scores[index]["hard_violation_pixels"] > 0,
            scores[index]["hard_violation_pixels"],
            scores[index]["total"],
            index,
        ),
    )
    hard_valid = [index for index in ranked if scores[index]["hard_violation_pixels"] == 0]
    if len(hard_valid) >= output_count:
        selection_pool = hard_valid[: min(shortlist_size, len(hard_valid))]
        selected = [selection_pool[0]]
    else:
        selection_pool = ranked[:shortlist_size]
        selected = hard_valid.copy()
        if not selected:
            selected = [selection_pool[0]]
    remaining = [index for index in selection_pool if index not in selected]
    height, width = observation.first.shape[:2]
    while len(selected) < output_count:
        diversity = {
            candidate: min(
                _trajectory_distance(motions[candidate], motions[chosen], width, height)
                for chosen in selected
            )
            for candidate in remaining
        }
        choice = max(
            remaining,
            key=lambda candidate: (
                diversity[candidate],
                -scores[candidate]["total"],
                -candidate,
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return {
        "contract": SELECTION_CONTRACT,
        "selected_indices": selected,
        "quality_ranking": ranked,
        "shortlist_size": shortlist_size,
        "hard_valid_candidates": len(hard_valid),
        "selection_pool_size": len(selection_pool),
        "scores": scores,
    }
