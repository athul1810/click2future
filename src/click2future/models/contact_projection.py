"""D028: position-based non-penetration projection for sampled pose trajectories.

This follows the constraint projection of position-based dynamics (Mueller et al.,
2007) and the physics-projection idea of PhysDiff (Yuan et al., 2023). Overlapping
observed convex hulls are separated along their minimum translation axis, and each
object moves in inverse proportion to its sampled mass. Hulls are also kept on the
canvas. Only violations are corrected. Rotations are unchanged and frames without a
violation keep their poses, so zero-force rest stays exact. It is an explicit geometric
constraint applied to a neural sample, labelled as a hybrid, not learned physics.
"""

import numpy as np

from click2future.evaluation.rigid_geometry import _hull

PROJECTION_CONTRACT = "observed-hull-position-projection-1"
# Corrections end this far inside a limit, so float32 pose storage cannot round back
# across it. One thousandth of a native pixel.
EPSILON_PIXELS = 1e-3


def _separation(first, second):
    """Depth and unit axis that moves ``second`` out of ``first``; depth 0 if apart."""
    edges = np.concatenate(
        (np.roll(first, -1, axis=0) - first, np.roll(second, -1, axis=0) - second)
    )
    axes = np.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    lengths = np.linalg.norm(axes, axis=-1)
    axes = axes[lengths > 0] / lengths[lengths > 0, None]
    a, b = first @ axes.T, second @ axes.T
    forward, backward = a.max(0) - b.min(0), b.max(0) - a.min(0)
    overlap = np.minimum(forward, backward)
    index = int(np.argmin(overlap))
    if overlap[index] <= 0:
        return 0.0, np.zeros(2)
    sign = 1.0 if forward[index] <= backward[index] else -1.0
    return float(overlap[index]), sign * axes[index]


def _wall_shift(hull, width, height):
    """Smallest translation that keeps observed pixel-centre hulls on the canvas."""
    shift = np.zeros(2)
    for axis, size in ((0, width), (1, height)):
        low, high = hull[:, axis].min(), hull[:, axis].max()
        if high - low > size:
            continue
        if low < 0:
            shift[axis] = EPSILON_PIXELS - low
        elif high > size:
            shift[axis] = size - EPSILON_PIXELS - high
    return shift


def project_motion(observation, motion, masses, *, carry=False, driven=None, iterations=8):
    """Return a copy of ``motion`` [F,2,4] whose observed hulls do not overlap.

    ``masses`` are the sampled masses in canonical object order. With ``carry`` a
    correction persists into later frames, as in a position-based integrator; without
    it each frame is projected from the neural pose alone. With ``driven`` set to an
    object index, that object keeps its neural pose in pair contacts and the other
    object takes the whole separation, as for a kinematically driven body.
    """
    motion = np.array(motion, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if motion.ndim != 3 or motion.shape[1:] != (2, 4) or not np.isfinite(motion).all():
        raise ValueError("Motion must be finite [F,2,4]")
    if masses.shape != (2,) or not (masses > 0).all():
        raise ValueError("Provide two positive masses")
    height, width = observation.first.shape[:2]
    scale = 0.5 * np.array([width, height])
    centers = [observation.centers[i] * [width, height] for i in range(2)]
    hulls = []
    for mask in observation.masks:
        y, x = np.nonzero(mask)
        hulls.append(_hull(np.stack((x + 0.5, y + 0.5), -1)))
    inverse = 1 / masses
    weights = inverse / inverse.sum()
    if driven is not None:
        if driven not in (0, 1):
            raise ValueError("The driven object must be 0 or 1")
        weights = np.eye(2)[1 - driven]
    carried = np.zeros((2, 2))
    for frame in range(motion.shape[0]):
        offsets = carried.copy() if carry else np.zeros((2, 2))
        rotations, base = [], []
        for i in range(2):
            value = motion[frame, i]
            angle = np.arctan2(value[2], value[3] * 2 + 1)
            c, s = np.cos(angle), np.sin(angle)
            rotations.append(np.array([[c, -s], [s, c]]))
            base.append(centers[i] + value[:2] * scale)
        for _ in range(iterations):
            placed = [
                (hulls[i] - centers[i]) @ rotations[i].T + base[i] + offsets[i] for i in range(2)
            ]
            depth, axis = _separation(placed[0], placed[1])
            moved = depth > 0
            if moved:
                offsets[0] -= weights[0] * (depth + EPSILON_PIXELS) * axis
                offsets[1] += weights[1] * (depth + EPSILON_PIXELS) * axis
                placed = [
                    (hulls[i] - centers[i]) @ rotations[i].T + base[i] + offsets[i]
                    for i in range(2)
                ]
            for i in range(2):
                shift = _wall_shift(placed[i], width, height)
                if shift.any():
                    offsets[i] += shift
                    moved = True
            if not moved:
                break
        motion[frame, :, :2] += offsets / scale
        if carry:
            carried = offsets
    return motion.astype(np.float32)
