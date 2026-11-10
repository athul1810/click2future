"""Post-force linear-momentum correction for sampled two-object trajectories.

The block assumes the dataset's declared zero initial velocities, then uses the
sampled masses and known external impulse. Before object contact, correction is
assigned to the force-selected object. After contact, a shared centre-of-mass
correction preserves the learned relative collision response. Rotation stays
unchanged. Trajectories that approach a wall are returned unchanged because the
unobserved wall impulse makes pair momentum an invalid constraint there.
"""

import numpy as np

from click2future.evaluation.rigid_geometry import _hull


MOMENTUM_BLOCK_CONTRACT = "wall-free-contact-aware-linear-momentum-1"


def _placed_hull(observation, motion, object_index):
    height, width = observation.first.shape[:2]
    center = observation.centers[object_index] * [width, height]
    y, x = np.nonzero(observation.masks[object_index])
    hull = _hull(np.stack((x + 0.5, y + 0.5), -1)) - center
    angle = np.arctan2(motion[2], 1 + 2 * motion[3])
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    offset = motion[:2] * 0.5 * np.array([width, height])
    return hull @ rotation.T + center + offset


def _approaches_wall(observation, motion, clearance_pixels):
    height, width = observation.first.shape[:2]
    zero = np.zeros(4)
    for frame in [np.stack((zero, zero)), *motion]:
        for object_index in range(2):
            hull = _placed_hull(observation, frame[object_index], object_index)
            if (
                hull[:, 0].min() <= clearance_pixels
                or hull[:, 0].max() >= width - clearance_pixels
                or hull[:, 1].min() <= clearance_pixels
                or hull[:, 1].max() >= height - clearance_pixels
            ):
                return True
    return False


def _overlap_depth(first, second):
    edges = np.concatenate(
        (np.roll(first, -1, axis=0) - first, np.roll(second, -1, axis=0) - second)
    )
    axes = np.stack((-edges[:, 1], edges[:, 0]), -1)
    lengths = np.linalg.norm(axes, axis=-1)
    axes = axes[lengths > 0] / lengths[lengths > 0, None]
    a, b = first @ axes.T, second @ axes.T
    return float(np.minimum(a.max(0) - b.min(0), b.max(0) - a.min(0)).min())


def _first_contact_frame(observation, motion, clearance_pixels):
    for frame_index, frame in enumerate(motion):
        hulls = [_placed_hull(observation, frame[index], index) for index in range(2)]
        if _overlap_depth(*hulls) >= -clearance_pixels:
            return frame_index
    return None


def apply_momentum_block(
    observation,
    motion,
    masses,
    force_xy,
    *,
    fps=8,
    force_duration=0.25,
    wall_clearance_pixels=1.0,
):
    """Correct post-force pair momentum before and after predicted contact.

    Motion is ``[future frames, 2 objects, 4 pose values]``. The first two
    values are displacement divided by 0.5. The remaining values encode
    rotation and are never changed.
    """
    motion = np.asarray(motion)
    masses = np.asarray(masses, dtype=np.float64)
    force = np.asarray(force_xy, dtype=np.float64)
    if (
        motion.ndim != 3
        or motion.shape[1:] != (2, 4)
        or not np.isfinite(motion).all()
        or masses.shape != (2,)
        or not np.isfinite(masses).all()
        or (masses <= 0).any()
        or force.shape != (2,)
        or not np.isfinite(force).all()
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or fps <= 0
        or not np.isfinite(force_duration)
        or force_duration < 0
        or not np.isfinite(wall_clearance_pixels)
        or wall_clearance_pixels < 0
    ):
        raise ValueError("Provide finite two-object motion, masses, force and timing")
    result = motion.astype(np.float64, copy=True)
    if _approaches_wall(observation, result, wall_clearance_pixels):
        return result.astype(np.float32)

    positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * result[..., :2]))
    increments = np.diff(positions, axis=0)
    first_post_force = int(np.ceil(force_duration * fps))
    first_contact = _first_contact_frame(observation, result, wall_clearance_pixels)
    target_momentum = force * force_duration
    selected = observation.selected_index
    total_mass = masses.sum()
    for interval in range(first_post_force, len(increments)):
        velocity = increments[interval] * fps
        momentum_error = (masses[:, None] * velocity).sum(0) - target_momentum
        if first_contact is None or interval < first_contact:
            velocity[selected] -= momentum_error / masses[selected]
        else:
            velocity -= momentum_error / total_mass
        increments[interval] = velocity / fps

    result[..., :2] = 2 * np.cumsum(increments, axis=0)
    return result.astype(np.float32)
