"""Amodal silhouette diagnostics for the separate rigid-trajectory model.

These use predicted poses, not RGB tracking. They expose overlap hidden by layer
compositing and do not correct motion or certify physical validity.
"""

import numpy as np
from PIL import Image

GEOMETRY_VERSION = "rigid-silhouettes-1"


def geometry_diagnostics(observation, motions):
    """Audit every future pose, including fully occluded or off-canvas objects.

    Masks use the rigid decoder's inverse affine transform and nearest sampling.
    Boundary checks transform all observed pixel centres before canvas clipping.
    The one-pixel rendered border defines the interior. Report raw overlap as
    well as overlap exceeding 10% of the smaller visible amodal silhouette;
    neither threshold is an accepted physical-quality gate.
    """
    motions = np.asarray(motions)
    if (
        motions.ndim != 4
        or motions.shape[2:] != (2, 4)
        or min(motions.shape[:2]) < 1
        or not np.isfinite(motions).all()
    ):
        raise ValueError("Provide finite future motions [K,F,2,4]")
    height, width = observation.first.shape[:2]
    sprites = [Image.fromarray(mask.astype(np.uint8) * 255) for mask in observation.masks]
    points = []
    for mask in observation.masks:
        y, x = np.nonzero(mask)
        points.append(np.stack((x + 0.5, y + 0.5), axis=-1))
    clips = []
    for motion in motions:
        intersections, fractions, outside = [], [], []
        for state in motion:
            masks, boundary_fractions = [], []
            for index, value in enumerate(state):
                dx, dy = value[:2] * 0.5 * np.array([width, height])
                angle = np.arctan2(value[2], value[3] * 2 + 1)
                cosine, sine = np.cos(angle), np.sin(angle)
                cx, cy = observation.centers[index] * [width, height]
                affine = (
                    cosine,
                    sine,
                    cx - cosine * (cx + dx) - sine * (cy + dy),
                    -sine,
                    cosine,
                    cy + sine * (cx + dx) - cosine * (cy + dy),
                )
                warped = sprites[index].transform(
                    (width, height),
                    Image.Transform.AFFINE,
                    affine,
                    resample=Image.Resampling.NEAREST,
                )
                masks.append(np.asarray(warped) != 0)
                rotation = np.array([[cosine, -sine], [sine, cosine]])
                transformed = (points[index] - [cx, cy]) @ rotation.T + [cx + dx, cy + dy]
                inside = (
                    (transformed[:, 0] >= 1)
                    & (transformed[:, 0] < width - 1)
                    & (transformed[:, 1] >= 1)
                    & (transformed[:, 1] < height - 1)
                )
                boundary_fractions.append(float((~inside).mean()))
            intersection = int((masks[0] & masks[1]).sum())
            smaller = min(int(mask.sum()) for mask in masks)
            intersections.append(intersection)
            fractions.append(intersection / smaller if smaller else 0.0)
            outside.append(boundary_fractions)
        intersections, fractions, outside = map(np.asarray, (intersections, fractions, outside))
        clips.append(
            {
                "overlap_frames": int((intersections > 0).sum()),
                "overlap_over_one_pixel_frames": int((intersections > 1).sum()),
                "overlap_over_ten_percent_frames": int((fractions > 0.1).sum()),
                "maximum_overlap_pixels": int(intersections.max()),
                "maximum_overlap_fraction": float(fractions.max()),
                "outside_interior_object_frames": int((outside > 0).sum()),
                "maximum_outside_pixel_fraction": float(outside.max()),
                "overlap_pixels_per_frame": intersections.tolist(),
                "overlap_fraction_per_frame": fractions.tolist(),
                "outside_pixel_fraction_per_object_frame": outside.tolist(),
            }
        )
    return {
        "version": GEOMETRY_VERSION,
        "source": "amodal rigid-pose silhouettes; separate from generated-RGB tracking",
        "frame_zero_scored": False,
        "samples": len(clips),
        "future_frames_per_sample": motions.shape[1],
        "expected_pair_frames": int(np.prod(motions.shape[:2])),
        "expected_object_frames": int(np.prod(motions.shape[:2]) * 2),
        "clips_with_overlap": sum(clip["overlap_frames"] > 0 for clip in clips),
        "clips_with_overlap_over_ten_percent": sum(
            clip["overlap_over_ten_percent_frames"] > 0 for clip in clips
        ),
        "overlap_frames": sum(clip["overlap_frames"] for clip in clips),
        "overlap_over_ten_percent_frames": sum(
            clip["overlap_over_ten_percent_frames"] for clip in clips
        ),
        "clips_outside_interior": sum(clip["outside_interior_object_frames"] > 0 for clip in clips),
        "outside_interior_object_frames": sum(
            clip["outside_interior_object_frames"] for clip in clips
        ),
        "clips": clips,
        "limitations": [
            "Rasterized silhouette overlap is not continuous rigid-body penetration depth.",
            "Rotated pixel-centre boundary tests have rasterization bias; compare oracle control.",
            "No momentum, force law or contact timing is certified by these diagnostics.",
        ],
    }
