"""Diagnose saved validation wall-track failures without changing video scores."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from click2future.data import load_rollout
from click2future.models.object_motion import observe, render_motion
from click2future.training.train_cvae import sha256
from click2future.training.trajectory_data import canonical_ids

from .tracking import TrackerConfig, track_video
from .trajectory_audit import verify_inputs


def _extended_mask(mask, affine, padding):
    """Extend the native nearest-neighbour grid without re-rounding its origin.

    Pillow uses 16.16 coefficients for rotated nearest-neighbour affine warps.
    Translating a floating-point matrix and rounding it again changes that grid.
    Quantize at the native origin, then evaluate the same integers off canvas.
    The caller checks the native crop against Pillow before using the result.
    """
    height, width = mask.shape
    y, x = np.ogrid[-padding : height + padding, -padding : width + padding]
    a, b, c, d, e, f = affine
    if b == 0 and d == 0:
        # A rigid transform with no off-diagonal terms is a pure translation.
        ix = np.floor(c + a * 0.5 + a * x).astype(np.int64)
        iy = np.floor(f + e * 0.5 + e * y).astype(np.int64)
    else:
        matrix = np.array([[a, b, c + a * 0.5 + b * 0.5],
                           [d, e, f + d * 0.5 + e * 0.5]])
        fixed = np.floor(matrix * 65536 + 0.5).astype(np.int64)
        ix = (fixed[0, 0] * x + fixed[0, 1] * y + fixed[0, 2]) >> 16
        iy = (fixed[1, 0] * x + fixed[1, 1] * y + fixed[1, 2]) >> 16
    valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    return valid & mask[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)]


def sprite_areas(observation, value, object_index, *, padding=128):
    """Compare native and padded rigid-sprite areas under the same affine transform.

    The padded view is a measurement only. It never replaces the generated clip.
    Reject an insufficient measurement canvas rather than silently truncate it.
    """
    value = np.asarray(value)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("Provide one finite four-coordinate motion token")
    if (
        isinstance(object_index, bool)
        or not isinstance(object_index, (int, np.integer))
        or not 0 <= object_index < len(observation.masks)
        or isinstance(padding, bool)
        or not isinstance(padding, int)
        or padding < 1
    ):
        raise ValueError("Provide a valid object index and positive integer padding")
    height, width = observation.first.shape[:2]
    mask = observation.masks[object_index]
    dx, dy = value[:2] * 0.5 * np.array([width, height])
    angle = np.arctan2(value[2], value[3] * 2 + 1)
    cosine, sine = np.cos(angle), np.sin(angle)
    cx, cy = observation.centers[object_index] * [width, height]
    y, x = np.nonzero(mask)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points = (np.stack((x + 0.5, y + 0.5), -1) - [cx, cy]) @ rotation.T
    points += [cx + dx, cy + dy]
    if np.any(points.min(0) < -padding) or np.any(
        points.max(0) >= np.array([width, height]) + padding
    ):
        raise ValueError("Padded measurement canvas is too small")
    affine = [
        cosine,
        sine,
        cx - cosine * (cx + dx) - sine * (cy + dy),
        -sine,
        cosine,
        cy + sine * (cx + dx) - cosine * (cy + dy),
    ]
    sprite = Image.fromarray(mask.astype(np.uint8) * 255)
    native = np.asarray(
        sprite.transform(
            (width, height), Image.Transform.AFFINE, affine, resample=Image.Resampling.NEAREST
        )
    ) > 0
    full = _extended_mask(mask, affine, padding)
    if not full.any() or full[[0, -1]].any() or full[:, [0, -1]].any():
        raise ValueError("Padded measurement canvas cannot contain the whole sprite")
    if not np.array_equal(full[padding : padding + height, padding : padding + width], native):
        raise ValueError("Padded and native rasterizations disagree")
    visible, unclipped = int(native.sum()), int(full.sum())
    return {
        "native_amodal_area": visible,
        "unclipped_amodal_area": unclipped,
        "clipped_fraction": 1 - visible / unclipped,
    }


def audit(plan_path, predictions_path, dataset, checkpoint, poses_dir):
    """Keep all frozen wall conditions/seeds and diagnose every missing observation."""
    dataset, poses_dir = Path(dataset), Path(poses_dir)
    plan, run, paths, hashes = verify_inputs(plan_path, predictions_path, dataset, checkpoint)
    rows, incomplete, wall_clips = [], 0, 0
    tracker = TrackerConfig()
    for case in plan["cases"]:
        if not case["condition_id"].endswith("-wall"):
            continue
        reference = load_rollout(dataset, case["sample_id"])
        observation = observe(reference["frames"][0], reference["selected_mask"])
        order = canonical_ids(observation, reference["selected_object_id"])
        pose_path = poses_dir / (case["sample_id"] + "-poses.npz")
        hashes[pose_path] = sha256(pose_path)
        with np.load(pose_path, allow_pickle=False) as archive:
            motions = archive["motion"]
        with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
            videos = archive["videos"]
        if motions.shape != (len(plan["sampling_seeds"]), len(videos[0]) - 1, 2, 4):
            raise ValueError("Saved poses must cover every original seed and future frame")
        for seed, motion, video in zip(plan["sampling_seeds"], motions, videos, strict=True):
            if not np.array_equal(render_motion(observation, motion), video):
                raise ValueError("Saved poses do not render the original video pixels")
            tracks = track_video(
                video, reference["instance_masks"][0], reference["object_valid"], tracker
            )
            missing = np.argwhere(~tracks.tracked[1:] & reference["object_valid"][None])
            wall_clips += 1
            incomplete += bool(len(missing))
            for future_index, object_id in missing:
                frame, object_id = int(future_index) + 1, int(object_id)
                index = order.index(object_id)
                areas = sprite_areas(observation, motion[future_index, index], index)
                initial_area = int(reference["instance_masks"][0, object_id].sum())
                native_ratio = areas["native_amodal_area"] / initial_area
                full_ratio = areas["unclipped_amodal_area"] / initial_area
                rows.append(
                    {
                        "sample_id": case["sample_id"],
                        "seed": seed,
                        "frame": frame,
                        "object_id": object_id,
                        "reason": str(tracks.reasons[frame, object_id]),
                        "initial_reference_area": initial_area,
                        **areas,
                        "native_area_ratio": native_ratio,
                        "unclipped_area_ratio": full_ratio,
                        "native_mask_fails_minimum_area": native_ratio < tracker.min_area_ratio,
                        "unclipped_mask_passes_area_limits": (
                            tracker.min_area_ratio <= full_ratio <= tracker.max_area_ratio
                        ),
                    }
                )
    if not wall_clips:
        raise ValueError("No wall conditions in the frozen validation plan")
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Audit input changed during the measurement: " + path.name)
    return {
        "status": "measured_saved_validation_raster_clipping",
        "raster_metric_version": "native-grid-extension-2",
        "method": run["method"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "scope": (
            "All frozen validation wall conditions and seeds; every missing future object "
            "observation. Padded masks never replace generated videos or failed tracks."
        ),
        "summary": {
            "wall_clips": wall_clips,
            "incomplete_wall_clips": incomplete,
            "missing_observations": len(rows),
            "reasons": dict(Counter(row["reason"] for row in rows)),
            "native_mask_fails_area_count": sum(
                row["native_mask_fails_minimum_area"] for row in rows
            ),
            "unclipped_mask_passes_area_count": sum(
                row["unclipped_mask_passes_area_limits"] for row in rows
            ),
            "clipped_fraction_min": min((row["clipped_fraction"] for row in rows), default=None),
            "clipped_fraction_max": max((row["clipped_fraction"] for row in rows), default=None),
        },
        "observations": rows,
        "input_sha256": {str(path): digest for path, digest in hashes.items()},
        "source_sha256": {
            name: sha256(path)
            for name, path in (
                ("wall_clipping.py", Path(__file__)),
                ("object_motion.py", Path(__file__).parents[1] / "models/object_motion.py"),
                ("tracking.py", Path(__file__).with_name("tracking.py")),
            )
        },
        "interpretation": (
            "Clipping does not validate physical wall contact or authorize tracker relaxation. "
            "Original video scores and every missing observation remain unchanged."
        ),
        "test_splits_evaluated": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "poses-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output path; preserve previous diagnostics")
    report = audit(args.plan, args.predictions, args.dataset, args.checkpoint, args.poses_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
