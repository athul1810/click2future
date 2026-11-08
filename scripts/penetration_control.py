"""Positive control for the rigid hull-penetration audit at known true overlaps.

The geometry audit has a negative control (oracle motion stays below its one-pixel
tolerance) but no positive one. This places two objects from each distinct
training/validation frame-zero layout at known continuous overlaps, using the
simulator's exact shapes, and scores the resulting poses with the unchanged
`geometry_diagnostics`. One object is rotated about its observed centre and slid
towards the other along their centre line, with a random lateral offset, until
the exact convex penetration depth equals the target. Only frame-zero masks and
scene specifications of training and validation clips are read; no model,
threshold or recorded score changes.

The simulator's rasterisation fixes the true shapes only to within a fraction of
a pixel, so every offset in --offsets places them that far from the pixel-index
origin. Penetration depends only on relative placement, so the offset matters
only through rotation; the summary reports each offset separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.evaluation import rigid_geometry
from click2future.evaluation.rigid_geometry import _penetration, geometry_diagnostics
from click2future.models.object_motion import observe
from click2future.sim.geometry import polygon_vertices

DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
TOLERANCE = 1.0  # the audit's fixed one-native-pixel tolerance
SPLITS = ("train", "validation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def true_polygon(obj, resolution, offset):
    """Exact convex outline in evaluator pixel coordinates; circles as 256-gons."""
    x, y = obj["position"]
    px, py = x * resolution + offset, y * resolution + offset
    if obj["shape"] == "circle":
        radius = obj["geometry"]["radius"] * resolution
        angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
        return np.stack((px + radius * np.cos(angles), py + radius * np.sin(angles)), -1)
    c, s = math.cos(obj["angle"]), math.sin(obj["angle"])
    vertices = polygon_vertices(obj["shape"], obj["geometry"])
    return np.array(
        [
            [px + (lx * c - ly * s) * resolution, py + (lx * s + ly * c) * resolution]
            for lx, ly in vertices
        ]
    )


def depth_along(fixed, mover, centre, theta, lateral, toward):
    """Exact convex penetration as `mover`, rotated about `centre`, slides along `toward`."""

    def depth(step):
        return _penetration(fixed, place(mover, centre, theta, lateral + step * toward))

    return depth


def place(polygon, centre, theta, shift):
    c, s = math.cos(theta), math.sin(theta)
    return (polygon - centre) @ np.array([[c, -s], [s, c]]).T + centre + shift


def layouts(dataset):
    """One entry per distinct frame-zero image among training/validation clips."""
    manifest = json.loads((Path(dataset) / "dataset_manifest.json").read_text())
    seen = set()
    for row in sorted(manifest["samples"], key=lambda r: r["sample_id"]):
        if row["split"] not in SPLITS:
            continue
        rollout = load_rollout(dataset, row["sample_id"])
        key = hashlib.sha256(rollout["frames"][0].tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        yield row["sample_id"], rollout


def run(dataset, *, offset, trials, seed, keep_rows):
    rng = np.random.default_rng(seed)
    rows, used = [], []
    for sample_id, rollout in layouts(dataset):
        resolution = rollout["frames"].shape[1]
        observation = observe(
            rollout["frames"][0], np.asarray(rollout["selected_mask"]).astype(np.uint8)
        )
        instances = np.asarray(rollout["instance_masks"][0])
        objects = {o["object_id"]: o for o in rollout["scene_spec"]["objects"]}
        order = [
            max(objects, key=lambda oid: int((instances[oid] & observation.masks[i]).sum()))
            for i in range(2)
        ]
        if order[0] == order[1]:
            raise ValueError(f"Could not match observed masks to objects in {sample_id}")
        polygons = [true_polygon(objects[oid], resolution, offset) for oid in order]
        centres = [observation.centers[i] * resolution for i in range(2)]
        used.append(sample_id)
        for depth in DEPTHS:
            for _ in range(trials):
                mover = int(rng.integers(2))
                fixed = 1 - mover
                theta = float(rng.uniform(-math.pi / 6, math.pi / 6))
                toward = centres[fixed] - centres[mover]
                toward = toward / np.linalg.norm(toward)
                lateral = (
                    np.array([-toward[1], toward[0]]) * rng.uniform(-0.5, 0.5) * 0.08 * resolution
                )

                low, high = -5.0, float(np.linalg.norm(centres[fixed] - centres[mover]))
                true_depth = depth_along(
                    polygons[fixed], polygons[mover], centres[mover], theta, lateral, toward
                )

                if true_depth(low) > 0 or true_depth(high) < depth:
                    continue
                for _ in range(60):
                    middle = (low + high) / 2
                    below = true_depth(middle) <= 1e-9 if depth == 0 else true_depth(middle) < depth
                    low, high = (middle, high) if below else (low, middle)
                step = low if depth == 0 else (low + high) / 2
                shift = lateral + step * toward
                motion = np.zeros((1, 1, 2, 4))
                motion[0, 0, mover, :2] = shift / (0.5 * resolution)
                motion[0, 0, mover, 2] = math.sin(theta)
                motion[0, 0, mover, 3] = (math.cos(theta) - 1) / 2
                clip = geometry_diagnostics(observation, motion)["clips"][0]
                rows.append(
                    {
                        "layout": sample_id,
                        "target_depth": depth,
                        "true_depth": true_depth(step),
                        "measured_depth": clip["hull_penetration_pixels_per_frame"][0],
                        "mask_overlap_pixels": clip["overlap_pixels_per_frame"][0],
                        "shapes": sorted(objects[oid]["shape"] for oid in order),
                        "rotation_degrees": math.degrees(theta),
                    }
                )
    return summarize(rows, used, offset, trials, seed, keep_rows)


def summarize(rows, used, offset, trials, seed, keep_rows):
    true = np.array([r["true_depth"] for r in rows])
    measured = np.array([r["measured_depth"] for r in rows])
    error = measured - true
    by_depth = {}
    for depth in DEPTHS:
        pick = np.array([r["target_depth"] == depth for r in rows])
        e, m = error[pick], measured[pick]
        by_depth[f"{depth:g}"] = {
            "trials": int(pick.sum()),
            "median_measured": float(np.median(m)),
            "error_percentiles_5_25_50_75_95": [
                float(np.percentile(e, q)) for q in (5, 25, 50, 75, 95)
            ],
            "flagged_over_tolerance": float(np.mean(m > TOLERANCE)),
            "mask_overlap_over_one_pixel": float(
                np.mean(
                    [
                        r["mask_overlap_pixels"] > 1
                        for r, keep in zip(rows, pick, strict=True)
                        if keep
                    ]
                )
            ),
        }
    clear = true >= 0.5
    slope, intercept = np.polyfit(true[clear], measured[clear], 1)
    shapes = {}
    for pair in sorted({tuple(r["shapes"]) for r in rows}):
        pick = clear & np.array([tuple(r["shapes"]) == pair for r in rows])
        shapes["-".join(pair)] = {
            "trials": int(pick.sum()),
            "median_error": float(np.median(error[pick])),
            "error_percentiles_5_95": [float(np.percentile(error[pick], q)) for q in (5, 95)],
        }
    record = {
        "offset_pixels": offset,
        "layouts": len(used),
        "trials_per_layout_and_depth": trials,
        "seed": seed,
        "trials": len(rows),
        "linear_fit_true_at_least_half_pixel": {
            "slope": float(slope),
            "intercept": float(intercept),
        },
        "median_error_true_at_least_half_pixel": float(np.median(error[clear])),
        "mean_absolute_error_true_at_least_half_pixel": float(np.abs(error[clear]).mean()),
        "share_within_half_pixel_true_at_least_half_pixel": float(
            np.mean(np.abs(error[clear]) <= 0.5)
        ),
        "false_flag_rate_true_at_most_one_pixel": float(
            np.mean(measured[true <= TOLERANCE] > TOLERANCE)
        ),
        "miss_rate_true_at_least_one_and_half_pixel": float(
            np.mean(measured[true >= 1.5] <= TOLERANCE)
        ),
        "by_target_depth": by_depth,
        "by_shape_pair_true_at_least_half_pixel": shapes,
    }
    if keep_rows:
        record["layout_sample_ids"] = used
        record["rows"] = rows
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.2, 0.0, 0.5])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve earlier results; use a fresh output path")
    runs = [
        run(args.dataset, offset=o, trials=args.trials, seed=args.seed, keep_rows=i == 0)
        for i, o in enumerate(args.offsets)
    ]
    record = {
        "status": "measured_positive_control",
        "scope": (
            "Known continuous overlaps scored with the unchanged rigid hull-penetration audit; "
            "training/validation frame-zero layouts only; no model, threshold or result changes."
        ),
        "tolerance_pixels": TOLERANCE,
        "target_depths_pixels": list(DEPTHS),
        "rotation_range_degrees": [-30, 30],
        "dataset_manifest_sha256": sha256(args.dataset / "dataset_manifest.json"),
        "source_sha256": {
            "rigid_geometry.py": sha256(rigid_geometry.__file__),
            "penetration_control.py": sha256(__file__),
        },
        "primary_offset_pixels": args.offsets[0],
        "runs": runs,
        "test_splits_read": [],
        "limitations": [
            "Circles and rectangles only: triangles occur only in the shape-OOD split.",
            "Single frames with one rigid move; no motion blur or multi-frame effects.",
            "Calibration of the audit, not of any model; recorded scores are unchanged.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    primary = runs[0]
    print(
        json.dumps(
            {
                k: primary[k]
                for k in primary
                if k not in ("rows", "by_target_depth", "layout_sample_ids")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
