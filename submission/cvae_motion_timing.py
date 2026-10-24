"""Time-resolved selected-object motion in saved validation samples versus references.

The action controls score only the first 0.5 seconds. This reads the saved sample
videos from a finished evaluation, tracks every frame with the evaluation tracker,
and reports the selected object's displacement from frame zero, its projection on
the force direction and its tracked-area ratio at fixed times. Simulator positions
of every hidden-physics reference give the same quantities for comparison.
Frame-0 masks align identities only, as in the evaluation; nothing is resampled.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from click2future.data import load_batch
from click2future.evaluation.tracking import TrackerConfig, track_video

FRAMES = (4, 8, 16, 31)
VARIANTS = ("forward", "reverse", "zero")


def _motion(positions, origin, direction):
    """positions [T,2] -> per-frame displacement norm and projection on the force."""
    shift = positions - origin
    return np.linalg.norm(shift, axis=-1), shift @ direction


def analyse(data, samples, run_index=0):
    data, samples = Path(data), Path(samples)
    predictions = json.loads((samples / "predictions.json").read_text())
    run = predictions["runs"][run_index]
    manifest = json.loads((data / "dataset_manifest.json").read_text())
    rows = {row["sample_id"]: row for row in manifest["samples"]}
    references = defaultdict(list)
    for row in manifest["samples"]:
        if row["split"] == "validation":
            references[row["condition_id"]].append(row["sample_id"])
    tracker = TrackerConfig()
    collected = {
        source: {variant: defaultdict(list) for variant in VARIANTS}
        for source in ("model", "reference")
    }
    for sample_id, relative in sorted(run["videos"].items()):
        row = rows[sample_id]
        variant = row["action_variant"]
        if variant not in VARIANTS:
            continue
        base = load_batch(data, [sample_id])
        selected = int(base["selected_object_id"][0])
        force = base["force_xy"][0].astype(np.float64)
        norm = np.linalg.norm(force)
        direction = force / norm if norm else np.zeros(2)
        masks, valid = base["instance_masks"][0, 0], base["object_valid"][0]
        videos = np.load(samples / relative)["videos"]
        for video in videos:
            tracks = track_video(video, masks, valid, tracker)
            path = tracks.positions[:, selected]
            area = tracks.masks[:, selected].sum(axis=(1, 2)).astype(np.float64)
            distance, along = _motion(path, path[0], direction)
            for frame in FRAMES:
                bucket = collected["model"][variant][frame]
                ok = bool(tracks.tracked[frame, selected]) and np.isfinite(distance[frame])
                bucket.append(
                    (distance[frame], along[frame], area[frame] / area[0]) if ok else None
                )
        for reference_id in references[row["condition_id"]]:
            ref = load_batch(data, [reference_id])["positions"][0, :, selected]
            distance, along = _motion(ref.astype(np.float64), ref[0], direction)
            for frame in FRAMES:
                collected["reference"][variant][frame].append((distance[frame], along[frame], 1.0))
    summary = {}
    for source, variants in collected.items():
        for variant, frames in variants.items():
            for frame, values in frames.items():
                good = np.array([v for v in values if v is not None], dtype=np.float64)
                summary.setdefault(source, {}).setdefault(variant, {})[str(frame)] = {
                    "tracked": int(len(good)),
                    "total": len(values),
                    "mean_displacement": float(good[:, 0].mean()) if len(good) else None,
                    "mean_along_force": float(good[:, 1].mean()) if len(good) else None,
                    "moving_fraction": float((good[:, 0] > 1 / 64).mean()) if len(good) else None,
                    "mean_area_ratio": float(good[:, 2].mean()) if len(good) else None,
                }
    return {
        "status": "measured_diagnostic",
        "run_id": run["run_id"],
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "frames": list(FRAMES),
        "units": "normalized image coordinates; one pixel = 1/64; moving means > one pixel",
        "summary": summary,
        "scope": "Validation diagnostic of saved samples; frame-0 masks align identities only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = analyse(args.data, args.samples)
    if args.output:
        if args.output.exists():
            raise ValueError("Use a fresh output file")
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"{'source':10s} {'action':8s} {'frame':>5s} {'tracked':>9s} {'|d|':>8s} "
          f"{'along f':>8s} {'moving':>7s} {'area':>6s}")
    for source, variants in record["summary"].items():
        for variant, frames in variants.items():
            for frame, s in frames.items():
                fmt = lambda v, p=4: "" if v is None else f"{v:.{p}f}"  # noqa: E731
                print(f"{source:10s} {variant:8s} {frame:>5s} {s['tracked']:>4d}/{s['total']:<4d} "
                      f"{fmt(s['mean_displacement']):>8s} {fmt(s['mean_along_force']):>8s} "
                      f"{fmt(s['moving_fraction'], 2):>7s} {fmt(s['mean_area_ratio'], 2):>6s}")


if __name__ == "__main__":
    main()
