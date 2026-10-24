"""Time-resolved object motion in saved evaluation samples versus their references.

The action controls score only the first 0.5 seconds. This reads the saved sample
videos of a finished evaluation suite (validation or selected-object), tracks every
frame with the evaluation tracker and reports, for the selected object and the
other object separately, the displacement from frame zero, its projection on the
force direction and the tracked-area ratio at fixed times. Simulator positions of
every reference in the suite's plan give the same quantities for comparison.
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


def _action(condition_id):
    """forward/reverse/zero for action cases, select for selected-object cases."""
    if "-select-" in condition_id:
        return "select"
    suffix = condition_id.rsplit("-", 1)[-1]
    return suffix if suffix in {"forward", "reverse", "zero"} else None


def _record(bucket, positions, tracked, areas, selected, direction):
    """positions [T,N,2], tracked [T,N], areas [T,N] or None for references."""
    for obj in range(positions.shape[1]):
        role = "selected" if obj == selected else "other"
        shift = positions[:, obj] - positions[0, obj]
        distance, along = np.linalg.norm(shift, axis=-1), shift @ direction
        for frame in FRAMES:
            ok = bool(tracked[frame, obj]) and np.isfinite(distance[frame])
            area = 1.0 if areas is None else areas[frame, obj] / max(areas[0, obj], 1.0)
            bucket[role][frame].append((distance[frame], along[frame], area) if ok else None)


def analyse(plan_path, data, samples, run_index=0):
    plan = json.loads(Path(plan_path).read_text())
    data, samples = Path(data), Path(samples)
    run = json.loads((samples / "predictions.json").read_text())["runs"][run_index]
    tracker = TrackerConfig()
    collected = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for case in plan["cases"]:
        action = _action(case["condition_id"])
        if action is None or case["sample_id"] not in run["videos"]:
            continue
        base = load_batch(data, [case["sample_id"]])
        selected = int(base["selected_object_id"][0])
        force = base["force_xy"][0].astype(np.float64)
        norm = np.linalg.norm(force)
        direction = force / norm if norm else np.zeros(2)
        masks, valid = base["instance_masks"][0, 0], base["object_valid"][0]
        for video in np.load(samples / run["videos"][case["sample_id"]])["videos"]:
            tracks = track_video(video, masks, valid, tracker)
            areas = tracks.masks.sum(axis=(2, 3)).astype(np.float64)
            _record(
                collected["model"][action],
                tracks.positions.astype(np.float64),
                tracks.tracked,
                areas,
                selected,
                direction,
            )
        for reference_id in case["reference_ids"]:
            ref = load_batch(data, [reference_id])
            positions = ref["positions"][0].astype(np.float64)
            tracked = np.broadcast_to(ref["object_valid"][0], positions.shape[:2])
            _record(collected["reference"][action], positions, tracked, None, selected, direction)
    summary = {}
    for source, actions in collected.items():
        for action, roles in actions.items():
            for role, frames in roles.items():
                for frame, values in frames.items():
                    good = np.array([v for v in values if v is not None], dtype=np.float64)
                    summary.setdefault(source, {}).setdefault(action, {}).setdefault(role, {})[
                        str(frame)
                    ] = {
                        "tracked": int(len(good)),
                        "total": len(values),
                        "mean_displacement": float(good[:, 0].mean()) if len(good) else None,
                        "mean_along_force": float(good[:, 1].mean()) if len(good) else None,
                        "moving_fraction": float((good[:, 0] > 1 / 64).mean())
                        if len(good)
                        else None,
                        "mean_area_ratio": float(good[:, 2].mean()) if len(good) else None,
                    }
    return {
        "status": "measured_diagnostic",
        "plan_sha256": plan.get("plan_sha256"),
        "split": plan.get("split"),
        "run_id": run["run_id"],
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "frames": list(FRAMES),
        "units": "normalized image coordinates; one pixel = 1/64; moving means > one pixel",
        "summary": summary,
        "scope": "Validation diagnostic of saved samples; frame-0 masks align identities only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="plan.json of the suite")
    parser.add_argument("--data", type=Path, required=True, help="local copy of its dataset")
    parser.add_argument("--samples", type=Path, required=True, help="its *-samples folder")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = analyse(args.plan, args.data, args.samples)
    if args.output:
        if args.output.exists():
            raise ValueError("Use a fresh output file")
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def fmt(value, places=4):
        return "" if value is None else f"{value:.{places}f}"

    print(
        f"{'source':10s} {'action':8s} {'object':9s} {'frame':>5s} {'tracked':>9s} "
        f"{'|d|':>8s} {'along f':>8s} {'moving':>7s} {'area':>6s}"
    )
    for source, actions in record["summary"].items():
        for action, roles in actions.items():
            for role, frames in roles.items():
                for frame, s in frames.items():
                    print(
                        f"{source:10s} {action:8s} {role:9s} {frame:>5s} "
                        f"{s['tracked']:>4d}/{s['total']:<4d} {fmt(s['mean_displacement']):>8s} "
                        f"{fmt(s['mean_along_force']):>8s} {fmt(s['moving_fraction'], 2):>7s} "
                        f"{fmt(s['mean_area_ratio'], 2):>6s}"
                    )


if __name__ == "__main__":
    main()
