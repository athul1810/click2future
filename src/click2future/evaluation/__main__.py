"""Evaluate an extracted-track/contact NPZ and write a JSON record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .metrics import collision_metrics, trajectory_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", choices=("video_tracking", "auxiliary_head", "control_fixture"), required=True
    )
    parser.add_argument("--extractor-version")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-id", required=True, help="Reference rollout ID")
    parser.add_argument("--scene-group-id", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--selected-object", type=int, required=True)
    parser.add_argument(
        "--event-tolerance", type=int, required=True, help="Recorded-frame intervals"
    )
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from the input archive")
    try:
        if args.source == "video_tracking" and not args.extractor_version:
            raise ValueError("video_tracking requires --extractor-version")
        fields = (
            "predicted_positions",
            "reference_positions",
            "object_valid",
            "tracked",
            "predicted_collision_onset",
            "reference_collision_onset",
            "collision_observed",
            "timestamps",
        )
        with np.load(args.input, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in fields}
        with args.input.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        positions = arrays["reference_positions"]
        predicted = arrays["predicted_positions"]
        if positions.ndim != 3 or predicted.ndim != 4:
            raise ValueError("Position arrays must have shapes [T,N,2] and [K,T,N,2]")
        frames, objects, _ = positions.shape
        if frames < 2 or objects < 1 or positions.shape[-1] != 2:
            raise ValueError("Reference positions require T >= 2, N >= 1, and two coordinates")
        expected = (frames - 1, objects, objects)
        if arrays["reference_collision_onset"].shape != expected:
            raise ValueError("Reference collisions must align with position frame intervals")
        if arrays["predicted_collision_onset"].shape != (len(predicted), *expected):
            raise ValueError("Predicted collision samples/intervals must align with positions")
        timestamps = arrays["timestamps"]
        if timestamps.dtype.kind not in "fiu" or timestamps.shape != (frames,):
            raise ValueError("timestamps must be a numeric vector with one value per frame")
        if (
            not np.isfinite(timestamps).all()
            or timestamps[0] != 0
            or not (np.diff(timestamps.astype(np.float64)) > 0).all()
        ):
            raise ValueError("timestamps must start at zero and be finite and strictly increasing")
        report = {
            "status": "passed",
            "metric_version": "1",
            "source": args.source,
            "extractor_version": args.extractor_version,
            "run_id": args.run_id,
            "sample_id": args.sample_id,
            "scene_group_id": args.scene_group_id,
            "condition_id": args.condition_id,
            "input_sha256": archive_hash,
            "selected_object": args.selected_object,
            "timestamps_seconds": timestamps.tolist(),
            "trajectory": trajectory_metrics(
                predicted,
                positions,
                arrays["object_valid"],
                arrays["tracked"],
                selected_object=args.selected_object,
            ),
            "collision": collision_metrics(
                arrays["predicted_collision_onset"],
                arrays["reference_collision_onset"],
                arrays["object_valid"],
                arrays["collision_observed"],
                tolerance=args.event_tolerance,
            ),
        }
    except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
