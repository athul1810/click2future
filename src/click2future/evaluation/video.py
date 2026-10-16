"""Extract and evaluate tracks/contacts from generated RGB clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import yaml

from click2future.data import load_rollout, validate_rollout

from .contacts import estimate_contacts
from .metrics import collision_metrics, trajectory_metrics
from .tracking import TRACKER_VERSION, TrackerConfig, track_video


def evaluate_videos(
    reference: dict,
    videos,
    *,
    tracker: TrackerConfig,
    max_gap_pixels: float,
    event_tolerance: int,
    run_id: str,
    motion_threshold: float | None = None,
    motion_window: int = 3,
    motion_rule: str = "opposing",
) -> tuple[dict, dict]:
    """Use only the reference's frame-0 masks during extraction; score afterwards."""
    validate_rollout(reference)
    videos = np.asarray(videos)
    if videos.ndim != 5 or videos.dtype != np.uint8 or len(videos) < 1:
        raise ValueError("videos must be uint8 [K,T,H,W,3] with K >= 1")
    if videos.shape[1:] != reference["frames"].shape:
        raise ValueError("Generated videos must match the reference frame count and resolution")
    if not np.all(videos[:, 0] == reference["frames"][0]):
        raise ValueError("Every sample must preserve the supplied first frame exactly")
    tracks = [
        track_video(clip, reference["instance_masks"][0], reference["object_valid"], tracker)
        for clip in videos
    ]
    contacts = [
        estimate_contacts(
            t.masks,
            t.tracked,
            reference["object_valid"],
            max_gap_pixels=max_gap_pixels,
            motion_threshold=motion_threshold,
            motion_window=motion_window,
            motion_rule=motion_rule,
        )
        for t in tracks
    ]
    arrays = {
        "predicted_positions": np.stack([t.positions for t in tracks]),
        "reference_positions": reference["positions"],
        "object_valid": reference["object_valid"],
        "tracked": np.stack([t.tracked for t in tracks]),
        "predicted_collision_onset": np.stack([c.onset for c in contacts]),
        "reference_collision_onset": reference["collision_onset"],
        "collision_observed": np.stack([c.observed for c in contacts]),
        "timestamps": reference["timestamps"],
    }
    reasons = Counter(
        str(reason) for t in tracks for reason in t.reasons[1:, reference["object_valid"]].flat
    )
    report = {
        "status": "passed",
        "source": "video_tracking",
        "metric_version": "1",
        "extractor_version": TRACKER_VERSION,
        "contact_estimator": "normal-motion-1"
        if motion_rule == "normal"
        else "opposing-motion-2"
        if motion_threshold is not None
        else "pixel-proximity-1",
        "run_id": run_id,
        "scene_group_id": reference["scene_spec"]["scene_group_id"],
        "condition_id": reference["scene_spec"]["condition_id"],
        "timestamps_seconds": reference["timestamps"].tolist(),
        "initialization": "shared_reference_frame_zero_identity_masks; evaluation_only",
        "settings": {
            "tracker": asdict(tracker),
            "max_gap_pixels": max_gap_pixels,
            "motion_threshold": motion_threshold,
            "motion_window": motion_window,
            "motion_rule": motion_rule,
            "event_tolerance_intervals": event_tolerance,
            "resolution_hw": list(videos.shape[2:4]),
        },
        "tracking_reasons_future": dict(sorted(reasons.items())),
        "trajectory": trajectory_metrics(
            arrays["predicted_positions"],
            arrays["reference_positions"],
            arrays["object_valid"],
            arrays["tracked"],
            selected_object=reference["selected_object_id"],
        ),
        "collision": collision_metrics(
            arrays["predicted_collision_onset"],
            arrays["reference_collision_onset"],
            arrays["object_valid"],
            arrays["collision_observed"],
            tolerance=event_tolerance,
        ),
        "limitation": (
            "Solid-colour baseline; between-frame contacts and occluded objects may be missed"
        ),
    }
    return report, arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--videos", type=Path, required=True, help="NPZ with uint8 videos [K,T,H,W,3]"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/pixel.yaml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extracted-output", type=Path)
    args = parser.parse_args(argv)
    inputs = [
        args.videos,
        args.config,
        args.reference_dir / (args.sample_id + ".npz"),
        args.reference_dir / (args.sample_id + ".json"),
    ]
    outputs = [args.output] + ([args.extracted_output] if args.extracted_output else [])
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        p.resolve() == q.resolve() for p in outputs for q in inputs
    ):
        parser.error("Outputs must be distinct and must not overwrite inputs")
    try:
        config = yaml.safe_load(args.config.read_text())
        reference = load_rollout(args.reference_dir, args.sample_id)
        if list(reference["frames"].shape[1:3]) != config["resolution_hw"]:
            raise ValueError("Resolution differs from the extractor calibration config")
        if "fps" in config:
            fps = float(config["fps"])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Configured fps must be finite and positive")
            if not np.allclose(np.diff(reference["timestamps"]), 1 / fps, rtol=0, atol=1e-8):
                raise ValueError("Frame timing differs from the extractor calibration config")
        with np.load(args.videos, allow_pickle=False) as archive:
            videos = archive["videos"]
        report, arrays = evaluate_videos(
            reference,
            videos,
            tracker=TrackerConfig(**config["tracker"]),
            max_gap_pixels=config["max_gap_pixels"],
            event_tolerance=config["event_tolerance_intervals"],
            run_id=args.run_id,
            motion_threshold=config.get("motion_threshold"),
            motion_window=config.get("motion_window", 3),
            motion_rule=config.get("motion_rule", "opposing"),
        )
        report["sample_id"] = args.sample_id
        report["input_sha256"] = {}
        for name, path in zip(
            ["videos", "config", "reference_arrays", "reference_metadata"], inputs, strict=True
        ):
            with path.open("rb") as stream:
                report["input_sha256"][name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if args.extracted_output:
            args.extracted_output.parent.mkdir(parents=True, exist_ok=True)
            with args.extracted_output.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
    except (ValueError, TypeError, KeyError, OSError, BadZipFile, yaml.YAMLError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
