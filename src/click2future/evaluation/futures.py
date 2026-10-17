"""Evaluate temporal consistency and ensemble diversity from generated RGB futures."""

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
from click2future.sim.scene import visible_scene

from .contacts import estimate_contacts
from .diversity import collision_frequency, ensemble_metrics
from .temporal import temporal_metrics
from .tracking import TRACKER_VERSION, TrackerConfig, track_video


def evaluate_futures(
    references,
    videos,
    *,
    tracker: TrackerConfig,
    contact_settings: dict,
    source: str,
    run_id: str,
    frame_valid=None,
    contact_margin: int = 1,
) -> dict:
    if source not in ("generated_video", "simulator_control"):
        raise ValueError("Source must be generated_video or simulator_control")
    if not references:
        raise ValueError("At least one reference rollout is required")
    base = references[0]
    for reference in references:
        validate_rollout(reference)
        if (
            visible_scene(reference["scene_spec"]) != visible_scene(base["scene_spec"])
            or reference["scene_spec"]["scene_group_id"] != base["scene_spec"]["scene_group_id"]
            or reference["scene_spec"]["condition_id"] != base["scene_spec"]["condition_id"]
            or reference["frames"].shape != base["frames"].shape
            or not np.array_equal(reference["frames"][0], base["frames"][0])
            or not np.array_equal(reference["instance_masks"][0], base["instance_masks"][0])
            or not np.array_equal(reference["timestamps"], base["timestamps"])
        ):
            raise ValueError(
                "Reference futures must share visible scene, action, identities and timing"
            )
    videos = np.asarray(videos)
    if (
        videos.dtype != np.uint8
        or videos.ndim != 5
        or len(videos) < 1
        or videos.shape[1:] != base["frames"].shape
    ):
        raise ValueError("videos must be matching uint8 [K,T,H,W,3]")
    if not np.all(videos[:, 0] == base["frames"][0]):
        raise ValueError("All futures must preserve the shared first frame")
    if frame_valid is None:
        frame_valid = np.ones(videos.shape[:2], bool)
    else:
        frame_valid = np.asarray(frame_valid)
        if (
            frame_valid.shape != videos.shape[:2]
            or frame_valid.dtype != np.bool_
            or not frame_valid[:, 0].all()
        ):
            raise ValueError("frame_valid must be boolean [K,T] with valid first frames")
    tracks = [
        track_video(clip, base["instance_masks"][0], base["object_valid"], tracker)
        for clip in videos
    ]
    for k, track in enumerate(tracks):
        track.tracked[~frame_valid[k]] = False
        track.positions[~frame_valid[k]] = np.nan
        track.masks[~frame_valid[k]] = False
        track.reasons[~frame_valid[k]] = "missing_rgb"
    positions = np.stack([track.positions for track in tracks])
    tracked = np.stack([track.tracked for track in tracks])
    masks = np.stack([track.masks for track in tracks])
    contacts = [
        estimate_contacts(track.masks, track.tracked, base["object_valid"], **contact_settings)
        for track in tracks
    ]
    reasons = Counter(
        str(reason) for track in tracks for reason in track.reasons[1:, base["object_valid"]].flat
    )
    return {
        "status": "passed",
        "source": source,
        "run_id": run_id,
        "extractor_version": TRACKER_VERSION,
        "scene_group_id": base["scene_spec"]["scene_group_id"],
        "condition_id": base["scene_spec"]["condition_id"],
        "reference_count": len(references),
        "timestamps_seconds": base["timestamps"].tolist(),
        "settings": {
            "tracker": asdict(tracker),
            "contacts": contact_settings,
            "contact_margin_intervals": contact_margin,
        },
        "tracking_reasons_future": dict(sorted(reasons.items())),
        "temporal": temporal_metrics(
            videos,
            positions,
            masks,
            tracked,
            base["object_valid"],
            base["timestamps"],
            collision_onset=np.any([r["collision_onset"] for r in references], axis=0),
            wall_contact_onset=np.any([r["wall_contact_onset"] for r in references], axis=0),
            selected_object=base["selected_object_id"],
            force_duration=base["scene_spec"]["action"]["force_duration"]
            if np.any(base["force_xy"])
            else 0,
            contact_margin=contact_margin,
        ),
        "ensemble": ensemble_metrics(
            positions,
            tracked,
            base["object_valid"],
            np.stack([r["positions"] for r in references]),
            selected_object=base["selected_object_id"],
        ),
        "collision_frequency": collision_frequency(
            np.stack([c.onset for c in contacts]),
            np.stack([c.observed for c in contacts]),
            np.stack([r["collision_onset"] for r in references]),
            base["object_valid"],
        ),
        "reference_note": (
            "Temporal exclusion windows use the union of declared reference contacts, "
            "fixed across methods and unaffected by generated trajectories"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-ids", nargs="+", required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/pixel.yaml"))
    parser.add_argument("--contact-margin", type=int, default=1)
    parser.add_argument("--source", choices=["generated_video", "simulator_control"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.reference_ids)) != len(args.reference_ids):
        parser.error("Reference IDs must be distinct")
    inputs = {"videos": args.videos, "config": args.config}
    for sample_id in args.reference_ids:
        for extension in (".json", ".npz"):
            inputs[sample_id + extension] = args.reference_dir / (sample_id + extension)
    if args.output.resolve() in {p.resolve() for p in inputs.values()}:
        parser.error("Output must not overwrite an input")
    try:
        references = [
            load_rollout(args.reference_dir, sample_id) for sample_id in args.reference_ids
        ]
        config = yaml.safe_load(args.config.read_text())
        if list(references[0]["frames"].shape[1:3]) != config["resolution_hw"]:
            raise ValueError("Resolution differs from the extractor configuration")
        fps = float(config.get("fps", 8))
        if (
            not np.isfinite(fps)
            or fps <= 0
            or not np.allclose(np.diff(references[0]["timestamps"]), 1 / fps, rtol=0, atol=1e-8)
        ):
            raise ValueError("Frame timing differs from the extractor configuration")
        with np.load(args.videos, allow_pickle=False) as archive:
            videos = archive["videos"]
            frame_valid = archive["frame_valid"] if "frame_valid" in archive else None
        settings = {"max_gap_pixels": config["max_gap_pixels"]}
        settings.update(
            {
                name: config[name]
                for name in ("motion_threshold", "motion_window", "motion_rule")
                if name in config
            }
        )
        report = evaluate_futures(
            references,
            videos,
            tracker=TrackerConfig(**config["tracker"]),
            contact_settings=settings,
            source=args.source,
            run_id=args.run_id,
            frame_valid=frame_valid,
            contact_margin=args.contact_margin,
        )
        report["reference_ids"] = args.reference_ids
        report["input_sha256"] = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()
        }
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        EOFError,
        BadZipFile,
        yaml.YAMLError,
    ) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
