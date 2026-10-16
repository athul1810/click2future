"""Run paired action checks from generated videos or reproduce simulator controls."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import yaml

from click2future.data import load_rollout, validate_rollout
from click2future.sim import make_scene, run_rollout

from .actions import action_metrics, reversal_metrics
from .tracking import TRACKER_VERSION, TrackerConfig, track_video

KINDS = ("forward", "reverse", "zero")


def validate_controls(references: dict, end_frame: int) -> None:
    """Certify the fixed window using simulator labels before scoring any output."""
    base = references["forward"]
    for kind in KINDS:
        ref = references[kind]
        validate_rollout(ref)
        if not 1 <= end_frame < len(ref["frames"]):
            raise ValueError("Control end frame must exist in every reference")
        if (
            ref["scene_spec"]["objects"] != base["scene_spec"]["objects"]
            or ref["selected_object_id"] != base["selected_object_id"]
            or ref["scene_spec"]["action"]["force_duration"]
            != base["scene_spec"]["action"]["force_duration"]
            or not np.array_equal(ref["timestamps"], base["timestamps"])
            or not np.array_equal(ref["frames"][0], base["frames"][0])
        ):
            raise ValueError("Controls must share objects, physics, first frame, timing and target")
        for obj in ref["scene_spec"]["objects"]:
            if any(obj["initial_velocity"]) or obj["initial_angular_velocity"] != 0:
                raise ValueError("Controls must start at rest")
        if ref["collision_onset"][:end_frame].any() or ref["wall_contact_onset"][:end_frame].any():
            raise ValueError("Control window must be contact-free for every reference")
    force = base["force_xy"]
    if (
        not np.any(force)
        or not np.array_equal(references["reverse"]["force_xy"], -force)
        or np.any(references["zero"]["force_xy"])
    ):
        raise ValueError("Controls require a nonzero force, its exact reverse, and zero force")


def evaluate_controls(
    references: dict,
    videos: dict,
    *,
    tracker: TrackerConfig,
    end_frame: int,
    motion_threshold: float,
    source: str,
    run_id: str,
    resolution_hw: tuple[int, int] = (64, 64),
    fps: int = 8,
) -> dict:
    if source not in ("simulator_control", "generated_video"):
        raise ValueError("Source must be simulator_control or generated_video")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Configured fps must be finite and positive")
    validate_controls(references, end_frame)
    arrays, scores = {}, {}
    count = None
    for kind in KINDS:
        ref = references[kind]
        if list(ref["frames"].shape[1:3]) != list(resolution_hw) or not np.allclose(
            np.diff(ref["timestamps"]), 1 / fps, rtol=0, atol=1e-8
        ):
            raise ValueError("Reference resolution/timing differs from the control configuration")
        clips = np.asarray(videos[kind])
        if (
            clips.dtype != np.uint8
            or clips.ndim != 5
            or len(clips) < 1
            or clips.shape[1:] != ref["frames"].shape
        ):
            raise ValueError("Control videos must be uint8 [K,T,H,W,3] matching references")
        if not np.all(clips[:, 0] == ref["frames"][0]):
            raise ValueError("Control videos must preserve their shared first frame")
        if count is not None and count != len(clips):
            raise ValueError("All controls must have the same number of paired sample seeds")
        count = len(clips)
        tracks = [
            track_video(clip, ref["instance_masks"][0], ref["object_valid"], tracker)
            for clip in clips
        ]
        positions = np.stack([track.positions for track in tracks])
        tracked = np.stack([track.tracked for track in tracks])
        arrays[kind] = positions, tracked
        scores[kind] = action_metrics(
            positions,
            tracked,
            ref["object_valid"],
            selected_object=ref["selected_object_id"],
            force_xy=ref["force_xy"],
            end_frame=end_frame,
            motion_threshold=motion_threshold,
        )
    base = references["forward"]
    scores["reversal"] = reversal_metrics(
        arrays["forward"][0],
        arrays["reverse"][0],
        arrays["forward"][1],
        arrays["reverse"][1],
        base["object_valid"],
        selected_object=base["selected_object_id"],
        force_xy=base["force_xy"],
        end_frame=end_frame,
        motion_threshold=motion_threshold,
    )
    return {
        "status": "passed",
        "source": source,
        "run_id": run_id,
        "extractor_version": TRACKER_VERSION,
        "action_metric_version": "1",
        "settings": {
            "tracker": asdict(tracker),
            "end_frame": end_frame,
            "end_time_seconds": float(base["timestamps"][end_frame]),
            "motion_threshold": motion_threshold,
            "resolution_hw": list(resolution_hw),
            "fps": fps,
        },
        "condition_ids": {kind: references[kind]["scene_spec"]["condition_id"] for kind in KINDS},
        "pairing": "Same sample index must use the same sampling seed across actions",
        "scores": scores,
    }


def simulator_controls(seed: int) -> dict:
    """A small contact-free fixture family; these are not model benchmark results."""
    scene = make_scene(seed, object_count=2, physics_seed=seed + 100000)
    selected = seed % 2
    for obj in scene["objects"]:
        obj["position"] = [0.5, 0.5] if obj["object_id"] == selected else [0.15, 0.15]
        obj["hidden_physics"]["mass"] = 0.8 + 0.1 * (seed % 5)
    angle = (seed % 8) * np.pi / 4
    force = 0.4 * np.array([np.cos(angle), np.sin(angle)])
    references = {}
    for kind, scale in zip(KINDS, [1, -1, 0], strict=True):
        variant = copy.deepcopy(scene)
        variant["action"]["selected_object_id"] = selected
        variant["action"]["force_xy"] = (force * scale).tolist()
        variant["condition_id"] = f"control-{seed}-{kind}"
        references[kind] = run_rollout(variant, frame_count=5, fps=8, resolution=64)
    return references


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-check", action="store_true")
    parser.add_argument("--reference-dir", type=Path)
    for kind in KINDS:
        parser.add_argument(f"--{kind}-id")
    parser.add_argument("--videos", type=Path, help="NPZ with forward/reverse/zero [K,T,H,W,3]")
    parser.add_argument("--sample-seeds", type=int, nargs="+", help="Paired model sampling seeds")
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/actions.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.simulator_check:
        if (
            args.videos
            or args.reference_dir
            or args.sample_seeds
            or any(getattr(args, kind + "_id") for kind in KINDS)
        ):
            parser.error("Simulator checks cannot be combined with generated-video inputs")
    elif not (
        args.videos
        and args.reference_dir
        and args.run_id
        and args.sample_seeds
        and all(getattr(args, kind + "_id") for kind in KINDS)
    ):
        parser.error("Provide reference directory/IDs, videos, paired sample seeds and run ID")
    inputs = [args.config]
    if not args.simulator_check:
        inputs.append(args.videos)
        for kind in KINDS:
            for ext in (".json", ".npz"):
                inputs.append(args.reference_dir / (getattr(args, kind + "_id") + ext))
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("Output must not overwrite an input")
    try:
        config = yaml.safe_load(args.config.read_text())
        settings = dict(
            tracker=TrackerConfig(**config["tracker"]),
            end_frame=config["end_frame"],
            motion_threshold=config["motion_threshold"],
            resolution_hw=config["resolution_hw"],
            fps=config["fps"],
        )
        if args.simulator_check:
            records = []
            for seed in range(50000, 50016):
                refs = simulator_controls(seed)
                records.append(
                    evaluate_controls(
                        refs,
                        {kind: refs[kind]["frames"][None] for kind in KINDS},
                        source="simulator_control",
                        run_id=f"control-{seed}",
                        **settings,
                    )
                )
            report = {
                "status": "passed",
                "purpose": "Extractor/control validation, not model results",
                "seed_start": 50000,
                "scene_groups": 16,
                "records": records,
            }
        else:
            refs = {
                kind: load_rollout(args.reference_dir, getattr(args, kind + "_id"))
                for kind in KINDS
            }
            with np.load(args.videos, allow_pickle=False) as archive:
                videos = {kind: archive[kind] for kind in KINDS}
            if any(len(clips) != len(args.sample_seeds) for clips in videos.values()) or len(
                set(args.sample_seeds)
            ) != len(args.sample_seeds):
                raise ValueError("Provide one distinct paired sampling seed per video sample")
            report = evaluate_controls(
                refs, videos, source="generated_video", run_id=args.run_id, **settings
            )
            report["sample_seeds"] = args.sample_seeds
        report["input_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
        }
    except (ValueError, TypeError, KeyError, OSError, BadZipFile, yaml.YAMLError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
