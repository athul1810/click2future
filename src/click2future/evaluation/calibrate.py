"""Calibrate pixel contact gaps on validation scenes, then measure held-out extraction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from click2future.sim.rollout import run_rollout
from click2future.sim.scene import make_scene

from .contacts import contacts_from_geometry, estimate_contacts, prepare_contact_geometry
from .metrics import collision_metrics
from .tracking import TRACKER_VERSION, TrackerConfig, track_video


def extract_cases(start: int, groups: int, variants: int, tracker: TrackerConfig) -> list:
    """Dedicated extractor scenes, independent of the model dataset manifests."""
    cases = []
    for index, seed in enumerate(range(start, start + groups)):
        for variant in range(variants):
            scene = make_scene(
                seed,
                object_count=2 + index % 3,
                physics_seed=1_000_000 + seed * 10 + variant,
                action_seed=seed,
            )
            reference = run_rollout(scene, frame_count=32, fps=8, resolution=64)
            tracks = track_video(
                reference["frames"],
                reference["instance_masks"][0],
                reference["object_valid"],
                tracker,
            )
            cases.append((reference, tracks))
    return cases


def summarize(
    cases: list, gap: float, tolerance: int, *, geometries=None, **contact_options
) -> dict:
    reasons = Counter()
    counts = {"event": Counter(), "interval": Counter(), "encounter": Counter()}
    errors = []
    expected = tracked_count = complete_clips = reference_events = 0
    if geometries is not None and len(geometries) != len(cases):
        raise ValueError("Geometry cache must match the case count")
    for index, (reference, tracks) in enumerate(cases):
        valid = reference["object_valid"]
        known = tracks.tracked[1:, valid]
        expected += known.size
        tracked_count += int(known.sum())
        complete_clips += int(known.all())
        reasons.update(str(reason) for reason in tracks.reasons[1:, valid].flat)
        errors.extend(
            np.linalg.norm(
                tracks.positions[1:, valid][known] - reference["positions"][1:, valid][known],
                axis=-1,
            ).tolist()
        )
        contact = (
            estimate_contacts(
                tracks.masks, tracks.tracked, valid, max_gap_pixels=gap, **contact_options
            )
            if geometries is None
            else contacts_from_geometry(geometries[index], max_gap_pixels=gap, **contact_options)
        )
        score = collision_metrics(
            contact.onset[None],
            reference["collision_onset"],
            valid,
            contact.observed[None],
            tolerance=tolerance,
        )["per_sample"][0]
        for kind in counts:
            for key, value in score[kind].items():
                if isinstance(value, int):
                    counts[kind][key] += value
        pairs = np.triu(valid[:, None] & valid[None, :], k=1)
        reference_events += int(reference["collision_onset"][:, pairs].sum())
    for kind, count in counts.items():
        tp, fp, fn = (count[key] for key in ("true_positive", "false_positive", "false_negative"))
        count["precision"] = tp / (tp + fp) if tp + fp else None
        count["recall"] = tp / (tp + fn) if tp + fn else None
        count["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        if kind in {"interval", "encounter"}:
            denominator = count["scored_bins"] if kind == "interval" else count["scored_pairs"]
            count["accuracy"] = (tp + count["true_negative"]) / denominator if denominator else None
    return {
        "clips": len(cases),
        "complete_tracking_clips": complete_clips,
        "expected_future_observations": expected,
        "tracked_future_observations": tracked_count,
        "missing_track_rate": 1 - tracked_count / expected,
        "mean_centroid_error_over_tracked_pixels": float(np.mean(errors) * 64) if errors else None,
        "p95_centroid_error_over_tracked_pixels": float(np.quantile(errors, 0.95) * 64)
        if errors
        else None,
        "tracking_reasons_future": dict(sorted(reasons.items())),
        "reference_events_all_pairs": reference_events,
        "reference_events_excluded_incomplete_pairs": reference_events
        - counts["event"]["true_positive"]
        - counts["event"]["false_negative"],
        "collision": {kind: dict(count) for kind, count in counts.items()},
    }


def select_gap(candidates: list[dict]) -> float:
    eligible = [
        row
        for row in candidates
        if row["metrics"]["collision"]["event"]["true_positive"]
        + row["metrics"]["collision"]["event"]["false_negative"]
        > 0
    ]
    if not eligible:
        raise ValueError(
            "Validation has no observable reference events; calibration is insufficient"
        )
    return min(
        eligible,
        key=lambda row: (-row["metrics"]["collision"]["event"]["f1"], row["max_gap_pixels"]),
    )["max_gap_pixels"]


def select_motion(candidates: list[dict]) -> dict:
    eligible = [
        row
        for row in candidates
        if row["metrics"]["collision"]["event"]["true_positive"]
        + row["metrics"]["collision"]["event"]["false_negative"]
        > 0
    ]
    if not eligible:
        raise ValueError(
            "Validation has no observable reference events; calibration is insufficient"
        )
    return min(
        eligible,
        key=lambda row: (
            -row["metrics"]["collision"]["event"]["f1"],
            row["motion_window"],
            row["motion_threshold"],
            row["max_gap_pixels"],
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-per-split", type=int, default=100)
    parser.add_argument("--variants-per-group", type=int, default=2)
    parser.add_argument("--validation-start", type=int, default=20000)
    parser.add_argument("--held-out-start", type=int)
    parser.add_argument(
        "--method", choices=["motion", "motion_gap", "normal_motion", "proximity"], default="motion"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.held_out_start is None:
        args.held_out_start = {"motion_gap": 110000, "normal_motion": 120000}.get(
            args.method, 40000
        )
    if args.method in {"motion_gap", "normal_motion"} and any(
        abs(args.held_out_start - start) < max(100, args.groups_per_split)
        for start in (
            (20000, 30000, 40000, 110000)
            if args.method == "normal_motion"
            else (20000, 30000, 40000)
        )
    ):
        parser.error("Use a fresh held-out seed range for the expanded gap calibration")
    if not 1 <= args.groups_per_split <= 10_000 or not 1 <= args.variants_per_group <= 10:
        parser.error("Use 1-10000 groups per split and 1-10 variants per group")
    if (
        min(args.validation_start, args.held_out_start) < 0
        or abs(args.validation_start - args.held_out_start) < args.groups_per_split
    ):
        parser.error("Validation and held-out seed ranges must be nonnegative and disjoint")
    if args.output.resolve() == args.config_output.resolve():
        parser.error("Report and config outputs must be distinct")
    tracker = TrackerConfig()
    validation = extract_cases(
        args.validation_start, args.groups_per_split, args.variants_per_group, tracker
    )
    validation_geometry = [
        prepare_contact_geometry(t.masks, t.tracked, r["object_valid"]) for r, t in validation
    ]
    if args.method in {"motion", "motion_gap", "normal_motion"}:
        candidates = []
        gaps = (0.0, 1.0, 2.0, 3.0) if args.method == "motion_gap" else (0.0,)
        if args.method == "normal_motion":
            gaps = (0.0, 1.0)
        windows = (3, 5, 7) if args.method == "normal_motion" else (3, 5)
        thresholds = (
            (0.04, 0.08, 0.12, 0.2, 0.3, 0.4)
            if args.method == "normal_motion"
            else (0.1, 0.2, 0.3, 0.4)
        )
        for window in windows:
            for threshold in thresholds:
                for candidate_gap in gaps:
                    options = {"motion_window": window, "motion_threshold": threshold}
                    if args.method == "normal_motion":
                        options["motion_rule"] = "normal"
                    candidates.append(
                        {
                            **options,
                            "max_gap_pixels": candidate_gap,
                            "metrics": summarize(
                                validation,
                                candidate_gap,
                                1,
                                geometries=validation_geometry,
                                **options,
                            ),
                        }
                    )
        selected = select_motion(candidates)
        gap = selected["max_gap_pixels"]
        options = {key: selected[key] for key in ("motion_window", "motion_threshold")}
        if args.method == "normal_motion":
            options["motion_rule"] = "normal"
        selection = "Maximum validation micro event F1; ties: smaller window, threshold, gap"
    else:
        candidates = [
            {
                "max_gap_pixels": gap,
                "metrics": summarize(validation, gap, 1, geometries=validation_geometry),
            }
            for gap in (0.0, 0.5, 1.0, 1.5)
        ]
        gap, options = select_gap(candidates), {}
        selection = "Maximum validation micro event F1; ties use the smaller gap"
    # The settings above are selected before constructing or inspecting the held-out cases.
    held_out = extract_cases(
        args.held_out_start, args.groups_per_split, args.variants_per_group, tracker
    )
    held_geometry = [
        prepare_contact_geometry(t.masks, t.tracked, r["object_valid"]) for r, t in held_out
    ]
    config = {
        "resolution_hw": [64, 64],
        "fps": 8,
        "tracker": asdict(tracker),
        "max_gap_pixels": gap,
        "event_tolerance_intervals": 1,
        **options,
    }
    report = {
        "purpose": "Extractor calibration on clean simulator RGB; not model evaluation",
        "extractor_version": TRACKER_VERSION,
        "contact_estimator": (
            "normal-motion-1"
            if options.get("motion_rule") == "normal"
            else "opposing-motion-2"
            if options
            else "pixel-proximity-1"
        ),
        "initialization": "Frame-0 identity masks only; future labels used only for scoring",
        "protocol": {
            "validation_seed_start": args.validation_start,
            "held_out_seed_start": args.held_out_start,
            "groups_per_split": args.groups_per_split,
            "variants_per_group": args.variants_per_group,
            "physics_seed_rule": "1000000 + visible_seed * 10 + variant",
            "object_count_rule": "2 + group_index % 3",
            "frames": 32,
            "fps": 8,
            "physics_hz": 240,
            "selection": selection,
            "reserved_scene_groups": "Exclude these seeds from model datasets",
        },
        "settings": config,
        "validation_candidates": candidates,
        "held_out": summarize(held_out, gap, 1, geometries=held_geometry, **options),
        "previous_motion_same_held_out": summarize(
            held_out, 0.0, 1, geometries=held_geometry, motion_threshold=0.3, motion_window=3
        ),
        "proximity_baseline_same_held_out": summarize(held_out, 0.0, 1, geometries=held_geometry),
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pymunk", "pillow")
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "limitations": [
            "Solid-colour simulator only; generated-video failures remain unmeasured",
            "Opposing motion is a contact proxy, not a reconstruction of physical forces",
            "Subpixel impulses and closely spaced repeated contacts can be missed",
            "Offline motion windows use future RGB context; timing can be shifted",
            "Missing pair timelines are excluded from event F1 and counted separately",
            "No uncertainty intervals; variants within a scene are correlated",
        ],
    }
    for path in (args.output, args.config_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.config_output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(
        json.dumps(
            {
                "settings": config,
                "held_out": report["held_out"],
                "baseline": report["proximity_baseline_same_held_out"]["collision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
