"""Diagnose saved validation collisions without changing the official scores."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.models.object_motion import observe, render_motion
from click2future.training.train_cvae import sha256

from .rigid_geometry import geometry_diagnostics
from .tracking import TrackerConfig
from .trajectory_audit import verify_inputs
from .uncertainty_strata import action_family
from .video import evaluate_videos


def event_details(predicted, reference, observed, *, tolerance):
    """Keep event indices and one-to-one offsets; incomplete timelines stay unscored.

    Indices identify zero-based intervals, not continuous collision times.
    A positive matched offset means the predicted interval occurs later.
    """
    predicted, reference, observed = map(np.asarray, (predicted, reference, observed))
    if (
        predicted.ndim != 1
        or not len(predicted)
        or any(
            x.shape != predicted.shape or x.dtype != np.bool_
            for x in (predicted, reference, observed)
        )
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, int)
        or tolerance < 0
    ):
        raise ValueError("Use matching nonempty boolean timelines and integer tolerance")
    p, q = np.flatnonzero(predicted).tolist(), np.flatnonzero(reference).tolist()
    result = {
        "predicted_intervals": p,
        "reference_intervals": q,
        "missing_intervals": np.flatnonzero(~observed).tolist(),
        "complete": bool(observed.all()),
        "matches": None,
        "unmatched_predicted": None,
        "unmatched_reference": None,
        "event": None,
    }
    if not result["complete"]:
        return result
    i = j = 0
    matches, extra, missed = [], [], []
    while i < len(p) and j < len(q):
        if p[i] < q[j] - tolerance:
            extra.append(p[i])
            i += 1
        elif q[j] < p[i] - tolerance:
            missed.append(q[j])
            j += 1
        else:
            matches.append({"predicted": p[i], "reference": q[j], "offset": p[i] - q[j]})
            i += 1
            j += 1
    extra.extend(p[i:])
    missed.extend(q[j:])
    result.update(
        matches=matches,
        unmatched_predicted=extra,
        unmatched_reference=missed,
        event={
            "true_positive": len(matches),
            "false_positive": len(extra),
            "false_negative": len(missed),
        },
    )
    return result


def event_summary(rows):
    complete = [r for r in rows if r["complete"]]
    counts = {
        k: sum(r["event"][k] for r in complete)
        for k in ("true_positive", "false_positive", "false_negative")
    }
    tp, fp, fn = (counts[k] for k in counts)
    return {
        "expected_timelines": len(rows),
        "complete_timelines": len(complete),
        "missing_timelines": len(rows) - len(complete),
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "scope": "Descriptive event counts; no independent-training or scene interval",
    }


def penetration_details(depths):
    depths = np.asarray(depths, dtype=float)
    if depths.ndim != 1 or not len(depths) or not np.isfinite(depths).all() or (depths < 0).any():
        raise ValueError("Provide nonnegative finite penetration depths")
    indices = np.flatnonzero(depths > 1)
    longest = length = 0
    for failed in depths > 1:
        length = length + 1 if failed else 0
        longest = max(longest, length)
    return {
        "maximum_depth_pixels": float(depths.max()),
        "frames_above_one_pixel": (indices + 1).tolist(),
        "longest_consecutive_frames_above_one_pixel": longest,
        "threshold_pixels": 1,
        "frame_zero_scored": False,
    }


def audit(plan_path, predictions_path, dataset, checkpoint, poses_dir, config_path):
    dataset, poses_dir, config_path = map(Path, (dataset, poses_dir, config_path))
    plan, run, paths, hashes = verify_inputs(plan_path, predictions_path, dataset, checkpoint)
    if sha256(config_path) != plan["input_sha256"]["config"]:
        raise ValueError("Collision settings must match the historical validation plan")
    for name in ("contacts.py", "metrics.py", "tracking.py", "video.py"):
        if sha256(Path(__file__).with_name(name)) != plan["evaluator_source_sha256"].get(name):
            raise ValueError("Collision metric source differs from the historical plan: " + name)
    hashes[config_path] = sha256(config_path)
    config = yaml.safe_load(config_path.read_text())
    settings = {
        "tracker": TrackerConfig(**config["tracker"]),
        "event_tolerance": config["event_tolerance_intervals"],
        **{
            key: config[key]
            for key in ("max_gap_pixels", "motion_window", "motion_threshold", "motion_rule")
            if key in config
        },
    }
    tolerance = settings["event_tolerance"]
    cases, reference_controls, reference_cross = [], [], []
    for case in plan["cases"]:
        refs = [load_rollout(dataset, name) for name in case["reference_ids"]]
        ref = refs[0]
        if not np.array_equal(ref["object_valid"], [True, True]):
            raise ValueError("This rigid-pose audit requires exactly two valid objects")
        observation = observe(ref["frames"][0], ref["selected_mask"])
        pose_path = poses_dir / (case["sample_id"] + "-poses.npz")
        hashes[pose_path] = sha256(pose_path)
        with np.load(pose_path, allow_pickle=False) as archive:
            motions = archive["motion"]
        with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
            videos = archive["videos"]
            frame_valid = archive["frame_valid"] if "frame_valid" in archive else None
        if motions.shape != (len(plan["sampling_seeds"]), len(ref["frames"]) - 1, 2, 4):
            raise ValueError("Poses must cover every frozen seed and future frame")
        if not np.array_equal(np.stack([render_motion(observation, m) for m in motions]), videos):
            raise ValueError("Saved poses must reproduce every original RGB pixel")
        report, arrays = evaluate_videos(
            ref, videos, run_id=run["method"], frame_valid=frame_valid, **settings
        )
        geometry = geometry_diagnostics(observation, motions)
        truth = [r["collision_onset"][:, 0, 1] for r in refs]
        control_rows = []
        for name, reference, labels in zip(case["reference_ids"], refs, truth, strict=True):
            if not np.array_equal(reference["frames"][0], observation.first):
                raise ValueError("Reference variants must share the initial image")
            _, extracted = evaluate_videos(
                reference, reference["frames"][None], run_id="rendered_reference", **settings
            )
            predicted = extracted["predicted_collision_onset"][0, :, 0, 1]
            known = extracted["collision_observed"][0, :, 0, 1]
            own = event_details(predicted, labels, known, tolerance=tolerance)
            cross = event_details(predicted, truth[0], known, tolerance=tolerance)
            control_rows.append(
                {"reference_id": name, "own_labels": own, "first_reference_labels": cross}
            )
            reference_controls.append(own)
            if name != case["sample_id"]:
                reference_cross.append(cross)
        samples = []
        for index, seed in enumerate(plan["sampling_seeds"]):
            comparisons = [
                {
                    "reference_id": name,
                    **event_details(
                        arrays["predicted_collision_onset"][index, :, 0, 1],
                        labels,
                        arrays["collision_observed"][index, :, 0, 1],
                        tolerance=tolerance,
                    ),
                }
                for name, labels in zip(case["reference_ids"], truth, strict=True)
            ]
            event = comparisons[0]["event"]
            if event is not None and any(
                event[key] != report["collision"]["per_sample"][index]["event"][key]
                for key in event
            ):
                raise ValueError("Detailed event matching must reproduce the official score")
            samples.append(
                {
                    "seed": seed,
                    "reference_comparisons": comparisons,
                    "penetration": penetration_details(
                        geometry["clips"][index]["hull_penetration_pixels_per_frame"]
                    ),
                }
            )
        cases.append(
            {
                **case,
                "action_family": action_family(case),
                "samples": samples,
                "rendered_reference_controls": control_rows,
            }
        )
        print(json.dumps({"case": case["sample_id"], "pixel_exact": True}), flush=True)
    groups = defaultdict(list)
    for case in cases:
        groups[case["action_family"]].extend(case["samples"])
    summaries = {}
    for family, samples in groups.items():
        penetrating = [
            s["penetration"] for s in samples if s["penetration"]["frames_above_one_pixel"]
        ]
        summaries[family] = {
            "samples": len(samples),
            "events_against_first_reference": event_summary(
                [s["reference_comparisons"][0] for s in samples]
            ),
            "penetrating_clips": len(penetrating),
            "penetration_frames": sum(len(p["frames_above_one_pixel"]) for p in penetrating),
            "maximum_depth_pixels": max(s["penetration"]["maximum_depth_pixels"] for s in samples),
            "longest_consecutive_penetration_frames": max(
                s["penetration"]["longest_consecutive_frames_above_one_pixel"] for s in samples
            ),
        }
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Audit input changed: " + path.name)
    return {
        "status": "measured_D018_collision_audit"
        if plan.get("study_id") == "D018"
        else "measured_validation_collision_audit",
        "method": run["method"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "historical_plan_sha256": plan["plan_sha256"],
        "event_tolerance_intervals": tolerance,
        "sampling_seeds": plan["sampling_seeds"],
        "model_events_against_first_reference": event_summary(
            [s["reference_comparisons"][0] for c in cases for s in c["samples"]]
        ),
        "rendered_reference_events_against_own_labels": event_summary(reference_controls),
        "other_reference_renders_against_first_labels": event_summary(reference_cross),
        "action_families": summaries,
        "cases": cases,
        "input_sha256": {str(p): digest for p, digest in hashes.items()},
        "source_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("contacts.py"),
                Path(__file__).with_name("video.py"),
                Path(__file__).with_name("metrics.py"),
                Path(__file__).with_name("rigid_geometry.py"),
                Path(__file__).with_name("tracking.py"),
            )
        },
        "interpretation": [
            "All declared cases, seeds, reference variants and missing timelines are retained.",
            "Own-render calibration measures the pixel estimator, not learned-model quality.",
            "Other-physics references illustrate uncertainty; they are not a performance bound.",
            "Per-reference matches retain every reference; the official score is unchanged.",
            "Pixel-centre hull penetration uses the existing approximate one-pixel tolerance.",
            "No thresholds, video bytes, model weights or original scores are changed.",
        ],
        "test_splits_evaluated": ["D018:id_test"] if plan.get("study_id") == "D018" else [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "poses-dir", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output path; preserve previous audits")
    result = audit(
        args.plan, args.predictions, args.dataset, args.checkpoint, args.poses_dir, args.config
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["action_families"], indent=2))


if __name__ == "__main__":
    main()
