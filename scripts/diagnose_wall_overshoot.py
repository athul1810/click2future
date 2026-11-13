"""Locate boundary excursions and lost tracks in saved D034 trajectories.

Re-score saved poses with the original geometry and tracker, grouping failures
by action family. This diagnostic does not change the comparison's decisions."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import action_family
from click2future.evaluation.video import evaluate_videos
from click2future.models.object_motion import observe, render_motion


def family_group(case):
    family = action_family(case)
    if family in ("forward", "reverse"):
        return "push"
    return "collision" if family.startswith("collision") else family


def summarize(clips):
    """Boundary and lost-track summary for per-clip rows of one arm."""
    boundary = [c for c in clips if c["boundary_frames"] > 0]
    depths = [c["boundary_depth"] for c in boundary]
    return {
        "clips": len(clips),
        "complete_clips": sum(c["complete"] for c in clips),
        "penetrating_clips": sum(c["penetration_frames"] > 0 for c in clips),
        "clips_by_family": dict(sorted(Counter(c["family"] for c in clips).items())),
        "boundary_clips": len(boundary),
        "boundary_clips_by_family": dict(sorted(Counter(c["family"] for c in boundary).items())),
        "boundary_object_frames": sum(c["boundary_frames"] for c in boundary),
        "boundary_depth_pixels": {
            "mean": statistics.mean(depths) if depths else None,
            "median": statistics.median(depths) if depths else None,
            "maximum": max(depths) if depths else None,
            "clips_over_2": sum(d > 2 for d in depths),
            "clips_over_4": sum(d > 4 for d in depths),
        },
        "lost_track_clips": [
            {
                key: c[key]
                for key in (
                    "sample_id",
                    "family",
                    "sample",
                    "boundary_depth",
                    "boundary_frames",
                    "penetration_depth",
                    "overlap_fraction",
                    "outside_fraction",
                    "reasons",
                )
            }
            for c in clips
            if not c["complete"]
        ],
    }


def score_arm(dataset, run, arm, plan, settings):
    clips = []
    for case in plan["cases"]:
        reference = load_rollout(dataset, case["sample_id"])
        observation = observe(reference["frames"][0], reference["selected_mask"])
        motion = np.load(run / arm / (case["sample_id"] + ".npz"))["motion"]
        geometry = geometry_diagnostics(observation, motion)
        videos = np.stack([render_motion(observation, value) for value in motion])
        report, arrays = evaluate_videos(
            reference,
            videos,
            tracker=TrackerConfig(**settings["tracker"]),
            max_gap_pixels=settings["max_gap_pixels"],
            event_tolerance=settings["event_tolerance_intervals"],
            motion_threshold=settings["motion_threshold"],
            motion_window=settings["motion_window"],
            motion_rule=settings["motion_rule"],
            run_id="D034-diagnosis",
        )
        valid = np.asarray(arrays["object_valid"]).astype(bool)
        for index, clip in enumerate(geometry["clips"]):
            clips.append(
                {
                    "sample_id": case["sample_id"],
                    "family": family_group(case),
                    "sample": index,
                    "complete": bool(arrays["tracked"][index][:, valid].all()),
                    "boundary_frames": clip["boundary_over_one_pixel_object_frames"],
                    "boundary_depth": clip["maximum_boundary_depth_pixels"],
                    "penetration_frames": clip["penetration_over_one_pixel_frames"],
                    "penetration_depth": clip["maximum_hull_penetration_pixels"],
                    "overlap_fraction": clip["maximum_overlap_fraction"],
                    "outside_fraction": clip["maximum_outside_pixel_fraction"],
                    "reasons": report["tracking_reasons_future"],
                }
            )
    return clips


def diagnose(dataset, runs, arms):
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "configs/evaluation/pixel.yaml").read_text())
    result = {}
    for label, run in runs.items():
        plan = json.loads((run / "plan.json").read_text())
        result[label] = {
            arm: summarize(score_arm(dataset, run, arm, plan, settings)) for arm in arms
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIRECTORY")
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    named = {label: Path(path) for label, path in (item.split("=", 1) for item in args.run)}
    args.output.write_text(
        json.dumps(diagnose(args.dataset, named, args.arm), indent=2, allow_nan=False) + "\n"
    )
