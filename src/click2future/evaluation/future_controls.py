"""Reproduce temporal/diversity controls with matched hidden-physics simulator futures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from click2future.sim import make_scene, run_rollout

from .futures import evaluate_futures
from .tracking import TrackerConfig


def run_controls() -> dict:
    records = []
    violations = []
    for seed in range(70000, 70004):
        references = [
            run_rollout(
                make_scene(seed, object_count=2 + seed % 3, physics_seed=seed * 10 + variant),
                frame_count=32,
            )
            for variant in range(8)
        ]
        original = references[0]["frames"]
        samples = np.stack([reference["frames"] for reference in references[:4]])
        duplicate = np.repeat(original[None], 4, axis=0)
        static = np.repeat(duplicate[:, :1], len(original), axis=1)
        flicker = duplicate.copy()
        for t in range(1, len(original)):
            flicker[:, t] = np.clip(
                duplicate[:, t].astype(int) + (8 if t % 2 else -8), 0, 255
            ).astype(np.uint8)
        groups = {
            "physics_variants": samples,
            "duplicate_future": duplicate,
            "frozen_first_frame": static,
            "colour_flicker": flicker,
        }
        summaries = {}
        for name, videos in groups.items():
            report = evaluate_futures(
                references,
                videos,
                tracker=TrackerConfig(),
                contact_settings={"max_gap_pixels": 0, "motion_threshold": 0.3, "motion_window": 3},
                source="simulator_control",
                run_id=f"{seed}-{name}",
            )
            ensemble = report["ensemble"]["all_objects"]
            temporal = report["temporal"]["per_sample"]
            summaries[name] = {
                "ensemble": ensemble,
                "temporal": temporal,
                "collision_frequency": report["collision_frequency"],
                "tracking_reasons_future": report["tracking_reasons_future"],
            }
            spread = ensemble["mean_pairwise_trajectory_distance"]
            if name in ("duplicate_future", "frozen_first_frame", "colour_flicker") and spread != 0:
                violations.append(f"{seed}: {name} changed trajectory spread")
            if name == "physics_variants" and (spread is None or spread <= 0):
                violations.append(f"{seed}: hidden-physics variants produced no measured spread")
            if name == "frozen_first_frame" and any(
                row["object_translation_mae_over_observed"] != 0 for row in temporal
            ):
                violations.append(f"{seed}: static object warp should be zero")
            if name == "colour_flicker" and any(
                row["background_mae_over_observed"] is None
                or row["background_mae_over_observed"] <= 0
                for row in temporal
            ):
                violations.append(f"{seed}: flicker was not detected")
        records.append(
            {
                "seed": seed,
                "scene_group_id": references[0]["scene_spec"]["scene_group_id"],
                "controls": summaries,
            }
        )
    return {
        "status": "passed" if not violations else "failed",
        "violations": violations,
        "purpose": "Metric controls on simulator RGB, not model results",
        "protocol": {
            "seed_start": 70000,
            "scene_groups": 4,
            "reference_futures": 8,
            "samples_per_control": 4,
            "frames": 32,
            "fps": 8,
            "resolution_hw": [64, 64],
            "physics_seed_rule": "visible_seed * 10 + variant",
            "object_count": "2 + seed % 3",
            "flicker": "alternate +8/-8 RGB intensity; preserve frame 0",
            "temporal_exclusion": "union of all declared reference contact windows",
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "future_controls.py",
                "futures.py",
                "temporal.py",
                "diversity.py",
                "sequences.py",
                "tracking.py",
                "contacts.py",
            )
        },
        "records": records,
        "limitations": [
            "Static videos can be temporally consistent",
            "Finite-reference coverage cannot prove physical plausibility",
            "Reference positions are oracle; sampled tracks have rasterization error",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_controls()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "violations": report["violations"],
                "output": str(args.output),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
