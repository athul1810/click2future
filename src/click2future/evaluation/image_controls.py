"""Reproduce image-quality control checks with seeded simulator RGB clips."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from click2future.sim import make_scene, run_rollout

from .images import image_quality
from .perceptual import LPIPSScorer


def control_videos(reference: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Keep frame 0 exact; distort all future frames deterministically."""
    exact = reference[None].copy()
    shifted = exact.copy()
    shifted[:, 1:] = np.roll(shifted[:, 1:], 4, axis=3)
    noisy = exact.copy()
    noise = np.random.default_rng(seed).normal(0, 12, reference[1:].shape)
    noisy[0, 1:] = np.clip(reference[1:].astype(float) + noise, 0, 255).astype(np.uint8)
    static = np.repeat(exact[:, :1], len(reference), axis=1)
    return {
        "exact": exact,
        "shift_four_pixels": shifted,
        "noise_std12": noisy,
        "frozen_first_frame": static,
    }


def run_controls(*, scorer: LPIPSScorer | None, metrics=("ssim", "lpips")) -> dict:
    records = []
    for seed in range(60000, 60004):
        reference = run_rollout(make_scene(seed, object_count=2 + seed % 3), frame_count=16)
        results = {
            name: image_quality(videos, reference["frames"], metrics=metrics, scorer=scorer)
            for name, videos in control_videos(reference["frames"], seed).items()
        }
        records.append(
            {
                "scene_group_id": reference["scene_spec"]["scene_group_id"],
                "seed": seed,
                "scores": results,
            }
        )
    violations = []
    for record in records:
        scores = {
            name: result["mean_over_complete_samples"] for name, result in record["scores"].items()
        }
        if "ssim" in metrics:
            if abs(scores["exact"]["ssim"] - 1) > 1e-10:
                violations.append(f"{record['seed']}: exact SSIM differs from one")
            for kind in scores.keys() - {"exact"}:
                if scores[kind]["ssim"] >= scores["exact"]["ssim"]:
                    violations.append(f"{record['seed']}: {kind} did not reduce SSIM")
        if "lpips" in metrics:
            if abs(scores["exact"]["lpips"]) > 1e-7:
                violations.append(f"{record['seed']}: exact LPIPS differs from zero")
            for kind in scores.keys() - {"exact"}:
                if scores[kind]["lpips"] <= scores["exact"]["lpips"]:
                    violations.append(f"{record['seed']}: {kind} did not increase LPIPS")
    return {
        "status": "passed" if not violations else "failed",
        "violations": violations,
        "purpose": "Image-metric controls on simulator RGB, not model evaluation",
        "source": "simulator_control",
        "scene_groups": 4,
        "protocol": {
            "seed_start": 60000,
            "frames": 16,
            "fps": 8,
            "resolution_hw": [64, 64],
            "object_count": "2 + seed % 3",
            "frame_zero_scored": False,
            "shift": "roll entire future frame four pixels right with wraparound",
            "noise": "Gaussian std 12 in uint8 intensity units; clip then cast uint8",
            "frozen": "repeat exact initial frame",
            "resize": "none",
        },
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "pymunk", "pillow")
        },
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("images.py", "perceptual.py", "image_controls.py")
        },
        "preprocessing": records[0]["scores"]["exact"]["preprocessing"],
        "records": [
            {
                "scene_group_id": row["scene_group_id"],
                "seed": row["seed"],
                "scores": {
                    name: {
                        key: result[key]
                        for key in (
                            "total_samples",
                            "complete_samples",
                            "expected_future_frames",
                            "scored_future_frames",
                            "missing_future_frames",
                            "mean_over_complete_samples",
                        )
                    }
                    for name, result in row["scores"].items()
                },
            }
            for row in records
        ],
        "limitations": [
            "Checks metric response to distortions, not physical validity",
            "Single-reference scores penalize some valid alternative futures",
            "Full-frame scores may be dominated by the tabletop background",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", nargs="+", choices=["ssim", "lpips"], default=["ssim", "lpips"]
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.download_weights and "lpips" not in args.metrics:
        parser.error("--download-weights requires LPIPS")
    try:
        scorer = (
            LPIPSScorer(allow_download=args.download_weights) if "lpips" in args.metrics else None
        )
        report = run_controls(scorer=scorer, metrics=tuple(args.metrics))
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
