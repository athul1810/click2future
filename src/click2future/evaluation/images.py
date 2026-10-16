"""Frame-aligned SSIM/LPIPS for future RGB frames, with explicit missing-frame counts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from click2future.data import load_rollout

from .perceptual import LPIPSScorer

SSIM_SETTINGS = {
    "implementation": "skimage.metrics.structural_similarity",
    "data_range": 255,
    "channel_axis": -1,
    "gaussian_weights": True,
    "sigma": 1.5,
    "win_size": 11,
    "use_sample_covariance": False,
    "K1": 0.01,
    "K2": 0.03,
    "channels": "RGB, arithmetic mean of channel SSIM",
    "resize": "none",
    "crop": "five-pixel SSIM window border excluded by skimage",
}


def _inputs(videos, reference, frame_valid):
    videos, reference = np.asarray(videos), np.asarray(reference)
    if (
        reference.dtype != np.uint8
        or reference.ndim != 4
        or reference.shape[-1] != 3
        or len(reference) < 2
        or min(reference.shape[1:3]) < 11
    ):
        raise ValueError("reference must be uint8 [T,H,W,3] with T >= 2 and H,W >= 11")
    if (
        videos.dtype != np.uint8
        or videos.ndim != 5
        or len(videos) < 1
        or videos.shape[1:] != reference.shape
    ):
        raise ValueError("videos must be uint8 [K,T,H,W,3] matching reference with K >= 1")
    if frame_valid is None:
        valid = np.ones(videos.shape[:2], bool)
    else:
        valid = np.asarray(frame_valid)
        if valid.dtype != np.bool_ or valid.shape != videos.shape[:2]:
            raise ValueError("frame_valid must be boolean [K,T]")
    return videos, reference, valid


def image_quality(
    videos,
    reference,
    *,
    metrics=("ssim", "lpips"),
    frame_valid=None,
    scorer: LPIPSScorer | None = None,
    batch_size: int = 32,
) -> dict:
    """Score full future frames; means require a complete sample, with no best-frame choice.

    One reference future is shared across K samples. Valid alternate futures may score
    poorly. Tracker failures do not mask image scores; frame_valid means missing RGB.
    """
    videos, reference, valid = _inputs(videos, reference, frame_valid)
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics) <= {"ssim", "lpips"}:
        raise ValueError("Request ssim, lpips, or both, without duplicates")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if "lpips" in metrics and min(reference.shape[1:3]) < 64:
        raise ValueError("LPIPS requires H,W >= 64; images are never resized automatically")
    settings = {}
    if "ssim" in metrics:
        try:
            from skimage.metrics import structural_similarity
        except ImportError as error:
            raise RuntimeError("Install the image-metrics extra to use SSIM") from error
        settings["ssim"] = {
            **SSIM_SETTINGS,
            "package_version": importlib.metadata.version("scikit-image"),
        }
    if "lpips" in metrics:
        scorer = scorer if scorer is not None else LPIPSScorer()
        settings["lpips"] = scorer.metadata
    samples, frames = videos.shape[:2]
    values = {name: np.full((samples, frames - 1), np.nan) for name in metrics}
    pairs = np.argwhere(valid[:, 1:])
    if "ssim" in metrics:
        for k, future in pairs:
            score = structural_similarity(
                reference[future + 1],
                videos[k, future + 1],
                **{
                    key: SSIM_SETTINGS[key]
                    for key in (
                        "data_range",
                        "channel_axis",
                        "gaussian_weights",
                        "sigma",
                        "win_size",
                        "use_sample_covariance",
                        "K1",
                        "K2",
                    )
                },
            )
            if not np.isfinite(score):
                raise ValueError("SSIM returned an invalid score")
            values["ssim"][k, future] = score
    if "lpips" in metrics:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            scores = scorer.score(videos[batch[:, 0], batch[:, 1] + 1], reference[batch[:, 1] + 1])
            values["lpips"][batch[:, 0], batch[:, 1]] = scores
    complete = valid[:, 1:].all(axis=1)
    records = []
    for k in range(samples):
        records.append(
            {
                "sample_index": k,
                "complete": bool(complete[k]),
                "scored_future_frames": int(valid[k, 1:].sum()),
                "expected_future_frames": frames - 1,
                "mean": {
                    name: float(scores[k].mean()) if complete[k] else None
                    for name, scores in values.items()
                },
                "per_future_frame": {
                    name: [float(x) if np.isfinite(x) else None for x in scores[k]]
                    for name, scores in values.items()
                },
            }
        )
    return {
        "image_metric_version": "1",
        "metrics_requested": list(metrics),
        "comparison": "frame_aligned_single_reference",
        "frame_zero_scored": False,
        "color_space": "stored RGB; no gamma conversion",
        "resolution_hw": list(reference.shape[1:3]),
        "preprocessing": settings,
        "batch_size": batch_size,
        "direction": {"ssim": "higher_is_better", "lpips": "lower_is_better"},
        "missing_policy": "complete_sample_means; all missing frame counts retained",
        "total_samples": samples,
        "complete_samples": int(complete.sum()),
        "expected_future_frames": samples * (frames - 1),
        "scored_future_frames": int(valid[:, 1:].sum()),
        "missing_future_frames": int((~valid[:, 1:]).sum()),
        "mean_over_complete_samples": {
            name: float(scores[complete].mean(axis=1).mean()) if complete.any() else None
            for name, scores in values.items()
        },
        "per_sample": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--videos", type=Path, required=True, help="NPZ videos and optional frame_valid"
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=["ssim", "lpips"], default=["ssim", "lpips"]
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source", choices=["generated_video", "simulator_control"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "videos": args.videos,
        "reference_arrays": args.reference_dir / (args.sample_id + ".npz"),
        "reference_metadata": args.reference_dir / (args.sample_id + ".json"),
    }
    if args.output.resolve() in {p.resolve() for p in paths.values()}:
        parser.error("Output must not overwrite an input")
    if args.download_weights and "lpips" not in args.metrics:
        parser.error("--download-weights requires LPIPS")
    try:
        reference = load_rollout(args.reference_dir, args.sample_id)
        with np.load(args.videos, allow_pickle=False) as archive:
            videos = archive["videos"]
            frame_valid = archive["frame_valid"] if "frame_valid" in archive else None
        _inputs(videos, reference["frames"], frame_valid)
        if not np.all(videos[:, 0] == reference["frames"][0]) or (
            frame_valid is not None and not frame_valid[:, 0].all()
        ):
            raise ValueError("Every sample must preserve its valid first frame")
        scorer = (
            LPIPSScorer(device=args.device, allow_download=args.download_weights)
            if "lpips" in args.metrics
            else None
        )
        report = {
            "status": "passed",
            "source": args.source,
            "run_id": args.run_id,
            "sample_id": args.sample_id,
            "scene_group_id": reference["scene_spec"]["scene_group_id"],
            "condition_id": reference["scene_spec"]["condition_id"],
            "timestamps_seconds": reference["timestamps"].tolist(),
            "input_sha256": {
                name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in paths.items()
            },
            "image_quality": image_quality(
                videos,
                reference["frames"],
                metrics=args.metrics,
                frame_valid=frame_valid,
                scorer=scorer,
                batch_size=args.batch_size,
            ),
        }
    except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile, RuntimeError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
