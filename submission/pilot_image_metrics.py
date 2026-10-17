"""Supplement frozen validation videos with SSIM/LPIPS and a static-copy control."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.images import image_quality
from click2future.evaluation.perceptual import LPIPSScorer
from click2future.evaluation.suite import _write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(values):
    values = np.asarray(values)
    draws = np.random.default_rng(0).integers(0, len(values), (2000, len(values)))
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(values[draws].mean(1), [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new image-analysis directory")
    plan_path = args.evaluation / "plan.json"
    sample_path = args.evaluation / "validation-samples/sampling.json"
    plan, sampling = (json.loads(p.read_text()) for p in (plan_path, sample_path))
    if plan["split"] != "validation" or not plan["cases"]:
        raise ValueError("This follow-up is restricted to frozen validation cases")
    expected = {c["sample_id"] for c in plan["cases"]}
    for run in sampling["runs"]:
        if set(run["videos"]) != expected or set(run["video_sha256"]) != expected:
            raise ValueError("Saved video coverage differs from the validation plan")
    torch.set_num_threads(4)
    scorer = LPIPSScorer(device="cpu")
    args.output.mkdir()
    models = []
    for run in [*sampling["runs"], {"method": "copy-frame0", "run_id": "copy-frame0"}]:
        cases = []
        for case in plan["cases"]:
            name = case["sample_id"]
            reference_path = args.dataset / (name + ".npz")
            if sha(reference_path) != plan["input_sha256"][name + ".npz"]:
                raise ValueError("Reference arrays changed after plan creation")
            reference = load_rollout(args.dataset, name)
            if run["method"] == "copy-frame0":
                videos = np.broadcast_to(
                    reference["frames"][:1],
                    (len(plan["sampling_seeds"]), *reference["frames"].shape),
                ).copy()
                archive = args.output / (run["method"] + "-" + name + ".npz")
                np.savez_compressed(archive, videos=videos)
            else:
                archive = args.evaluation / "validation-samples" / run["videos"][name]
                if sha(archive) != run["video_sha256"][name]:
                    raise ValueError("Generated video changed after sampling")
                with np.load(archive, allow_pickle=False) as saved:
                    videos = saved["videos"]
            scores = image_quality(
                videos, reference["frames"], metrics=("ssim", "lpips"), scorer=scorer
            )
            cases.append(
                {
                    "sample_id": name,
                    "scene_group_id": case["scene_group_id"],
                    "video_sha256": sha(archive),
                    "reference_arrays_sha256": sha(reference_path),
                    "means": scores["mean_over_complete_samples"],
                }
            )
            _write(args.output / (run["method"] + "-" + name + ".json"), scores)
        groups = sorted({c["scene_group_id"] for c in cases})
        summary = {
            metric: estimate(
                [
                    np.mean([c["means"][metric] for c in cases if c["scene_group_id"] == g])
                    for g in groups
                ]
            )
            for metric in ("ssim", "lpips")
        }
        models.append(
            {
                "method": run["method"],
                "run_id": run["run_id"],
                "source": "control" if run["method"] == "copy-frame0" else "generated_video",
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "summary": summary,
                "cases": cases,
            }
        )
        print(json.dumps({"method": run["method"], "summary": summary}), flush=True)
    _write(
        args.output / "summary.json",
        {
            "status": "measured_supplementary_cpu_image_analysis",
            "scope": "Single-reference distances do not establish physical validity.",
            "plan_sha256": sha(plan_path),
            "sampling_manifest_sha256": sha(sample_path),
            "weights": scorer.metadata,
            "script_sha256": sha(__file__),
            "frame_zero_scored": False,
            "sampling_seeds": plan["sampling_seeds"],
            "scene_groups": len(groups),
            "cases_per_model": len(plan["cases"]),
            "aggregation": (
                "equal cases within scene, then equal scenes; 2000 scene bootstrap draws, seed 0"
            ),
            "models": models,
        },
    )


if __name__ == "__main__":
    main()
