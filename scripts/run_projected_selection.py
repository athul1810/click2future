"""D037: select four repair-aware futures from 16 projected D020 candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import action_family, zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.contact_projection import PROJECTION_CONTRACT, project_motion
from click2future.models.object_motion import observe, render_motion
from click2future.models.projected_selection import SELECTION_CONTRACT, select_projected
from click2future.models.world_motion import sample_world
from click2future.training.sample_trajectory import load_trajectory, sampling_geometry
from click2future.training.train_cvae import sha256


def _geometry(reports):
    clips = [clip for report in reports for clip in report["clips"]]
    return {
        "clips": len(clips),
        "penetrating_clips": sum(x["penetration_over_one_pixel_frames"] > 0 for x in clips),
        "penetrating_frames": sum(x["penetration_over_one_pixel_frames"] for x in clips),
        "boundary_clips": sum(x["boundary_over_one_pixel_object_frames"] > 0 for x in clips),
        "boundary_object_frames": sum(x["boundary_over_one_pixel_object_frames"] for x in clips),
    }


def _zero(rows):
    keys = ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
    return {key: sum(row[key] for row in rows) for key in keys}


def run(dataset, checkpoint, result_path, *, device="cuda", candidate_count=16):
    dataset, checkpoint, result_path = map(Path, (dataset, checkpoint, result_path))
    if result_path.exists():
        raise ValueError("Use a fresh result path")
    if candidate_count < 4:
        raise ValueError("D037 requires at least four candidates")

    root = Path(__file__).parents[1]
    settings_path = root / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    plan = prepare_plan(dataset, settings_path, split="id_test", metrics=("ssim",))
    model, state = load_trajectory(checkpoint, device)
    model_device = next(model.parameters()).device
    seeds = list(range(candidate_count))
    reports = {"projected_first_four": [], "projected_selected": []}
    geometries = {key: [] for key in reports}
    zero_rows = {key: [] for key in reports}
    cases = []

    for number, case in enumerate(plan["cases"], start=1):
        reference = load_rollout(dataset, case["sample_id"])
        observation = observe(reference["frames"][0], reference["selected_mask"])
        features = torch.from_numpy(observation.features[None]).to(model_device)
        force = torch.from_numpy(reference["force_xy"][None]).to(model_device)
        with torch.inference_mode():
            raw = (
                model.sample(
                    features,
                    force,
                    seeds=seeds,
                    steps=1 if getattr(model, "world_prior_only", False) else 16,
                    sigma_min=model.sigma_min,
                    sigma_max=model.sigma_max,
                    **sampling_geometry(model, observation, model_device),
                )
                .cpu()
                .numpy()
            )
        projected = np.stack(
            [
                project_motion(observation, motion, sample_world(seed)[0, :, 0].numpy())
                for seed, motion in zip(seeds, raw, strict=True)
            ]
        )
        selection = select_projected(
            observation, raw, projected, reference["force_xy"], output_count=4
        )
        indices = {
            "projected_first_four": list(range(4)),
            "projected_selected": selection["selected_indices"],
        }
        for name, chosen in indices.items():
            motion = projected[chosen]
            videos = np.stack([render_motion(observation, item) for item in motion])
            report, _ = evaluate_videos(
                reference,
                videos,
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id="D037-" + name,
            )
            reports[name].append(report)
            geometries[name].append(geometry_diagnostics(observation, motion))
            if action_family(case) == "zero":
                zero_rows[name].append(zero_horizon(reference, videos))
        cases.append(
            {
                "sample_id": case["sample_id"],
                "action_family": action_family(case),
                "selected_indices": selection["selected_indices"],
                "mean_projection_pixels": [
                    score["mean_projection_pixels"] for score in selection["scores"]
                ],
            }
        )
        print(f"{number}/{len(plan['cases'])} {case['sample_id']}", flush=True)

    result = {
        "study_id": "D037",
        "status": "local_exploratory",
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": state["step"],
        "checkpoint_source_revision": state["provenance"]["source_revision"],
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "plan_sha256": plan["plan_sha256"],
        "candidate_seeds": seeds,
        "returned_futures": 4,
        "control": "D028 projection of candidates 0-3",
        "method": "D028 projection of every candidate, then repair-aware physics selection",
        "projection_contract": PROJECTION_CONTRACT,
        "selection_contract": SELECTION_CONTRACT,
        "selection_excludes": "reference futures, hidden physics and evaluation metrics",
        "source_sha256": {
            "contact_projection.py": sha256(root / "src/click2future/models/contact_projection.py"),
            "projected_selection.py": sha256(
                root / "src/click2future/models/projected_selection.py"
            ),
            "run_projected_selection.py": sha256(Path(__file__)),
        },
        "runs": {
            name: {
                "quality": summarize_quality(reports[name]),
                "geometry": _geometry(geometries[name]),
                "zero_force": _zero(zero_rows[name]),
            }
            for name in reports
        },
        "cases": cases,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--candidate-count", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run(
        args.dataset,
        args.checkpoint,
        args.result,
        device=args.device,
        candidate_count=args.candidate_count,
    )
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
