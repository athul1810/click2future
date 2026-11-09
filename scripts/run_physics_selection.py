"""Compare fixed-seed sampling with reference-free physics-guided selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.suite import _write, prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import action_family, zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.object_motion import observe, render_motion
from click2future.models.physics_selection import SELECTION_CONTRACT, select_physics_guided
from click2future.training.sample_trajectory import load_trajectory, sampling_geometry
from click2future.training.train_cvae import sha256


def _geometry_summary(reports):
    clips = [clip for report in reports for clip in report["clips"]]
    return {
        "clips": len(clips),
        "penetrating_clips": sum(clip["penetration_over_one_pixel_frames"] > 0 for clip in clips),
        "penetrating_frames": sum(clip["penetration_over_one_pixel_frames"] for clip in clips),
        "boundary_clips": sum(clip["boundary_over_one_pixel_object_frames"] > 0 for clip in clips),
        "boundary_object_frames": sum(
            clip["boundary_over_one_pixel_object_frames"] for clip in clips
        ),
        "maximum_penetration_pixels": max(
            clip["maximum_hull_penetration_pixels"] for clip in clips
        ),
        "maximum_boundary_depth_pixels": max(
            clip["maximum_boundary_depth_pixels"] for clip in clips
        ),
    }


def _zero_summary(rows):
    keys = ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
    return {key: sum(row[key] for row in rows) for key in keys}


def run(dataset, checkpoint, output, *, device, candidate_count=16, output_count=4):
    dataset, checkpoint, output = map(Path, (dataset, checkpoint, output))
    if output.exists():
        raise ValueError("Use a fresh output folder")
    if candidate_count < output_count or output_count < 1:
        raise ValueError("candidate_count must be at least output_count")

    project = Path(__file__).parents[1]
    settings_path = project / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    plan = prepare_plan(dataset, settings_path, split="id_test", metrics=("ssim",))
    model, state = load_trajectory(checkpoint, device)
    seeds = list(range(candidate_count))
    output.mkdir(parents=True)
    _write(output / "plan.json", plan)

    video_reports = {"control": [], "selected": []}
    geometry_reports = {"control": [], "selected": []}
    zero_reports = {"control": [], "selected": []}
    cases = []
    for case_number, case in enumerate(plan["cases"], start=1):
        reference = load_rollout(dataset, case["sample_id"])
        observation = observe(reference["frames"][0], reference["selected_mask"])
        model_device = next(model.parameters()).device
        features = torch.from_numpy(observation.features[None]).to(model_device)
        force = torch.from_numpy(reference["force_xy"][None]).to(model_device)
        with torch.inference_mode():
            motion = (
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
        selection = select_physics_guided(
            observation, motion, reference["force_xy"], output_count=output_count
        )
        indices = {
            "control": list(range(output_count)),
            "selected": selection["selected_indices"],
        }
        videos = {
            name: np.stack([render_motion(observation, motion[index]) for index in chosen])
            for name, chosen in indices.items()
        }
        archive = output / f"{case['sample_id']}.npz"
        np.savez_compressed(
            archive,
            candidate_motion=motion,
            control_indices=np.asarray(indices["control"]),
            selected_indices=np.asarray(indices["selected"]),
            control_videos=videos["control"],
            selected_videos=videos["selected"],
        )
        for name in ("control", "selected"):
            report, _ = evaluate_videos(
                reference,
                videos[name],
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id=name,
            )
            video_reports[name].append(report)
            geometry_reports[name].append(geometry_diagnostics(observation, motion[indices[name]]))
            if action_family(case) == "zero":
                zero_reports[name].append(zero_horizon(reference, videos[name]))
        cases.append(
            {
                "sample_id": case["sample_id"],
                "scene_group_id": case["scene_group_id"],
                "action_family": action_family(case),
                "archive_sha256": sha256(archive),
                "control_indices": indices["control"],
                **selection,
            }
        )
        print(f"{case_number}/{len(plan['cases'])} {case['sample_id']}", flush=True)

    result = {
        "study": "reference-free physics-guided candidate selection",
        "selection_contract": SELECTION_CONTRACT,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": state["step"],
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "plan_sha256": plan["plan_sha256"],
        "candidate_seeds": seeds,
        "control_rule": f"first {output_count} candidates from the identical pool",
        "selection_inputs": "visible frame zero, selected mask, force vector, generated poses",
        "excluded_from_selection": "reference futures, hidden physics and evaluation metrics",
        "runs": {
            name: {
                "quality": summarize_quality(video_reports[name]),
                "geometry": _geometry_summary(geometry_reports[name]),
                "zero_force": _zero_summary(zero_reports[name]),
            }
            for name in ("control", "selected")
        },
        "cases": cases,
    }
    _write(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--output-count", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run(
        args.dataset,
        args.checkpoint,
        args.output,
        device=args.device,
        candidate_count=args.candidate_count,
        output_count=args.output_count,
    )
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
