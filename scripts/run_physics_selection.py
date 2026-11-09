"""Compare fixed-seed sampling with reference-free physics-guided selection."""

import argparse
import itertools
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


def _uniform_four(reports, geometries, zero_rows, output_count):
    case_ades = []
    pass_probabilities = []
    complete = total = 0
    expected_observations = tracked_observations = 0
    for report in reports:
        all_objects = report["trajectory"]["all_objects"]
        expected_observations += all_objects["expected_future_observations"]
        tracked_observations += all_objects["tracked_future_observations"]
        records = report["trajectory"]["selected_object"]["per_sample"]
        ades = [row["ade"] for row in records]
        known = [value for value in ades if value is not None]
        complete += len(known)
        total += len(ades)
        case_ades.append(float(np.mean(known)) if known else None)
        subsets = list(itertools.combinations(range(len(ades)), output_count))
        passed = sum(
            all(ades[index] is not None for index in subset)
            and np.mean([ades[index] for index in subset]) <= 0.03125
            for subset in subsets
        )
        pass_probabilities.append(passed / len(subsets))

    pool_geometry = _geometry_summary(geometries)
    pool_zero = _zero_summary(zero_rows)
    scale = output_count / (total / len(reports))
    return {
        "quality": {
            "case_count": len(reports),
            "expected_selected_clips": len(reports) * output_count,
            "expected_complete_selected_clips": complete * scale,
            "expected_missing_selected_clips": (total - complete) * scale,
            "tracking_fraction": tracked_observations / expected_observations,
            "selected_ade_mean_cases_over_complete_candidates": float(
                np.mean([value for value in case_ades if value is not None])
            ),
            "mean_per_case_gate_pass_probability": float(np.mean(pass_probabilities)),
            "all_cases_gate_pass_probability": float(np.prod(pass_probabilities)),
            "scope": "Exact expectation over uniform four-of-K subsets; no random draw.",
        },
        "geometry": {
            key: value * scale
            for key, value in pool_geometry.items()
            if key
            in {
                "clips",
                "penetrating_clips",
                "penetrating_frames",
                "boundary_clips",
                "boundary_object_frames",
            }
        },
        "zero_force": {key: value * scale for key, value in pool_zero.items()},
    }


def _evaluate(reference, videos, settings, run_id):
    report, _ = evaluate_videos(
        reference,
        videos,
        tracker=TrackerConfig(**settings["tracker"]),
        max_gap_pixels=settings["max_gap_pixels"],
        event_tolerance=settings["event_tolerance_intervals"],
        motion_threshold=settings["motion_threshold"],
        motion_window=settings["motion_window"],
        motion_rule=settings["motion_rule"],
        run_id=run_id,
    )
    return report


def run(
    dataset,
    output,
    *,
    checkpoint=None,
    candidate_source=None,
    device="cpu",
    candidate_count=16,
    output_count=4,
):
    dataset, output = map(Path, (dataset, output))
    if output.exists():
        raise ValueError("Use a fresh output folder")
    if candidate_count < output_count or output_count < 1:
        raise ValueError("candidate_count must be at least output_count")
    if (checkpoint is None) == (candidate_source is None):
        raise ValueError("Provide either a checkpoint or saved candidate source")

    project = Path(__file__).parents[1]
    settings_path = project / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    plan = prepare_plan(dataset, settings_path, split="id_test", metrics=("ssim",))
    if candidate_source is None:
        checkpoint = Path(checkpoint)
        model, state = load_trajectory(checkpoint, device)
        seeds = list(range(candidate_count))
        checkpoint_sha256, checkpoint_step = sha256(checkpoint), state["step"]
    else:
        candidate_source = Path(candidate_source)
        previous = json.loads((candidate_source / "results.json").read_text())
        seeds = previous["candidate_seeds"]
        if (
            previous["dataset_manifest_sha256"] != sha256(dataset / "dataset_manifest.json")
            or previous["plan_sha256"] != plan["plan_sha256"]
            or len(seeds) != candidate_count
        ):
            raise ValueError("Saved candidates do not match this dataset, plan or pool size")
        checkpoint_sha256 = previous["checkpoint_sha256"]
        checkpoint_step = previous["checkpoint_step"]
    output.mkdir(parents=True)
    _write(output / "plan.json", plan)

    arms = ("first_four", "selected", "oracle_four")
    video_reports = {name: [] for name in arms}
    geometry_reports = {name: [] for name in arms}
    zero_reports = {name: [] for name in arms}
    pool_reports, pool_geometry, pool_zero = [], [], []
    cases = []
    for case_number, case in enumerate(plan["cases"], start=1):
        reference = load_rollout(dataset, case["sample_id"])
        observation = observe(reference["frames"][0], reference["selected_mask"])
        if candidate_source is None:
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
            archive = output / f"{case['sample_id']}.npz"
            np.savez_compressed(archive, candidate_motion=motion)
        else:
            archive = candidate_source / f"{case['sample_id']}.npz"
            with np.load(archive, allow_pickle=False) as arrays:
                motion = arrays["candidate_motion"]
        selection = select_physics_guided(
            observation, motion, reference["force_xy"], output_count=output_count
        )
        all_videos = np.stack([render_motion(observation, item) for item in motion])
        pool_report = _evaluate(reference, all_videos, settings, "candidate_pool")
        ades = pool_report["trajectory"]["selected_object"]["per_sample"]
        complete = [row for row in ades if row["ade"] is not None]
        if len(complete) < output_count:
            raise ValueError("Oracle diagnostic requires at least four complete candidates")
        oracle = [row["sample_index"] for row in sorted(complete, key=lambda row: row["ade"])][
            :output_count
        ]
        indices = {
            "first_four": list(range(output_count)),
            "selected": selection["selected_indices"],
            "oracle_four": oracle,
        }
        pool_reports.append(pool_report)
        pool_geometry.append(geometry_diagnostics(observation, motion))
        if action_family(case) == "zero":
            pool_zero.append(zero_horizon(reference, all_videos))
        for name in arms:
            videos = all_videos[indices[name]]
            report = _evaluate(reference, videos, settings, name)
            video_reports[name].append(report)
            geometry_reports[name].append(geometry_diagnostics(observation, motion[indices[name]]))
            if action_family(case) == "zero":
                zero_reports[name].append(zero_horizon(reference, videos))
        cases.append(
            {
                "sample_id": case["sample_id"],
                "scene_group_id": case["scene_group_id"],
                "action_family": action_family(case),
                "archive_sha256": sha256(archive),
                "first_four_indices": indices["first_four"],
                "oracle_indices": indices["oracle_four"],
                **selection,
            }
        )
        print(f"{case_number}/{len(plan['cases'])} {case['sample_id']}", flush=True)

    result = {
        "study": "reference-free physics-guided candidate selection",
        "selection_contract": SELECTION_CONTRACT,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "plan_sha256": plan["plan_sha256"],
        "candidate_seeds": seeds,
        "primary_control": f"exact uniform {output_count}-of-{candidate_count} expectation",
        "operational_baseline": f"first {output_count} candidates from the identical pool",
        "oracle_scope": "reference-dependent diagnostic only; unavailable at inference",
        "selection_inputs": "visible frame zero, selected mask, force vector, generated poses",
        "selection_does_not_use": "reference futures, hidden physics, ADE or tracker output",
        "shared_with_physical_audit": "observed-hull penetration and boundary definitions",
        "runs": {
            name: {
                "quality": summarize_quality(video_reports[name]),
                "geometry": _geometry_summary(geometry_reports[name]),
                "zero_force": _zero_summary(zero_reports[name]),
            }
            for name in arms
        },
        "cases": cases,
    }
    result["runs"]["uniform_four"] = _uniform_four(
        pool_reports, pool_geometry, pool_zero, output_count
    )
    _write(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--reuse-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--output-count", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = run(
        args.dataset,
        args.output,
        checkpoint=args.checkpoint,
        candidate_source=args.reuse_candidates,
        device=args.device,
        candidate_count=args.candidate_count,
        output_count=args.output_count,
    )
    print(json.dumps(result["runs"], indent=2))


if __name__ == "__main__":
    main()
