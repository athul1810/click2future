"""Matched pixel/geometry screen for an explicit physics baseline and pose model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.models.object_motion import observe, render_motion
from click2future.models.physics_baseline import (
    PHYSICS_BASELINE,
    PhysicsBaseline,
    rollout_observation,
)
from click2future.models.world_motion import sample_world
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.train_cvae import sha256

from .passive_stationarity import passive_horizon, stationary_reference
from .pilot import summarize_quality
from .rigid_geometry import geometry_diagnostics
from .suite import _write, prepare_plan
from .tracking import TrackerConfig
from .uncertainty_strata import action_family, zero_horizon
from .video import evaluate_videos

FRESH_ID_SEED = 910000
FRESH_COUNTS = {
    "train": 1,
    "validation": 1,
    "id_test": 10,
    "shape_ood": 1,
    "texture_ood": 1,
    "force_ood": 1,
    "physics_test": 1,
}


def screen(dataset, output, *, checkpoint=None, fresh_id=False, training_data=None):
    output, dataset = Path(output), Path(dataset).resolve()
    if output.exists():
        raise ValueError("Use a fresh screen folder; retain every previous attempt")
    project = Path(__file__).parents[3]
    pixel = project / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(pixel.read_text())
    if fresh_id:
        manifest = json.loads((dataset / "dataset_manifest.json").read_text())
        if (
            training_data is None
            or manifest["seed"] != FRESH_ID_SEED
            or manifest["group_counts"] != FRESH_COUNTS
        ):
            raise ValueError("D021 requires its registered fresh corpus and original training data")
    plan = prepare_plan(
        dataset,
        pixel,
        split="id_test" if fresh_id else "validation",
        metrics=("ssim",),
        training_corpus=training_data if fresh_id else None,
    )
    output.mkdir(parents=True)
    _write(output / "plan.json", plan)
    seeds = plan["sampling_seeds"]
    models = {"physics_assisted": None}
    if checkpoint:
        models["neural"], _ = load_trajectory(checkpoint)
    record = {
        "scope": "source-labelled solver baseline versus pose model; all conditions and seeds",
        "study_id": "D021" if fresh_id else "validation_screen",
        "split": plan["split"],
        "plan_sha256": plan["plan_sha256"],
        "physics_contract": PHYSICS_BASELINE,
        "physics_source_sha256": PhysicsBaseline().metadata["sampling_source_sha256"],
        "checkpoint_sha256": sha256(Path(checkpoint)) if checkpoint else None,
        "sampling_seeds": seeds,
        "models": {},
        "test_splits_evaluated": ["D021:id_test"] if fresh_id else [],
    }
    for name, model in models.items():
        folder = output / name
        folder.mkdir()
        reports, cases = [], []
        rest = {
            key: 0
            for key in ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
        }
        passive = dict(rest)
        events = {
            key: 0
            for key in (
                "true_positive",
                "false_positive",
                "false_negative",
                "complete_pair_timelines",
                "expected_pair_timelines",
            )
        }
        for case in plan["cases"]:
            reference = load_rollout(dataset, case["sample_id"])
            obs = observe(reference["frames"][0], reference["selected_mask"])
            if model is None:
                motion = np.stack(
                    [
                        rollout_observation(
                            obs, reference["force_xy"], sample_world(seed)[0].numpy()
                        )["motion"]
                        for seed in seeds
                    ]
                )
            else:
                with torch.inference_mode():
                    motion = (
                        model.sample(
                            torch.from_numpy(obs.features[None]),
                            torch.from_numpy(reference["force_xy"][None]),
                            seeds=seeds,
                            steps=1 if getattr(model, "world_prior_only", False) else 16,
                        )
                        .cpu()
                        .numpy()
                    )
            videos = np.stack([render_motion(obs, value) for value in motion])
            archive = folder / (case["sample_id"] + ".npz")
            np.savez_compressed(archive, videos=videos, motion=motion)
            report, _ = evaluate_videos(
                reference,
                videos,
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id=name,
            )
            _write(folder / (case["sample_id"] + ".json"), report)
            reports.append(report)
            geometry = geometry_diagnostics(obs, motion)
            row = {
                "sample_id": case["sample_id"],
                "scene_group_id": case["scene_group_id"],
                "archive_sha256": sha256(archive),
                "penetrating_clips": geometry["clips_with_penetration_over_one_pixel"],
                "penetrating_frames": geometry["penetration_over_one_pixel_frames"],
                "max_depth_pixels": max(
                    r["maximum_hull_penetration_pixels"] for r in geometry["clips"]
                ),
            }
            if action_family(case) == "zero":
                value = zero_horizon(reference, videos)
                for key in rest:
                    rest[key] += value[key]
            if action_family(case) in {"forward", "reverse"} and all(
                stationary_reference(load_rollout(dataset, identifier))
                for identifier in case["reference_ids"]
            ):
                value = passive_horizon(reference, videos)
                for key in passive:
                    passive[key] += value[key]
            for value in report["collision"]["per_sample"]:
                for key in events:
                    events[key] += value["event"][key]
            cases.append(row)
        denominator = (
            2 * events["true_positive"] + events["false_positive"] + events["false_negative"]
        )
        record["models"][name] = {
            "source": "explicit physics from inferred silhouettes and sampled prior"
            if model is None
            else "trained neural poses; unchanged sampler",
            "quality": summarize_quality(reports),
            "penetrating_clips": sum(r["penetrating_clips"] for r in cases),
            "penetrating_frames": sum(r["penetrating_frames"] for r in cases),
            "max_depth_pixels": max(r["max_depth_pixels"] for r in cases),
            "zero_force": rest,
            "passive": passive,
            "rgb_contact_against_first_reference": {
                **events,
                "f1": 2 * events["true_positive"] / denominator if denominator else None,
                "scope": (
                    "approximate RGB estimator against one realized hidden world; "
                    "not physical-validity proof"
                ),
            },
            "cases": cases,
        }
        _write(output / "screen.json", record)
        print(
            json.dumps(
                {
                    name: {
                        key: value
                        for key, value in record["models"][name].items()
                        if key != "cases"
                    }
                }
            ),
            flush=True,
        )
    record["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write(output / "screen.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fresh-id", action="store_true")
    parser.add_argument("--training-data", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    screen(
        args.dataset,
        args.output,
        checkpoint=args.checkpoint,
        fresh_id=args.fresh_id,
        training_data=args.training_data,
    )


if __name__ == "__main__":
    main()
