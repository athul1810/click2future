"""D028: the neural D020 sampler with and without the position-based contact projection.

Selection (default) uses the original validation split only and scores two projection
modes. The fresh run (--fresh) uses the mode frozen in FROZEN_MODE on a new seed-940000
corpus and applies the acceptance rules in accept(), fixed before generation.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.evaluation.passive_stationarity import passive_horizon, stationary_reference
from click2future.evaluation.physics_screen import FRESH_COUNTS
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.contact_projection import PROJECTION_CONTRACT, project_motion
from click2future.models.object_motion import observe, render_motion
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.trajectory_data import canonical_ids

FRESH_SEED = 940000
# Set from the validation selection before the fresh corpus is generated.
FROZEN_MODE = "frame"
# Non-inferiority margins, fixed before the fresh run. The projection exists to remove
# penetration, so accuracy only has to stay within these normalized-unit margins.
SELECTED_ADE_MARGIN = 0.0005
SAME_WORLD_ADE_MARGIN = 0.0005


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def accept(p, h, interval):
    """D028 rules: remove penetration without new failures, within the fixed margins."""
    return bool(
        p["quality"]["complete_all_object_clips"] == 240
        and p["penetrating_clips"] == 0
        and p["boundary_clips"] <= h["boundary_clips"]
        and p["zero_force"]["drifting_samples"] == 0
        and p["passive"]["drifting_samples"] == 0
        and p["passive"]["missing_samples"] == 0
        and p["quality"]["selected_ade_mean_cases_if_all_complete"]
        <= h["quality"]["selected_ade_mean_cases_if_all_complete"] + SELECTED_ADE_MARGIN
        and p["contact_f1"] is not None
        and h["contact_f1"] is not None
        and p["contact_f1"] >= h["contact_f1"]
        and interval[1] < SAME_WORLD_ADE_MARGIN
    )


def compare(dataset, output, *, checkpoint, training_data=None, previous_data=(), fresh=False):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Retain previous comparisons and choose a fresh output")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(config.read_text())
    if fresh:
        manifest = json.loads((dataset / "dataset_manifest.json").read_text())
        if manifest["seed"] != FRESH_SEED or manifest["group_counts"] != FRESH_COUNTS:
            raise ValueError("D028 requires its registered fresh corpus")
        if FROZEN_MODE not in ("frame", "carry", "driven") or training_data is None:
            raise ValueError("Freeze the selected mode and bind the training corpus first")
        plan = prepare_plan(
            dataset, config, split="id_test", metrics=("ssim",), training_corpus=training_data
        )
        modes = {"neural": None, "projected": FROZEN_MODE}
    else:
        plan = prepare_plan(dataset, config, split="validation", metrics=("ssim",))
        modes = {
            "neural": None,
            "projected_frame": "frame",
            "projected_carry": "carry",
            "projected_driven": "driven",
        }
    model, _ = load_trajectory(checkpoint)
    model.eval()
    old_pixels = set()
    for folder in previous_data:
        previous = json.loads((Path(folder) / "dataset_manifest.json").read_text())
        for row in previous["samples"]:
            with np.load(Path(folder) / (row["sample_id"] + ".npz")) as arrays:
                old_pixels.add(hashlib.sha256(arrays["frames"][0].tobytes()).hexdigest())
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        if hashlib.sha256(ref["frames"][0].tobytes()).hexdigest() in old_pixels:
            raise ValueError("Fresh D028 image duplicates an opened corpus")
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    totals = {}
    for name in modes:
        (output / name).mkdir()
        totals[name] = dict(
            reports=[],
            cases=[],
            events=dict(tp=0, fp=0, fn=0),
            rest={
                key: 0
                for key in (
                    "total_samples",
                    "complete_samples",
                    "missing_samples",
                    "drifting_samples",
                )
            },
            passive={
                key: 0
                for key in (
                    "total_samples",
                    "complete_samples",
                    "missing_samples",
                    "drifting_samples",
                )
            },
        )
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        predictions = {name: [] for name in totals}
        errors = {name: [] for name in totals}
        other_errors = {name: [] for name in totals}
        seeds = plan["sampling_seeds"]
        with torch.inference_mode():
            raw = (
                model.sample(
                    torch.from_numpy(obs.features[None]),
                    torch.from_numpy(ref["force_xy"][None]),
                    seeds=seeds,
                    steps=1 if getattr(model, "world_prior_only", False) else 16,
                )
                .cpu()
                .numpy()
            )
        for k, seed in enumerate(seeds):
            world = sample_world(seed)[0].numpy()
            # Predict first. True scene geometry enters only the same-world reference.
            for name, mode in modes.items():
                motion = (
                    raw[k]
                    if mode is None
                    else project_motion(
                        obs,
                        raw[k],
                        world[:, 0],
                        carry=mode == "carry",
                        driven=obs.selected_index if mode == "driven" else None,
                    )
                )
                predictions[name].append((motion, render_motion(obs, motion)))
            scene = copy.deepcopy(ref["scene_spec"])
            objects = {o["object_id"]: o for o in scene["objects"]}
            for slot, identifier in enumerate(order):
                objects[identifier]["hidden_physics"] = dict(
                    zip(
                        ("mass", "friction", "restitution"),
                        world[slot].astype(float).tolist(),
                        strict=True,
                    )
                )
            target = run_rollout(
                scene,
                **{
                    key: ref["metadata"][key]
                    for key in ("frame_count", "fps", "physics_hz", "resolution")
                },
            )
            if not np.array_equal(target["frames"][0], ref["frames"][0]):
                raise ValueError("World intervention changed the visible condition")
            true_delta = target["positions"][1:, order] - target["positions"][:1, order]
            for name in totals:
                delta = predictions[name][-1][0][..., :2] * 0.5
                errors[name].append(
                    float(
                        np.linalg.norm(
                            delta[:, obs.selected_index] - true_delta[:, obs.selected_index],
                            axis=-1,
                        ).mean()
                    )
                )
                other = 1 - obs.selected_index
                other_errors[name].append(
                    float(np.linalg.norm(delta[:, other] - true_delta[:, other], axis=-1).mean())
                )
        for name, values in totals.items():
            motion = np.stack([v[0] for v in predictions[name]])
            videos = np.stack([v[1] for v in predictions[name]])
            np.savez_compressed(
                output / name / (case["sample_id"] + ".npz"), motion=motion, videos=videos
            )
            report, _ = evaluate_videos(
                ref,
                videos,
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id="D028-" + name,
            )
            values["reports"].append(report)
            write(output / name / (case["sample_id"] + ".json"), report)
            for sample in report["collision"]["per_sample"]:
                for key, field in (
                    ("tp", "true_positive"),
                    ("fp", "false_positive"),
                    ("fn", "false_negative"),
                ):
                    values["events"][key] += sample["event"][field]
            if "-zero-" in case["sample_id"]:
                value = zero_horizon(ref, videos)
                for key in values["rest"]:
                    values["rest"][key] += value[key]
                if not np.array_equal(videos, np.broadcast_to(obs.first, videos.shape)):
                    raise ValueError("Zero force changed observed pixels")
            if any(tag in case["sample_id"] for tag in ("-forward-", "-reverse-")) and all(
                stationary_reference(load_rollout(dataset, r)) for r in case["reference_ids"]
            ):
                value = passive_horizon(ref, videos)
                for key in values["passive"]:
                    values["passive"][key] += value[key]
            geo = geometry_diagnostics(obs, motion)
            values["cases"].append(
                dict(
                    sample_id=case["sample_id"],
                    scene_group_id=case["scene_group_id"],
                    matched_world_ade=errors[name],
                    other_object_matched_world_ade=other_errors[name],
                    penetrating_clips=geo["clips_with_penetration_over_one_pixel"],
                    boundary_clips=sum(
                        r["boundary_over_one_pixel_object_frames"] > 0 for r in geo["clips"]
                    ),
                )
            )
    models = {}
    for name, values in totals.items():
        events = values["events"]
        den = 2 * events["tp"] + events["fp"] + events["fn"]
        models[name] = dict(
            quality=summarize_quality(values["reports"]),
            matched_world_ade=float(np.mean([r["matched_world_ade"] for r in values["cases"]])),
            other_object_matched_world_ade=float(
                np.mean([r["other_object_matched_world_ade"] for r in values["cases"]])
            ),
            penetrating_clips=sum(r["penetrating_clips"] for r in values["cases"]),
            boundary_clips=sum(r["boundary_clips"] for r in values["cases"]),
            zero_force=values["rest"],
            passive=values["passive"],
            contact_counts=events,
            contact_f1=2 * events["tp"] / den if den else None,
            cases=values["cases"],
        )
    groups = sorted({r["scene_group_id"] for r in models["neural"]["cases"]})
    result = dict(
        study_id="D028",
        split=plan["split"],
        plan_sha256=plan["plan_sha256"],
        projection_contract=PROJECTION_CONTRACT,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        models=models,
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    for name in models:
        if name == "neural":
            continue
        differences = []
        for group in groups:
            means = {
                key: np.mean(
                    [
                        r["matched_world_ade"]
                        for r in models[key]["cases"]
                        if r["scene_group_id"] == group
                    ]
                )
                for key in ("neural", name)
            }
            differences.append(float(means[name] - means["neural"]))
        draws = (
            np.random.default_rng(20261108)
            .choice(differences, (10000, len(groups)), replace=True)
            .mean(1)
        )
        interval = np.quantile(draws, [0.025, 0.975]).tolist()
        result[name + "_paired_scene_differences"] = differences
        result[name + "_paired_scene_bootstrap_95_interval"] = interval
        if fresh:
            result["frozen_mode"] = FROZEN_MODE
            result["accept_projection"] = accept(models[name], models["neural"], interval)
    write(output / "results.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "models"}
            | {
                name: {k: v for k, v in model.items() if k not in ("cases",)}
                for name, model in models.items()
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--previous-data", type=Path, nargs="*", default=())
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    compare(
        args.dataset,
        args.output,
        checkpoint=args.checkpoint,
        training_data=args.training_data,
        previous_data=args.previous_data,
        fresh=args.fresh,
    )
