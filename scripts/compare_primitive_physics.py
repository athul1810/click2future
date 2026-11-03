"""D023 frozen fresh-ID comparison of observed hull and circle/hull physics."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
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
from click2future.models.object_motion import observe, render_motion
from click2future.models.physics_baseline import PhysicsBaseline, rollout_observation
from click2future.models.primitive_physics import PrimitivePhysicsBaseline, primitive_prediction
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare(dataset, training_data, previous_data, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Retain previous comparisons and choose a fresh output")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    if manifest["seed"] != 920000 or manifest["group_counts"] != FRESH_COUNTS:
        raise ValueError("D023 requires its registered fresh corpus")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(config.read_text())
    plan = prepare_plan(
        dataset, config, split="id_test", metrics=("ssim",), training_corpus=training_data
    )
    # Also reject visible frames from the already opened D021 corpus, not only training.
    previous = json.loads((Path(previous_data) / "dataset_manifest.json").read_text())
    old_pixels = set()
    for row in previous["samples"]:
        with np.load(Path(previous_data) / (row["sample_id"] + ".npz")) as arrays:
            old_pixels.add(hashlib.sha256(arrays["frames"][0].tobytes()).hexdigest())
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        if hashlib.sha256(ref["frames"][0].tobytes()).hexdigest() in old_pixels:
            raise ValueError("Fresh D023 image duplicates the opened D021 corpus")
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    totals = {}
    for name in ("hull", "primitive"):
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
        for seed in plan["sampling_seeds"]:
            world = sample_world(seed)[0].numpy()
            # Predict first, using the observable-only APIs. True scene geometry
            # enters only the independent, same-world evaluation reference below.
            a = rollout_observation(obs, ref["force_xy"], world)
            b = primitive_prediction(obs, ref["force_xy"], world)
            predictions["hull"].append((a["motion"], render_motion(obs, a["motion"])))
            predictions["primitive"].append((b["motion"], b["videos"]))
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
                run_id="D023-" + name,
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
            penetrating_clips=sum(r["penetrating_clips"] for r in values["cases"]),
            boundary_clips=sum(r["boundary_clips"] for r in values["cases"]),
            zero_force=values["rest"],
            passive=values["passive"],
            contact_counts=events,
            contact_f1=2 * events["tp"] / den if den else None,
            cases=values["cases"],
        )
    groups = sorted({r["scene_group_id"] for r in models["hull"]["cases"]})
    differences = []
    for group in groups:
        means = {
            name: np.mean(
                [r["matched_world_ade"] for r in model["cases"] if r["scene_group_id"] == group]
            )
            for name, model in models.items()
        }
        differences.append(float(means["primitive"] - means["hull"]))
    draws = (
        np.random.default_rng(20261104)
        .choice(differences, (10000, len(groups)), replace=True)
        .mean(1)
    )
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    p, h = models["primitive"], models["hull"]
    accepted = bool(
        p["quality"]["complete_all_object_clips"] == 240
        and p["penetrating_clips"] == 0
        and p["boundary_clips"] <= h["boundary_clips"]
        and p["zero_force"]["drifting_samples"] == 0
        and p["passive"]["drifting_samples"] == 0
        and p["passive"]["missing_samples"] == 0
        and p["quality"]["selected_ade_mean_cases_if_all_complete"]
        <= h["quality"]["selected_ade_mean_cases_if_all_complete"]
        and p["contact_f1"] is not None
        and h["contact_f1"] is not None
        and p["contact_f1"] >= h["contact_f1"]
        and interval[1] < 0
    )
    result = dict(
        study_id="D023",
        split="fresh_id_test",
        plan_sha256=plan["plan_sha256"],
        models=models,
        paired_scene_group_ade_differences=differences,
        paired_scene_bootstrap_95_interval=interval,
        accept_primitive_as_demo_baseline=accepted,
        sources={
            "hull": PhysicsBaseline().metadata,
            "primitive": PrimitivePhysicsBaseline().metadata,
        },
        scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation=(
            "Ten fresh ID groups and four fixed prior worlds. No neural success, OOD or "
            "uncertainty-calibration claim. Contact F1 remains a limited RGB proxy. "
            "Historical gates and results remain unchanged."
        ),
    )
    write(output / "results.json", result)
    print(
        json.dumps(
            {
                "accepted": accepted,
                "paired_scene_bootstrap_95_interval": interval,
                **{
                    name: {k: v for k, v in model.items() if k != "cases"}
                    for name, model in models.items()
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "training-data", "previous-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.dataset, args.training_data, args.previous_data, args.output)
