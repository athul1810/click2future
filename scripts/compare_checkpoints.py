"""D031-D038: pose checkpoints, physics corrections and simulator controls, scored
with the unchanged scorers plus the multi-reference energy score.

Modes (all rules below were fixed before any of these runs):
- ``validation`` (D032): original validation split. ``select_scale`` picks at most one
  geometry-loss scale per training family. No test split is read.
- ``d031``: the opened seed-940000 D028 corpus. Descriptive energy scores for the D020
  sampler, its projection and the controls; it also reproduces the D028 numbers.
- ``fresh`` (D033): a new seed-950000 corpus, generated after this file was committed.
  ``projection_holds`` and ``learned_geometry_holds`` are the acceptance rules.
- ``d035``: a new seed-960000 corpus, generated after ``combined_holds`` was committed.
  Tests whether the scaled contact-dynamics losses plus the projection hold together.
- ``d037``: a new seed-970000 corpus, generated after ``dynamics_residual_holds`` was
  committed. Tests a Newtonian free-motion training residual against its matched control.
- ``d038``: a new seed-980000 corpus, generated only after this rule was committed.
  Tests the residual-trained model with momentum correction and contact projection
  against its matched projected control.

Every checkpoint is scored raw and with the D028 projection in its frozen per-frame
mode. Projection weights use the sampled masses when the model is conditioned on the
sampled world, and equal masses otherwise. Controls, all with the plan's sampling seeds:
- simulator_sampled_world: true geometry and prior-sampled hidden physics. Its samples
  come from the same distribution as the references, so in expectation no generator
  should beat its energy score (``metric_check``).
- oracle_first_reference: four copies of the first reference (true hidden physics).
  It has zero single-reference error and no spread.
- physics_baseline_0.5px and physics_baseline_0.15px: the explicit hull baseline.

All sampling runs on CPU, as in D028, so results do not depend on the GPU.
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
from click2future.evaluation.energy_score import (
    ENERGY_SCORE_CONTRACT,
    energy_score,
    paired_scene_interval,
)
from click2future.evaluation.passive_stationarity import passive_horizon, stationary_reference
from click2future.evaluation.physics_screen import FRESH_COUNTS
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import action_family, zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.contact_projection import PROJECTION_CONTRACT, project_motion
from click2future.models.momentum_block import MOMENTUM_BLOCK_CONTRACT, apply_momentum_block
from click2future.models.object_motion import encode_motion, observe, render_motion
from click2future.models.physics_baseline import CALIBRATED_BEVEL_PIXELS, rollout_observation
from click2future.models.rest_anchored_prior import RestAnchoredPriorModel
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.contact_geometry import dynamics_residual, observed_hulls
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.trajectory_data import canonical_ids, replay_angles

FRESH_SEED = 950000
D031_SEED = 940000
D035_SEED = 960000
D037_SEED = 970000
D038_SEED = 980000
# D035 compares these two arms, per training seed: control, then scaled losses.
COMBINED = ("direct_x0", "direct_dyn_x1024")
DYNAMICS_RESIDUAL = ("control", "residual")
FULL_PIPELINE = ("control+projection", "residual+momentum+projection")
PROJECTION_MODE = "frame"
SELECTED_ADE_MARGIN = 0.0005
SAME_WORLD_ADE_MARGIN = 0.0005
# The same 0.0005 normalized margin, in native pixels (64 px per unit).
ENERGY_MARGIN_PIXELS = 0.032
# Training families of D032: control first, then candidates in increasing scale.
FAMILIES = {
    "rest_anchored": ("rest_x1", ("rest_x16", "rest_x128", "rest_x1024")),
    "direct_dynamics": ("direct_x0", ("direct_dyn_x1", "direct_dyn_x128", "direct_dyn_x1024")),
}
CONTROLS = (
    "simulator_sampled_world",
    "oracle_first_reference",
    "physics_baseline_0.5px",
    "physics_baseline_0.15px",
)
TIMING = ("frame_count", "fps", "physics_hz", "resolution")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family_group(case):
    family = action_family(case)
    if family in ("forward", "reverse"):
        return "push"
    return "collision" if family.startswith("collision") else family


def _no_new_drift(c, h):
    return (
        c["zero_force"]["drifting_samples"] <= h["zero_force"]["drifting_samples"]
        and c["passive"]["drifting_samples"] <= h["passive"]["drifting_samples"]
        and c["zero_force"]["missing_samples"] == 0
        and c["passive"]["missing_samples"] == 0
    )


def _selected_ade_within(c, h):
    """Selected ADE rises by at most the margin. An undefined ADE (incomplete tracks)
    on either side cannot show that, so it counts as not satisfied."""
    first = c["quality"]["selected_ade_mean_cases_if_all_complete"]
    second = h["quality"]["selected_ade_mean_cases_if_all_complete"]
    return first is not None and second is not None and first <= second + SELECTED_ADE_MARGIN


def eligible(c, h):
    """D032: a candidate must cut penetration without new failures, within the margins."""
    return bool(
        c["quality"]["complete_all_object_clips"] == h["quality"]["complete_all_object_clips"]
        and _no_new_drift(c, h)
        and _selected_ade_within(c, h)
        and c["matched_world_ade"] <= h["matched_world_ade"] + SAME_WORLD_ADE_MARGIN
        and c["energy"]["selected"] <= h["energy"]["selected"] + ENERGY_MARGIN_PIXELS
        and c["penetrating_clips"] < h["penetrating_clips"]
    )


def select_scale(models, control, candidates):
    """Fewest penetrating clips, then fewest boundary clips, then the smaller scale."""
    passing = [name for name in candidates if eligible(models[name], models[control])]
    if not passing:
        return None
    return min(
        passing,
        key=lambda n: (
            models[n]["penetrating_clips"],
            models[n]["boundary_clips"],
            candidates.index(n),
        ),
    )


def projection_holds(p, h):
    """D033 rule for every checkpoint: the projection removes violations at no real cost."""
    return bool(
        p["quality"]["complete_all_object_clips"] == h["quality"]["complete_all_object_clips"]
        and p["penetrating_clips"] == 0
        and p["boundary_clips"] == 0
        and _no_new_drift(p, h)
        and _selected_ade_within(p, h)
        and p["energy"]["selected"] <= h["energy"]["selected"] + ENERGY_MARGIN_PIXELS
    )


def learned_geometry_holds(c, h, same_world_interval):
    """D033 rule for a D032-selected scale against its matched control, both raw."""
    return bool(
        c["quality"]["complete_all_object_clips"] == h["quality"]["complete_all_object_clips"]
        and c["penetrating_clips"] < h["penetrating_clips"]
        and c["boundary_clips"] <= h["boundary_clips"]
        and _no_new_drift(c, h)
        and _selected_ade_within(c, h)
        and same_world_interval[1] < SAME_WORLD_ADE_MARGIN
        and c["energy"]["selected"] <= h["energy"]["selected"] + ENERGY_MARGIN_PIXELS
    )


def combined_holds(raw, raw_control, projected, projected_control, same_world_interval):
    """D035 rule for one training seed, against the same-seed control.

    The scaled losses must keep their raw penetration gain. With the projection on both
    arms, they must have no violations, no lost tracks and no new drift, within the same
    margins as ``learned_geometry_holds``.
    """
    return bool(
        raw["penetrating_clips"] < raw_control["penetrating_clips"]
        and projected["quality"]["complete_all_object_clips"]
        == projected_control["quality"]["complete_all_object_clips"]
        and projected["penetrating_clips"] == 0
        and projected["boundary_clips"] == 0
        and _no_new_drift(projected, projected_control)
        and _selected_ade_within(projected, projected_control)
        and same_world_interval[1] < SAME_WORLD_ADE_MARGIN
        and projected["energy"]["selected"]
        <= projected_control["energy"]["selected"] + ENERGY_MARGIN_PIXELS
    )


def dynamics_residual_holds(candidate, control, interval):
    """D037 rule: improve free-motion dynamics without degrading existing safeguards."""
    return bool(
        interval[1] < 0
        and candidate["dynamics_residual"] < control["dynamics_residual"]
        and candidate["quality"]["complete_all_object_clips"]
        == control["quality"]["complete_all_object_clips"]
        and candidate["penetrating_clips"] <= control["penetrating_clips"]
        and candidate["boundary_clips"] <= control["boundary_clips"]
        and _no_new_drift(candidate, control)
        and _selected_ade_within(candidate, control)
        and candidate["matched_world_ade"]
        <= control["matched_world_ade"] + SAME_WORLD_ADE_MARGIN
        and candidate["energy"]["selected"]
        <= control["energy"]["selected"] + ENERGY_MARGIN_PIXELS
    )


def full_pipeline_checks(candidate, control, same_world_interval, residual_interval):
    return {
        "tracking_not_worse": candidate["quality"]["complete_all_object_clips"]
        >= control["quality"]["complete_all_object_clips"],
        "zero_geometry_violations": candidate["penetrating_clips"] == 0
        and candidate["boundary_clips"] == 0,
        "no_new_drift": _no_new_drift(candidate, control),
        "pose_ade_noninferior": candidate["pose_selected_ade"]
        <= control["pose_selected_ade"] + SELECTED_ADE_MARGIN,
        "same_world_ade_improves": same_world_interval[1] < 0,
        "dynamics_residual_improves": residual_interval[1] < 0,
        "momentum_conserved": candidate["post_force_momentum_error"] <= 1e-6,
        "energy_noninferior": candidate["energy"]["selected"]
        <= control["energy"]["selected"] + ENERGY_MARGIN_PIXELS,
    }


def metric_check(models, intervals):
    """D031 checks of the score itself.

    (a) The zero-spread oracle of the first reference must score worse than the
        prior-matched simulator, although its single-reference ADE is far lower.
    (b) No generator may be significantly better than that simulator: its paired
        scene interval (generator minus simulator) must not lie entirely below zero.
    A plain "lowest mean" rule was dropped after the first validation look: the hull
    baselines sit within noise of the simulator on four scenes (docs/ENERGY_SCORE.md).
    """
    control = models["simulator_sampled_world"]["energy"]["selected"]
    oracle_worse = models["oracle_first_reference"]["energy"]["selected"] > control
    better = sorted(
        name
        for name in models
        if name != "simulator_sampled_world"
        and intervals[name + "_minus_simulator_energy_selected"][1] < 0
    )
    return {
        "oracle_worse_than_simulator": bool(oracle_worse),
        "significantly_better_than_simulator": better,
        "passes": bool(oracle_worse and not better),
    }


def _scene_differences(models, first, second, key):
    groups = sorted({r["scene_group_id"] for r in models[first]["cases"]})
    values = []
    for group in groups:
        means = [
            np.mean([r[key] for r in models[name]["cases"] if r["scene_group_id"] == group])
            for name in (first, second)
        ]
        values.append(float(means[1] - means[0]))
    return values


def _with_world(scene, order, world):
    scene = copy.deepcopy(scene)
    objects = {o["object_id"]: o for o in scene["objects"]}
    for slot, identifier in enumerate(order):
        objects[identifier]["hidden_physics"] = dict(
            zip(
                ("mass", "friction", "restitution"), world[slot].astype(float).tolist(), strict=True
            )
        )
    return scene


def _arm():
    counts = ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
    return dict(
        reports=[],
        cases=[],
        dynamics_residual=[],
        momentum_error=[],
        events=dict(tp=0, fp=0, fn=0),
        rest=dict.fromkeys(counts, 0),
        passive=dict.fromkeys(counts, 0),
        zero_pixels=dict(samples=0, changed=0),
    )


def matched_world_motion(observation, targets):
    motions = []
    for target in targets:
        order = canonical_ids(observation, target["selected_object_id"])
        motions.append(
            encode_motion(target["positions"][:, order], replay_angles(target)[:, order])
        )
    return np.stack(motions)


def free_motion_residual(observation, motion, target_motion, worlds, force):
    """Score generated acceleration using matched-world contact and wall masks."""
    geometry = {
        key: torch.from_numpy(np.repeat(value[None], len(motion), axis=0))
        for key, value in observed_hulls(observation).items()
    }
    features = torch.from_numpy(np.repeat(observation.features[None], len(motion), axis=0))
    forces = torch.from_numpy(np.repeat(np.asarray(force, np.float32)[None], len(motion), axis=0))
    masses = torch.from_numpy(np.stack([world[:, 0] for world in worlds]))
    return dynamics_residual(
        torch.from_numpy(motion),
        torch.from_numpy(target_motion),
        features,
        forces,
        masses,
        geometry,
        contact_window_pixels=2.0,
    ).numpy()


def momentum_error(motion, masses, force, *, duration=0.25, fps=8):
    positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * motion[..., :2]))
    velocity = np.diff(positions, axis=0) * fps
    start = int(np.ceil(duration * fps))
    total = (masses[:, None] * velocity[start:]).sum(1)
    return float(np.linalg.norm(total - np.asarray(force) * duration, axis=-1).mean())


def zero_pixel_check(models):
    """Arms whose zero-force output is exactly the first frame by construction.

    The rest-anchored model subtracts its own zero-force response, and the hull
    baseline starts from rest, so any changed pixel there is a bug. Direct and E10
    models have no such guarantee; their zero-force motion is measured by the
    unchanged zero_force drift counts instead. Recorded, not raised, so a run is
    never lost; a violation must be explained before any result is used.
    """
    exact = sorted(
        name
        for name, m in models.items()
        if m["zero_force_pixels"] is not None
        and (m["rest_anchored"] or name.startswith("physics_baseline"))
    )
    violations = [name for name in exact if models[name]["zero_force_pixels"]["changed"]]
    return {"expected_exact": exact, "violations": violations, "passes": not violations}


def compare(dataset, output, *, mode, checkpoints, training_data=None, previous_data=()):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Retain previous comparisons and choose a fresh output")
    if mode not in ("validation", "d031", "fresh", "d035", "d037", "d038"):
        raise ValueError("Unknown mode")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(config.read_text())
    if mode == "validation":
        plan = prepare_plan(dataset, config, split="validation", metrics=("ssim",))
    else:
        manifest = json.loads((dataset / "dataset_manifest.json").read_text())
        expected = {
            "fresh": FRESH_SEED,
            "d031": D031_SEED,
            "d035": D035_SEED,
            "d037": D037_SEED,
            "d038": D038_SEED,
        }[mode]
        if manifest["seed"] != expected or manifest["group_counts"] != FRESH_COUNTS:
            raise ValueError(f"{mode} requires its registered seed-{expected} corpus")
        if training_data is None:
            raise ValueError("Bind the original training corpus")
        plan = prepare_plan(
            dataset, config, split="id_test", metrics=("ssim",), training_corpus=training_data
        )
    if mode in ("fresh", "d035", "d037", "d038"):
        old = set()
        for folder in previous_data:
            previous = json.loads((Path(folder) / "dataset_manifest.json").read_text())
            for row in previous["samples"]:
                with np.load(Path(folder) / (row["sample_id"] + ".npz")) as arrays:
                    old.add(hashlib.sha256(arrays["frames"][0].tobytes()).hexdigest())
        for case in plan["cases"]:
            frame = load_rollout(dataset, case["sample_id"])["frames"][0]
            if hashlib.sha256(frame.tobytes()).hexdigest() in old:
                raise ValueError("Fresh image duplicates an opened corpus")
    models = {}
    for name, path in checkpoints.items():
        model, _ = load_trajectory(path)
        models[name] = model.eval()
    arms = {}
    for name, model in models.items():
        conditioned = bool(getattr(model, "world_condition", False))
        anchored = isinstance(model, RestAnchoredPriorModel)
        variants = [("", False, False), ("+projection", True, False)]
        if mode == "d038":
            variants += [("+momentum", False, True), ("+momentum+projection", True, True)]
        for suffix, projected, momentum in variants:
            arms[name + suffix] = dict(
                checkpoint=name,
                projected=projected,
                momentum=momentum,
                world_conditioned=conditioned,
                rest_anchored=anchored,
            )
    for name in CONTROLS:
        arms[name] = dict(
            checkpoint=None,
            projected=False,
            momentum=False,
            world_conditioned=name != "oracle_first_reference",
            rest_anchored=False,
        )
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    totals = {}
    for name in arms:
        (output / name).mkdir()
        totals[name] = _arm()
    seeds = plan["sampling_seeds"]
    worlds = [sample_world(seed)[0].numpy() for seed in seeds]
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        sel, other = obs.selected_index, 1 - obs.selected_index
        meta = {key: ref["metadata"][key] for key in TIMING}
        scale = float(ref["metadata"]["resolution"])
        references = [load_rollout(dataset, r) for r in case["reference_ids"]]
        reference_paths = np.stack(
            [(r["positions"][1:, order] - r["positions"][:1, order]) * scale for r in references]
        )
        # Same-world targets: true geometry with each sampled world (also the control).
        targets = []
        for world in worlds:
            target = run_rollout(_with_world(ref["scene_spec"], order, world), **meta)
            if not np.array_equal(target["frames"][0], ref["frames"][0]):
                raise ValueError("World intervention changed the visible condition")
            targets.append(target)
        same_world = np.stack(
            [(t["positions"][1:, order] - t["positions"][:1, order]) * scale for t in targets]
        )
        target_motion = matched_world_motion(obs, targets)
        family = action_family(case)
        outputs = {}
        for name, model in models.items():
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
            # Masses weight the projection only for models that saw the sampled world.
            masses = [
                world[:, 0] if arms[name]["world_conditioned"] else np.ones(2) for world in worlds
            ]
            outputs[name] = dict(motion=raw)
            outputs[name + "+projection"] = dict(
                motion=np.stack(
                    [project_motion(obs, raw[k], masses[k]) for k in range(len(worlds))]
                )
            )
            if mode == "d038":
                corrected = np.stack(
                    [
                        apply_momentum_block(obs, raw[k], masses[k], ref["force_xy"])
                        for k in range(len(worlds))
                    ]
                )
                outputs[name + "+momentum"] = dict(motion=corrected)
                outputs[name + "+momentum+projection"] = dict(
                    motion=np.stack(
                        [project_motion(obs, corrected[k], masses[k]) for k in range(len(worlds))]
                    )
                )
        for bevel, label in (
            (0.5, "physics_baseline_0.5px"),
            (CALIBRATED_BEVEL_PIXELS, "physics_baseline_0.15px"),
        ):
            outputs[label] = dict(
                motion=np.stack(
                    [
                        rollout_observation(obs, ref["force_xy"], w, bevel_pixels=bevel)["motion"]
                        for w in worlds
                    ]
                )
            )
        outputs["simulator_sampled_world"] = dict(
            videos=np.stack([t["frames"] for t in targets]), paths=same_world
        )
        outputs["oracle_first_reference"] = dict(
            videos=np.stack([ref["frames"]] * len(seeds)),
            paths=np.stack([reference_paths[0]] * len(seeds)),
        )
        for name, values in totals.items():
            item = outputs[name]
            motion = item.get("motion")
            if motion is not None:
                videos = np.stack([render_motion(obs, value) for value in motion])
                paths = motion[..., :2] * 0.5 * scale
                residual = free_motion_residual(obs, motion, target_motion, worlds, ref["force_xy"])
                values["dynamics_residual"].extend(residual.tolist())
                if family != "wall":
                    values["momentum_error"].extend(
                        momentum_error(motion[k], worlds[k][:, 0], ref["force_xy"])
                        for k in range(len(worlds))
                    )
            else:
                videos, paths = item["videos"], item["paths"]
            np.savez_compressed(
                output / name / (case["sample_id"] + ".npz"),
                displacement_pixels=paths.astype(np.float32),
                **({} if motion is None else {"motion": motion}),
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
                run_id="D033-" + name,
            )
            values["reports"].append(report)
            for sample in report["collision"]["per_sample"]:
                for key, field in (
                    ("tp", "true_positive"),
                    ("fp", "false_positive"),
                    ("fn", "false_negative"),
                ):
                    values["events"][key] += sample["event"][field]
            if family == "zero":
                value = zero_horizon(ref, videos)
                for key in values["rest"]:
                    values["rest"][key] += value[key]
                if motion is not None:
                    values["zero_pixels"]["samples"] += len(videos)
                    values["zero_pixels"]["changed"] += sum(
                        not np.array_equal(video, np.broadcast_to(obs.first, video.shape))
                        for video in videos
                    )
            if family in ("forward", "reverse") and all(
                stationary_reference(r) for r in references
            ):
                value = passive_horizon(ref, videos)
                for key in values["passive"]:
                    values["passive"][key] += value[key]
            row = dict(sample_id=case["sample_id"], scene_group_id=case["scene_group_id"])
            row["family"] = family_group(case)
            if motion is not None:
                geo = geometry_diagnostics(obs, motion)
                row["penetrating_clips"] = geo["clips_with_penetration_over_one_pixel"]
                row["boundary_clips"] = sum(
                    r["boundary_over_one_pixel_object_frames"] > 0 for r in geo["clips"]
                )
                row["boundary_clips_over_4_pixels"] = sum(
                    r["maximum_boundary_depth_pixels"] > 4 for r in geo["clips"]
                )
                row["dynamics_residual"] = float(residual.mean())
            # Same-world error, normalized units as in D028 (pixels / resolution).
            errors = np.linalg.norm(paths - same_world, axis=-1).mean(1) / scale
            row["matched_world_ade"] = float(errors[:, sel].mean())
            row["other_object_matched_world_ade"] = float(errors[:, other].mean())
            first_errors = np.linalg.norm(paths - reference_paths[0], axis=-1).mean(1) / scale
            row["pose_selected_ade"] = float(first_errors[:, sel].mean())
            selected = energy_score(paths[:, :, sel], reference_paths[:, :, sel])
            both = energy_score(paths, reference_paths)
            row["energy_selected"] = selected["score"]
            row["energy_all_objects"] = both["score"]
            row["energy_selected_terms"] = selected
            values["cases"].append(row)
    summary = {}
    for name, values in totals.items():
        events, cases = values["events"], values["cases"]
        den = 2 * events["tp"] + events["fp"] + events["fn"]
        has_geometry = "penetrating_clips" in cases[0]
        families = sorted({r["family"] for r in cases})
        summary[name] = dict(
            **arms[name],
            quality=summarize_quality(values["reports"]),
            matched_world_ade=float(np.mean([r["matched_world_ade"] for r in cases])),
            other_object_matched_world_ade=float(
                np.mean([r["other_object_matched_world_ade"] for r in cases])
            ),
            dynamics_residual=(
                float(np.mean(values["dynamics_residual"])) if has_geometry else None
            ),
            post_force_momentum_error=(
                float(np.mean(values["momentum_error"])) if has_geometry else None
            ),
            pose_selected_ade=float(np.mean([r["pose_selected_ade"] for r in cases])),
            penetrating_clips=sum(r["penetrating_clips"] for r in cases) if has_geometry else None,
            boundary_clips=sum(r["boundary_clips"] for r in cases) if has_geometry else None,
            boundary_clips_over_4_pixels=(
                sum(r["boundary_clips_over_4_pixels"] for r in cases) if has_geometry else None
            ),
            zero_force=values["rest"],
            zero_force_pixels=values["zero_pixels"] if has_geometry else None,
            passive=values["passive"],
            contact_counts=events,
            contact_f1=2 * events["tp"] / den if den else None,
            energy=dict(
                selected=float(np.mean([r["energy_selected"] for r in cases])),
                all_objects=float(np.mean([r["energy_all_objects"] for r in cases])),
                selected_by_family={
                    f: float(np.mean([r["energy_selected"] for r in cases if r["family"] == f]))
                    for f in families
                },
                selected_sample_spread=float(
                    np.mean([r["energy_selected_terms"]["sample_spread"] for r in cases])
                ),
                selected_reference_spread=float(
                    np.mean(
                        [
                            r["energy_selected_terms"]["reference_spread"]
                            for r in cases
                            if r["energy_selected_terms"]["reference_spread"] is not None
                        ]
                    )
                ),
            ),
            cases=cases,
        )
    result = dict(
        study_id={
            "validation": "D032",
            "d031": "D031",
            "fresh": "D033",
            "d035": "D035",
            "d037": "D037",
            "d038": "D038",
        }[mode],
        mode=mode,
        split=plan["split"],
        plan_sha256=plan["plan_sha256"],
        dataset_manifest_sha256=sha256(dataset / "dataset_manifest.json"),
        projection_contract=PROJECTION_CONTRACT,
        projection_mode=PROJECTION_MODE,
        momentum_contract=MOMENTUM_BLOCK_CONTRACT,
        energy_score_contract=ENERGY_SCORE_CONTRACT,
        checkpoint_sha256={name: sha256(path) for name, path in checkpoints.items()},
        scorer_sha256=sha256(__file__),
        sampling_seeds=seeds,
        models=summary,
    )
    intervals = {}
    for name in summary:
        if name != "simulator_sampled_world":
            values = _scene_differences(summary, "simulator_sampled_world", name, "energy_selected")
            intervals[name + "_minus_simulator_energy_selected"] = paired_scene_interval(values)
    for name in models:
        for key in ("matched_world_ade", "energy_selected"):
            values = _scene_differences(summary, name, name + "+projection", key)
            intervals[f"{name}+projection_minus_raw_{key}"] = paired_scene_interval(values)
    result["paired_scene_bootstrap_95_intervals"] = intervals
    result["metric_check"] = metric_check(summary, intervals)
    result["zero_pixel_check"] = zero_pixel_check(summary)
    if mode == "validation":
        # A candidate whose training failed is simply not eligible.
        result["selection"] = {
            family: select_scale(summary, control, [c for c in candidates if c in summary])
            for family, (control, candidates) in FAMILIES.items()
            if control in summary
        }
        result["candidates_not_scored"] = sorted(
            c for _, candidates in FAMILIES.values() for c in candidates if c not in summary
        )
    if mode in ("fresh", "d035", "d037", "d038"):
        result["projection_holds"] = {
            name: projection_holds(summary[name + "+projection"], summary[name]) for name in models
        }
    if mode == "d035":
        control, candidate = COMBINED
        values = _scene_differences(
            summary, control + "+projection", candidate + "+projection", "matched_world_ade"
        )
        interval = paired_scene_interval(values)
        result["combined"] = dict(
            control=control,
            candidate=candidate,
            projected_same_world_interval=interval,
            holds=combined_holds(
                summary[candidate],
                summary[control],
                summary[candidate + "+projection"],
                summary[control + "+projection"],
                interval,
            ),
        )
    if mode == "d037":
        control, candidate = DYNAMICS_RESIDUAL
        if not {control, candidate} <= models.keys():
            raise ValueError("D037 requires control and residual checkpoints")
        comparisons = {}
        for suffix in ("", "+projection"):
            first, second = control + suffix, candidate + suffix
            values = _scene_differences(summary, first, second, "dynamics_residual")
            interval = paired_scene_interval(values)
            comparisons["raw" if not suffix else "projected"] = dict(
                control=first,
                candidate=second,
                paired_scene_interval=interval,
                holds=dynamics_residual_holds(summary[second], summary[first], interval),
            )
        result["dynamics_residual"] = comparisons
    if mode == "d038":
        control, candidate = FULL_PIPELINE
        same_world = paired_scene_interval(
            _scene_differences(summary, control, candidate, "matched_world_ade")
        )
        residual = paired_scene_interval(
            _scene_differences(summary, control, candidate, "dynamics_residual")
        )
        checks = full_pipeline_checks(summary[candidate], summary[control], same_world, residual)
        result["full_pipeline"] = {
            "control": control,
            "candidate": candidate,
            "same_world_ade_interval": same_world,
            "dynamics_residual_interval": residual,
            "checks": checks,
            "holds": all(checks.values()),
        }
    if mode == "fresh":
        learned = {}
        for control, candidates in FAMILIES.values():
            for name in models:
                if name in candidates and control in models:
                    values = _scene_differences(summary, control, name, "matched_world_ade")
                    interval = paired_scene_interval(values)
                    learned[name] = dict(
                        control=control,
                        same_world_interval=interval,
                        holds=learned_geometry_holds(summary[name], summary[control], interval),
                    )
        result["learned_geometry"] = learned
    write(output / "results.json", result)
    return result


def parse_named(values):
    named = {}
    for item in values:
        name, _, path = item.partition("=")
        if not name or not path or name in named or name in CONTROLS or "+" in name:
            raise ValueError(f"Use unique NAME=PATH checkpoint arguments, not {item!r}")
        named[name] = Path(path)
    return named


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("validation", "d031", "fresh", "d035", "d037", "d038"),
        required=True,
    )
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--previous-data", type=Path, nargs="*", default=())
    args = parser.parse_args()
    torch.set_num_threads(4)
    r = compare(
        args.dataset,
        args.output,
        mode=args.mode,
        checkpoints=parse_named(args.checkpoint),
        training_data=args.training_data,
        previous_data=args.previous_data,
    )
    for name, m in r["models"].items():
        print(
            name,
            "pen",
            m["penetrating_clips"],
            "bnd",
            m["boundary_clips"],
            "selADE",
            m["quality"]["selected_ade_mean_cases_if_all_complete"],
            "same",
            round(m["matched_world_ade"], 5),
            "ES",
            round(m["energy"]["selected"], 4),
        )
    for key in (
        "selection",
        "projection_holds",
        "learned_geometry",
        "combined",
        "metric_check",
        "zero_pixel_check",
    ):
        if key in r:
            print(key, json.dumps(r[key]))
