"""Freeze selected-object interventions without changing the training corpus."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from click2future.data import load_rollout, save_rollout
from click2future.sim import run_rollout

from .actions import _displacement, _force, _inputs
from .metrics import _ratio
from .paired_actions import action_settings, summarize_actions
from .tracking import TRACKER_VERSION, track_video

KINDS = ("first", "second")


def target_metrics(
    first,
    second,
    first_tracked,
    second_tracked,
    object_valid,
    *,
    selected_object_ids,
    force_xy,
    end_frame,
    motion_threshold,
):
    """Require the clicked object to move and the unclicked objects to stay still.

    Both conditions use the same force and paired seeds. A separate contrast
    checks that each object's force projection increases when it is selected.
    Missing tracks and weak motion keep their denominators and failure counts.
    """
    if len(selected_object_ids) != 2 or selected_object_ids[0] == selected_object_ids[1]:
        raise ValueError("Choose two distinct valid selected objects")
    arrays, known = [], []
    for positions, tracked, target in zip(
        (first, second), (first_tracked, second_tracked), selected_object_ids, strict=True
    ):
        values, valid_tracks, valid = _inputs(
            positions, tracked, object_valid, target, end_frame, motion_threshold
        )
        arrays.append(values)
        known.append(valid_tracks)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Target controls must have matching shapes")
    initial_known = known[0][:, 0] & known[1][:, 0] & valid
    if not np.array_equal(arrays[0][:, 0][initial_known], arrays[1][:, 0][initial_known]):
        raise ValueError("Target controls must have identical initial positions")
    force, norm = _force(force_xy)
    if norm == 0:
        raise ValueError("Selected-object controls require a nonzero force")
    unit = force / norm
    records = []
    for index in range(len(first)):
        row = {"sample_index": index, "status": "missing_track"}
        if all(tracks[index, :, valid].all() for tracks in known):
            displacements, drifts, selected_lengths, selected_projections = [], [], [], []
            for values, target in zip(arrays, selected_object_ids, strict=True):
                delta, lengths = _displacement(np.where(valid[None, :, None], values[index], 0))
                others = valid.copy()
                others[target] = False
                displacements.append(delta[-1])
                drifts.append(float(lengths[1:, others].max()))
                selected_lengths.append(float(lengths[-1, target]))
                selected_projections.append(float(np.dot(delta[-1, target], unit)))
            a, b = selected_object_ids
            contrasts = [
                float(np.dot(displacements[0][a] - displacements[1][a], unit)),
                float(np.dot(displacements[1][b] - displacements[0][b], unit)),
            ]
            row.update(
                status="stationary" if min(selected_lengths) < motion_threshold else "moving",
                selected_projections=selected_projections,
                nonselected_max_drifts=drifts,
                contrast_projections=contrasts,
                correct_selected_objects=bool(
                    min(selected_projections) >= motion_threshold and max(drifts) < motion_threshold
                ),
                correct_response_contrast=bool(min(contrasts) >= motion_threshold),
                nonselected_drift=bool(max(drifts) >= motion_threshold),
            )
        records.append(row)
    complete = [row for row in records if row["status"] != "missing_track"]
    return {
        "total_pairs": len(records),
        "complete_pairs": len(complete),
        "missing_pairs": len(records) - len(complete),
        "stationary_pairs": sum(row["status"] == "stationary" for row in complete),
        "moving_pairs": sum(row["status"] == "moving" for row in complete),
        "correct_target_rate_over_complete": _ratio(
            sum(row["correct_selected_objects"] for row in complete), len(complete)
        ),
        "correct_contrast_rate_over_complete": _ratio(
            sum(row["correct_response_contrast"] for row in complete), len(complete)
        ),
        "drift_rate_over_complete": _ratio(
            sum(row["nonselected_drift"] for row in complete), len(complete)
        ),
        "end_frame": end_frame,
        "motion_threshold": motion_threshold,
        "units": "normalized_image_coordinates",
        "selected_object_ids": list(selected_object_ids),
        "force_xy": force.tolist(),
        "per_sample": records,
    }


def _score_rgb(refs, videos, settings, frame_valid=None):
    arrays, known = [], []
    for kind in KINDS:
        ref, clips = refs[kind], np.asarray(videos[kind])
        if (
            clips.dtype != np.uint8
            or clips.ndim != 5
            or len(clips) < 1
            or clips.shape[1:] != ref["frames"].shape
            or not np.all(clips[:, 0] == ref["frames"][0])
        ):
            raise ValueError("Target videos must be uint8 [K,T,H,W,3] with exact frame 0")
        valid = None if frame_valid is None else frame_valid.get(kind)
        valid = np.ones(clips.shape[:2], dtype=bool) if valid is None else np.asarray(valid)
        if valid.dtype != np.bool_ or valid.shape != clips.shape[:2] or not valid[:, 0].all():
            raise ValueError("frame_valid must be boolean [K,T] with valid first frames")
        tracks = [
            track_video(
                clip[: settings["end_frame"] + 1],
                ref["instance_masks"][0],
                ref["object_valid"],
                settings["tracker"],
            )
            for clip in clips
        ]
        positions = np.stack([track.positions for track in tracks])
        tracked = np.stack([track.tracked for track in tracks])
        tracked &= valid[:, : settings["end_frame"] + 1, None]
        positions[~tracked] = np.nan
        arrays.append(positions)
        known.append(tracked)
    return target_metrics(
        *arrays,
        *known,
        refs["first"]["object_valid"],
        selected_object_ids=[refs[kind]["selected_object_id"] for kind in KINDS],
        force_xy=refs["first"]["force_xy"],
        end_frame=settings["end_frame"],
        motion_threshold=settings["motion_threshold"],
    )


def prepare_target_plan(
    parent_path, output_dir, *, config="configs/evaluation/actions.yaml", force_magnitude=None
):
    """Derive both clicked-object cases from every held-out forward condition.

    Corpus/split hashes still identify the ORIGINAL training corpus. Derived
    references live separately and never enter its manifests or train loader.
    The new plan binds the parent, original inputs and every generated file.
    """
    # Lazy import keeps the ordinary suite and this optional preparation CLI independent.
    from .suite import _digest, _sha, _write, read_plan

    parent_path, output_dir, config = map(
        lambda path: Path(path).resolve(), (parent_path, output_dir, config)
    )
    if output_dir.exists():
        raise ValueError("Use a fresh target-control output directory")
    if force_magnitude is not None and (
        isinstance(force_magnitude, bool)
        or not np.isfinite(force_magnitude)
        or force_magnitude <= 0
    ):
        raise ValueError("force_magnitude must be finite and positive")
    parent = read_plan(parent_path)
    if "intervention" in parent:
        raise ValueError("Derive target controls from an ordinary held-out corpus plan")
    settings = action_settings(config)
    sources = {}
    for case in parent["cases"]:
        ref = load_rollout(parent["dataset"], case["sample_id"])
        if ref["scene_spec"].get("action_variant") == "forward":
            if case["scene_group_id"] in sources:
                raise ValueError("Expected one forward condition per held-out scene group")
            sources[case["scene_group_id"]] = case
    if set(sources) != {case["scene_group_id"] for case in parent["cases"]}:
        raise ValueError("Every held-out scene group needs a forward condition")
    original_inputs = dict(parent["input_sha256"])
    simulator_sources = {
        str(path): _sha(path) for path in sorted((Path(__file__).parents[1] / "sim").glob("*.py"))
    }
    plan = copy.deepcopy(parent)
    plan.pop("plan_sha256")
    plan.pop("action_groups", None)
    plan.update(
        dataset=str(output_dir / "references"),
        split=parent["split"] + "-selected-object",
        cases=[],
        target_groups=[],
        intervention={
            "kind": "selected_object_only",
            "recipe_version": 1,
            "source_dataset": parent["dataset"],
            "source_split": parent["split"],
            "parent_plan_sha256": parent["plan_sha256"],
            "training_corpus_unchanged": True,
            "simulator_source_sha256": {
                Path(path).name: value for path, value in simulator_sources.items()
            },
            "force_policy": "Same forward direction, duration and physics under both clicks",
            "force_magnitude_override": force_magnitude,
        },
    )
    plan["files"].update(parent_plan=str(parent_path), actions_config=str(config))
    output_dir.mkdir(parents=True)
    try:
        for group_id, case in sorted(sources.items()):
            refs = [load_rollout(parent["dataset"], name) for name in case["reference_ids"]]
            base = refs[0]
            force, norm = _force(base["force_xy"])
            if norm == 0:
                raise ValueError("Selected-object controls require a nonzero forward force")
            if force_magnitude is not None:
                force = force / norm * force_magnitude
            force = force.astype(np.float32)
            targets = [base["selected_object_id"]]
            others = [int(i) for i in np.flatnonzero(base["object_valid"]) if i != targets[0]]
            if not others:
                raise ValueError("Selected-object controls require at least two valid objects")
            targets.append(others[0])
            grouped, contact = {kind: [] for kind in KINDS}, False
            names = {kind: [] for kind in KINDS}
            physics = set()
            for source_id, ref in zip(case["reference_ids"], refs, strict=True):
                profile = json.dumps(ref["scene_spec"]["objects"], sort_keys=True)
                if profile in physics:
                    raise ValueError("Source physics variants must be unique")
                physics.add(profile)
                if (
                    ref["selected_object_id"] != targets[0]
                    or ref["force_duration"] != base["force_duration"]
                    or not np.array_equal(ref["force_xy"], base["force_xy"])
                    or not np.any(ref["force_xy"])
                    or not np.array_equal(ref["frames"][0], base["frames"][0])
                    or not np.array_equal(ref["instance_masks"][0], base["instance_masks"][0])
                    or not np.array_equal(ref["timestamps"], base["timestamps"])
                    or settings["end_frame"] >= len(ref["frames"])
                    or list(ref["frames"].shape[1:3]) != list(settings["resolution_hw"])
                    or not np.allclose(
                        np.diff(ref["timestamps"]), 1 / settings["fps"], rtol=0, atol=1e-8
                    )
                ):
                    raise ValueError("Source conditions must share image, masks, action and timing")
                if any(
                    any(obj["initial_velocity"]) or obj["initial_angular_velocity"] != 0
                    for obj in ref["scene_spec"]["objects"]
                ):
                    raise ValueError("Selected-object controls must start at rest")
                for kind, target in zip(KINDS, targets, strict=True):
                    scene = copy.deepcopy(ref["scene_spec"])
                    scene["action"]["selected_object_id"] = target
                    scene["action"]["force_xy"] = force.tolist()
                    scene["action_variant"] = f"selected_target_{target}"
                    scene["condition_id"] = f"{case['condition_id']}-select-{target}"
                    regenerated = run_rollout(
                        scene,
                        **{
                            key: ref["metadata"][key]
                            for key in ("frame_count", "fps", "physics_hz", "resolution")
                        },
                    )
                    if (
                        not np.array_equal(regenerated["frames"][0], base["frames"][0])
                        or not np.array_equal(regenerated["force_xy"], force)
                        or regenerated["force_duration"] != base["force_duration"]
                    ):
                        raise ValueError("Derived control changed the image or planned force")
                    end = settings["end_frame"]
                    contact |= bool(
                        regenerated["collision_onset"][:end].any()
                        or regenerated["wall_contact_onset"][:end].any()
                    )
                    name = f"{source_id}-select-{target}"
                    save_rollout(regenerated, plan["dataset"], name)
                    names[kind].append(name)
                    grouped[kind].append(regenerated)
                    for suffix in (".json", ".npz"):
                        plan["files"][name + suffix] = str(Path(plan["dataset"]) / (name + suffix))
            for kind in KINDS:
                plan["cases"].append(
                    {
                        "scene_group_id": group_id,
                        "condition_id": grouped[kind][0]["scene_spec"]["condition_id"],
                        "sample_id": names[kind][0],
                        "reference_ids": names[kind],
                        "source_case_id": case["sample_id"],
                    }
                )
            group = {
                "scene_group_id": group_id,
                "eligible": not contact,
                "exclusion_reason": "reference_contact_within_fixed_window" if contact else None,
                "sample_ids": {kind: names[kind][0] for kind in KINDS},
                "reference_ids": names,
                "selected_object_ids": targets,
                "force_xy": force.tolist(),
            }
            if not contact:
                positions = [
                    np.stack([ref["positions"] for ref in grouped[kind]]) for kind in KINDS
                ]
                label_score = target_metrics(
                    *positions,
                    *(np.ones(value.shape[:-1], dtype=bool) for value in positions),
                    base["object_valid"],
                    selected_object_ids=targets,
                    force_xy=force,
                    end_frame=settings["end_frame"],
                    motion_threshold=settings["motion_threshold"],
                )
                rgb_score = _score_rgb(
                    {kind: grouped[kind][0] for kind in KINDS},
                    {kind: np.stack([ref["frames"] for ref in grouped[kind]]) for kind in KINDS},
                    settings,
                )
                group.update(
                    reference_label_scores={"selection": label_score},
                    reference_rgb_scores={"selection": rgb_score},
                )
            plan["target_groups"].append(group)
        read_plan(parent_path)
        if any(_sha(path) != value for path, value in simulator_sources.items()):
            raise ValueError("Simulator source changed during control preparation")
        plan["input_sha256"] = {key: _sha(path) for key, path in plan["files"].items()}
        if any(plan["input_sha256"][key] != value for key, value in original_inputs.items()):
            raise ValueError("Original corpus or parent inputs changed during preparation")
        plan["plan_sha256"] = _digest(plan)
        _write(output_dir / "plan.json", plan)
        return plan
    except Exception as error:
        _write(output_dir / "failure.json", {"status": "failed", "error": str(error)})
        raise


def evaluate_target_groups(plan, paths):
    settings = action_settings(plan["files"]["actions_config"])
    groups = []
    for group in plan["target_groups"]:
        record = dict(group)
        if group["eligible"]:
            refs, videos, valid = {}, {}, {}
            for kind, name in group["sample_ids"].items():
                refs[kind] = load_rollout(plan["dataset"], name)
                with np.load(paths[name], allow_pickle=False) as archive:
                    videos[kind] = archive["videos"]
                    valid[kind] = archive["frame_valid"] if "frame_valid" in archive else None
            record["scores"] = {"selection": _score_rgb(refs, videos, settings, valid)}
        groups.append(record)
    return groups


def target_report(plan, runs, *, source, untrained, bootstrap, confidence, seed):
    settings = dict(bootstrap=bootstrap, confidence=confidence, seed=seed)
    controls = action_settings(plan["files"]["actions_config"])
    return {
        "status": "passed",
        "source": source,
        "target_metric_version": "1",
        "extractor_version": TRACKER_VERSION,
        "evaluation_plan_sha256": plan["plan_sha256"],
        "config_sha256": plan["input_sha256"]["actions_config"],
        "settings": {**controls, "tracker": asdict(controls["tracker"])},
        "intervention": plan["intervention"],
        "sampling_seeds": plan["sampling_seeds"],
        "untrained_fixture_runs": untrained,
        "expected_scene_groups": len(plan["target_groups"]),
        "eligible_scene_groups": sum(group["eligible"] for group in plan["target_groups"]),
        "eligibility": (
            "All reference physics must start at rest and avoid contact in the fixed window"
        ),
        "pairing": (
            "Identical image, force, duration and sample seeds; only the selected object changes"
        ),
        "reference_calibration": {
            "source": "simulator reference labels and rendered RGB, not model output",
            "summary": summarize_actions(
                [
                    {
                        "method": method,
                        "groups": [
                            {**group, "scores": group.get(field, {})}
                            for group in plan["target_groups"]
                        ],
                    }
                    for method, field in (
                        ("reference-labels", "reference_label_scores"),
                        ("reference-rgb", "reference_rgb_scores"),
                    )
                ],
                **settings,
            ),
        },
        "runs": runs,
        "summary": summarize_actions(runs, **settings),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Ordinary frozen held-out plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/actions.yaml"))
    parser.add_argument(
        "--force-magnitude",
        type=float,
        help="Optional shared magnitude; choose on validation, then freeze for test",
    )
    args = parser.parse_args(argv)
    try:
        plan = prepare_target_plan(
            args.plan, args.output_dir, config=args.config, force_magnitude=args.force_magnitude
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Target-control preparation stopped: {error}\n")
    print(
        json.dumps(
            {"status": "passed", "cases": len(plan["cases"]), "groups": len(plan["target_groups"])}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
