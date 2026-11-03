"""Paired force/selection diagnostics, separate from historical evaluation scores."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.action_horizon import contact_free
from click2future.evaluation.targets import target_metrics
from click2future.evaluation.tracking import TrackerConfig, track_video
from click2future.evaluation.trajectory_audit import verify_inputs
from click2future.sim import run_rollout
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_cvae import require_tc2, sha256

MAGNITUDES = (0.15, 0.25, 0.5)
HORIZONS = {"window": 4, "full": 31}


def tracks(reference, videos):
    values = [
        track_video(v, reference["instance_masks"][0], reference["object_valid"], TrackerConfig())
        for v in videos
    ]
    return np.stack([v.positions for v in values]), np.stack([v.tracked for v in values])


def scaling(low, high, unit, selected, *, end_frame):
    """Descriptive paired response; undefined and backwards motion remain visible."""
    rows = []
    for index in range(len(low[0])):
        complete = bool(
            low[1][index, : end_frame + 1, selected].all()
            and high[1][index, : end_frame + 1, selected].all()
        )
        row = {"sample_index": index, "complete": complete}
        if complete:
            a, b = [
                float((values[index, end_frame, selected] - values[index, 0, selected]) @ unit)
                for values, _ in (low, high)
            ]
            row.update(
                low_displacement=a,
                high_displacement=b,
                difference=b - a,
                increases_by_at_least_one_pixel=b - a >= 1 / 64,
                decreases_by_at_least_one_pixel=a - b >= 1 / 64,
                high_to_low_ratio=b / a if a >= 1 / 64 else None,
            )
        rows.append(row)
    return {
        "total_samples": len(rows),
        "complete_samples": sum(r["complete"] for r in rows),
        "missing_samples": sum(not r["complete"] for r in rows),
        "per_sample": rows,
        "scope": (
            "paired projected displacement; monotonicity is descriptive, "
            "not exact force proportionality"
        ),
    }


def diagnose(plan_path, predictions_path, dataset, checkpoint, output, *, device="cpu"):
    plan, _, _, hashes = verify_inputs(plan_path, predictions_path, dataset, checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, _ = load_trajectory(checkpoint, device)
    if getattr(model, "world_condition", False) is not True:
        raise ValueError("This paired-world probe requires conditioned E7/E8/E9")
    groups = {}
    for case in plan["cases"]:
        if "-forward-" in case["sample_id"]:
            if case["scene_group_id"] in groups:
                raise ValueError("Duplicate forward condition")
            groups[case["scene_group_id"]] = case
    if len(groups) != len({c["scene_group_id"] for c in plan["cases"]}):
        raise ValueError("Retain every validation group")
    selections, force_rows = [], []
    generated_files = {}
    for group, case in sorted(groups.items()):
        originals = [load_rollout(dataset, name) for name in case["reference_ids"]]
        if len(originals) != 4:
            raise ValueError("Require every original physical reference")
        base = originals[0]
        for sign in (-1, 1):
            unit = np.array([sign, 0.0], dtype=np.float64)
            generated, reference, tracked = {}, {}, {}
            for magnitude in MAGNITUDES:
                for selected in (0, 1):
                    key = (magnitude, selected)
                    force = (unit * magnitude).astype(np.float32)
                    # References are an isolated diagnostic control, never model inputs.
                    refs = []
                    for original in originals:
                        scene = copy.deepcopy(original["scene_spec"])
                        scene["action"].update(selected_object_id=selected, force_xy=force.tolist())
                        refs.append(
                            run_rollout(scene, frame_count=32, fps=8, physics_hz=240, resolution=64)
                        )
                    reference[key] = refs
                    videos = sample_trajectory_videos(
                        model,
                        base["frames"][0],
                        base["instance_masks"][0, selected],
                        force,
                        seeds=plan["sampling_seeds"],
                        sampling_steps=1 if getattr(model, "world_prior_only", False) else 16,
                    )
                    generated[key] = videos
                    tracked[key] = tracks(refs[0], videos)
                    name = f"{group}-sign{sign}-m{magnitude}-s{selected}.npz"
                    np.savez_compressed(
                        output / name,
                        videos=videos,
                        reference_videos=np.stack([r["frames"] for r in refs]),
                    )
                    generated_files[name] = sha256(output / name)
            for horizon, end_frame in HORIZONS.items():
                for magnitude in MAGNITUDES:
                    all_refs = [r for selected in (0, 1) for r in reference[(magnitude, selected)]]
                    eligible = contact_free(all_refs, end_frame)
                    row = {
                        "scene_group_id": group,
                        "sign": sign,
                        "magnitude": magnitude,
                        "horizon": horizon,
                        "end_frame": end_frame,
                        "eligible": eligible,
                        "exclusion_reason": None
                        if eligible
                        else "reference_contact_within_horizon",
                    }
                    if eligible:
                        a, b = (tracked[(magnitude, selected)] for selected in (0, 1))
                        row["model"] = target_metrics(
                            a[0],
                            b[0],
                            a[1],
                            b[1],
                            base["object_valid"],
                            selected_object_ids=(0, 1),
                            force_xy=(unit * magnitude),
                            end_frame=end_frame,
                            motion_threshold=1 / 64,
                        )
                        values = [
                            tracks(
                                reference[(magnitude, s)][0],
                                np.stack([r["frames"] for r in reference[(magnitude, s)]]),
                            )
                            for s in (0, 1)
                        ]
                        row["rendered_references"] = target_metrics(
                            values[0][0],
                            values[1][0],
                            values[0][1],
                            values[1][1],
                            base["object_valid"],
                            selected_object_ids=(0, 1),
                            force_xy=(unit * magnitude),
                            end_frame=end_frame,
                            motion_threshold=1 / 64,
                        )
                    selections.append(row)
                for low, high in zip(MAGNITUDES[:-1], MAGNITUDES[1:], strict=True):
                    for selected in (0, 1):
                        refs = reference[(low, selected)] + reference[(high, selected)]
                        eligible = contact_free(refs, end_frame)
                        row = {
                            "scene_group_id": group,
                            "sign": sign,
                            "selected_object_id": selected,
                            "low_force": low,
                            "high_force": high,
                            "horizon": horizon,
                            "end_frame": end_frame,
                            "eligible": eligible,
                            "exclusion_reason": None
                            if eligible
                            else "reference_contact_within_horizon",
                        }
                        if eligible:
                            row["model"] = scaling(
                                tracked[(low, selected)],
                                tracked[(high, selected)],
                                unit,
                                selected,
                                end_frame=end_frame,
                            )
                            ref_tracks = [
                                tracks(
                                    reference[(m, selected)][0],
                                    np.stack([r["frames"] for r in reference[(m, selected)]]),
                                )
                                for m in (low, high)
                            ]
                            row["rendered_references"] = scaling(
                                *ref_tracks, unit, selected, end_frame=end_frame
                            )
                        force_rows.append(row)
        print(json.dumps({"group": group, "state": "diagnosed"}), flush=True)
    for path, checksum in hashes.items():
        if sha256(path) != checksum:
            raise ValueError("Original audit input changed")
    summary = {}
    for horizon in HORIZONS:
        selected = [r for r in selections if r["horizon"] == horizon]
        force = [r for r in force_rows if r["horizon"] == horizon]
        summary[horizon] = {
            "selection_conditions": len(selected),
            "eligible_selection_conditions": sum(r["eligible"] for r in selected),
            "excluded_selection_conditions": sum(not r["eligible"] for r in selected),
            "force_conditions": len(force),
            "eligible_force_conditions": sum(r["eligible"] for r in force),
        }
        for family, metric_rows in (("selection", selected), ("force", force)):
            summary[horizon][family] = {}
            for method in ("model", "rendered_references"):
                rows = [r[method] for r in metric_rows if r["eligible"]]
                samples = [s for r in rows for s in r["per_sample"]]
                summary[horizon][family][method] = {
                    "expected_eligible_samples": len(rows) * 4,
                    "complete_samples": sum(
                        r["complete_pairs" if family == "selection" else "complete_samples"]
                        for r in rows
                    ),
                    "missing_samples": sum(
                        r["missing_pairs" if family == "selection" else "missing_samples"]
                        for r in rows
                    ),
                }
                if family == "selection":
                    summary[horizon][family][method].update(
                        correct_targets=sum(
                            s.get("correct_selected_objects", False) for s in samples
                        ),
                        nonselected_drift=sum(s.get("nonselected_drift", False) for s in samples),
                    )
                else:
                    summary[horizon][family][method].update(
                        increases=sum(
                            s.get("increases_by_at_least_one_pixel", False) for s in samples
                        ),
                        decreases=sum(
                            s.get("decreases_by_at_least_one_pixel", False) for s in samples
                        ),
                    )
    record = {
        "status": "measured_development_diagnostic",
        "scope": (
            "paired force/selection response on existing validation images; "
            "not historical official scores or calibrated probabilities"
        ),
        "checkpoint_sha256": sha256(checkpoint),
        "plan_sha256": plan["plan_sha256"],
        "sampling_seeds": plan["sampling_seeds"],
        "physical_reference_ids": {k: v["reference_ids"] for k, v in groups.items()},
        "magnitudes": MAGNITUDES,
        "selections": selections,
        "force_scaling": force_rows,
        "summary": summary,
        "generated_files_sha256": generated_files,
        "source_sha256": sha256(Path(__file__)),
        "test_splits_evaluated": [],
        "inference_corrections": False,
    }
    (output / "response.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.device == "cuda":
        require_tc2()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    result = diagnose(
        args.plan, args.predictions, args.dataset, args.checkpoint, args.output, device=args.device
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
