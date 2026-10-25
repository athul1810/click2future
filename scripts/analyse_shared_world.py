"""Verify a collected E7 allocation and retain all validation denominators."""

import argparse
import json
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.evaluation.trajectory_audit import verify_inputs
from click2future.models.object_motion import observe, render_motion
from click2future.training.train_cvae import sha256


def read(path):
    return json.loads(path.read_text())


def analyse(root, dataset, anchor):
    root = Path(root).resolve()
    verified = 0
    for line in (root / "manifest.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Collected file identity differs: " + name)
        verified += 1
    evaluation = root / "recovery" if (root / "recovery/comparison.json").exists() else root
    comparison = read(evaluation / "comparison.json")
    if comparison["status"] != "completed" or comparison["test_splits_evaluated"]:
        raise ValueError("Require a complete validation-only comparison")
    if evaluation != root and (
        comparison["retained_comparison_sha256"] != sha256(root / "comparison.json")
    ):
        raise ValueError("Original failed comparison changed during evaluation recovery")
    arms = ("local", "relational", "conditioned")
    training = {name: read(root / name / "summary.json") for name in arms}
    for name, value in training.items():
        if (
            value["status"] != "completed"
            or value["steps_completed"] != 8000
            or value["parameter_count"] != 1064580
            or value["provenance"] != training["local"]["provenance"]
            or value["selection"] != "fixed final update, not validation loss or physics scores"
        ):
            raise ValueError("Unequal or incomplete training: " + name)
    for left, right, switch in (
        ("local", "relational", "relational"),
        ("relational", "conditioned", "world_condition"),
    ):
        a, b = training[left]["config"], training[right]["config"]
        if {key for key in a if a[key] != b[key]} != {switch}:
            raise ValueError("Adjacent recipes changed more than one switch")
    cache = read(root / "cache/manifest.json")
    supplement = training["local"]["provenance"]["training_supplement"]
    result = {
        "status": "measured_post_D014_exploratory_validation",
        "scope": (
            "one warm-start seed; four scenes; extra physical supervision; "
            "no external SOTA comparison"
        ),
        "training_source_revision": comparison.get(
            "training_source_revision", comparison["source_revision"]
        ),
        "evaluation_source_revision": comparison["source_revision"],
        "training_slurm_job_id": comparison.get(
            "training_slurm_job_id", comparison["slurm_job_id"]
        ),
        "evaluation_slurm_job_id": comparison["slurm_job_id"],
        "original_attempt": read(root / "comparison.json")["status"],
        "manifest_files_verified": verified,
        "comparison_sha256": sha256(evaluation / "comparison.json"),
        "analysis_source_sha256": sha256(Path(__file__)),
        "elapsed_seconds": comparison["elapsed_seconds"],
        "phases": comparison["phases"],
        "warm_start": training["local"]["provenance"]["initialization"],
        "data_identity": {
            **{
                key: value
                for key, value in training["local"]["provenance"].items()
                if key not in {"initialization", "training_supplement", "source_sha256"}
            },
            "cache_manifest_sha256": sha256(root / "cache/manifest.json"),
            "cache_arrays_sha256": sha256(root / "cache/arrays.npz"),
            "example_counts": {key: len(value) for key, value in cache["examples"].items()},
            "training_supplement": {
                "manifest_sha256": sha256_json(supplement["manifest"]),
                "sample_count": len(supplement["manifest"]["samples"]),
                "recipe": supplement["manifest"]["recipe"],
            },
        },
        "models": {},
        "test_splits_evaluated": [],
        "inference_corrections": False,
    }
    if comparison["source_revision"] == "unrecorded":
        result["evaluation_source_attestation"] = read(root / "operations/source-attestation.json")
    for name in ("anchor", *arms):
        folder = evaluation / (name + "-validation")
        checkpoint = Path(anchor) if name == "anchor" else root / name / "best.pt"
        digest = sha256(checkpoint)
        if digest != comparison["models"][name]["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs: " + name)
        plan, _, paths, _ = verify_inputs(
            folder / "plan.json",
            folder / "validation-samples/predictions.json",
            dataset,
            checkpoint,
        )
        trajectory = read(folder / "trajectory-results.json")
        geometry = read(folder / "geometry-audit/geometry.json")
        collisions = read(folder / "collision-audit.json")
        if not geometry["pixel_exact_pose_replay"] or len(geometry["cases"]) != 24:
            raise ValueError("Native geometry replay is incomplete")
        if len(collisions["cases"]) != 24 or collisions["checkpoint_sha256"] != digest:
            raise ValueError("Native collision audit is incomplete")
        replays, mismatches, failed_cases = 0, [], []
        for case in plan["cases"]:
            ref = load_rollout(dataset, case["sample_id"])
            observation = observe(ref["frames"][0], ref["selected_mask"])
            with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
                videos = archive["videos"]
            with np.load(folder / "geometry-audit" / (case["sample_id"] + "-poses.npz")) as archive:
                motions = archive["motion"]
            if not np.isfinite(motions).all() or len(motions) != 4:
                raise ValueError("Missing/invalid saved poses")
            for seed, motion, video in zip(plan["sampling_seeds"], motions, videos, strict=True):
                generated = render_motion(observation, motion)
                different = np.any(generated != video, axis=-1)
                if different.any():
                    mismatches.append(
                        {
                            "sample_id": case["sample_id"],
                            "seed": seed,
                            "different_pixels": int(different.sum()),
                            "different_frames": np.flatnonzero(different.any(axis=(1, 2))).tolist(),
                        }
                    )
                else:
                    replays += 1
        for path in sorted((folder / "validation-metrics").glob("*-trajectory.json")):
            report = read(path)
            selected = report["trajectory"]["selected_object"]
            ade = selected["mean_ade_over_complete_samples"]
            if (
                selected["complete_samples"] != selected["total_samples"]
                or ade is None
                or ade > 0.03125
            ):
                failed_cases.append({"sample_id": report["sample_id"], "selected_ade": ade})
        families = collisions["action_families"]
        row = {
            "checkpoint_sha256": digest,
            "evaluated_checkpoint_step": trajectory["training_steps"],
            "quality": trajectory["quality"],
            "failed_trajectory_cases": failed_cases,
            "geometry": geometry["summary"],
            "maximum_penetration_pixels": max(v["maximum_depth_pixels"] for v in families.values()),
            "longest_penetration_frames": max(
                v["longest_consecutive_penetration_frames"] for v in families.values()
            ),
            "zero_force_full_horizon": read(folder / "contact-results.json")[
                "zero_force_full_horizon"
            ]["summary"],
            "passive_stationarity": read(folder / "passive-stationarity.json")["summary"],
            "clipping": read(folder / "wall-clipping.json")["summary"],
            "collision_micro": collisions["model_events_against_first_reference"],
            "action_families": families,
            "strata": read(folder / "uncertainty-strata.json")["action_strata"],
            "early_actions": read(folder / "validation-metrics/actions.json")["summary"],
            "early_selection": read(folder / "selected-object-metrics/targets.json")["summary"],
            "horizons": read(folder / "action-horizons.json")["summary"],
            "native_pixel_exact_pose_replays": 96,
            "windows_pose_replay": {
                "exact_clips": replays,
                "expected_clips": 96,
                "mismatches": mismatches,
            },
            "collision_audit_sha256": sha256(folder / "collision-audit.json"),
        }
        original_predictions = root / (name + "-validation/validation-samples/predictions.json")
        if evaluation != root and original_predictions.exists():
            before = read(original_predictions)["runs"][0]["video_sha256"]
            after = read(folder / "validation-samples/predictions.json")["runs"][0]["video_sha256"]
            row["original_attempt_video_reproduction"] = {
                "expected_conditions": len(before),
                "byte_identical_conditions": sum(before[key] == after[key] for key in before),
                "scope": "unchanged checkpoint/sampler across A40 allocations; no outputs replaced",
            }
        if name != "anchor":
            row["training"] = {
                key: value for key, value in training[name].items() if key != "provenance"
            }
            row["resume_check"] = read(root / name / "resume-check.json")
            intervention = read(folder / "world-interventions/summary.json")
            row["mass_interventions"] = intervention
            summary = intervention["summary"]
            if summary["samples"] != 32 or (
                name != "conditioned" and summary["equal"] != summary["samples"]
            ):
                raise ValueError("Incomplete or changed blind intervention")
        result["models"][name] = row
        print(
            json.dumps(
                {
                    "model": name,
                    "quality": row["quality"],
                    "geometry": row["geometry"]["learned"],
                    "replay": row["windows_pose_replay"],
                }
            ),
            flush=True,
        )
    return result


def sha256_json(value):
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "dataset", "anchor", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyse(args.root, args.dataset, args.anchor)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
