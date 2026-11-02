"""Verify the paired visible-coverage run, preserving every failed outcome."""

import argparse
import json
from pathlib import Path

import numpy as np

from click2future.evaluation.trajectory_audit import verify_inputs
from click2future.training.train_cvae import sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_changes(root, name):
    """Measure actual saved predictions, including unchanged clips and all seeds."""
    anchor = root / "anchor-validation"
    candidate = root / (name + "-validation")
    plan = read(anchor / "plan.json")
    count = changed = pixels = values = 0
    squared = maximum = 0.0
    for case in plan["cases"]:
        filename = case["sample_id"]
        with (
            np.load(anchor / "geometry-audit" / (filename + "-poses.npz")) as a,
            np.load(candidate / "geometry-audit" / (filename + "-poses.npz")) as b,
        ):
            delta = (b["motion"][..., :2].astype(np.float64) - a["motion"][..., :2]) * 32
            squared += float(np.square(delta).sum())
            values += delta.size
            maximum = max(maximum, float(np.abs(delta).max()))
        with (
            np.load(anchor / "validation-samples" / (filename + ".npz")) as a,
            np.load(candidate / "validation-samples" / (filename + ".npz")) as b,
        ):
            different = np.any(a["videos"] != b["videos"], axis=-1)
            count += len(different)
            changed += int(different.reshape(len(different), -1).any(1).sum())
            pixels += int(different.sum())
    return {
        "clips": count,
        "rgb_changed_clips": changed,
        "rgb_changed_pixels_over_all_frames": pixels,
        "center_coordinate_rms_change_pixels": (squared / values) ** 0.5,
        "center_coordinate_maximum_absolute_change_pixels": maximum,
        "scope": "paired saved predictions against E7; effect size, not quality",
    }


def analyse(root, dataset):
    root = Path(root).resolve()
    verified = 0
    for line in (root / "manifest.sha256").read_text().splitlines():
        checksum, filename = line.split("  ", 1)
        path = (root / filename).resolve()
        if not path.is_relative_to(root) or sha256(path) != checksum:
            raise ValueError("Collected file changed: " + filename)
        verified += 1
    comparison = read(root / "comparison.json")
    if comparison["status"] != "completed" or comparison["test_splits_evaluated"]:
        raise ValueError("Require a complete validation comparison")
    for stage in ("prejob", "after"):
        lines = (root / "operations" / ("source-check-" + stage + ".out")).read_text().splitlines()
        if len(lines) != 470 or not all(line.endswith(": OK") for line in lines):
            raise ValueError("Immutable source verification is incomplete")
    training = {name: read(root / name / "summary.json") for name in ("narrow", "broad")}
    a, b = training.values()
    if a["config"] != b["config"]:
        raise ValueError("Unmatched coverage recipes")
    for key in (
        "parameter_count",
        "trainable_parameter_count",
        "initial_model_tensors_sha256",
        "initial_private_rng_sha256",
        "sampling_weights_sha256",
    ):
        if a[key] != b[key]:
            raise ValueError("Unmatched coverage initialization: " + key)
    for row in training.values():
        if row["steps_completed"] != 8000 or row["status"] != "completed":
            raise ValueError("Incomplete coverage training")
    teacher_checks = {
        "matched_initial_tensors": True,
        "matched_private_rng": True,
        "matched_sampling_weights": True,
    }
    result = {
        "status": "measured_post_D014_exploratory_validation",
        "scope": (
            "one warm-start seed; four validation scenes; "
            "paired old versus new visible distribution"
        ),
        "source_revision": comparison["source_revision"],
        "slurm_job_id": comparison["slurm_job_id"],
        "elapsed_seconds": comparison["elapsed_seconds"],
        "phases": comparison["phases"],
        "manifest_files_verified": verified,
        "comparison_sha256": sha256(root / "comparison.json"),
        "analysis_source_sha256": sha256(Path(__file__)),
        "teacher_sha256": sha256(root / "teacher.pt"),
        "cache_manifest_sha256": sha256(root / "base-cache/manifest.json"),
        "cache_arrays_sha256": sha256(root / "base-cache/arrays.npz"),
        "matched_training_checks": teacher_checks,
        "immutable_source_files_verified_before_and_after": 470,
        "models": {},
        "test_splits_evaluated": [],
        "inference_corrections": False,
    }
    for key, value in (
        ("teacher_sha256", result["teacher_sha256"]),
        ("base_cache_manifest_sha256", result["cache_manifest_sha256"]),
        ("base_cache_arrays_sha256", result["cache_arrays_sha256"]),
    ):
        if comparison[key] != value:
            raise ValueError("Original input changed: " + key)
    for name in ("anchor", "narrow", "broad"):
        folder = root / (name + "-validation")
        checkpoint = root / "teacher.pt" if name == "anchor" else root / name / "final.pt"
        plan, _, _, _ = verify_inputs(
            folder / "plan.json",
            folder / "validation-samples/predictions.json",
            dataset,
            checkpoint,
        )
        geometry = read(folder / "geometry-audit/geometry.json")
        collision = read(folder / "collision-audit.json")
        if not geometry["pixel_exact_pose_replay"] or len(plan["cases"]) != 24:
            raise ValueError("Native replay is incomplete")
        families = collision["action_families"]
        failed = []
        for path in sorted((folder / "validation-metrics").glob("*-trajectory.json")):
            report = read(path)
            selected = report["trajectory"]["selected_object"]
            ade = selected["mean_ade_over_complete_samples"]
            if (
                selected["complete_samples"] != selected["total_samples"]
                or ade is None
                or ade > 0.03125
            ):
                failed.append({"sample_id": report["sample_id"], "selected_ade": ade})
        interventions = read(folder / "world-interventions/summary.json")
        if interventions["summary"]["samples"] != 32:
            raise ValueError("Mass diagnostic is incomplete")
        row = {
            "checkpoint_sha256": sha256(checkpoint),
            "quality": read(folder / "trajectory-results.json")["quality"],
            "failed_trajectory_cases": failed,
            "geometry": geometry["summary"],
            "maximum_penetration_pixels": max(v["maximum_depth_pixels"] for v in families.values()),
            "longest_penetration_frames": max(
                v["longest_consecutive_penetration_frames"] for v in families.values()
            ),
            "collision_micro": collision["model_events_against_first_reference"],
            "action_families": families,
            "zero_force_full_horizon": read(folder / "contact-results.json")[
                "zero_force_full_horizon"
            ]["summary"],
            "passive_stationarity": read(folder / "passive-stationarity.json")["summary"],
            "clipping": read(folder / "wall-clipping.json")["summary"],
            "strata": read(folder / "uncertainty-strata.json")["action_strata"],
            "horizons": read(folder / "action-horizons.json")["summary"],
            "early_selection": read(folder / "selected-object-metrics/targets.json")["summary"],
            "mass_interventions": interventions,
            "native_pixel_exact_pose_replays": 96,
        }
        if sha256(checkpoint) != comparison["models"][name]["checkpoint_sha256"]:
            raise ValueError("Evaluated checkpoint differs")
        if name != "anchor":
            row["training"] = {
                key: value for key, value in training[name].items() if key != "provenance"
            }
            row["initialization"] = training[name]["provenance"]["initialization"]
            row["resume_check"] = read(root / name / "resume-check.json")
            row["paired_change_against_anchor"] = paired_changes(root, name)
        result["models"][name] = row
        print(
            json.dumps(
                {
                    "model": name,
                    "quality": row["quality"],
                    "penetrating": row["geometry"]["learned"][
                        "clips_with_penetration_over_one_pixel"
                    ],
                    "maximum_depth": row["maximum_penetration_pixels"],
                }
            ),
            flush=True,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(
        json.dumps(analyse(args.root, args.dataset), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
