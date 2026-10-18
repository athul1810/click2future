"""Collect all declared contact candidates; retain failed gates and denominators."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(evaluation, passive, geometry, zero):
    paths = {
        "quality": evaluation / "trajectory-results.json",
        "summary": evaluation / "validation-metrics/summary.json",
        "actions": evaluation / "validation-metrics/actions.json",
        "selection": evaluation / "selected-object-metrics/targets.json",
        "passive": passive,
    }
    raw = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    action = raw["actions"]["summary"]["methods"][0]
    selection = raw["selection"]["summary"]["methods"][0]
    summary = raw["summary"]["methods"][0]
    if raw["quality"]["test_splits_evaluated"] or raw["quality"]["quality"]["case_count"] != 24:
        raise ValueError("Require every frozen validation case and no final tests")
    return {
        "checkpoint_sha256": raw["quality"]["checkpoint_sha256"],
        "training_steps": raw["quality"]["training_steps"],
        "quality": raw["quality"]["quality"],
        "early_action_counts": action["counts"],
        "early_reversal": action["metrics"]["reversal.correct_force_sign_rate_over_complete"],
        "early_selection_counts": selection["counts"],
        "early_selection": selection["metrics"]["selection.correct_target_rate_over_complete"],
        "full_horizon_zero": zero,
        "full_horizon_passive": raw["passive"]["summary"],
        "full_horizon_passive_definition": {
            "metric_version": raw["passive"].get("passive_metric_version", "1"),
            "motion_rule": raw["passive"].get(
                "motion_rule", "maximum excursion > motion_threshold"
            ),
        },
        "geometry": geometry,
        "image_ssim": summary["metrics"]["image.ssim"],
        "all_original_metrics": summary["metrics"],
        "collision_metrics": summary["collision_micro"],
        "original_summary_counts": summary["counts"],
        "raw_input_sha256": {str(path): digest(path) for path in paths.values()},
        "promoted": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/tc2-contact-20261018"))
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/trajectory-handoff/trajectory-20261017-validation-r2"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline_path = Path("docs/results/contact-uncertainty-audit.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    root = args.artifacts
    records = {
        "E5": collect(
            args.baseline,
            root / "passive-E5.json",
            baseline["geometry"]["summary"],
            baseline["uncertainty"]["zero_force_full_horizon"]["summary"],
        )
    }
    for label, folder, passive in (
        ("E6-first", "contact-20261018", root / "passive-E6-first.json"),
        ("E6-token", "contact-token-20261018", root / "passive-E6-token.json"),
        (
            "E6-grid",
            "contact-grid-20261018",
            root / "contact-grid-20261018-validation/passive-stationarity.json",
        ),
    ):
        evaluation = root / (folder + "-validation")
        contact_path = evaluation / "contact-results.json"
        contact = json.loads(contact_path.read_text(encoding="utf-8"))
        records[label] = collect(
            evaluation,
            passive,
            contact["geometry"],
            contact["zero_force_full_horizon"]["summary"],
        )
        training_path = root / folder / "summary.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        provenance = training["provenance"]
        records[label].update(
            source_revision=training["provenance"]["source_revision"],
            slurm_job_id=training["slurm_job_id"],
            config=training["config"],
            training_status=training["status"],
            gpu=training["gpu"],
            peak_cuda_allocated_bytes=training["peak_cuda_allocated_bytes"],
            process_seconds_including_preparation=training["elapsed_seconds"],
            data_identity={
                key: provenance[key]
                for key in (
                    "dataset_manifest_sha256",
                    "split_manifest_sha256",
                    "train_validation_content_sha256",
                    "calibration",
                    "angular_labels_sha256",
                    "geometry_sha256",
                    "source_sha256",
                )
            },
        )
        if "training_supplement" in provenance:
            supplement = provenance["training_supplement"]
            records[label]["data_identity"]["training_supplement"] = {
                "manifest_hash": supplement["manifest"]["manifest_hash"],
                "sample_count": supplement["manifest"]["sample_count"],
                "recipe": supplement["manifest"]["recipe"],
                "labels": supplement["labels"],
                "combined_training_examples": provenance["combined_training_examples"],
            }
        records[label]["raw_input_sha256"].update(
            {str(path): digest(path) for path in (contact_path, training_path)}
        )
        strata_path = evaluation / "uncertainty-strata.json"
        records[label]["uncertainty_strata"] = json.loads(strata_path.read_text(encoding="utf-8"))
        records[label]["raw_input_sha256"][str(strata_path)] = digest(strata_path)
    records["E5"]["uncertainty_strata"] = baseline["uncertainty"]
    interactive_path = root / "user-replay-final/checks.json"
    cache_path = root / "user-replay-final/cache-check.json"
    result = {
        "scope": "one training seed per declared package; four validation scenes; no sota claim",
        "records": records,
        "interactive_request": json.loads(interactive_path.read_text(encoding="utf-8")),
        "historical_cache_check": json.loads(cache_path.read_text(encoding="utf-8")),
        "additional_input_sha256": {
            str(path): digest(path)
            for path in (baseline_path, interactive_path, cache_path, Path(__file__))
        },
        "test_splits_evaluated": [],
        "no_sampling_correction": True,
        "limitations": [
            "four samples do not establish calibrated physical uncertainty",
            "training packages change several factors; not isolated loss ablations",
            "extra passive and interactive diagnostics were added after baseline inspection",
            "boundary violations are not claimed: oracle calibration fails that metric",
            "tracker v2 remains pending independent review in pr 42",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, record in records.items():
        print(
            label,
            {
                "complete_clips": record["quality"]["complete_all_object_clips"],
                "penetrating_clips": record["geometry"]["learned"][
                    "clips_with_penetration_over_one_pixel"
                ],
                "passive_drift": record["full_horizon_passive"]["drifting_samples"],
                "contact_f1": record["collision_metrics"]["collision.event.f1"]["estimate"],
            },
        )


if __name__ == "__main__":
    main()
