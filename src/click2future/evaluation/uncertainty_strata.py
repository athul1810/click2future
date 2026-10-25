"""Report every frozen E5 action family without hiding incomplete ensembles."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.training.train_cvae import sha256

from .actions import action_metrics
from .aggregate import aggregate
from .suite import _write
from .tracking import TRACKER_VERSION, TrackerConfig, track_video
from .trajectory_audit import verify_inputs


def zero_horizon(reference, videos):
    """Supplement the unchanged 0.5s check with a full-horizon stationary control."""
    if (
        np.any(reference["force_xy"])
        or reference["collision_onset"].any()
        or reference["wall_contact_onset"].any()
        or not np.allclose(reference["positions"], reference["positions"][:1], atol=1e-8, rtol=0)
        or not np.array_equal(
            reference["frames"],
            np.repeat(reference["frames"][:1], len(reference["frames"]), axis=0),
        )
    ):
        raise ValueError("Full-horizon zero controls require a verified stationary reference")
    videos = np.asarray(videos)
    if (
        videos.dtype != np.uint8
        or videos.ndim != 5
        or len(videos) < 1
        or videos.shape[1:] != reference["frames"].shape
        or not np.all(videos[:, 0] == reference["frames"][0])
    ):
        raise ValueError("Provide matching videos with exact frame zero")
    tracks = [
        track_video(
            video, reference["instance_masks"][0], reference["object_valid"], TrackerConfig()
        )
        for video in videos
    ]
    report = action_metrics(
        np.stack([track.positions for track in tracks]),
        np.stack([track.tracked for track in tracks]),
        reference["object_valid"],
        selected_object=reference["selected_object_id"],
        force_xy=[0, 0],
        end_frame=len(reference["frames"]) - 1,
        motion_threshold=1 / reference["frames"].shape[2],
    )
    report.update(
        extractor_version=TRACKER_VERSION,
        scope=(
            "maximum excursion of any valid object across the full stationary clip; "
            "separate from the original initial-window result"
        ),
    )
    return report


def action_family(case):
    prefix = case["scene_group_id"] + "-"
    if not case["condition_id"].startswith(prefix):
        raise ValueError("Condition must retain its scene-group identity")
    return case["condition_id"][len(prefix) :]


def stratify(comparison_path, *, bootstrap=2000):
    comparison_path = Path(comparison_path)
    manifest = json.loads(comparison_path.read_text())
    if (
        manifest["suite"] != "validation"
        or manifest["source"] != "video_tracking"
        or len(manifest["runs"]) != 1
        or manifest["runs"][0]["method"] not in {"E5", "E6", "E7"}
    ):
        raise ValueError("This diagnostic is for saved validation RGB reports only")
    # Validate the entire historical comparison first. Strata retain all cases,
    # including failures, and use exactly the same scene-bootstrap implementation.
    total = aggregate(manifest, comparison_path.parent, bootstrap=bootstrap)
    families = sorted({action_family(case) for case in manifest["cases"]})
    strata = {}
    for family in families:
        indices = [i for i, case in enumerate(manifest["cases"]) if action_family(case) == family]
        subset = copy.deepcopy(manifest)
        subset["cases"] = [manifest["cases"][i] for i in indices]
        if len({case["scene_group_id"] for case in subset["cases"]}) != len(indices):
            raise ValueError("Use one condition per action family and independent scene")
        for run in subset["runs"]:
            entries = {row["case_index"]: row for row in run["reports"]}
            run["reports"] = [{**entries[i], "case_index": j} for j, i in enumerate(indices)]
        result = aggregate(subset, comparison_path.parent, bootstrap=bootstrap)
        methods = []
        for run, method in zip(subset["runs"], result["methods"], strict=True):
            frequencies, references, variances, complete, expected = [], [], [], 0, 0
            for entry in run["reports"]:
                report = json.loads((comparison_path.parent / entry["futures"]).read_text())
                if len(report["collision_frequency"]["per_pair"]) != 1:
                    raise ValueError("E5 frequency strata require the declared two-object pair")
                for pair in report["collision_frequency"]["per_pair"]:
                    complete += pair["complete_sample_timelines"]
                    expected += pair["expected_sample_timelines"]
                    known = pair["complete_sample_timelines"] == pair["expected_sample_timelines"]
                    frequencies.append(pair["sample_any_event_rate"] if known else None)
                    references.append(pair["reference_any_event_rate"])
                    variances.append(pair["sample_bernoulli_variance"] if known else None)

            def strict_mean(values):
                return (
                    float(np.mean(values))
                    if values and all(x is not None for x in values)
                    else None
                )

            methods.append(
                {
                    "method": method["method"],
                    "run_ids": method["run_ids"],
                    "scene_groups": method["scene_groups"],
                    "metrics": {
                        key: value
                        for key, value in method["metrics"].items()
                        if "pairwise" in key or "nearest" in key or "missing_track" in key
                    },
                    "counts": {
                        key: value
                        for key, value in method["counts"].items()
                        if key.startswith("ensemble.")
                    },
                    "collision_frequency": {
                        "complete_sample_pair_timelines": complete,
                        "expected_sample_pair_timelines": expected,
                        "mean_generated_any_contact_rate_if_complete": strict_mean(frequencies),
                        "mean_reference_any_contact_rate": strict_mean(references),
                        "mean_generated_bernoulli_variance_if_complete": strict_mean(variances),
                        "scope": (
                            "descriptive finite-ensemble frequency; not calibrated probability"
                        ),
                    },
                }
            )
        strata[family] = {"case_count": len(indices), "methods": methods}
    if sum(row["case_count"] for row in strata.values()) != len(manifest["cases"]):
        raise ValueError("Every case must appear in exactly one action stratum")
    return {
        "status": "measured_validation_uncertainty_strata",
        "source": "saved generated-RGB reports; candidate tracker version remains unchanged",
        "comparison_sha256": sha256(comparison_path),
        "source_sha256": {
            "uncertainty_strata.py": sha256(Path(__file__)),
            "aggregate.py": sha256(Path(__file__).with_name("aggregate.py")),
        },
        "report_sha256": total["provenance"],
        "total_case_count": len(manifest["cases"]),
        "action_strata": strata,
        "scope": (
            "All action families shown. Conditional spread uses complete sample pairs; coverage "
            "requires all K trajectories. Nulls are retained. Zero-force reference spread can "
            "be undefined when only one reference exists. This is not calibrated uncertainty."
        ),
        "test_splits_evaluated": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("plan", "predictions", "dataset", "checkpoint"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Keep historical diagnostics; use a fresh output path")
    report = stratify(args.comparison)
    extra = (args.plan, args.predictions, args.dataset, args.checkpoint)
    if any(extra):
        if not all(extra):
            parser.error("Full-horizon controls need plan, predictions, dataset and checkpoint")
        plan, run, paths, hashes = verify_inputs(*extra)
        comparison = json.loads(args.comparison.read_text())
        if (
            comparison["protocol_id"] != plan["plan_sha256"]
            or comparison["runs"][0]["checkpoint_sha256"] != run["checkpoint_sha256"]
        ):
            raise ValueError("Stationarity inputs and saved comparison disagree")
        rows = []
        for case in plan["cases"]:
            if action_family(case) != "zero":
                continue
            ref = load_rollout(args.dataset, case["sample_id"])
            with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
                videos = archive["videos"]
            copied = np.repeat(ref["frames"][None], len(videos), axis=0)
            rows.append(
                {
                    "sample_id": case["sample_id"],
                    "scene_group_id": case["scene_group_id"],
                    "model": zero_horizon(ref, videos),
                    "static_control": zero_horizon(ref, copied),
                }
            )
        if not rows:
            raise ValueError("No zero-force conditions in the frozen plan")
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise ValueError("Stationarity input changed during evaluation")
        report["zero_force_full_horizon"] = {
            "cases": rows,
            "summary": {
                family: {
                    key: sum(row[family][key] for row in rows)
                    for key in (
                        "total_samples",
                        "complete_samples",
                        "missing_samples",
                        "drifting_samples",
                    )
                }
                for family in ("model", "static_control")
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(json.dumps({"cases": report["total_case_count"], "strata": len(report["action_strata"])}))


if __name__ == "__main__":
    main()
