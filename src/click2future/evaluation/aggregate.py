"""Build comparison tables from declared cases, resampling whole scene groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCES = {"video_tracking", "auxiliary_head", "control_fixture", "simulator_control"}
FAMILIES = {"trajectory", "images", "futures"}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError("Metric values must be finite numbers or null")
    return float(value)


def _mean(values):
    known = [x for x in values if x is not None]
    return float(np.mean(known)) if known else None


def _complete_mean(values):
    return _mean(values) if all(x is not None for x in values) else None


def _signature(report, family):
    """Include settings that cannot change between compared records."""
    if family == "trajectory":
        return [
            report["metric_version"],
            report.get("extractor_version"),
            report.get("contact_estimator"),
            report.get("settings"),
            report["trajectory"]["units"],
            report["trajectory"]["missing_track_policy"],
            report["collision"]["tolerance_intervals"],
            report["collision"]["pair_rule"],
            report["collision"]["missing_policy"],
        ]
    if family == "images":
        image = report["image_quality"]
        return [
            image[key]
            for key in (
                "image_metric_version",
                "metrics_requested",
                "resolution_hw",
                "preprocessing",
                "comparison",
                "frame_zero_scored",
                "missing_policy",
            )
        ]
    return [
        report.get("extractor_version"),
        report["settings"],
        report["temporal"]["temporal_metric_version"],
        report["ensemble"]["ensemble_metric_version"],
        report["ensemble"]["future_frames"],
        report["ensemble"]["missing_policy"],
    ]


def _extract(report, family, k):
    values, counts = {}, {}
    if family == "trajectory":
        for scope in ("all_objects", "selected_object"):
            data = report["trajectory"][scope]
            if data["total_samples"] != k or len(data["per_sample"]) != k:
                raise ValueError("Trajectory K differs from the declared sampling seeds")
            for metric in ("ade", "fde"):
                values[f"{scope}.mean_{metric}"] = data[f"mean_{metric}_over_complete_samples"]
            best = data["best_of_k"]
            for metric in ("min_ade", "min_fde", "fde_at_min_ade"):
                values[f"{scope}.{metric}"] = best[metric] if best else None
            values[f"{scope}.missing_track_rate"] = data["missing_track_rate"]
            values[f"{scope}.out_of_bounds_rate"] = data["out_of_bounds_rate_over_tracked"]
            for key in (
                "complete_samples",
                "total_samples",
                "expected_future_observations",
                "tracked_future_observations",
            ):
                counts[f"{scope}.{key}"] = data[key]
        records = report["collision"]["per_sample"]
        if len(records) != k:
            raise ValueError("Collision K differs from the declared sampling seeds")
        for section, keys in {
            "interval": (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "scored_bins",
                "expected_bins",
                "missing_bins",
            ),
            "event": (
                "true_positive",
                "false_positive",
                "false_negative",
                "complete_pair_timelines",
                "expected_pair_timelines",
            ),
        }.items():
            for key in keys:
                counts[f"collision.{section}.{key}"] = sum(
                    _integer(x[section][key], key) for x in records
                )
    elif family == "images":
        data = report["image_quality"]
        if data["total_samples"] != k or len(data["per_sample"]) != k:
            raise ValueError("Image K differs from the declared sampling seeds")
        for name, value in data["mean_over_complete_samples"].items():
            values[f"image.{name}"] = value
        for key in (
            "complete_samples",
            "total_samples",
            "scored_future_frames",
            "expected_future_frames",
            "missing_future_frames",
        ):
            counts[f"image.{key}"] = data[key]
    else:
        for scope in ("all_objects", "selected_object"):
            data = report["ensemble"][scope]
            if data["total_samples"] != k:
                raise ValueError("Ensemble K differs from the declared sampling seeds")
            for key in (
                "mean_pairwise_trajectory_distance",
                "mean_pairwise_endpoint_distance",
                "mean_reference_pairwise_trajectory_distance",
                "mean_reference_to_nearest_sample",
                "mean_sample_to_nearest_reference",
            ):
                values[f"{scope}.{key}"] = data[key]
            for key in (
                "complete_samples",
                "total_samples",
                "scored_sample_pairs",
                "expected_sample_pairs",
                "reference_futures",
            ):
                counts[f"ensemble.{scope}.{key}"] = data[key]
        records = report["temporal"]["per_sample"]
        if len(records) != k:
            raise ValueError("Temporal K differs from the declared sampling seeds")
        for key in (
            "object_translation_mae_over_observed",
            "background_mae_over_observed",
            "mean_acceleration_norm_over_scored",
            "warp_valid_pixel_fraction",
        ):
            values[f"temporal.{key}"] = _mean([_number(x[key]) for x in records])
        for key in (
            "scored_object_transitions",
            "expected_object_transitions",
            "scored_background_transitions",
            "expected_background_transitions",
            "scored_acceleration_windows",
            "expected_acceleration_windows",
            "excluded_reference_contact_or_force_windows",
            "missing_acceleration_windows",
        ):
            counts[f"temporal.{key}"] = sum(_integer(x[key], key) for x in records)
    return (
        {key: _number(value) for key, value in values.items()},
        {key: _integer(value, key) for key, value in counts.items()},
    )


def _collision_scores(counts):
    values = {}

    def ratio(a, b):
        return float(a / b) if b else None

    for section in ("interval", "event"):
        prefix = f"collision.{section}."
        if prefix + "true_positive" not in counts:
            continue
        tp, fp, fn = (
            counts[prefix + key] for key in ("true_positive", "false_positive", "false_negative")
        )
        values.update(
            {
                prefix + "precision": ratio(tp, tp + fp),
                prefix + "recall": ratio(tp, tp + fn),
                prefix + "f1": ratio(2 * tp, 2 * tp + fp + fn),
            }
        )
        if section == "interval":
            tn = counts[prefix + "true_negative"]
            values[prefix + "accuracy"] = ratio(tp + tn, tp + fp + fn + tn)
            values[prefix + "positive_prevalence"] = ratio(tp + fn, tp + fp + fn + tn)
    return values


def _estimate(groups, draws, confidence):
    """Null groups remain absent; report their count with every conditional estimate."""
    values = np.array([np.nan if x is None else x for x in groups], float)
    known = np.isfinite(values)
    point = float(values[known].mean()) if known.any() else None
    boot = []
    if known.sum() >= 2:
        for draw in draws:
            sample = values[draw]
            if np.isfinite(sample).any():
                boot.append(float(sample[np.isfinite(sample)].mean()))
    alpha = (1 - confidence) / 2
    bounds = np.quantile(boot, [alpha, 1 - alpha]).tolist() if boot else None
    return {
        "estimate": point,
        "ci": bounds,
        "scored_scene_groups": int(known.sum()),
        "expected_scene_groups": len(values),
        "defined_bootstrap_replicates": len(boot),
    }


def aggregate(manifest: dict, base_dir: Path, *, bootstrap=2000, seed=0, confidence=0.95):
    """Read every declared report. A missing/failed report fails the whole comparison.

    Cases are fixed across runs. Means weight scenes equally, after averaging their
    cases and then training runs. Collision scores pool counts and bootstrap whole
    scenes. Intervals condition on the supplied trained checkpoints, not new training.
    """
    _integer(bootstrap, "bootstrap", 1)
    _integer(seed, "seed")
    if isinstance(confidence, bool) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if manifest["schema_version"] != 1 or manifest["source"] not in SOURCES:
        raise ValueError("Unsupported manifest version or metric source")
    for name in ("dataset_id", "protocol_id", "suite"):
        _text(manifest[name], name)
    sampling_seeds = manifest["sampling_seeds"]
    if not sampling_seeds or len(set(sampling_seeds)) != len(sampling_seeds):
        raise ValueError("Declare distinct sampling seeds, one for every K sample")
    for value in sampling_seeds:
        _integer(value, "sampling seed")
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Declare all expected cases before aggregation")
    identities = []
    for case in cases:
        identities.append(
            tuple(_text(case[key], key) for key in ("scene_group_id", "condition_id", "sample_id"))
        )
        if "reference_ids" in case and (
            not case["reference_ids"]
            or len(set(case["reference_ids"])) != len(case["reference_ids"])
            or case["sample_id"] not in case["reference_ids"]
        ):
            raise ValueError("Reference IDs must be distinct and include the single reference")
    if len(set(identities)) != len(cases):
        raise ValueError("Duplicate evaluation case")
    condition_groups = {}
    for group, condition, _ in identities:
        if condition in condition_groups and condition_groups[condition] != group:
            raise ValueError("A condition cannot belong to several scene groups")
        condition_groups[condition] = group
    scene_groups = sorted({x[0] for x in identities})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scene_groups), (bootstrap, len(scene_groups)))
    methods, run_ids, signatures, provenance, rows = defaultdict(list), set(), {}, [], []
    families_expected = None
    reference_hashes = {}
    time_grid = None
    for run in manifest["runs"]:
        method, run_id = (_text(run[key], key) for key in ("method", "run_id"))
        if run_id in run_ids:
            raise ValueError("Run IDs must be unique")
        run_ids.add(run_id)
        control = manifest["source"] in {"control_fixture", "simulator_control"}
        if not control:
            _integer(run["training_seed"], "training seed")
            digest = run["checkpoint_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Each model run needs a checkpoint SHA-256")
        records = run["reports"]
        indices = [_integer(x["case_index"], "case_index") for x in records]
        if sorted(indices) != list(range(len(cases))):
            raise ValueError("Every run must cover every declared case exactly once")
        group_values, group_counts = defaultdict(list), defaultdict(list)
        case_rows = []
        for entry in sorted(records, key=lambda x: x["case_index"]):
            index = entry["case_index"]
            case, (group, condition, sample_id) = cases[index], identities[index]
            families = set(entry) - {"case_index"}
            if not families or not families <= FAMILIES:
                raise ValueError("Use trajectory, images, and/or futures report paths")
            if manifest["source"] == "auxiliary_head" and families != {"trajectory"}:
                raise ValueError("Auxiliary-head tables must contain head metrics only")
            if families_expected is None:
                families_expected = families
            if families != families_expected:
                raise ValueError("All cases and runs must declare the same metric families")
            values, counts = {}, {}
            video_hash = None
            single_reference_hashes = None
            for family in sorted(families):
                path = base_dir / entry[family]
                payload = path.read_bytes()
                report = json.loads(payload)
                if report["status"] != "passed":
                    raise ValueError(f"Failed report for {run_id}, case {index}: {path.name}")
                expected_source = manifest["source"]
                if family == "trajectory" and expected_source == "simulator_control":
                    expected_source = "video_tracking"
                elif family != "trajectory" and expected_source == "video_tracking":
                    expected_source = "generated_video"
                if report["source"] != expected_source or report["run_id"] != run_id:
                    raise ValueError("Report source/run ID differs from the manifest")
                if (report["scene_group_id"], report["condition_id"]) != (group, condition):
                    raise ValueError("Report scene/condition differs from the declared case")
                if family == "futures":
                    if report["reference_ids"] != case["reference_ids"]:
                        raise ValueError("Reference sets must match the declared order and IDs")
                elif report["sample_id"] != sample_id:
                    raise ValueError("Single-reference rollout differs from the declared case")
                signature = json.dumps(_signature(report, family), sort_keys=True, allow_nan=False)
                if family in signatures and signature != signatures[family]:
                    raise ValueError(f"Incompatible {family} metric settings across reports")
                signatures[family] = signature
                # The same reference bytes and video batch must be used by all metric families.
                hashes = report.get("input_sha256")
                if not hashes:
                    raise ValueError("Reports must retain their input hashes")
                times = report["timestamps_seconds"]
                if (
                    not isinstance(times, list)
                    or len(times) < 2
                    or times[0] != 0
                    or not all(_number(x) is not None for x in times)
                    or not (np.diff(times) > 0).all()
                ):
                    raise ValueError("Reports need a finite, increasing time grid starting at zero")
                if time_grid is not None and times != time_grid:
                    raise ValueError("Frame timing differs across reports")
                time_grid = times
                if isinstance(hashes, dict):
                    if "videos" not in hashes:
                        raise ValueError("RGB reports must identify their video input hash")
                    if video_hash is not None and hashes["videos"] != video_hash:
                        raise ValueError("Metric families must score the same video batch")
                    video_hash = hashes["videos"]
                    ref_hashes = {
                        key: value
                        for key, value in hashes.items()
                        if key not in {"videos", "config"}
                    }
                    if not ref_hashes:
                        raise ValueError("RGB reports must identify their reference hashes")
                    single_hashes = (
                        {
                            key: hashes[case["sample_id"] + extension]
                            for key, extension in (
                                ("reference_arrays", ".npz"),
                                ("reference_metadata", ".json"),
                            )
                        }
                        if family == "futures"
                        else {
                            key: value
                            for key, value in ref_hashes.items()
                            if key in {"reference_arrays", "reference_metadata"}
                        }
                    )
                    if single_reference_hashes is not None and (
                        single_reference_hashes != single_hashes
                    ):
                        raise ValueError("Metric families must use the same reference bytes")
                    single_reference_hashes = single_hashes
                    hash_key = (index, family)
                    if hash_key in reference_hashes and reference_hashes[hash_key] != ref_hashes:
                        raise ValueError("Reference inputs differ between compared runs")
                    reference_hashes[hash_key] = ref_hashes
                provenance.append(
                    {
                        "run_id": run_id,
                        "case_index": index,
                        "family": family,
                        "report": entry[family],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                v, c = _extract(report, family, len(sampling_seeds))
                values.update(v)
                counts.update(c)
            group_values[group].append(values)
            group_counts[group].append(counts)
            case_rows.append(
                {"case_index": index, "scene_group_id": group, "values": values, "counts": counts}
            )
        group_summary = {}
        for group in scene_groups:
            keys = group_values[group][0]
            group_summary[group] = {
                "values": {
                    key: _complete_mean([x[key] for x in group_values[group]]) for key in keys
                },
                "counts": {
                    key: sum(x[key] for x in group_counts[group]) for key in group_counts[group][0]
                },
            }
        methods[method].append((run, group_summary))
        rows.append(
            {
                "method": method,
                "run_id": run_id,
                "training_seed": run.get("training_seed"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "cases": case_rows,
            }
        )
    if not methods:
        raise ValueError("At least one run is required")
    summaries = []
    for method, runs in sorted(methods.items()):
        training_seeds = [r.get("training_seed") for r, _ in runs]
        if not control and len(set(training_seeds)) != len(training_seeds):
            raise ValueError("Repeated training seeds within a method would double-count a run")
        value_keys = next(iter(runs[0][1].values()))["values"]
        group_means = {
            key: [
                _complete_mean([data[g]["values"][key] for _, data in runs]) for g in scene_groups
            ]
            for key in value_keys
        }
        metrics = {key: _estimate(values, draws, confidence) for key, values in group_means.items()}
        count_keys = next(iter(runs[0][1].values()))["counts"]
        group_totals = [
            {key: sum(data[g]["counts"][key] for _, data in runs) for key in count_keys}
            for g in scene_groups
        ]
        totals = {key: sum(x[key] for x in group_totals) for key in count_keys}
        collision = {}
        for key, point in _collision_scores(totals).items():
            boot = []
            if len(scene_groups) >= 2:
                for draw in draws:
                    summed = {name: sum(group_totals[i][name] for i in draw) for name in count_keys}
                    value = _collision_scores(summed)[key]
                    if value is not None:
                        boot.append(value)
            alpha = (1 - confidence) / 2
            collision[key] = {
                "estimate": point,
                "ci": np.quantile(boot, [alpha, 1 - alpha]).tolist() if boot else None,
                "defined_bootstrap_replicates": len(boot),
            }
        summaries.append(
            {
                "method": method,
                "run_ids": [r["run_id"] for r, _ in runs],
                "training_seeds": training_seeds,
                "runs": len(runs),
                "cases_per_run": len(cases),
                "samples_per_case": len(sampling_seeds),
                "scene_groups": len(scene_groups),
                "metrics": metrics,
                "collision_micro": collision,
                "counts": totals,
            }
        )
    return {
        "status": "passed",
        "aggregation_version": "1",
        "dataset_id": manifest["dataset_id"],
        "protocol_id": manifest["protocol_id"],
        "suite": manifest["suite"],
        "source": manifest["source"],
        "sampling_seeds": sampling_seeds,
        "bootstrap": {
            "seed": seed,
            "replicates": bootstrap,
            "confidence": confidence,
            "unit": "scene_group_id",
            "method": "percentile",
            "scope": "conditional on supplied checkpoints; no training-seed uncertainty",
        },
        "aggregation": "equal scenes; mean cases within scene, then mean runs; micro collisions",
        "missing_policy": (
            "failed/missing reports reject table; a null case/run metric makes its scene "
            "unscorable for that metric; expected/scored scene counts retained"
        ),
        "methods": summaries,
        "runs": rows,
        "provenance": provenance,
    }


def csv_table(report):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "method",
            "run_ids",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "scored_scene_groups",
            "expected_scene_groups",
            "runs",
            "cases_per_run",
            "k",
        ]
    )
    for row in report["methods"]:
        records = {
            **row["metrics"],
            **row["collision_micro"],
            **{
                f"count.{name}": {"estimate": value, "ci": None}
                for name, value in row["counts"].items()
            },
        }
        for name, metric in records.items():
            low, high = metric["ci"] or (None, None)
            writer.writerow(
                [
                    row["method"],
                    ";".join(row["run_ids"]),
                    name,
                    metric["estimate"],
                    low,
                    high,
                    metric.get("scored_scene_groups"),
                    row["scene_groups"],
                    row["runs"],
                    row["cases_per_run"],
                    row["samples_per_case"],
                ]
            )
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    inputs = [args.manifest] + [
        args.manifest.parent / entry[family]
        for run in manifest["runs"]
        for entry in run["reports"]
        for family in FAMILIES
        if family in entry
    ]
    if args.output.resolve() == args.csv.resolve() or any(
        p.resolve() == q.resolve() for p in (args.output, args.csv) for q in inputs
    ):
        parser.error("Outputs must be distinct and must not overwrite any input")
    try:
        report = aggregate(
            manifest,
            args.manifest.parent,
            bootstrap=args.bootstrap,
            seed=args.seed,
            confidence=args.confidence,
        )
        report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Aggregation failed: {error}\n")
    for path, payload in (
        (args.output, json.dumps(report, indent=2, allow_nan=False) + "\n"),
        (args.csv, csv_table(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "methods": len(report["methods"]),
                "output": str(args.output),
                "csv": str(args.csv),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
