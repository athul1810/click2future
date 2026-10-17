"""Prepare fixed held-out cases and evaluate every declared run against them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.data.splits import load_split_manifest

from .aggregate import aggregate, csv_table
from .futures import evaluate_futures
from .images import image_quality
from .perceptual import LPIPSScorer
from .tracking import TrackerConfig
from .video import evaluate_videos


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _evaluator_sources():
    return {path.name: _sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def _seeds(values):
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in values)
        or len(set(values)) != 4
    ):
        raise ValueError("Use four distinct nonnegative integer sampling seeds")
    return values


def prepare_plan(dataset, config, *, split="test", metrics=("ssim", "lpips"), protocol=None):
    """Include every condition and hidden-physics reference in the chosen split.

    Preparing a test plan does not choose a checkpoint or tune any metric.
    Store plans outside the source submission, alongside the generated dataset.
    """
    dataset, config = Path(dataset).resolve(), Path(config).resolve()
    if split == "train":
        raise ValueError("Evaluation plans require a held-out split")
    if not metrics or len(set(metrics)) != len(metrics) or not set(metrics) <= {"ssim", "lpips"}:
        raise ValueError("Choose distinct ssim and/or lpips metrics")
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if _digest(payload) != manifest["manifest_hash"]:
        raise ValueError("Dataset manifest hash mismatch")
    comparison = yaml.safe_load(Path(protocol).read_text()) if protocol is not None else {}
    if comparison:
        settings = manifest["config"]
        if (
            settings.get("frame_count") != comparison["total_frames"]
            or [settings.get("resolution")] * 2 != comparison["resolution_hw"]
            or settings.get("fps") != comparison["fps"]
            or settings.get("force_duration", 0.25) != comparison["force_duration"]
        ):
            raise ValueError("Dataset timing/resolution differs from the comparison protocol")
    splits = load_split_manifest(dataset / "splits.json")["splits"]
    if split not in splits or not splits[split]:
        raise ValueError("The selected split is empty or unknown")
    grouped, seen = defaultdict(list), set()
    for row in manifest["samples"]:
        name = row["sample_id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Sample IDs must be safe and unique")
        seen.add(name)
        if row["scene_group_id"] not in splits.get(row["split"], []):
            raise ValueError("Dataset and scene-group splits disagree")
        if row["split"] == split:
            grouped[(row["scene_group_id"], row["condition_id"])].append(name)
    cases = []
    for (group, condition), names in sorted(grouped.items()):
        names = sorted(names)
        cases.append(
            {
                "scene_group_id": group,
                "condition_id": condition,
                "sample_id": names[0],
                "reference_ids": names,
            }
        )
    if {case["scene_group_id"] for case in cases} != set(splits[split]):
        raise ValueError("Some held-out scene groups have no samples")
    files = {
        "dataset_manifest": str(manifest_path),
        "splits": str(dataset / "splits.json"),
        "config": str(config),
    }
    if protocol is not None:
        files["comparison_protocol"] = str(Path(protocol).resolve())
    for case in cases:
        for name in case["reference_ids"]:
            for suffix in (".json", ".npz"):
                files[name + suffix] = str(dataset / (name + suffix))
    plan = {
        "schema_version": 1,
        "dataset": str(dataset),
        "config": str(config),
        "split": split,
        "sampling_seeds": _seeds(comparison.get("sampling_seeds", [0, 1, 2, 3])),
        "comparison_protocol": comparison,
        "evaluator_source_sha256": _evaluator_sources(),
        "image_metrics": list(metrics),
        "cases": cases,
        "files": files,
        "input_sha256": {key: _sha(path) for key, path in files.items()},
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def read_plan(plan_path):
    """Verify a frozen plan, its evaluator revision and every referenced input."""
    plan = json.loads(Path(plan_path).read_text())
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["schema_version"] != 1 or _digest(payload) != plan["plan_sha256"]:
        raise ValueError("Evaluation plan changed; prepare a new plan explicitly")
    if plan["evaluator_source_sha256"] != _evaluator_sources():
        raise ValueError("Evaluator source changed after preparing the plan")
    for key, path in plan["files"].items():
        if _sha(path) != plan["input_sha256"][key]:
            raise ValueError(f"Frozen evaluation input changed: {key}")
    _seeds(plan["sampling_seeds"])
    return plan


def run_suite(plan_path, predictions_path, output_dir, *, bootstrap=None):
    """Score pre-generated videos; never train, select, or silently drop a case.

    Model runs need a local checkpoint for its hash. Simulator controls have a
    separate source label and cannot use E0-E4 method names.
    """
    plan_path, predictions_path, output_dir = map(Path, (plan_path, predictions_path, output_dir))
    plan = read_plan(plan_path)
    predictions = json.loads(predictions_path.read_text())
    source = predictions["source"]
    if source not in {"generated_video", "simulator_control"}:
        raise ValueError("Declare generated_video or simulator_control source")
    if _seeds(predictions["sampling_seeds"]) != _seeds(plan["sampling_seeds"]):
        raise ValueError("Sampling seeds differ from the evaluation plan")
    if not predictions["runs"]:
        raise ValueError("At least one run is required")
    expected = {case["sample_id"] for case in plan["cases"]}
    prepared, seen = [], set()
    for run in predictions["runs"]:
        name = run["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Run IDs must be safe and unique")
        seen.add(name)
        if set(run["videos"]) != expected:
            raise ValueError("Every run must cover every frozen case exactly once")
        model_method = run["method"] in {"E0", "E1", "E2", "E3", "E4"}
        if model_method != (source == "generated_video"):
            raise ValueError("Use E0-E4 only for model-generated video")
        record = {"method": run["method"], "run_id": name, "reports": []}
        if source == "generated_video":
            seed = run["training_seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Training seed must be a nonnegative integer")
            record.update(
                training_seed=seed,
                checkpoint_sha256=_sha(predictions_path.parent / run["checkpoint"]),
            )
            if (
                "checkpoint_sha256" in run
                and run["checkpoint_sha256"] != record["checkpoint_sha256"]
            ):
                raise ValueError("Checkpoint changed after sampling")
            if "training_steps" in run:
                steps = run["training_steps"]
                fixture = run.get("untrained_fixture")
                if (
                    isinstance(steps, bool)
                    or not isinstance(steps, int)
                    or steps < 0
                    or not isinstance(fixture, bool)
                    or fixture != (steps == 0)
                ):
                    raise ValueError("Training steps and untrained-fixture label disagree")
                record.update(training_steps=steps, untrained_fixture=fixture)
        videos = {
            key: (predictions_path.parent / path).resolve() for key, path in run["videos"].items()
        }
        if "video_sha256" in run and set(run["video_sha256"]) != expected:
            raise ValueError("Video hashes must cover every frozen case")
        for sample_id, path in videos.items():
            if not path.is_file():
                raise ValueError(f"Missing prediction file: {path.name}")
            if "video_sha256" in run and _sha(path) != run["video_sha256"][sample_id]:
                raise ValueError("Video changed after sampling")
        prepared.append((record, videos))
    config = yaml.safe_load(Path(plan["config"]).read_text())
    tracker = TrackerConfig(**config["tracker"])
    contacts = {
        key: config[key]
        for key in ("max_gap_pixels", "motion_window", "motion_threshold", "motion_rule")
        if key in config
    }
    # Initialize once, without downloading weights or changing sampling RNG.
    scorer = LPIPSScorer() if "lpips" in plan["image_metrics"] else None
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "source": "video_tracking" if source == "generated_video" else source,
        "dataset_id": plan["input_sha256"]["dataset_manifest"],
        "protocol_id": plan["plan_sha256"],
        "suite": plan["split"],
        "sampling_seeds": plan["sampling_seeds"],
        "cases": plan["cases"],
        "runs": [],
    }
    try:
        for record, paths in prepared:
            for index, case in enumerate(plan["cases"]):
                refs = [load_rollout(plan["dataset"], name) for name in case["reference_ids"]]
                reference = refs[0]
                if any(
                    reference["scene_spec"][key] != case[key]
                    for key in ("scene_group_id", "condition_id")
                ):
                    raise ValueError("Reference identity differs from the frozen case")
                if list(reference["frames"].shape[1:3]) != config[
                    "resolution_hw"
                ] or not np.allclose(
                    np.diff(reference["timestamps"]), 1 / config["fps"], atol=1e-8, rtol=0
                ):
                    raise ValueError("Reference resolution/timing differs from the metric config")
                video_path = paths[case["sample_id"]]
                with np.load(video_path, allow_pickle=False) as archive:
                    videos = archive["videos"]
                    valid = archive["frame_valid"] if "frame_valid" in archive else None
                if len(videos) != len(plan["sampling_seeds"]):
                    raise ValueError("Prediction K differs from the frozen sampling seeds")
                hashes = {
                    "videos": _sha(video_path),
                    "config": plan["input_sha256"]["config"],
                    "reference_arrays": plan["input_sha256"][case["sample_id"] + ".npz"],
                    "reference_metadata": plan["input_sha256"][case["sample_id"] + ".json"],
                }
                trajectory, _ = evaluate_videos(
                    reference,
                    videos,
                    tracker=tracker,
                    event_tolerance=config["event_tolerance_intervals"],
                    run_id=record["run_id"],
                    frame_valid=valid,
                    **contacts,
                )
                trajectory.update(sample_id=case["sample_id"], input_sha256=hashes)
                images = {
                    "status": "passed",
                    "source": source,
                    "run_id": record["run_id"],
                    "sample_id": case["sample_id"],
                    "scene_group_id": case["scene_group_id"],
                    "condition_id": case["condition_id"],
                    "timestamps_seconds": reference["timestamps"].tolist(),
                    "input_sha256": hashes,
                    "image_quality": image_quality(
                        videos,
                        reference["frames"],
                        metrics=plan["image_metrics"],
                        frame_valid=valid,
                        scorer=scorer,
                    ),
                }
                futures = evaluate_futures(
                    refs,
                    videos,
                    tracker=tracker,
                    contact_settings=contacts,
                    source=source,
                    run_id=record["run_id"],
                    frame_valid=valid,
                )
                future_hashes = {"videos": hashes["videos"], "config": hashes["config"]}
                for name in case["reference_ids"]:
                    for suffix in (".json", ".npz"):
                        future_hashes[name + suffix] = plan["input_sha256"][name + suffix]
                futures.update(reference_ids=case["reference_ids"], input_sha256=future_hashes)
                entry = {"case_index": index}
                for family, report in (
                    ("trajectory", trajectory),
                    ("images", images),
                    ("futures", futures),
                ):
                    name = f"{record['run_id']}-{index:05d}-{family}.json"
                    _write(output_dir / name, report)
                    entry[family] = name
                record["reports"].append(entry)
            manifest["runs"].append(record)
        _write(output_dir / "comparison.json", manifest)
        settings = plan["comparison_protocol"].get("bootstrap", {})
        report = aggregate(
            manifest,
            output_dir,
            bootstrap=bootstrap if bootstrap is not None else settings.get("replicates", 2000),
            confidence=settings.get("confidence", 0.95),
            seed=settings.get("seed", 0),
        )
        report["evaluation_plan_sha256"] = plan["plan_sha256"]
        report["predictions_manifest_sha256"] = _sha(predictions_path)
        report["untrained_fixture_runs"] = [
            run["run_id"] for run in manifest["runs"] if run.get("untrained_fixture")
        ]
        _write(output_dir / "summary.json", report)
        (output_dir / "summary.csv").write_text(csv_table(report))
        return report
    except Exception as error:
        _write(output_dir / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--config", type=Path, default=Path("configs/evaluation/pixel.yaml"))
    prepare.add_argument(
        "--protocol", type=Path, default=Path("configs/evaluation/comparison.yaml")
    )
    prepare.add_argument("--split", default="test")
    prepare.add_argument("--metrics", nargs="+", default=["ssim", "lpips"])
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            plan = prepare_plan(
                args.dataset,
                args.config,
                split=args.split,
                metrics=args.metrics,
                protocol=args.protocol,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(json.dumps(plan, indent=2) + "\n")
            print(json.dumps({"status": "passed", "cases": len(plan["cases"])}))
        else:
            result = run_suite(args.plan, args.predictions, args.output_dir)
            print(json.dumps({"status": result["status"], "methods": len(result["methods"])}))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
