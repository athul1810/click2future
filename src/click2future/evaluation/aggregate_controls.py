"""Check table export on simulator futures and frozen videos, without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from click2future.data import generate_dataset, load_rollout

from .aggregate import main as aggregate_main
from .futures import evaluate_futures
from .images import image_quality
from .tracking import TrackerConfig
from .video import evaluate_videos


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output_dir: Path, config_path: Path):
    if output_dir.exists():
        raise ValueError("Use a fresh output directory so previous controls are not overwritten")
    output_dir.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_text())
    data_dir = output_dir / "reference"
    dataset = generate_dataset(
        data_dir,
        group_count=4,
        variants_per_group=4,
        seed=92000,
        config={"resolution": 64, "frame_count": 16, "fps": 8, "physics_hz": 240},
    )["manifest"]
    settings = {key: config[key] for key in ("max_gap_pixels", "motion_threshold", "motion_window")}
    if "motion_rule" in config:
        settings["motion_rule"] = config["motion_rule"]
    tracker = TrackerConfig(**config["tracker"])
    manifest = {
        "schema_version": 1,
        "source": "simulator_control",
        "dataset_id": dataset["manifest_hash"],
        "protocol_id": "aggregation-controls-1",
        "suite": "metric_correctness_controls",
        "sampling_seeds": [0, 1, 2, 3],
        "cases": [],
        "runs": [
            {"method": method, "run_id": method, "reports": []}
            for method in ("simulator-variants", "frozen-video")
        ],
    }
    for index in range(4):
        ids = [x["sample_id"] for x in dataset["samples"][index * 4 : (index + 1) * 4]]
        refs = [load_rollout(data_dir, sample_id) for sample_id in ids]
        base = refs[0]
        manifest["cases"].append(
            {
                "scene_group_id": base["scene_spec"]["scene_group_id"],
                "condition_id": base["scene_spec"]["condition_id"],
                "sample_id": ids[0],
                "reference_ids": ids,
            }
        )
        for run in manifest["runs"]:
            videos = np.stack([ref["frames"] for ref in refs])
            if run["method"] == "frozen-video":
                videos[:] = videos[:, :1]
            video_path = output_dir / f"{run['run_id']}-{index}.npz"
            np.savez_compressed(video_path, videos=videos)
            common = {
                "status": "passed",
                "source": "simulator_control",
                "run_id": run["run_id"],
                **{
                    key: manifest["cases"][-1][key]
                    for key in ("scene_group_id", "condition_id", "sample_id")
                },
                "timestamps_seconds": base["timestamps"].tolist(),
            }
            hashes = {
                "videos": _hash(video_path),
                "reference_arrays": _hash(data_dir / (ids[0] + ".npz")),
                "reference_metadata": _hash(data_dir / (ids[0] + ".json")),
            }
            trajectory, _ = evaluate_videos(
                base,
                videos,
                tracker=tracker,
                run_id=run["run_id"],
                event_tolerance=config["event_tolerance_intervals"],
                **settings,
            )
            trajectory.update(
                sample_id=ids[0], input_sha256={**hashes, "config": _hash(config_path)}
            )
            images = {
                **common,
                "input_sha256": hashes,
                "image_quality": image_quality(videos, base["frames"], metrics=("ssim",)),
            }
            futures = evaluate_futures(
                refs,
                videos,
                tracker=tracker,
                contact_settings=settings,
                source="simulator_control",
                run_id=run["run_id"],
            )
            futures.update(
                reference_ids=ids,
                input_sha256={
                    "videos": _hash(video_path),
                    "config": _hash(config_path),
                    **{
                        sample_id + ext: _hash(data_dir / (sample_id + ext))
                        for sample_id in ids
                        for ext in (".npz", ".json")
                    },
                },
            )
            entry = {"case_index": index}
            for family, report in (
                ("trajectory", trajectory),
                ("images", images),
                ("futures", futures),
            ):
                path = output_dir / f"{run['run_id']}-{index}-{family}.json"
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                entry[family] = path.name
            run["reports"].append(entry)
    path = output_dir / "comparison.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    aggregate_main(
        [
            "--manifest",
            str(path),
            "--output",
            str(output_dir / "summary.json"),
            "--csv",
            str(output_dir / "table.csv"),
            "--seed",
            "7",
        ]
    )
    report = json.loads((output_dir / "summary.json").read_text())
    rows = {row["method"]: row for row in report["methods"]}
    moving, frozen = rows["simulator-variants"], rows["frozen-video"]
    if not (
        moving["metrics"]["all_objects.mean_ade"]["estimate"]
        < frozen["metrics"]["all_objects.mean_ade"]["estimate"]
    ):
        raise RuntimeError("Simulator futures must improve trajectory error over frozen output")
    if frozen["metrics"]["all_objects.mean_pairwise_trajectory_distance"]["estimate"] != 0:
        raise RuntimeError("Frozen duplicates must have zero trajectory spread")
    return {
        "status": "passed",
        "purpose": "simulator metric checks, not trained-model results",
        "scene_groups": 4,
        "samples_per_case": 4,
        "report_inputs": 24,
        "source_sha256": {
            name: _hash(Path(__file__).with_name(name))
            for name in ("aggregate.py", "aggregate_controls.py")
        },
        "manifest_sha256": _hash(path),
        "methods": [
            {
                "method": row["method"],
                "metrics": {
                    key: row["metrics"][key]
                    for key in (
                        "all_objects.mean_ade",
                        "all_objects.min_ade",
                        "image.ssim",
                        "all_objects.mean_pairwise_trajectory_distance",
                    )
                },
            }
            for row in rows.values()
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", default=Path("configs/evaluation/pixel.yaml"), type=Path)
    parser.add_argument("--record", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.record.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("Keep the small check record outside the generated-data directory")
    result = check(args.output_dir, args.config)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "record": str(args.record)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
