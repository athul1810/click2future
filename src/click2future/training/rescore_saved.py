"""Re-score unchanged saved pixels with an explicitly versioned extractor."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.evaluation.tracking import TRACKER_VERSION, TrackerConfig

from .compare_representation import score_samples


def digest(path, *, text=False):
    payload = Path(path).read_bytes()
    if text:
        payload = payload.replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def rescore(spec, *, root):
    root = Path(root)
    expected_hashes = spec["reference_sha256"]
    for extension in ("npz", "json"):
        source = root / spec["dataset"] / (spec["sample_id"] + "." + extension)
        if digest(source) != expected_hashes[extension]:
            raise ValueError("Reference differs from the recorded experiment")
    source_records = {}
    for name, expected in spec["historical_records_sha256"].items():
        if digest(root / name, text=True) != expected:
            raise ValueError("Historical result record changed")
        source_records[name] = expected
    reference = load_rollout(root / spec["dataset"], spec["sample_id"])
    records, ids = [], set()
    for row in spec["videos"]:
        if row["id"] in ids:
            raise ValueError("Saved video IDs must be distinct")
        ids.add(row["id"])
        source = root / row["file"]
        if digest(source) != row["file_sha256"]:
            raise ValueError("Saved video archive differs from the recorded artifact")
        with np.load(source, allow_pickle=False) as archive:
            videos = archive["videos"]
        if (
            videos.dtype != np.uint8
            or videos.shape != (len(row["seeds"]), *reference["frames"].shape)
            or not np.all(videos[:, 0] == reference["frames"][0])
            or not row["seeds"]
            or len(set(row["seeds"])) != len(row["seeds"])
        ):
            raise ValueError("Saved pixels must match the original clip and ordered seeds")
        scores = score_samples(reference, videos)
        all_objects, selected = scores["all_objects"], scores["selected_object"]
        ade = selected["mean_ade_over_complete_samples"]
        gate = (
            all_objects["tracked_future_observations"] / all_objects["expected_future_observations"]
            >= 0.9
            and selected["complete_samples"] == len(row["seeds"])
            and ade is not None
            and ade <= 0.03125
        )
        records.append(
            {
                **row,
                "pixels_sha256": hashlib.sha256(videos.tobytes()).hexdigest(),
                "shape": list(videos.shape),
                "scores": scores,
                "declared_training_gate_passed": gate,
            }
        )
    if not records:
        raise ValueError("Provide at least one saved artifact")
    return {
        "status": "saved_pixel_rescore",
        "scope": "training-clip diagnostic only; no retraining or held-out action success",
        "extractor_version": TRACKER_VERSION,
        "extractor_adoption": "version 2; calibration reviewed",
        "tracker_settings": asdict(TrackerConfig()),
        "source_sha256": {
            "rescore_saved.py": digest(Path(__file__)),
            "compare_representation.py": digest(
                Path(__file__).with_name("compare_representation.py")
            ),
            "tracking.py": digest(Path(__file__).parents[1] / "evaluation/tracking.py"),
            "metrics.py": digest(Path(__file__).parents[1] / "evaluation/metrics.py"),
        },
        "reference": {
            "dataset": spec["dataset"],
            "sample_id": spec["sample_id"],
            "files_sha256": expected_hashes,
        },
        "historical_records_sha256": source_records,
        "historical_hash_policy": "UTF-8 bytes with CRLF normalized to LF; JSON is not rewritten",
        "quality_gate": {
            "tracking_fraction_min": 0.9,
            "selected_ade_max": 0.03125,
            "all_selected_samples_must_be_complete": True,
            "scope": "existing tiny-training gate, not physical validity",
        },
        "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh report path; keep historical scores unchanged")
    report = rescore(json.loads(args.spec.read_text()), root=args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"extractor": TRACKER_VERSION, "records": len(report["records"])}))


if __name__ == "__main__":
    main()
