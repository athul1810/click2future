"""Inspect unselected-object drift in a saved selection probe.

For each case and seed, report both objects' displacement, the unselected
object's force projection and first above-one-pixel frame, and minimum centre
gap. Count selected identities in training/validation metadata for comparison.
Frame-zero masks align identities; no videos are resampled."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from click2future.data import load_batch
from click2future.evaluation.tracking import TrackerConfig, track_video

PIXEL = 1 / 64


def _find(record, key):
    if isinstance(record, dict):
        if key in record:
            return record[key]
        for value in record.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def selection_counts(dataset):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    counts = Counter()
    for row in manifest["samples"]:
        if row["split"] in ("train", "validation"):
            meta = json.loads((dataset / row["metadata"]).read_text())
            counts[(row["split"], row["action_variant"], _find(meta, "selected_object_id"))] += 1
    table = {}
    for (split, variant, selected), count in sorted(counts.items()):
        table.setdefault(split, {}).setdefault(variant, {})[str(selected)] = count
    return table


def drift(evaluation):
    evaluation = Path(evaluation)
    plan = json.loads((evaluation / "target-controls/plan.json").read_text())
    samples = evaluation / "selected-object-samples"
    run = json.loads((samples / "predictions.json").read_text())["runs"][0]
    references = evaluation / "target-controls/references"
    rows = []
    for case in plan["cases"]:
        base = load_batch(references, [case["sample_id"]])
        selected = int(base["selected_object_id"][0])
        other = 1 - selected
        force = base["force_xy"][0].astype(np.float64)
        direction = force / np.linalg.norm(force)
        videos = np.load(samples / run["videos"][case["sample_id"]])["videos"]
        for seed, video in enumerate(videos):
            tracks = track_video(
                video, base["instance_masks"][0, 0], base["object_valid"][0], TrackerConfig()
            )
            row = {"sample_id": case["sample_id"], "seed": seed, "selected": selected}
            if not tracks.tracked[:, [selected, other]].all():
                rows.append({**row, "complete": False})
                continue
            positions = tracks.positions.astype(np.float64)
            moved = positions[:, other] - positions[0, other]
            distance = np.linalg.norm(moved, axis=-1)
            above = np.flatnonzero(distance > PIXEL)
            gap = np.linalg.norm(positions[:, selected] - positions[:, other], axis=-1)
            rows.append(
                {
                    **row,
                    "complete": True,
                    "selected_final": float(
                        np.linalg.norm(positions[-1, selected] - positions[0, selected])
                    ),
                    "other_final": float(distance[-1]),
                    "other_final_along_force": float(moved[-1] @ direction),
                    "other_first_frame_above_one_pixel": int(above[0]) if len(above) else None,
                    "minimum_centre_gap": float(gap.min()),
                }
            )
    summary = {}
    for selected in (0, 1):
        done = [r for r in rows if r["selected"] == selected and r["complete"]]
        summary[f"selected_object_{selected}"] = {
            "complete_samples": len(done),
            "samples": sum(r["selected"] == selected for r in rows),
            "other_moving_over_one_pixel": sum(r["other_final"] > PIXEL for r in done),
            "other_moving_along_force": sum(
                r["other_final"] > PIXEL and r["other_final_along_force"] > 0 for r in done
            ),
            "other_final_range": [
                min((r["other_final"] for r in done), default=None),
                max((r["other_final"] for r in done), default=None),
            ],
            "minimum_centre_gap": min((r["minimum_centre_gap"] for r in done), default=None),
        }
    return {
        "run_id": run["run_id"],
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "summary": summary,
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="cvae_evaluation output")
    parser.add_argument("--dataset", type=Path, required=True, help="pilot dataset folder")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = {
        "status": "measured_diagnostic",
        "units": "normalized image coordinates; one pixel = 1/64",
        "selection_probe": drift(args.evaluation),
        "selected_object_by_action": selection_counts(args.dataset),
        "scope": "Validation diagnostic of saved samples and clip metadata; no test split.",
    }
    if args.output:
        if args.output.exists():
            raise ValueError("Use a fresh output file")
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record["selection_probe"]["summary"], indent=2))


if __name__ == "__main__":
    main()
