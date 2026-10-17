"""Oracle-motion renderer ceiling; these videos are never learned predictions."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.models.object_motion import encode_motion, observe, render_motion

from .compare_representation import score_samples
from .train_cvae import dataset_index, sha256
from .trajectory_data import canonical_ids, replay_angles


def calibrate(dataset):
    started = time.perf_counter()
    selected, _, provenance = dataset_index(dataset)
    cases = []
    for split, ids in selected.items():
        for sample_id in ids:
            ref = load_rollout(dataset, sample_id)
            row = {"sample_id": sample_id, "split": split}
            try:
                obs = observe(ref["frames"][0], ref["selected_mask"])
                order = canonical_ids(obs, ref["selected_object_id"])
                masks = ref["instance_masks"][0, order]
                row["mask_iou"] = (
                    (obs.masks & masks).sum((1, 2)) / (obs.masks | masks).sum((1, 2))
                ).tolist()
                motion = encode_motion(ref["positions"][:, order], replay_angles(ref)[:, order])
                video = render_motion(obs, motion)
                scores = score_samples(ref, video[None])
                row.update(
                    tracked=scores["all_objects"]["tracked_future_observations"],
                    expected=scores["all_objects"]["expected_future_observations"],
                    all_complete=scores["all_objects"]["complete_samples"] == 1,
                    ade=scores["all_objects"]["mean_ade_over_complete_samples"],
                    mae=scores["rgb_mae_future_0_1"],
                )
            except ValueError as error:
                row["failure"] = str(error)
            cases.append(row)
    summary = {}
    for split in selected:
        rows = [row for row in cases if row["split"] == split]
        valid = [row for row in rows if "failure" not in row]
        summary[split] = {
            "cases": len(rows),
            "failures": sum("failure" in row for row in rows),
            "minimum_mask_iou": min((min(row["mask_iou"]) for row in valid), default=None),
            "tracked": sum(row["tracked"] for row in valid),
            "expected": sum(row["expected"] for row in valid),
            "complete_cases": sum(row["all_complete"] for row in valid),
            "mean_ade_complete": float(
                np.mean([row["ade"] for row in valid if row["ade"] is not None])
            )
            if any(row["ade"] is not None for row in valid)
            else None,
        }
    return {
        "scope": "oracle-motion renderer ceiling; not learned predictions or action success",
        "provenance": provenance,
        "source_sha256": {
            "object_motion.py": sha256(Path(__file__).parents[1] / "models/object_motion.py"),
            "trajectory_data.py": sha256(Path(__file__).with_name("trajectory_data.py")),
            "calibrate_trajectory.py": sha256(Path(__file__)),
        },
        "summary": summary,
        "cases": cases,
        "seconds": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve earlier calibration; use a fresh output path")
    report = calibrate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
