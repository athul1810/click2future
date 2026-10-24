"""Compare hidden-physics references' exact collision labels with the first reference.

This separates genuine hidden-physics timing variation from contact-estimator error
in the reference calibration. No video, tracker, estimator or model is used: every
alternative reference's exact simulator onsets are matched against the first
reference's onsets with the collision audit's one-to-one event rule and the
historical tolerance. Only validation clips are read.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.training.train_cvae import sha256

from .collision_audit import event_details, event_summary


def compare_labels(first, others, *, tolerance):
    """Match each alternative [T] onset timeline against the first; all are complete."""
    first = np.asarray(first, dtype=bool)
    observed = np.ones_like(first)
    return [
        event_details(np.asarray(other, dtype=bool), first, observed, tolerance=tolerance)
        for other in others
    ]


def label_agreement(dataset, config_path, split="validation"):
    dataset, config_path = Path(dataset), Path(config_path)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    tolerance = yaml.safe_load(config_path.read_text())["event_tolerance_intervals"]
    conditions = defaultdict(list)
    for row in manifest["samples"]:
        if row["split"] == split:
            conditions[row["condition_id"]].append(row["sample_id"])
    rows, cases = [], []
    for condition, names in sorted(conditions.items()):
        names = sorted(names)  # p00 first, matching the evaluation case's first reference
        labels, first_frame = [], None
        for name in names:
            rollout = load_rollout(dataset, name)
            if not np.array_equal(rollout["object_valid"], [True, True]):
                raise ValueError("Label agreement requires exactly two valid objects")
            if first_frame is None:
                first_frame = rollout["frames"][0]
            elif not np.array_equal(rollout["frames"][0], first_frame):
                raise ValueError("Reference variants must share the initial image")
            labels.append(rollout["collision_onset"][:, 0, 1])
        compared = compare_labels(labels[0], labels[1:], tolerance=tolerance)
        rows.extend(compared)
        cases.append(
            {
                "condition_id": condition,
                "first_reference": names[0],
                "alternatives": [
                    {"reference_id": name, **result}
                    for name, result in zip(names[1:], compared, strict=True)
                ],
            }
        )
    return {
        "status": "measured_reference_label_agreement",
        "split": split,
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "config_sha256": sha256(config_path),
        "event_tolerance_intervals": tolerance,
        "conditions": len(cases),
        "alternative_labels_against_first_labels": event_summary(rows),
        "cases": cases,
        "interpretation": [
            "Exact simulator onsets only; no rendering, tracking or contact estimation.",
            "Measures how far valid hidden-physics outcomes disagree in event timing.",
            "Not a model score, a performance bound or a replacement for any reported result.",
        ],
        "source_sha256": {p.name: sha256(p) for p in (Path(__file__),)},
        "test_splits_evaluated": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output path; preserve previous results")
    result = label_agreement(args.dataset, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["alternative_labels_against_first_labels"], indent=2))


if __name__ == "__main__":
    main()
