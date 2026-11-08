"""Original-validation reference control, with unchanged RGB scoring settings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.video import evaluate_videos

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Retain the previous audit and choose a fresh output")
root = Path(__file__).resolve().parents[1]
dataset = args.dataset.resolve()
config = root / "configs/evaluation/pixel.yaml"
settings = yaml.safe_load(config.read_text())
plan = prepare_plan(dataset, config, split="validation", metrics=("ssim",))
names = sorted({name for case in plan["cases"] for name in case["reference_ids"]})
rows = []
for name in names:
    ref = load_rollout(dataset, name)
    report, arrays = evaluate_videos(
        ref,
        ref["frames"][None],
        tracker=TrackerConfig(**settings["tracker"]),
        max_gap_pixels=settings["max_gap_pixels"],
        event_tolerance=settings["event_tolerance_intervals"],
        motion_threshold=settings["motion_threshold"],
        motion_window=settings["motion_window"],
        motion_rule=settings["motion_rule"],
        run_id="unchanged-reference-control",
    )
    truth = np.flatnonzero(arrays["reference_collision_onset"][:, 0, 1])
    estimate = np.flatnonzero(arrays["predicted_collision_onset"][0, :, 0, 1])
    complete = bool(arrays["collision_observed"][0, :, 0, 1].all())
    rows.append(
        dict(
            sample_id=name,
            complete=complete,
            truth=truth.tolist(),
            estimate=estimate.tolist(),
            close_native_onset_pairs=int((np.diff(truth) <= settings["motion_window"]).sum()),
            nearest_interval_offsets=[
                int(estimate[np.argmin(abs(estimate - t))] - t) if len(estimate) else None
                for t in truth
            ],
            event=report["collision"]["per_sample"][0]["event"],
        )
    )
counts = {
    key: sum(row["event"][key] for row in rows)
    for key in ("true_positive", "false_positive", "false_negative")
}
tp, fp, fn = (counts[k] for k in ("true_positive", "false_positive", "false_negative"))
enc = {"tp": 0, "fp": 0, "fn": 0}
for row in rows:
    if row["complete"]:
        a, b = bool(row["estimate"]), bool(row["truth"])
        enc["tp"] += int(a and b)
        enc["fp"] += int(a and not b)
        enc["fn"] += int(b and not a)
result = dict(
    scope=(
        "all original-validation simulator references scored as RGB; no model or threshold changes"
    ),
    clips=len(rows),
    complete_clips=sum(r["complete"] for r in rows),
    event_counts=counts,
    event_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    encounter_counts=enc,
    encounter_f1=2 * enc["tp"] / (2 * enc["tp"] + enc["fp"] + enc["fn"])
    if sum(enc.values())
    else None,
    native_events=sum(len(r["truth"]) for r in rows),
    close_native_onset_pairs=sum(r["close_native_onset_pairs"] for r in rows),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    validation_plan_sha256=plan["plan_sha256"],
    settings=settings,
    cases=rows,
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "settings")}))
