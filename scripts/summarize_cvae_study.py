"""Tabulate a CVAE study directory written by cluster/cvae_study.sbatch.

Reads each <run>/{train/run.json, eval/..., force-probe.json} and prints a
Markdown table; --output also writes the rows as JSON. Missing pieces are shown
as blanks rather than skipped, so failed or partial runs stay visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _method(report):
    methods = (report or {}).get("summary", {}).get("methods") or [{}]
    return methods[0].get("counts", {}), methods[0].get("metrics", {})


def _estimate(metrics, key):
    value = metrics.get(key, {}).get("estimate")
    return None if value is None else round(value, 3)


def summarize(run):
    train = _load(run / "train/run.json") or {}
    config = train.get("config", {})
    result = _load(run / "eval/cvae-results.json") or {}
    quality = result.get("quality", {})
    action_counts, action = _method(_load(run / "eval/validation-metrics/actions.json"))
    target_counts, target = _method(_load(run / "eval/selected-object-metrics/targets.json"))
    diagnostics = (_load(run / "eval/cvae-diagnostics.json") or {}).get("splits", {})
    validation = diagnostics.get("validation", {})
    probe = (_load(run / "force-probe.json") or {}).get("mae", {})
    foreground = {name: round(value["foreground"], 4) for name, value in probe.items()}
    return {
        "run": run.name,
        "beta": config.get("training", {}).get("beta"),
        "latent_dim": config.get("model", {}).get("latent_dim"),
        "train_status": train.get("status"),
        "steps": train.get("steps"),
        "best_validation": train.get("best_validation"),
        "training_minutes": round(train["total_training_seconds"] / 60, 1)
        if "total_training_seconds" in train
        else None,
        "dataset_manifest_sha256": train.get("dataset_manifest_sha256"),
        "tracked": quality.get("tracked_observations"),
        "expected": quality.get("expected_observations"),
        "complete_all_object_clips": quality.get("complete_all_object_clips"),
        "forward_moving": action_counts.get("forward.moving_samples"),
        "forward_complete": action_counts.get("forward.complete_samples"),
        "reversal_complete_pairs": action_counts.get("reversal.complete_pairs"),
        "reversal_correct_sign_rate": _estimate(
            action, "reversal.correct_force_sign_rate_over_complete"
        ),
        "selection_complete_pairs": target_counts.get("selection.complete_pairs"),
        "selection_correct_target_rate": _estimate(
            target, "selection.correct_target_rate_over_complete"
        ),
        "validation_posterior_mae": validation.get("posterior_future_mae"),
        "validation_prior_mae": validation.get("prior_future_mae"),
        "validation_prior_pairwise_mae": validation.get("prior_pairwise_rgb_mae"),
        "validation_kl_per_latent": validation.get("kl_mean_per_latent"),
        "probe_foreground_mae": foreground,
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(rows):
    columns = [
        ("run", "run"),
        ("beta", "β"),
        ("latent_dim", "z"),
        ("steps", "steps"),
        ("tracked", "tracked /5952"),
        ("complete_all_object_clips", "complete /96"),
        ("reversal_correct_sign_rate", "reversal sign ok"),
        ("selection_correct_target_rate", "selection ok"),
        ("validation_prior_pairwise_mae", "prior spread"),
        ("validation_kl_per_latent", "KL/latent"),
    ]
    probe_columns = [
        "force_total",
        "force_via_features",
        "force_via_prior",
        "prior_seed_spread",
        "posterior_minus_prior",
        "true_motion",
    ]
    header = [label for _, label in columns] + [f"probe {name}" for name in probe_columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [_cell(row[key]) for key, _ in columns]
        cells += [_cell(row["probe_foreground_mae"].get(name)) for name in probe_columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = sorted(p for p in args.study.iterdir() if p.is_dir())
    rows = [summarize(run) for run in runs]
    print(table(rows))
    hashes = {row["dataset_manifest_sha256"] for row in rows if row["dataset_manifest_sha256"]}
    print(f"\nDataset manifest hashes: {sorted(hashes)}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
