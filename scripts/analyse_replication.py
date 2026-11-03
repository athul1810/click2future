"""Collect the D017 held-out replication scores from the three saved job directories."""

import json
import sys
from pathlib import Path

SPLITS = ("id_test", "shape_ood", "texture_ood", "force_ood", "physics_test")
MEASURES = (
    "selected_object.mean_ade",
    "selected_object.mean_fde",
    "selected_object.missing_track_rate",
    "image.ssim",
    "image.lpips",
)
SELECTION = (
    "selection.correct_target_rate_over_complete",
    "selection.drift_rate_over_complete",
)


def read(path):
    return json.loads(Path(path).read_text())


def estimates(methods, names, scope):
    rows = {}
    for method in methods:
        rows[method["method"]] = {
            name: method["metrics"][name]["estimate"] for name in names if name in method["metrics"]
        }
        rows[method["method"]]["runs"] = method["runs"]
        rows[method["method"]]["scene_groups"] = method.get("scene_groups")
    return {"scope": scope, "methods": rows}


def collect(root, label, split):
    """Return score rows for one job directory and split, keyed by model group."""
    base = Path(root) / split
    if not base.is_dir():
        return {}
    out = {}
    for kind, folder, names, file in (
        ("primary", base, MEASURES, "summary.json"),
        ("selection", base / "target-controls", SELECTION, "targets.json"),
    ):
        for directory in sorted(folder.glob("*metrics")):
            path = directory / file
            if not path.is_file():
                continue
            data = read(path)
            methods = data["methods"] if "methods" in data else data["summary"]["methods"]
            prefix = directory.name.removesuffix("metrics").rstrip("-")
            for method in methods:
                group = label if not prefix else prefix
                key = method["method"] if not prefix else prefix
                out.setdefault(kind, {})[key if prefix else f"{key}-{label}"] = {
                    name: method["metrics"][name]["estimate"]
                    for name in names
                    if name in method["metrics"]
                } | {"runs": method["runs"], "group": group}
    return out


def main(argv):
    jobs = {"job1": argv[1], "job2": argv[2], "job2b": argv[3]}
    result = {
        "scope": "D017 held-out replication; scores only",
        "source_revision": "ce4132fb48adc66c37583fd30ccc9a49d73b7197",
        "jobs": {
            "job1": "50265: E0 seeds 0-2 and E2/E3 40k, all five splits",
            "job2": "50321: E2/E3 5k and six E10 runs for id_test, shape_ood and texture_ood; "
            "stopped by a disk quota error in force_ood, its replication.json is truncated",
            "job2b": "50527: the same for force_ood and physics_test, node-local output",
        },
        "note": "job2 main plan hashes equal job1's. The selection plan hashes differ between "
        "jobs only because each embeds its own output path; the cases are identical. "
        "A null ADE or FDE means some clip had a missing track for that scene group.",
        "splits": {},
    }
    for split in SPLITS:
        merged = {}
        for label, root in jobs.items():
            for kind, rows in collect(root, label, split).items():
                merged.setdefault(kind, {}).update(rows)
        result["splits"][split] = merged
    Path(argv[4]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
