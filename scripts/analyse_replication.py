"""Collect the D017 held-out replication scores from the three saved job directories."""

import hashlib
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(metric):
    return {
        "estimate": metric["estimate"],
        "scored_scene_groups": metric["scored_scene_groups"],
        "expected_scene_groups": metric["expected_scene_groups"],
    }


def collect(root, label, split):
    """Return score rows, plan hashes and checkpoint hashes for one job and split."""
    base = Path(root) / split
    if not base.is_dir():
        return {}
    out = {}
    for kind, folder, names, file in (
        ("primary", base, MEASURES, "summary.json"),
        ("selection", base / "target-controls", SELECTION, "targets.json"),
    ):
        out.setdefault("plan_sha256", {})[kind] = read(folder / "plan.json")["plan_sha256"]
        for directory in sorted(folder.glob("*metrics")):
            path = directory / file
            if not path.is_file():
                continue
            data = read(path)
            methods = data["methods"] if "methods" in data else data["summary"]["methods"]
            prefix = directory.name.removesuffix("metrics").rstrip("-")
            checkpoints = {}
            if kind == "primary":
                comparison = read(directory / "comparison.json")
                checkpoints = {
                    run["method"] + ":" + str(run["training_seed"]): run["checkpoint_sha256"]
                    for run in comparison["runs"]
                }
            for method in methods:
                key = prefix or f"{method['method']}-{label}"
                row = {
                    name: score(method["metrics"][name])
                    for name in names
                    if name in method["metrics"]
                }
                row["runs"] = method["runs"]
                row["source_sha256"] = sha256(path)
                if kind == "selection":
                    row["pair_counts"] = method["counts"]
                else:
                    row["checkpoint_sha256"] = checkpoints
                    row["sample_counts"] = method["counts"]
                out.setdefault(kind, {})[key] = row
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
        "ADE/FDE average complete trajectories within each case. A defined estimate "
        "does not imply all clips are complete. Scene-group aggregation requires a "
        "defined case/run value; report both sample counts and scored scene groups.",
        "splits": {},
    }
    for split in SPLITS:
        merged, plans = {}, {}
        for label, root in jobs.items():
            found = collect(root, label, split)
            plans[label] = found.pop("plan_sha256", None)
            for kind, rows in found.items():
                merged.setdefault(kind, {}).update(rows)
        merged["plan_sha256_by_job"] = plans
        result["splits"][split] = merged
    Path(argv[4]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv)
