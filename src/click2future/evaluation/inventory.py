"""Count evaluation work from frozen metadata without opening reference arrays."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from click2future.data.splits import load_split_manifest
from click2future.training.train_cvae import sha256

from .suite import _digest, _seeds
from .uncertainty_strata import action_family

RESERVED = ("id_test", "shape_ood", "texture_ood", "force_ood", "physics_test")


def inventory(dataset, protocol):
    dataset, protocol = Path(dataset), Path(protocol)
    index_path, splits_path = dataset / "dataset_manifest.json", dataset / "splits.json"
    manifest = json.loads(index_path.read_text())
    if (
        manifest["schema_version"] != "dataset-0.3"
        or _digest({k: v for k, v in manifest.items() if k != "manifest_hash"})
        != manifest["manifest_hash"]
        or sha256(splits_path) != manifest["split_manifest_sha256"]
    ):
        raise ValueError("Provide an unchanged dataset-0.3 index and split manifest")
    splits = load_split_manifest(splits_path)["splits"]
    if set(splits) != {"train", "validation", *RESERVED}:
        raise ValueError("Expected train, validation and all five reserved suites")
    config = yaml.safe_load(protocol.read_text())
    seeds = _seeds(config["sampling_seeds"])
    seen, groups = set(), defaultdict(lambda: defaultdict(list))
    for row in manifest["samples"]:
        name, split, scene = row["sample_id"], row["split"], row["scene_group_id"]
        if name in seen or scene not in splits.get(split, []):
            raise ValueError("Duplicate sample or scene-group split mismatch")
        seen.add(name)
        action_family(row)  # Check that the condition retains its scene identity.
        groups[split][(scene, row["condition_id"])].append(name)
    if len(seen) != manifest["sample_count"]:
        raise ValueError("Sample count disagrees with the index")
    rows = {}
    for split, scenes in splits.items():
        conditions = groups[split]
        if {scene for scene, _ in conditions} != set(scenes):
            raise ValueError("Some declared scene groups have no indexed conditions")
        forward = {scene for scene, condition in conditions if condition == scene + "-forward"}
        rows[split] = {
            "scene_groups": len(scenes),
            "reference_clips": sum(map(len, conditions.values())),
            "conditions": len(conditions),
            "primary_videos_per_run": len(conditions) * len(seeds),
            "proposed_selection_conditions": 2 * len(forward),
            "proposed_selection_videos_per_run": 2 * len(forward) * len(seeds),
            "selection_eligibility": "Requires later reference checks; no arrays opened here",
            "action_controls": "omitted_for_force_ood" if split == "force_ood" else "declared",
        }
    return {
        "status": "metadata_inventory_only_final_selection_pending",
        "dataset_manifest_sha256": sha256(index_path),
        "splits_sha256": sha256(splits_path),
        "comparison_protocol_sha256": sha256(protocol),
        "sampling_seeds": seeds,
        "core_methods": config["core_methods"],
        "planned_training_seeds": config["training_seeds"],
        "splits": rows,
        "reserved_totals": {
            key: sum(rows[split][key] for split in RESERVED)
            for key in (
                "scene_groups",
                "reference_clips",
                "conditions",
                "primary_videos_per_run",
                "proposed_selection_conditions",
                "proposed_selection_videos_per_run",
            )
        },
        "reference_arrays_opened": 0,
        "test_splits_evaluated": [],
        "limitations": [
            "Inventory is workload planning, not a checkpoint freeze or test authorization.",
            "Proposed target controls require reference eligibility checks after the freeze.",
            "Training-seed entries describe the protocol; they do not assert completed runs.",
            "No GPU allocation or unmeasured latency is inferred from these counts.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve existing inventories; provide a fresh output")
    result = inventory(args.dataset, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["reserved_totals"], indent=2))


if __name__ == "__main__":
    main()
