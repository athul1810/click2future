"""Freeze a small scene-disjoint corpus and run the declared TC2 validation pilots."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
import yaml

from click2future.data import load_batch
from click2future.data.generalization import (
    generate_generalization_dataset,
    validate_generalization_dataset,
)

from .auxiliary import collision_balance
from .targets import build_auxiliary_targets
from .train_cvae import require_tc2, sha256, train, validate_config
from .train_diffusion import TrainConfig
from .train_diffusion import main as train_diffusion

PROJECT = Path(__file__).resolve().parents[3]


def _write(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def training_balance(dataset, manifest):
    """Count valid contact labels once using only the training split."""
    ids = [r["sample_id"] for r in manifest["samples"] if r["split"] == "train"]

    def targets():
        for start in range(0, len(ids), 8):
            batch = load_batch(dataset, ids[start : start + 8])
            yield build_auxiliary_targets(
                torch.as_tensor(batch["positions"]),
                torch.as_tensor(batch["collision_onset"]),
                torch.as_tensor(batch["object_valid"]),
                torch.as_tensor(batch["instance_masks"][:, 0]),
                max_objects=4,
            )

    return collision_balance(targets())


def prepare(root, specification):
    root, specification = Path(root), Path(specification).resolve()
    if root.exists():
        raise ValueError("Use a new pilot directory; attempted runs are never overwritten")
    spec = json.loads(specification.read_text())
    if spec["dataset_version"] != "dataset-0.3" or spec["evaluation_split"] != "validation":
        raise ValueError("This runner is a dataset-0.3 validation pilot")
    configs = {}
    source_hashes = {}
    for method, name in spec["training_configs"].items():
        path = PROJECT / name
        source_hashes[name] = sha256(path)
        config = yaml.safe_load(path.read_text())
        if method == "E0":
            validate_config(config)
        elif method in {"E2", "E3", "E4"}:
            TrainConfig(**config).validate()
        else:
            raise ValueError("Use E0/E2/E3/E4 pilot methods")
        configs[method] = config
    if set(configs) != {"E0", "E2", "E3", "E4"}:
        raise ValueError("The declared pilot requires all four methods")
    root.mkdir(parents=True)
    try:
        generated = generate_generalization_dataset(
            root / "data", seed=spec["seed"], group_counts=spec["group_counts"]
        )
        checked = validate_generalization_dataset(root / "data", replay=True)
        if not checked["ok"]:
            raise ValueError("Pilot replay failed: " + "; ".join(checked["errors"]))
        balance = training_balance(root / "data", generated["manifest"])
        folder = root / "configs"
        folder.mkdir()
        config_hashes = {}
        for method, config in configs.items():
            if method != "E0":
                config = asdict(
                    replace(
                        TrainConfig(**config), collision_positive_weight=balance["positive_weight"]
                    )
                )
            path = folder / (method + ".json")
            _write(path, config)
            config_hashes[method] = sha256(path)
        record = {
            "status": "prepared_validation_pilot",
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "runner_sha256": sha256(Path(__file__)),
            "specification": spec,
            "specification_sha256": sha256(specification),
            "source_config_sha256": source_hashes,
            "effective_config_sha256": config_hashes,
            "dataset_manifest_sha256": sha256(root / "data/dataset_manifest.json"),
            "split_manifest_sha256": sha256(root / "data/splits.json"),
            "clips_by_split": dict(Counter(r["split"] for r in generated["manifest"]["samples"])),
            "replay": checked,
            "training_collision_balance": balance,
        }
        _write(root / "pilot.json", record)
        return record
    except BaseException as error:
        _write(root / "failure.json", {"stage": "prepare", "error": str(error)})
        raise


def read_pilot(root):
    root = Path(root)
    record = json.loads((root / "pilot.json").read_text())
    if record["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("Pilot runner source changed")
    for method, digest in record["effective_config_sha256"].items():
        if sha256(root / "configs" / (method + ".json")) != digest:
            raise ValueError("Frozen pilot config changed")
    for name, field in (
        ("dataset_manifest.json", "dataset_manifest_sha256"),
        ("splits.json", "split_manifest_sha256"),
    ):
        if sha256(root / "data" / name) != record[field]:
            raise ValueError("Frozen pilot dataset index changed")
    manifest = json.loads((root / "data/dataset_manifest.json").read_text())
    for row in manifest["samples"]:
        if row["split"] not in {"train", "validation"}:
            continue
        for field in ("arrays", "metadata"):
            if sha256(root / "data" / row[field]) != row[field + "_sha256"]:
                raise ValueError("Frozen pilot training/validation sample changed")
    return record


def train_all(root):
    require_tc2()
    root = Path(root)
    record = read_pilot(root)
    if any((root / method).exists() for method in record["effective_config_sha256"]):
        raise ValueError("Training already attempted; keep it and create a separate pilot")
    progress = {"status": "running", "slurm_job_id": os.environ["SLURM_JOB_ID"], "methods": {}}
    try:
        for method in ("E0", "E2", "E3", "E4"):
            config_path = root / "configs" / (method + ".json")
            print(json.dumps({"starting": method}), flush=True)
            if method == "E0":
                result = train(root / "data", root / method, json.loads(config_path.read_text()))
            else:
                train_diffusion(
                    [
                        "--dataset",
                        str(root / "data"),
                        "--run-dir",
                        str(root / method),
                        "--config",
                        str(config_path),
                        "--device",
                        "cuda",
                        "--require-slurm",
                    ]
                )
                steps = json.loads(config_path.read_text())["steps"]
                result = json.loads((root / method / f"summary-{steps}.json").read_text())
            progress["methods"][method] = result
            _write(root / "training.json", progress)
        progress["status"] = (
            "completed"
            if progress["methods"]["E0"]["status"] == "completed"
            else "completed_with_checkpointed_cvae"
        )
        _write(root / "training.json", progress)
    except BaseException as error:
        progress.update(status="failed", error=str(error))
        _write(root / "training.json", progress)
        raise
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, default=PROJECT / "configs/training/multiscene_pilot.json"
    )
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true")
    phase.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only:
        require_tc2()
    if not args.train_only:
        prepare(args.root, args.spec)
    if not args.prepare_only:
        train_all(args.root)


if __name__ == "__main__":
    main()
