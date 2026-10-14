"""Small command-line entry points for local simulator smoke runs."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from .data import (
    build_split_manifest,
    generate_dataset,
    load_rollout,
    save_rollout,
    save_split_manifest,
)
from .sim import make_scene, replay_saved_rollout, run_rollout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="click2future")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-smoke", help="generate small rollout samples")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--count", type=int, default=1)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--object-count", type=int, default=2)
    generate.add_argument("--config", type=Path, default=Path("configs/simulator/smoke.toml"))
    generate.set_defaults(handler=_generate_smoke)

    dataset = subparsers.add_parser(
        "generate-dataset",
        help="generate grouped visible-scene/physics-variant rollouts",
    )
    dataset.add_argument("--output-dir", type=Path, required=True)
    dataset.add_argument("--group-count", type=int, required=True)
    dataset.add_argument("--variants-per-group", type=int, required=True)
    dataset.add_argument("--seed", type=int, default=0)
    dataset.add_argument("--split-seed", type=int, default=None)
    dataset.add_argument("--object-count", type=int, default=2)
    dataset.add_argument("--config", type=Path, default=Path("configs/simulator/smoke.toml"))
    dataset.set_defaults(handler=_generate_dataset)

    split = subparsers.add_parser("make-splits", help="make a scene-group split manifest")
    split.add_argument("--input-dir", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(handler=_make_splits)

    validate = subparsers.add_parser("validate-sample", help="validate one saved rollout")
    validate.add_argument("--input-dir", type=Path, required=True)
    validate.add_argument("--sample-id", required=True)
    validate.add_argument("--replay", action="store_true")
    validate.add_argument("--atol", type=float, default=1e-6)
    validate.add_argument("--rtol", type=float, default=1e-6)
    validate.set_defaults(handler=_validate_sample)

    args = parser.parse_args(argv)
    return int(args.handler(args))


def _generate_smoke(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    config = _load_simulator_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = []
    for offset in range(args.count):
        seed = args.seed + offset
        scene = make_scene(seed, object_count=args.object_count)
        rollout = run_rollout(
            scene,
            frame_count=int(config.get("frame_count", 16)),
            fps=int(config.get("fps", 8)),
            physics_hz=int(config.get("physics_hz", 240)),
            resolution=int(config.get("resolution", 64)),
        )
        sample_id = f"sample_{seed:08d}"
        paths = save_rollout(rollout, args.output_dir, sample_id)
        sample_ids.append({"sample_id": sample_id, "arrays": str(paths["arrays"]), "metadata": str(paths["metadata"])})
    print(json.dumps(sample_ids, indent=2))
    return 0


def _make_splits(args: argparse.Namespace) -> int:
    scene_group_ids = []
    for metadata_path in sorted(args.input_dir.rglob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene_spec = metadata.get("scene_spec")
        if scene_spec and "scene_group_id" in scene_spec:
            scene_group_ids.append(scene_spec["scene_group_id"])
    if not scene_group_ids:
        raise ValueError("no rollout metadata with scene_group_id found")
    manifest = build_split_manifest(scene_group_ids, seed=args.seed)
    path = save_split_manifest(manifest, args.output)
    print(json.dumps({"path": str(path), "manifest_hash": manifest["manifest_hash"]}, indent=2))
    return 0


def _generate_dataset(args: argparse.Namespace) -> int:
    config = _load_simulator_config(args.config)
    result = generate_dataset(
        args.output_dir,
        group_count=args.group_count,
        variants_per_group=args.variants_per_group,
        seed=args.seed,
        split_seed=args.split_seed,
        object_count=args.object_count,
        config=config,
    )
    manifest = result["manifest"]
    print(
        json.dumps(
            {
                "manifest_path": str(result["manifest_path"]),
                "split_path": str(result["split_path"]),
                "group_count": manifest["group_count"],
                "variants_per_group": manifest["variants_per_group"],
                "sample_count": manifest["sample_count"],
            },
            indent=2,
        )
    )
    return 0


def _validate_sample(args: argparse.Namespace) -> int:
    try:
        rollout = load_rollout(args.input_dir, args.sample_id)
    except (OSError, KeyError, ValueError) as error:
        print(json.dumps({"ok": False, "sample_id": args.sample_id, "error": str(error)}))
        return 1

    report: dict[str, Any] = {
        "ok": True,
        "sample_id": args.sample_id,
        "schema_version": rollout["schema_version"],
        "frames_shape": list(rollout["frames"].shape),
        "instance_masks_shape": list(rollout["instance_masks"].shape),
        "positions_shape": list(rollout["positions"].shape),
    }
    if args.replay:
        replay_report = replay_saved_rollout(
            str(args.input_dir),
            args.sample_id,
            atol=args.atol,
            rtol=args.rtol,
        )
        report["replay"] = replay_report
        report["ok"] = bool(replay_report["ok"])
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def _load_simulator_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data.get("simulator", {})


if __name__ == "__main__":
    raise SystemExit(main())
