"""Sample K CVAE futures from frame-0 conditions and export the evaluation video format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.models.cvae import ConditionalVideoVAE

from .train_cvae import require_tc2, sha256, validate_config


def sample_checkpoint(checkpoint, data_dir, sample_ids, output_dir, *, seeds, device="cpu", run_id):
    if device == "cuda":
        require_tc2()
    if device not in {"cpu", "cuda"} or not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Choose cpu/cuda and distinct sample IDs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format")
    config = payload["config"]
    validate_config(config)
    training_sources = payload["provenance"].get("source_sha256", {})
    model_source = Path(__file__).parents[1] / "models/cvae.py"
    if training_sources and training_sources["cvae.py"] != sha256(model_source):
        raise ValueError("Checkpoint model source differs from the current implementation")
    if payload["model_config"] != config["model"]:
        raise ValueError("Checkpoint model and run configuration disagree")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("Sample seeds must be distinct nonnegative integers")
    # Loading a checkpoint should not consume the caller's experiment RNG state.
    with torch.random.fork_rng(devices=[]):
        model = ConditionalVideoVAE(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    if output_dir.exists():
        raise ValueError("Use a fresh sample output directory")
    references = []
    for sample_id in sample_ids:
        ref = load_rollout(data_dir, sample_id)
        d = config["data"]
        if ref["frames"].shape != (
            d["total_frames"],
            d["resolution"],
            d["resolution"],
            3,
        ) or not np.allclose(np.diff(ref["timestamps"]), 1 / d["fps"], rtol=0, atol=1e-8):
            raise ValueError(
                "Reference clip dimensions/timing differ from the trained configuration"
            )
        if not np.isclose(ref["force_duration"], d["force_duration"], atol=1e-7, rtol=0):
            raise ValueError("Reference force duration differs from the trained contract")
        references.append((sample_id, ref))
    output_dir.mkdir(parents=True)
    records = []
    for sample_id, ref in references:
        # Only frame 0, the selected mask, and the numeric force become inference inputs.
        first = (
            torch.from_numpy(ref["frames"][:1].transpose(0, 3, 1, 2)).to(device).float() / 127.5 - 1
        )
        mask = torch.from_numpy(ref["selected_mask"][None, None]).to(device)
        force = torch.from_numpy(ref["force_xy"][None]).to(device) / config["data"]["force_scale"]
        future = model.sample(first, mask, force, seeds=seeds)[:, 0]
        if not torch.isfinite(future).all():
            raise ValueError("The checkpoint generated nonfinite RGB")
        rgb = (
            ((future.cpu().permute(0, 2, 3, 4, 1) + 1) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .numpy()
        )
        videos = np.concatenate(
            (np.broadcast_to(ref["frames"][:1], (len(seeds), 1, *ref["frames"].shape[1:])), rgb),
            axis=1,
        )
        path = output_dir / (sample_id + ".npz")
        np.savez_compressed(path, videos=videos)
        records.append(
            {
                "sample_id": sample_id,
                "scene_group_id": ref["scene_spec"]["scene_group_id"],
                "condition_id": ref["scene_spec"]["condition_id"],
                "sampling_seeds": seeds,
                "videos": path.name,
                "videos_sha256": sha256(path),
                "reference_arrays_sha256": sha256(data_dir / (sample_id + ".npz")),
                "reference_metadata_sha256": sha256(data_dir / (sample_id + ".json")),
            }
        )
    record = {
        "source": "generated_video",
        "method": "E0_CVAE",
        "run_id": run_id,
        "checkpoint_sha256": sha256(checkpoint),
        "training_source_sha256": training_sources,
        "sampling_source_sha256": {
            "sample_cvae.py": sha256(Path(__file__)),
            "cvae.py": sha256(model_source),
        },
        "training_steps": payload["state"]["step"],
        "training_seed": config["training"]["seed"],
        "device": device,
        "conditioning": "frame 0 RGB, selected mask, numeric force; learned prior only",
        "samples": records,
    }
    (output_dir / "samples.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    try:
        record = sample_checkpoint(
            args.checkpoint,
            args.data_dir,
            args.sample_ids,
            args.output_dir,
            seeds=args.seeds,
            device=args.device,
            run_id=args.run_id,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"CVAE sampling stopped: {error}\n")
    print(json.dumps({"status": "passed", "conditions": len(record["samples"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
