"""Sample frozen checkpoints on every evaluation case with common private seeds."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.training.sample_cvae import sample_checkpoint
from click2future.training.sample_diffusion import load_generator, sample_videos
from click2future.training.train_cvae import require_tc2, validate_config
from click2future.training.train_diffusion import TrainConfig

from .suite import _digest, _sha, _write, read_plan


def _describe_run(run, spec_path, plan, sampling_steps, allow_untrained):
    if set(run) != {"method", "run_id", "checkpoint"}:
        raise ValueError("Each run requires only method, run_id and checkpoint")
    name, method = run["run_id"], run["method"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run IDs must be safe filename components")
    if method not in {"E0", "E1", "E2", "E3", "E4"}:
        raise ValueError("Use E0-E4 method names")
    checkpoint = (spec_path.parent / run["checkpoint"]).resolve()
    checkpoint_hash = _sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if method == "E0":
        if state.get("format_version") != 1 or "model_config" not in state:
            raise ValueError("E0 requires a CVAE checkpoint")
        config = state["config"]
        validate_config(config)
        seed, step = config["training"]["seed"], state["state"]["step"]
        data_hash = state["provenance"]["dataset_manifest_sha256"]
        split_hash = state["provenance"]["split_manifest_sha256"]
        settings = {"sampler": "learned_prior", "sampling_steps": None}
    else:
        if state.get("checkpoint_version") != 1 or "ema" not in state:
            raise ValueError("Diffusion methods require an EMA diffusion checkpoint")
        trained = TrainConfig(**state["config"])
        trained.validate()
        if not isinstance(trained.auxiliary, bool):
            raise ValueError("Diffusion auxiliary setting must be boolean")
        actual = (
            "E4"
            if trained.auxiliary and trained.conditioning == "spatial"
            else {"first_frame": "E1", "numeric": "E2", "spatial": "E3"}.get(trained.conditioning)
        )
        if actual != method or (trained.auxiliary and method != "E4"):
            raise ValueError("Declared method differs from checkpoint conditioning/heads")
        if sampling_steps > trained.diffusion_steps:
            raise ValueError("Sampling steps exceed the checkpoint diffusion schedule")
        seed, step = trained.seed, state["step"]
        hashes = state["provenance"]["file_hashes"]
        data_hash, split_hash = hashes["dataset_manifest.json"], hashes["splits.json"]
        config = state["config"]
        settings = {
            "sampler": "DDIM eta=0, clipped x0 with recomputed epsilon",
            "sampling_steps": sampling_steps,
            "weights": "ema",
            "prediction_type": trained.prediction_type,
            "diffusion_steps": trained.diffusion_steps,
            "base_channels": trained.base_channels,
            "force_limit": trained.force_limit,
        }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (seed, step)
    ):
        raise ValueError("Checkpoint training seed and step must be nonnegative integers")
    if step == 0 and not allow_untrained:
        raise ValueError("Untrained checkpoints require --allow-untrained-fixtures")
    if (
        data_hash != plan["input_sha256"]["dataset_manifest"]
        or split_hash != plan["input_sha256"]["splits"]
    ):
        raise ValueError("Checkpoint training dataset/splits differ from the evaluation corpus")
    return {
        "method": method,
        "run_id": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_seed": seed,
        "training_steps": step,
        "untrained_fixture": step == 0,
        "training_config": config,
        "training_config_sha256": _digest(config),
        "force_scale": config["data"]["force_scale"] if method == "E0" else 0.5,
        **settings,
    }


def _check_case_contracts(plan, runs):
    metrics = yaml.safe_load(Path(plan["config"]).read_text())
    if metrics["fps"] != 8:
        raise ValueError("Checkpoint sampling requires the native 8-FPS metric contract")
    common_scale = plan["comparison_protocol"].get("force_scale", 0.5)
    if any(run["force_scale"] != common_scale for run in runs):
        raise ValueError("Checkpoint force normalization differs from the shared protocol")
    diffusion = [run for run in runs if run["method"] != "E0"]
    if (
        len({(r["prediction_type"], r["diffusion_steps"], r["base_channels"]) for r in diffusion})
        > 1
    ):
        raise ValueError(
            "Diffusion comparison requires matched target, schedule and backbone width"
        )
    for case in plan["cases"]:
        ref = load_rollout(plan["dataset"], case["sample_id"])
        if any(ref["scene_spec"][key] != case[key] for key in ("scene_group_id", "condition_id")):
            raise ValueError("Frozen case identity differs from the saved rollout")
        if (
            list(ref["frames"].shape[1:3]) != metrics["resolution_hw"]
            or not np.allclose(
                ref["timestamps"], np.arange(len(ref["frames"])) / 8, rtol=0, atol=1e-8
            )
            or not np.isclose(ref["force_duration"], 0.25, rtol=0, atol=1e-7)
        ):
            raise ValueError("Frozen case dimensions/timing differ from the sampling contract")
        for run in runs:
            if run["method"] == "E0":
                data = run["training_config"]["data"]
                if ref["frames"].shape != (
                    data["total_frames"],
                    data["resolution"],
                    data["resolution"],
                    3,
                ):
                    raise ValueError("CVAE checkpoint dimensions differ from the frozen cases")
            elif np.linalg.norm(ref["force_xy"]) > run["force_limit"] + 1e-6:
                raise ValueError("Frozen case exceeds the checkpoint's declared force limit")


def generate_samples(plan_path, spec_path, output_dir, *, device="cpu", allow_untrained=False):
    """Write predictions.json only after every run/case has completed successfully.

    This never trains models or selects checkpoints. Partial outputs retain a
    failure record and cannot be mistaken for a complete comparison manifest.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    if device == "cuda":
        require_tc2()
    plan_path, spec_path, output_dir = map(Path, (plan_path, spec_path, output_dir))
    if output_dir.exists():
        raise ValueError("Use a new sample output directory")
    plan = read_plan(plan_path)
    spec_hash = _sha(spec_path)
    spec = json.loads(spec_path.read_text())
    if (
        set(spec) != {"sampling_steps", "runs"}
        or not isinstance(spec["runs"], list)
        or not spec["runs"]
    ):
        raise ValueError("Run specification requires sampling_steps and a nonempty runs list")
    steps = spec["sampling_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("sampling_steps must be an integer >= 2")
    runs = [_describe_run(run, spec_path, plan, steps, allow_untrained) for run in spec["runs"]]
    if len({run["run_id"] for run in runs}) != len(runs):
        raise ValueError("Run IDs must be unique")
    if len({(run["method"], run["training_seed"]) for run in runs}) != len(runs):
        raise ValueError("Repeated training seeds within a method would double-count a run")
    _check_case_contracts(plan, runs)
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "source": "generated_video",
        "status": "running",
        "evaluation_plan_sha256": plan["plan_sha256"],
        "run_spec_sha256": spec_hash,
        "sampling_seeds": plan["sampling_seeds"],
        "device": device,
        "allow_untrained_fixtures": allow_untrained,
        "scope": "Model-generated samples; trained quality requires evaluation",
        "runs": [],
    }
    started = time.perf_counter()
    try:
        for run in runs:
            if _sha(run["checkpoint"]) != run["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed after preflight")
            folder = output_dir / run["run_id"]
            before = time.perf_counter()
            if run["method"] == "E0":
                result = sample_checkpoint(
                    checkpoint=Path(run["checkpoint"]),
                    data_dir=Path(plan["dataset"]),
                    sample_ids=[case["sample_id"] for case in plan["cases"]],
                    output_dir=folder,
                    seeds=plan["sampling_seeds"],
                    device=device,
                    run_id=run["run_id"],
                )
                videos = {item["sample_id"]: item["videos"] for item in result["samples"]}
            else:
                # Checkpoint model construction must not consume the caller's CPU RNG.
                with torch.random.fork_rng(devices=[]):
                    model, diffusion, _ = load_generator(Path(run["checkpoint"]), device)
                folder.mkdir()
                videos = {}
                for case in plan["cases"]:
                    ref = load_rollout(plan["dataset"], case["sample_id"])
                    generated = sample_videos(
                        model,
                        diffusion,
                        ref["frames"][0],
                        ref["selected_mask"],
                        ref["force_xy"],
                        seeds=plan["sampling_seeds"],
                        future_frames=len(ref["frames"]) - 1,
                        sampling_steps=steps,
                        force_limit=run["force_limit"],
                    )
                    name = case["sample_id"] + ".npz"
                    np.savez_compressed(folder / name, videos=generated)
                    videos[case["sample_id"]] = name
                del model, diffusion
            run.update(
                seconds_including_load_and_export=time.perf_counter() - before,
                videos={key: str(Path(run["run_id"]) / name) for key, name in videos.items()},
                video_sha256={key: _sha(folder / name) for key, name in videos.items()},
            )
            record["runs"].append(run)
        # Recheck immutable inputs before publishing a usable prediction manifest.
        read_plan(plan_path)
        if _sha(spec_path) != spec_hash or any(
            _sha(run["checkpoint"]) != run["checkpoint_sha256"] for run in runs
        ):
            raise ValueError("Run specification or checkpoint changed during sampling")
        record.update(status="passed", seconds=time.perf_counter() - started)
        _write(output_dir / "sampling.json", record)
        _write(
            output_dir / "predictions.json",
            {
                "source": "generated_video",
                "sampling_seeds": plan["sampling_seeds"],
                "runs": [
                    {
                        key: run[key]
                        for key in (
                            "method",
                            "run_id",
                            "training_seed",
                            "checkpoint",
                            "checkpoint_sha256",
                            "videos",
                            "video_sha256",
                            "training_steps",
                            "untrained_fixture",
                        )
                    }
                    for run in runs
                ],
            },
        )
        return record
    except Exception as error:
        _write(output_dir / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--allow-untrained-fixtures", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = generate_samples(
            args.plan,
            args.runs,
            args.output_dir,
            device=args.device,
            allow_untrained=args.allow_untrained_fixtures,
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Sampling failed: {error}\n")
    print(json.dumps({"status": result["status"], "runs": len(result["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
