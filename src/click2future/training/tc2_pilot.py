"""Two-allocation resume check and tiny-data learning/sampling experiment."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from click2future.data import generate_dataset, load_rollout

from .sample_diffusion import load_generator, sample_videos, save_preview
from .train_diffusion import TrainConfig, Trainer
from .train_diffusion import main as train_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first", "resume"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("This experiment runs only in a Slurm allocation")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    root = args.root
    dataset, run = root / "data", root / "run"
    config_path = root / "config.json"
    if args.stage == "first":
        if config_path.exists():
            parser.error("Pilot already exists; choose a new root")
        root.mkdir(parents=True, exist_ok=True)
        config = TrainConfig(precision="bfloat16", limit_train=4)
        config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        generate_dataset(
            dataset,
            group_count=16,
            variants_per_group=2,
            seed=610132,
            config={
                "frame_count": 32,
                "resolution": 64,
                "fps": 8,
                "physics_hz": 240,
                "force_duration": 0.25,
            },
        )
        train_main(
            [
                "--dataset",
                str(dataset),
                "--run-dir",
                str(run),
                "--config",
                str(config_path),
                "--device",
                "cuda",
                "--require-slurm",
                "--until",
                "200",
            ]
        )
        trainer = Trainer(dataset, config, device="cuda")
        trainer.resume(run / "checkpoint.pt")
        expected_record = trainer.train_step()
        trainer.save(run / "expected-next-step.pt")
        (run / "expected-next-step.json").write_text(
            json.dumps({"record": expected_record, "job": os.environ["SLURM_JOB_ID"]}, indent=2),
            encoding="utf-8",
        )
        return

    config = TrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    trainer = Trainer(dataset, config, device="cuda")
    trainer.resume(run / "checkpoint.pt")
    actual_record = trainer.train_step()
    expected = torch.load(run / "expected-next-step.pt", weights_only=True, map_location="cpu")
    expected_record = json.loads((run / "expected-next-step.json").read_text(encoding="utf-8"))
    # Compare optimizer moments, EMA, model and RNG, not merely a weight reload.
    trainer.save(run / "resumed-next-step.pt")
    actual = torch.load(run / "resumed-next-step.pt", weights_only=True, map_location="cpu")
    largest = 0.0

    def compare(left, right):
        nonlocal largest
        if isinstance(left, torch.Tensor):
            if left.is_floating_point():
                largest = max(largest, float((left - right).abs().max()))
                torch.testing.assert_close(left, right, atol=1e-7, rtol=1e-6)
            else:
                assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                compare(a, b)
        else:
            assert left == right

    for key in (
        "model",
        "ema",
        "optimizer",
        "scheduler",
        "rng",
        "order_rng",
        "order",
        "cursor",
        "step",
    ):
        compare(expected[key], actual[key])
    assert actual_record["sample_ids"] == expected_record["record"]["sample_ids"]
    resume_report = {
        "status": "passed",
        "first_job": expected_record["job"],
        "resume_job": os.environ["SLURM_JOB_ID"],
        "compared_step": trainer.step,
        "largest_tensor_absolute_difference": largest,
        "comparison_tolerance": {"atol": 1e-7, "rtol": 1e-6},
        "expected_loss": expected_record["record"]["loss"],
        "resumed_loss": actual_record["loss"],
        "components": ["model", "ema", "optimizer", "scheduler", "all_rng", "data_order"],
    }
    (root / "resume-check.json").write_text(json.dumps(resume_report, indent=2), encoding="utf-8")
    del trainer
    torch.cuda.empty_cache()
    train_main(
        [
            "--dataset",
            str(dataset),
            "--run-dir",
            str(run),
            "--config",
            str(config_path),
            "--device",
            "cuda",
            "--require-slurm",
            "--resume",
            str(run / "checkpoint.pt"),
        ]
    )
    model, diffusion, _ = load_generator(run / "checkpoint.pt", "cuda")
    sample_id = expected["provenance"]["training_sample_ids"][0]
    reference = load_rollout(dataset, sample_id)
    save_preview(reference["frames"][None], root / "reference.gif")
    import time

    timings = {}
    for kind, force in (
        ("forward", reference["force_xy"]),
        ("reverse", -reference["force_xy"]),
        ("zero", np.zeros(2, dtype=np.float32)),
    ):
        started = time.perf_counter()
        videos = sample_videos(
            model,
            diffusion,
            reference["frames"][0],
            reference["selected_mask"],
            force,
            seeds=(100, 101, 102, 103),
            future_frames=31,
            sampling_steps=32,
        )
        timings[kind] = time.perf_counter() - started
        np.savez_compressed(root / f"{kind}.npz", videos=videos)
        save_preview(videos, root / f"{kind}.gif")
    (root / "sampling.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "split": "train",
                "seeds": [100, 101, 102, 103],
                "steps": 32,
                "weights": "ema",
                "four_sample_seconds": timings,
                "scope": "Training-scene plumbing diagnostic; action success not established",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
