"""Bounded supervised object-trajectory EDM pilot; TC2 training only."""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT, TrajectoryDiffusion

from .train_cvae import require_tc2, sha256
from .trajectory_data import prepare_trajectories


@dataclass(frozen=True)
class TrajectoryConfig:
    seed: int = 19
    steps: int = 6000
    batch_size: int = 64
    width: int = 128
    layers: int = 4
    learning_rate: float = 0.0003
    ema_decay: float = 0.995
    sigma_data: float = 0.25
    log_sigma_mean: float = -1.2
    log_sigma_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 20.0
    temporal_weight: float = 0.1
    checkpoint_every: int = 500
    max_seconds: int = 1200
    augment: bool = True
    force_limit: float = 1.1

    def validate(self):
        for key in ("steps", "batch_size", "width", "layers", "checkpoint_every", "max_seconds"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Positive integer configuration required: " + key)
        if (
            self.width < 16
            or self.width % 4
            or not 0 < self.sigma_min < self.sigma_max
            or not 0 <= self.ema_decay < 1
            or self.sigma_data <= 0
            or self.learning_rate <= 0
            or self.log_sigma_std <= 0
            or self.temporal_weight < 0
            or self.max_seconds > 19800
            or not 0 < self.force_limit <= 1.5
            or not isinstance(self.augment, bool)
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not all(math.isfinite(value) for value in asdict(self).values())
        ):
            raise ValueError("Invalid trajectory configuration")


def model_config(config):
    return {
        "future_frames": 31,
        "width": config.width,
        "layers": config.layers,
        "sigma_data": config.sigma_data,
    }


def edm_objective(model, clean, features, force, sigma, noise, temporal_weight):
    noisy = clean + sigma[:, None, None, None] * noise
    prediction = model(noisy, sigma, features, force)
    error = prediction - clean
    weight = (sigma.square() + model.sigma_data**2) / (sigma * model.sigma_data).square()
    point = error.square().flatten(1).mean(1)
    temporal = (
        (error[:, 1:] - error[:, :-1]).square().flatten(1).mean(1)
        if error.shape[1] > 1
        else torch.zeros_like(point)
    )
    return (weight * (point + temporal_weight * temporal)).mean()


class TrajectoryTrainer:
    """Explicit CPU construction is for unit tests; CLI training requires TC2."""

    def __init__(self, data, config, *, device="cpu", future_frames=31):
        config.validate()
        self.config, self.device = config, device
        self.data = {
            split: {
                key: torch.as_tensor(row[key], device=device, dtype=torch.float32)
                for key in ("features", "force", "motion")
            }
            for split, row in data.items()
        }
        if set(self.data) != {"train", "validation"}:
            raise ValueError("Provide training and validation only")
        for row in self.data.values():
            count = len(row["motion"])
            if (
                count < 1
                or row["motion"].shape != (count, future_frames, 2, 4)
                or row["features"].shape != (count, 2, 11)
                or row["force"].shape != (count, 2)
                or any(not torch.isfinite(value).all() for value in row.values())
                or (row["force"].norm(dim=-1) > config.force_limit + 1e-6).any()
            ):
                raise ValueError("Invalid trajectory training arrays or force range")
        torch.manual_seed(config.seed)
        self.model = TrajectoryDiffusion(
            future_frames=future_frames,
            width=config.width,
            layers=config.layers,
            sigma_data=config.sigma_data,
        ).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )
        self.generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        self.step = 0

    def train_step(self):
        c, row = self.config, self.data["train"]
        indices = torch.randint(
            len(row["motion"]), (c.batch_size,), device=self.device, generator=self.generator
        )
        clean, features, force = (row[key][indices] for key in ("motion", "features", "force"))
        sigma = (
            (
                torch.randn(c.batch_size, device=self.device, generator=self.generator)
                * c.log_sigma_std
                + c.log_sigma_mean
            )
            .exp()
            .clamp(c.sigma_min, c.sigma_max)
        )
        noise = torch.randn(clean.shape, device=self.device, generator=self.generator)
        loss = edm_objective(self.model, clean, features, force, sigma, noise, c.temporal_weight)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite trajectory loss")
        rate = c.learning_rate * (0.03 + 0.97 * (1 + math.cos(math.pi * self.step / c.steps)) / 2)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(
                self.ema.parameters(), self.model.parameters(), strict=True
            ):
                averaged.lerp_(current, 1 - c.ema_decay)
        self.step += 1
        return {
            "step": self.step,
            "loss": float(loss.detach()),
            "lr": rate,
            "gradient_norm": float(norm),
        }

    @torch.inference_mode()
    def validate(self):
        row, c = self.data["validation"], self.config
        generator = torch.Generator(device=self.device).manual_seed(c.seed + 100000)
        losses = []
        for start in range(0, len(row["motion"]), c.batch_size):
            clean, features, force = (
                row[key][start : start + c.batch_size] for key in ("motion", "features", "force")
            )
            sigma = (
                (
                    torch.randn(len(clean), device=self.device, generator=generator)
                    * c.log_sigma_std
                    + c.log_sigma_mean
                )
                .exp()
                .clamp(c.sigma_min, c.sigma_max)
            )
            noise = torch.randn(clean.shape, device=self.device, generator=generator)
            value = edm_objective(self.ema, clean, features, force, sigma, noise, c.temporal_weight)
            losses.append((len(clean), float(value)))
        return sum(size * loss for size, loss in losses) / sum(size for size, _ in losses)

    def save(self, path, provenance, *, elapsed_seconds=0, best_validation=None):
        state = {
            "trajectory_format_version": 1,
            "motion_contract": MOTION_CONTRACT,
            "config": asdict(self.config),
            "model_config": {
                "future_frames": self.model.future_frames,
                "width": self.config.width,
                "layers": self.config.layers,
                "sigma_data": self.config.sigma_data,
            },
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "provenance": provenance,
            "elapsed_seconds": elapsed_seconds,
            "best_validation": best_validation,
        }
        temporary = Path(path).with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path, provenance):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if (
            state["config"] != asdict(self.config)
            or state["provenance"] != provenance
            or state["trajectory_format_version"] != 1
            or state["motion_contract"] != MOTION_CONTRACT
            or state["model_config"]["future_frames"] != self.model.future_frames
        ):
            raise ValueError("Resume requires identical configuration, sources and data")
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.step = state["step"]
        return state


def check_resume_on_device(data, config, path, *, device):
    """Four fixture updates test actual device/optimizer state, not quality."""
    small = {
        split: {key: row[key][:4] for key in ("features", "force", "motion")}
        for split, row in data.items()
    }
    probe = replace(config, steps=4, width=16, layers=1, batch_size=2)
    trainer = TrajectoryTrainer(small, probe, device=device)
    trainer.train_step()
    trainer.train_step()
    identity = {"scope": "four-update checkpoint continuation fixture; not a model result"}
    trainer.save(path, identity)
    expected_losses = [trainer.train_step()["loss"], trainer.train_step()["loss"]]
    restored = TrajectoryTrainer(small, probe, device=device)
    restored.resume(path, identity)
    actual_losses = [restored.train_step()["loss"], restored.train_step()["loss"]]
    maximum = max(
        float((value - restored.model.state_dict()[key]).abs().max())
        for key, value in trainer.model.state_dict().items()
    )
    ema_maximum = max(
        float((value - restored.ema.state_dict()[key]).abs().max())
        for key, value in trainer.ema.state_dict().items()
    )
    if maximum != 0 or ema_maximum != 0 or expected_losses != actual_losses:
        raise ValueError("Device checkpoint continuation is not exact")
    return {
        "device": str(device),
        "model_max_abs_difference": maximum,
        "ema_max_abs_difference": ema_maximum,
        "losses_equal": expected_losses == actual_losses,
        "scope": identity["scope"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    require_tc2()
    config = TrajectoryConfig(**json.loads(args.config.read_text()))
    config.validate()
    if args.run_dir.exists() and not args.resume:
        parser.error("Use a new run directory; preserve attempted experiments")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data, provenance = prepare_trajectories(args.dataset, augment=config.augment)
    provenance.pop("preparation_seconds", None)
    provenance["source_revision"] = os.environ.get("C2F_SOURCE_REVISION", "unrecorded")
    provenance["source_sha256"] = {
        name: sha256(path)
        for name, path in {
            "object_motion.py": Path(__file__).parents[1] / "models/object_motion.py",
            "trajectory_data.py": Path(__file__).with_name("trajectory_data.py"),
            "train_trajectory.py": Path(__file__),
        }.items()
    }
    if not args.resume:
        check = check_resume_on_device(
            data, config, args.run_dir / "resume-unit-fixture.pt", device="cuda"
        )
        (args.run_dir / "resume-check.json").write_text(json.dumps(check, indent=2) + "\n")
    trainer = TrajectoryTrainer(data, config, device="cuda")
    best, previous = None, 0
    if args.resume:
        state = trainer.resume(args.resume, provenance)
        best, previous = state["best_validation"], state["elapsed_seconds"]
    summary = {
        "scope": "supervised trajectory diffusion; one seed; validation pilot only",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "config": asdict(config),
        "provenance": provenance,
        "parameters": sum(p.numel() for p in trainer.model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    with (args.run_dir / "training.jsonl").open("a", buffering=1) as log:
        while trainer.step < config.steps:
            record = trainer.train_step()
            if trainer.step % 50 == 0:
                record["elapsed_seconds"] = time.perf_counter() - started
                log.write(json.dumps(record) + "\n")
            if trainer.step % config.checkpoint_every == 0 or trainer.step == config.steps:
                validation = trainer.validate()
                elapsed = previous + time.perf_counter() - started
                if best is None or validation < best:
                    best = validation
                    trainer.save(
                        args.run_dir / "best.pt",
                        provenance,
                        elapsed_seconds=elapsed,
                        best_validation=best,
                    )
                trainer.save(
                    args.run_dir / "latest.pt",
                    provenance,
                    elapsed_seconds=elapsed,
                    best_validation=best,
                )
                log.write(
                    json.dumps({"step": trainer.step, "validation_denoising_loss": validation})
                    + "\n"
                )
            if previous + time.perf_counter() - started >= config.max_seconds:
                break
    elapsed = previous + time.perf_counter() - started
    trainer.save(
        args.run_dir / "latest.pt", provenance, elapsed_seconds=elapsed, best_validation=best
    )
    summary.update(
        status="completed" if trainer.step == config.steps else "checkpointed_time_limit",
        steps_completed=trainer.step,
        elapsed_seconds=elapsed,
        best_validation_denoising_loss=best,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "step": trainer.step}), flush=True)


if __name__ == "__main__":
    main()
