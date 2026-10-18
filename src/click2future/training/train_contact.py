"""Separate contact/stationarity training pilot, with unchanged neural sampling."""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT

from .contact_geometry import contact_objective, prepare_contact_geometry, signed_penetration
from .train_cvae import require_tc2, sha256
from .train_trajectory import TrajectoryConfig, TrajectoryTrainer, model_config
from .trajectory_data import prepare_trajectories


@dataclass(frozen=True)
class ContactConfig(TrajectoryConfig):
    penetration_weight: float = 64.0
    contact_weight: float = 16.0
    acceleration_weight: float = 2.0
    stationary_weight: float = 4.0
    physics_sigma_max: float = 0.3
    penetration_margin_pixels: float = 1.0
    contact_window_pixels: float = 2.0
    zero_probability: float = 0.25
    collision_probability: float = 0.4

    def validate(self):
        super().validate()
        if (
            min(
                self.penetration_weight,
                self.contact_weight,
                self.acceleration_weight,
                self.stationary_weight,
            )
            < 0
            or not self.sigma_min < self.physics_sigma_max <= self.sigma_max
            or not 0 < self.penetration_margin_pixels <= 2
            or not 0 < self.contact_window_pixels <= 4
            or not 0 < self.zero_probability < 1
            or not 0 < self.collision_probability < 1 - self.zero_probability
        ):
            raise ValueError("Invalid contact/stationary training configuration")


class ContactTrainer(TrajectoryTrainer):
    def __init__(self, data, geometry, config, *, device="cpu", future_frames=31):
        super().__init__(data, config, device=device, future_frames=future_frames)
        self.geometry = {
            split: {key: torch.as_tensor(value, device=device) for key, value in row.items()}
            for split, row in geometry.items()
        }
        if set(self.geometry) != set(self.data):
            raise ValueError("Geometry must match both declared splits")
        for split, row in self.geometry.items():
            count = len(self.data[split]["motion"])
            if (
                set(row) != {"hulls", "normals", "axis_valid"}
                or row["hulls"].shape != (count, 2, 32, 2)
                or row["normals"].shape != row["hulls"].shape
                or row["axis_valid"].shape != (count, 2, 32)
                or row["axis_valid"].dtype != torch.bool
                or (row["axis_valid"].sum(-1) < 3).any()
                or not torch.isfinite(row["hulls"]).all()
                or not torch.isfinite(row["normals"]).all()
            ):
                raise ValueError("Invalid visible hull geometry")
        force = self.data["train"]["force"]
        zero = (force == 0).all(-1)
        collision = (
            torch.tensor(
                ["-collision_" in item["sample_id"] for item in data["train"]["examples"]],
                device=device,
            )
            & ~zero
        )
        pools = (zero, collision, ~(zero | collision))
        self.weights = torch.zeros(len(force), device=device)
        for pool, probability in zip(
            pools,
            (
                config.zero_probability,
                config.collision_probability,
                1 - config.zero_probability - config.collision_probability,
            ),
            strict=True,
        ):
            if not pool.any():
                raise ValueError("Training needs zero, collision and other action families")
            self.weights[pool] = probability / pool.sum()
        if self.data["train"]["motion"][zero].abs().max() > 1e-7:
            raise ValueError("Zero-force training controls must be stationary")

    def train_step(self):
        c, row = self.config, self.data["train"]
        indices = torch.multinomial(
            self.weights, c.batch_size, replacement=True, generator=self.generator
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
        geometry = {key: value[indices] for key, value in self.geometry["train"].items()}
        loss, parts = contact_objective(
            self.model, clean, features, force, sigma, noise, geometry, c
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite contact-aware loss")
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
            **{key: float(value) for key, value in parts.items()},
        }

    def save(self, path, provenance, *, elapsed_seconds=0, best_validation=None):
        state = {
            "trajectory_format_version": 2,
            "training_variant": "contact_regularized",
            "motion_contract": MOTION_CONTRACT,
            "config": asdict(self.config),
            "model_config": {
                **model_config(self.config),
                "future_frames": self.model.future_frames,
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
            state["trajectory_format_version"] != 2
            or state["training_variant"] != "contact_regularized"
            or state["motion_contract"] != MOTION_CONTRACT
            or state["config"] != asdict(self.config)
            or state["provenance"] != provenance
            or state["model_config"]
            != {**model_config(self.config), "future_frames": self.model.future_frames}
        ):
            raise ValueError("Resume requires identical contact config, sources and data")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.step = state["step"]
        return state


def resume_probe(data, geometry, config, path, *, device):
    probe = replace(config, steps=4, width=16, layers=1, batch_size=2)
    trainer = ContactTrainer(data, geometry, probe, device=device)
    trainer.train_step()
    trainer.train_step()
    identity = {"scope": "four unit-fixture updates; not model-quality evidence"}
    trainer.save(path, identity)
    expected = [trainer.train_step()["loss"], trainer.train_step()["loss"]]
    restored = ContactTrainer(data, geometry, probe, device=device)
    restored.resume(path, identity)
    actual = [restored.train_step()["loss"], restored.train_step()["loss"]]
    maxima = {
        key: max(
            float((value - getattr(restored, key).state_dict()[name]).abs().max())
            for name, value in getattr(trainer, key).state_dict().items()
        )
        for key in ("model", "ema")
    }
    if expected != actual or any(maxima.values()):
        raise ValueError("Contact-aware optimizer continuation is not exact")
    return {
        "device": device,
        "maximum_abs_difference": maxima,
        "losses_equal": True,
        "scope": identity["scope"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run-dir", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    require_tc2()
    config = ContactConfig(**json.loads(args.config.read_text()))
    config.validate()
    if args.run_dir.exists() and not args.resume:
        parser.error("Preserve earlier attempts; use a fresh run directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data, provenance = prepare_trajectories(args.dataset, augment=config.augment)
    provenance.pop("preparation_seconds", None)
    geometry = prepare_contact_geometry(args.dataset, data)
    provenance["geometry_sha256"] = {
        split: {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in row.items()}
        for split, row in geometry.items()
    }
    provenance["source_revision"] = os.environ.get("C2F_SOURCE_REVISION", "unrecorded")
    source = Path(__file__).parents[1]
    provenance["source_sha256"] = {
        str(path.relative_to(source)): sha256(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("contact_geometry.py"),
            Path(__file__).with_name("trajectory_data.py"),
            Path(__file__).with_name("train_trajectory.py"),
            source / "models/object_motion.py",
            source / "evaluation/rigid_geometry.py",
        )
    }
    if not args.resume:
        check = resume_probe(
            data, geometry, config, args.run_dir / "resume-unit-fixture.pt", device="cuda"
        )
        (args.run_dir / "resume-check.json").write_text(json.dumps(check, indent=2) + "\n")
    trainer = ContactTrainer(data, geometry, config, device="cuda")
    best, previous = None, 0
    if args.resume:
        state = trainer.resume(args.resume, provenance)
        best, previous = state["best_validation"], state["elapsed_seconds"]
    calibration = {}
    with torch.no_grad():
        for split, row in trainer.data.items():
            maxima = []
            for start in range(0, len(row["motion"]), 64):
                stop = start + 64
                depth = signed_penetration(
                    row["motion"][start:stop],
                    row["features"][start:stop],
                    {key: value[start:stop] for key, value in trainer.geometry[split].items()},
                )
                maxima.append(float(depth.max() * 64))
            calibration[split] = max(maxima)
    summary = {
        "scope": "E6 contact/stationarity training package; one seed; validation only",
        "config": asdict(config),
        "provenance": provenance,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "oracle_geometry_max_penetration_pixels": calibration,
        "selection": "fixed-seed validation EDM loss, unchanged from E5",
        "sampler": "unchanged E5 Heun; no projection, clamp or zero override",
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    with (args.run_dir / "training.jsonl").open("a", buffering=1) as log:
        while trainer.step < config.steps:
            record = trainer.train_step()
            if trainer.step % 50 == 0:
                record["elapsed_seconds"] = time.perf_counter() - started
                log.write(json.dumps(record) + "\n")
            elapsed = previous + time.perf_counter() - started
            if trainer.step % config.checkpoint_every == 0 or trainer.step == config.steps:
                value = trainer.validate()
                if best is None or value < best:
                    best = value
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
                    json.dumps({"step": trainer.step, "validation_denoising_loss": value}) + "\n"
                )
            if elapsed >= config.max_seconds:
                break
    trainer.save(
        args.run_dir / "latest.pt",
        provenance,
        elapsed_seconds=previous + time.perf_counter() - started,
        best_validation=best,
    )
    summary.update(
        status="completed" if trainer.step == config.steps else "checkpointed_time_limit",
        steps_completed=trainer.step,
        elapsed_seconds=previous + time.perf_counter() - started,
        best_validation_denoising_loss=best,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "steps": trainer.step}), flush=True)


if __name__ == "__main__":
    main()
