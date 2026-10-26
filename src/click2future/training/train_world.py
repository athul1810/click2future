"""Matched E7 refinement with physical-world labels used only during training."""

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.world_motion import WORLD_CONTRACT, BoundWorld, WorldTrajectoryDiffusion

from .contact_geometry import contact_objective
from .train_contact import ContactConfig, ContactTrainer
from .train_cvae import require_tc2, sha256
from .train_trajectory import edm_objective, model_config
from .world_data import load_cache


@dataclass(frozen=True)
class WorldConfig(ContactConfig):
    world_condition: bool = True
    relational: bool = True

    def validate(self):
        if not isinstance(self.world_condition, bool) or not isinstance(self.relational, bool):
            raise ValueError("world_condition must be boolean")
        super().validate()


def source_identity():
    root = Path(__file__).parents[1]
    return {
        name: sha256(root / name)
        for name in (
            "models/object_motion.py",
            "models/world_motion.py",
            "training/train_world.py",
            "training/world_data.py",
            "training/contact_geometry.py",
            "training/train_contact.py",
            "training/train_trajectory.py",
        )
    }


class WorldTrainer(ContactTrainer):
    def __init__(self, data, geometry, worlds, config, *, device="cpu", future_frames=31):
        super().__init__(data, geometry, config, device=device, future_frames=future_frames)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 333)
            model = WorldTrajectoryDiffusion(
                **{**model_config(config), "future_frames": future_frames},
                world_condition=config.world_condition,
                relational=config.relational,
            )
        self.model = model.to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.worlds = {key: torch.as_tensor(value, device=device) for key, value in worlds.items()}
        if set(self.worlds) != set(self.data) or any(
            value.shape != (len(self.data[split]["motion"]), 2, 3)
            or not torch.isfinite(value).all()
            for split, value in self.worlds.items()
        ):
            raise ValueError("World labels must align with both checked splits")

    def initialize(self, checkpoint, provenance):
        from .contact_checkpoint import load_contact

        if self.step or self.optimizer.state:
            raise ValueError("Initialize a fresh world trainer only")
        base, state = load_contact(checkpoint, self.device)
        if state["model_config"] != model_config(self.config):
            raise ValueError("Warm-start backbone differs")
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
        ):
            if key not in provenance or provenance[key] != state["provenance"].get(key):
                raise ValueError("Warm-start data differ: " + key)
        missing, unexpected = self.model.load_state_dict(base.state_dict(), strict=False)
        if (
            unexpected
            or not missing
            or any(not key.startswith(("world_context.", "messages.")) for key in missing)
        ):
            raise ValueError("Unexpected backbone migration")
        self.ema.load_state_dict(self.model.state_dict(), strict=True)
        return {
            "kind": "wall EMA backbone; zero-output new adapters; fresh optimizer/private RNG",
            "checkpoint_sha256": sha256(checkpoint),
            "source_step": state["step"],
            "source_revision": state["provenance"].get("source_revision", "unrecorded"),
        }

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
        loss, parts = contact_objective(
            BoundWorld(self.model, self.worlds["train"][indices]),
            clean,
            features,
            force,
            sigma,
            noise,
            {key: value[indices] for key, value in self.geometry["train"].items()},
            c,
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite world loss")
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

    @torch.inference_mode()
    def validate(self):
        c, row = self.config, self.data["validation"]
        generator = torch.Generator(device=self.device).manual_seed(c.seed + 100000)
        values = []
        for start in range(0, len(row["motion"]), c.batch_size):
            stop = start + c.batch_size
            clean, features, force = (
                row[key][start:stop] for key in ("motion", "features", "force")
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
            loss = edm_objective(
                BoundWorld(self.ema, self.worlds["validation"][start:stop]),
                clean,
                features,
                force,
                sigma,
                noise,
                c.temporal_weight,
            )
            values.append((len(clean), float(loss)))
        return sum(size * loss for size, loss in values) / sum(size for size, _ in values)

    def save(self, path, provenance, *, elapsed_seconds=0, best_validation=None):
        state = {
            "trajectory_format_version": 3,
            "training_variant": "shared_world_relational",
            "motion_contract": MOTION_CONTRACT,
            "world_contract": WORLD_CONTRACT,
            "config": asdict(self.config),
            "model_config": {
                **model_config(self.config),
                "world_condition": self.config.world_condition,
                "relational": self.config.relational,
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
            state["trajectory_format_version"] != 3
            or state["world_contract"] != WORLD_CONTRACT
            or state["motion_contract"] != MOTION_CONTRACT
            or state["config"] != asdict(self.config)
            or state["provenance"] != provenance
            or state["model_config"]
            != {
                **model_config(self.config),
                "world_condition": self.config.world_condition,
                "relational": self.config.relational,
            }
        ):
            raise ValueError("Resume requires identical world config, data and source")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.step = state["step"]
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "warm-start", "config", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    config = WorldConfig(**json.loads(args.config.read_text()))
    config.validate()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    data, geometry, worlds, provenance = load_cache(args.cache)
    provenance = {
        **provenance,
        "source_sha256": source_identity(),
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
    }
    # Exercise the new architecture and loss on this device before the actual run.
    from dataclasses import replace

    probe_config = replace(config, steps=4, width=16, layers=1, batch_size=2)
    probe = WorldTrainer(data, geometry, worlds, probe_config, device="cuda")
    probe.train_step()
    probe.save(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    expected = probe.train_step()
    restored = WorldTrainer(data, geometry, worlds, probe_config, device="cuda")
    restored.resume(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    actual = restored.train_step()
    maximum = max(
        float((v - restored.model.state_dict()[k]).abs().max())
        for k, v in probe.model.state_dict().items()
    )
    ema_maximum = max(
        float((v - restored.ema.state_dict()[k]).abs().max())
        for k, v in probe.ema.state_dict().items()
    )
    if expected != actual or maximum or ema_maximum:
        raise ValueError("World CUDA continuation differs")
    (args.run_dir / "resume-check.json").write_text(
        json.dumps(
            {
                "scope": "two-update device fixture; not full-size quality",
                "losses_equal": True,
                "model_max_abs_difference": maximum,
                "ema_max_abs_difference": ema_maximum,
            },
            indent=2,
        )
        + "\n"
    )
    del probe, restored
    trainer = WorldTrainer(data, geometry, worlds, config, device="cuda")
    provenance["initialization"] = trainer.initialize(args.warm_start, provenance)
    summary = {
        "config": asdict(config),
        "provenance": provenance,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "selection": "fixed final update, not validation loss or physics scores",
        "validation_loss_scope": "known-world diagnostic; not prior predictive likelihood",
        "scope": "post-D014 exploratory warm-start; physical labels are extra supervision",
        "test_splits_evaluated": [],
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.run_dir / "training.jsonl").open("w", buffering=1) as log:
        while trainer.step < config.steps:
            row = trainer.train_step()
            if trainer.step % 50 == 0:
                row["elapsed_seconds"] = time.perf_counter() - started
                log.write(json.dumps(row) + "\n")
            if trainer.step % config.checkpoint_every == 0:
                log.write(
                    json.dumps(
                        {"step": trainer.step, "known_world_validation_loss": trainer.validate()}
                    )
                    + "\n"
                )
                trainer.save(args.run_dir / "latest.pt", provenance)
            if time.perf_counter() - started >= config.max_seconds:
                break
    trainer.save(args.run_dir / "latest.pt", provenance)
    completed = trainer.step == config.steps
    if completed:
        trainer.save(args.run_dir / "best.pt", provenance)
    summary.update(
        status="completed" if completed else "checkpointed_time_limit",
        steps_completed=trainer.step,
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "steps": trainer.step}), flush=True)


if __name__ == "__main__":
    main()
