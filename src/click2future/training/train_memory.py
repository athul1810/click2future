"""Frozen E7 backbone, matched reset/carried temporal residual training on TC2."""

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from click2future.models.contact_memory import MEMORY_CONTRACT, ContactMemoryDiffusion
from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.world_motion import WORLD_CONTRACT

from .train_cvae import require_tc2, sha256
from .train_trajectory import model_config
from .train_world import WorldConfig, WorldTrainer
from .world_checkpoint import load_world
from .world_data import load_cache


@dataclass(frozen=True)
class MemoryConfig(WorldConfig):
    temporal_memory: bool = True
    memory_width: int = 64

    def validate(self):
        super().validate()
        if (
            not isinstance(self.temporal_memory, bool)
            or isinstance(self.memory_width, bool)
            or not isinstance(self.memory_width, int)
            or not 8 <= self.memory_width <= 256
            or not self.world_condition
            or not self.relational
        ):
            raise ValueError("Invalid temporal residual configuration")


def memory_model_config(config):
    return {
        **model_config(config),
        "relational": config.relational,
        "world_condition": config.world_condition,
        "temporal_memory": config.temporal_memory,
        "memory_width": config.memory_width,
    }


def source_identity():
    root = Path(__file__).parents[1]
    from .train_world import source_identity as world_sources

    return {
        **world_sources(),
        **{
            name: sha256(root / name)
            for name in (
                "models/contact_memory.py",
                "training/train_memory.py",
                "training/memory_checkpoint.py",
            )
        },
    }


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.backbone.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class MemoryTrainer(WorldTrainer):
    """Reuse the checked sampler, loss, batches and private training RNG."""

    def __init__(self, data, geometry, worlds, config, *, device="cpu", future_frames=31):
        super().__init__(data, geometry, worlds, config, device=device, future_frames=future_frames)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 444)
            self.model = ContactMemoryDiffusion(**memory_model_config(config)).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.frozen_sha256 = frozen_digest(self.model)

    def initialize(self, checkpoint, provenance):
        if self.step or self.optimizer.state:
            raise ValueError("Initialize a fresh residual trainer only")
        base, state = load_world(checkpoint, self.device)
        expected = {
            key: value
            for key, value in memory_model_config(self.config).items()
            if key not in {"memory_width", "temporal_memory"}
        }
        if state["model_config"] != expected:
            raise ValueError("E7 backbone differs")
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
            "world_label_sha256",
        ):
            if key not in provenance or provenance[key] != state["provenance"].get(key):
                raise ValueError("Warm-start data differ: " + key)
        self.model.backbone.load_state_dict(base.state_dict(), strict=True)
        self.ema.load_state_dict(self.model.state_dict(), strict=True)
        self.frozen_sha256 = frozen_digest(self.model)
        return {
            "kind": "frozen conditioned E7 EMA; zero-output residual; fresh optimizer/private RNG",
            "checkpoint_sha256": sha256(checkpoint),
            "source_step": state["step"],
            "source_revision": state["provenance"].get("source_revision", "unrecorded"),
            "frozen_backbone_sha256": self.frozen_sha256,
        }

    def save(self, path, provenance):
        if (
            frozen_digest(self.model) != self.frozen_sha256
            or frozen_digest(self.ema) != self.frozen_sha256
        ):
            raise ValueError("Frozen backbone changed")
        state = {
            "trajectory_format_version": 4,
            "training_variant": "temporal_residual",
            "motion_contract": MOTION_CONTRACT,
            "world_contract": WORLD_CONTRACT,
            "memory_contract": MEMORY_CONTRACT,
            "config": asdict(self.config),
            "model_config": memory_model_config(self.config),
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "provenance": provenance,
            "frozen_backbone_sha256": self.frozen_sha256,
        }
        temporary = Path(path).with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path, provenance):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if (
            state["trajectory_format_version"] != 4
            or state["training_variant"] != "temporal_residual"
            or state["memory_contract"] != MEMORY_CONTRACT
            or state["motion_contract"] != MOTION_CONTRACT
            or state["world_contract"] != WORLD_CONTRACT
            or state["config"] != asdict(self.config)
            or state["model_config"] != memory_model_config(self.config)
            or state["provenance"] != provenance
        ):
            raise ValueError("Resume requires identical memory config, data and source")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.step, self.frozen_sha256 = state["step"], state["frozen_backbone_sha256"]
        if (
            frozen_digest(self.model) != self.frozen_sha256
            or frozen_digest(self.ema) != self.frozen_sha256
        ):
            raise ValueError("Frozen backbone identity differs")
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "warm-start", "config", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    config = MemoryConfig(**json.loads(args.config.read_text()))
    config.validate()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    data, geometry, worlds, provenance = load_cache(args.cache)
    provenance = {
        **provenance,
        "source_sha256": source_identity(),
        "source_revision": os.environ["C2F_SOURCE_REVISION"],
    }
    tiny = replace(config, steps=4, width=16, layers=1, batch_size=2, memory_width=8)
    probe = MemoryTrainer(data, geometry, worlds, tiny, device="cuda")
    probe.train_step()
    probe.save(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    expected = probe.train_step()
    restored = MemoryTrainer(data, geometry, worlds, tiny, device="cuda")
    restored.resume(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    actual = restored.train_step()
    differences = {
        name: max(
            float((v - getattr(restored, name).state_dict()[k]).abs().max())
            for k, v in getattr(probe, name).state_dict().items()
        )
        for name in ("model", "ema")
    }
    if expected != actual or any(differences.values()):
        raise ValueError("Device continuation did not match exactly")
    (args.run_dir / "resume-check.json").write_text(
        json.dumps(
            {
                "losses_equal": True,
                "max_abs_difference": differences,
                "scope": "two-update device fixture; not model quality",
            },
            indent=2,
        )
        + "\n"
    )
    del probe, restored
    trainer = MemoryTrainer(data, geometry, worlds, config, device="cuda")
    provenance["initialization"] = trainer.initialize(args.warm_start, provenance)
    summary = {
        "config": asdict(config),
        "provenance": provenance,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "selection": "fixed final update; not validation loss or physics scores",
        "scope": "post-D014 exploratory; frozen E7; physical labels are extra supervision",
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
        trainer.save(args.run_dir / "final.pt", provenance)
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
