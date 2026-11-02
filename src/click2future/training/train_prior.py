"""Matched E7 warm-start: conditional diffusion versus prior-only decoding."""

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.prior_motion import PRIOR_CONTRACT, PriorTrajectoryModel
from click2future.models.world_motion import WORLD_CONTRACT

from .train_cvae import require_tc2, sha256
from .train_trajectory import model_config
from .train_world import WorldConfig, WorldTrainer, source_identity
from .world_checkpoint import load_world
from .world_data import load_cache


@dataclass(frozen=True)
class PriorConfig(WorldConfig):
    world_prior_only: bool = True

    def validate(self):
        super().validate()
        if not isinstance(self.world_prior_only, bool) or not (
            self.world_condition and self.relational
        ):
            raise ValueError("Require a boolean prior arm with conditioned relational backbone")


def prior_config(config):
    return {
        **model_config(config),
        "world_condition": config.world_condition,
        "relational": config.relational,
        "world_prior_only": config.world_prior_only,
    }


def prior_sources():
    root = Path(__file__).parents[1]
    return {
        **source_identity(),
        **{
            name: sha256(root / name)
            for name in (
                "models/prior_motion.py",
                "training/train_prior.py",
                "training/prior_checkpoint.py",
            )
        },
    }


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PriorTrainer(WorldTrainer):
    def __init__(self, data, geometry, worlds, config, *, device="cpu", future_frames=31):
        config.validate()
        super().__init__(data, geometry, worlds, config, device=device, future_frames=future_frames)
        if future_frames != 31:
            raise ValueError("Use the frozen 31-future-frame contract")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed + 333)
            self.model = PriorTrajectoryModel(**prior_config(config)).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

    def initialize(self, checkpoint, provenance):
        if self.step or self.optimizer.state:
            raise ValueError("Initialize a fresh prior trainer only")
        base, state = load_world(checkpoint, self.device)
        expected = {k: v for k, v in prior_config(self.config).items() if k != "world_prior_only"}
        if state["model_config"] != expected:
            raise ValueError("Prior teacher architecture differs")
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
            "geometry_sha256",
            "world_label_sha256",
            "training_supplement",
            "combined_training_examples",
        ):
            if key not in provenance or provenance[key] != state["provenance"].get(key):
                raise ValueError("Prior teacher data differ: " + key)
        self.model.load_state_dict(base.state_dict(), strict=True)
        self.ema.load_state_dict(self.model.state_dict(), strict=True)
        return {
            "kind": (
                "E7 EMA, fresh optimizer/private RNG; "
                "fixed-sigma zero-motion input in direct arm"
            ),
            "checkpoint_sha256": sha256(checkpoint),
            "source_step": state["step"],
            "teacher_tensors_sha256": tensor_digest(base.state_dict()),
            "initial_tensors_sha256": tensor_digest(self.model.state_dict()),
            "initial_private_rng_sha256": hashlib.sha256(
                self.generator.get_state().cpu().numpy().tobytes()
            ).hexdigest(),
            "sampling_weights_sha256": hashlib.sha256(
                self.weights.detach().cpu().numpy().tobytes()
            ).hexdigest(),
        }

    def save(self, path, provenance):
        state = {
            "trajectory_format_version": 7,
            "training_variant": "prior_only_comparison",
            "motion_contract": MOTION_CONTRACT,
            "world_contract": WORLD_CONTRACT,
            "prior_contract": PRIOR_CONTRACT,
            "config": asdict(self.config),
            "model_config": prior_config(self.config),
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "provenance": provenance,
        }
        temporary = Path(path).with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path, provenance):
        state = torch.load(path, map_location=self.device, weights_only=True)
        expected = {
            "trajectory_format_version": 7,
            "training_variant": "prior_only_comparison",
            "motion_contract": MOTION_CONTRACT,
            "world_contract": WORLD_CONTRACT,
            "prior_contract": PRIOR_CONTRACT,
            "config": asdict(self.config),
            "model_config": prior_config(self.config),
            "provenance": provenance,
        }
        if (
            any(state[k] != v for k, v in expected.items())
            or isinstance(state["step"], bool)
            or not isinstance(state["step"], int)
            or not 0 <= state["step"] <= self.config.steps
        ):
            raise ValueError("Resume requires identical prior config, data and source")
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
    config = PriorConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    config.validate()
    revision = os.environ["C2F_SOURCE_REVISION"]
    if not revision or revision == "unrecorded":
        raise ValueError("Require immutable source revision")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    data, geometry, worlds, provenance = load_cache(args.cache)
    provenance = {**provenance, "source_sha256": prior_sources(), "source_revision": revision}
    tiny = replace(config, steps=4, width=16, layers=1, batch_size=2)
    probe = PriorTrainer(data, geometry, worlds, tiny, device="cuda")
    probe.train_step()
    probe.save(args.run_dir / "resume-fixture.pt", {"scope": "two-update device fixture"})
    expected = probe.train_step()
    restored = PriorTrainer(data, geometry, worlds, tiny, device="cuda")
    restored.resume(args.run_dir / "resume-fixture.pt", {"scope": "two-update device fixture"})
    actual = restored.train_step()
    differences = {
        kind: max(
            float((v - getattr(restored, kind).state_dict()[k]).abs().max())
            for k, v in getattr(probe, kind).state_dict().items()
        )
        for kind in ("model", "ema")
    }
    if expected != actual or any(differences.values()):
        raise ValueError("Prior CUDA continuation differs")
    (args.run_dir / "resume-check.json").write_text(
        json.dumps(
            {
                "losses_equal": True,
                "max_abs_difference": differences,
                "scope": "two-update device fixture; not model quality",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del probe, restored
    trainer = PriorTrainer(data, geometry, worlds, config, device="cuda")
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
        "selection": "fixed final update; validation loss diagnostic only",
        "test_splits_evaluated": [],
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.run_dir / "training.jsonl").open("w", encoding="utf-8", buffering=1) as log:
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
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "steps": trainer.step}), flush=True)


if __name__ == "__main__":
    main()
