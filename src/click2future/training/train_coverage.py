"""Explicit E7-to-E7 refinement with an attested training-only cache addition."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .train_cvae import require_tc2, sha256
from .train_trajectory import model_config
from .train_world import WorldConfig, WorldTrainer, source_identity
from .world_checkpoint import load_world
from .world_data import load_cache


class CoverageTrainer(WorldTrainer):
    def initialize(self, checkpoint, provenance):
        if self.step or self.optimizer.state:
            raise ValueError("Initialize a fresh coverage trainer only")
        addition = provenance["coverage_addition"]
        parent = addition["parent_provenance"]
        base, state = load_world(checkpoint, self.device)
        if state["model_config"] != {
            **model_config(self.config),
            "world_condition": True,
            "relational": True,
        } or not (self.config.world_condition and self.config.relational):
            raise ValueError("Coverage refinement requires the identical conditioned E7 model")
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
            if key not in parent or parent[key] != state["provenance"].get(key):
                raise ValueError("Coverage parent differs from teacher data: " + key)
        for key in (
            "dataset_manifest_sha256",
            "split_manifest_sha256",
            "train_validation_content_sha256",
            "angular_labels_sha256",
        ):
            if provenance[key] != parent[key]:
                raise ValueError("Original corpus identity changed: " + key)
        if any(
            provenance[key]["validation"] != parent[key]["validation"]
            for key in ("geometry_sha256", "world_label_sha256")
        ):
            raise ValueError("Validation labels changed")
        self.model.load_state_dict(base.state_dict(), strict=True)
        self.ema.load_state_dict(base.state_dict(), strict=True)
        return {
            "kind": "conditioned E7 EMA; all parameters trainable; fresh optimizer/private RNG",
            "checkpoint_sha256": sha256(checkpoint),
            "source_step": state["step"],
            "source_revision": state["provenance"].get("source_revision", "unrecorded"),
            "scope": "explicit training-only data extension, not resume on the original cache",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "warm-start", "config", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    config = WorldConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    config.validate()
    revision = os.environ["C2F_SOURCE_REVISION"]
    if not revision or revision == "unrecorded":
        raise ValueError("Require immutable source revision")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    data, geometry, worlds, provenance = load_cache(args.cache)
    sources = source_identity()
    for name in ("coverage_data.py", "train_coverage.py"):
        sources["training/" + name] = sha256(Path(__file__).with_name(name))
    provenance = {**provenance, "source_sha256": sources, "source_revision": revision}
    tiny = replace(config, steps=4, width=16, layers=1, batch_size=2)
    probe = CoverageTrainer(data, geometry, worlds, tiny, device="cuda")
    probe.train_step()
    probe.save(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    expected = probe.train_step()
    restored = CoverageTrainer(data, geometry, worlds, tiny, device="cuda")
    restored.resume(args.run_dir / "resume-fixture.pt", {"scope": "device fixture"})
    actual = restored.train_step()
    differences = {
        kind: max(
            float((v - getattr(restored, kind).state_dict()[k]).abs().max())
            for k, v in getattr(probe, kind).state_dict().items()
        )
        for kind in ("model", "ema")
    }
    if expected != actual or any(differences.values()):
        raise ValueError("Coverage CUDA continuation differs")
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
    trainer = CoverageTrainer(data, geometry, worlds, config, device="cuda")
    provenance["initialization"] = trainer.initialize(args.warm_start, provenance)
    initial = hashlib.sha256()
    for key, value in sorted(trainer.model.state_dict().items()):
        initial.update(key.encode())
        initial.update(value.detach().cpu().numpy().tobytes())
    summary = {
        "config": asdict(config),
        "provenance": provenance,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
        "initial_model_tensors_sha256": initial.hexdigest(),
        "initial_private_rng_sha256": hashlib.sha256(
            trainer.generator.get_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "sampling_weights_sha256": hashlib.sha256(
            trainer.weights.cpu().numpy().tobytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "selection": "fixed final update; no validation selection",
        "scope": "post-D014 exploratory; paired training coverage; reused development validation",
        "test_splits_evaluated": [],
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.run_dir / "training.jsonl").open("w", buffering=1, encoding="utf-8") as log:
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
