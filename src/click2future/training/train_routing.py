"""E9: matched numeric versus spatial force routing on the E6 contact recipe."""

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
from click2future.models.routed_motion import ROUTINGS, RoutedTrajectoryDiffusion

from .contact_geometry import canvas_excess, prepare_contact_geometry
from .train_contact import ContactConfig, ContactTrainer
from .train_cvae import require_tc2, sha256
from .train_trajectory import model_config
from .trajectory_data import add_training_supplement, prepare_trajectories

FORMAT_VERSION = 6
VARIANT = "force_routing"


@dataclass(frozen=True)
class RoutingConfig(ContactConfig):
    # False keeps E6's spatial routing; True gives both objects the numeric force.
    broadcast_force: bool = False

    def validate(self):
        if not isinstance(self.broadcast_force, bool):
            raise ValueError("broadcast_force must be boolean")
        super().validate()


def routing_model_config(config, future_frames=31):
    return {
        **model_config(config),
        "future_frames": future_frames,
        "force_routing": ROUTINGS[int(config.broadcast_force)],
    }


def source_identity():
    root = Path(__file__).parents[1]
    return {
        name: sha256(root / name)
        for name in (
            "models/object_motion.py",
            "models/routed_motion.py",
            "training/train_routing.py",
            "training/train_contact.py",
            "training/train_trajectory.py",
            "training/contact_geometry.py",
            "training/trajectory_data.py",
            "data/action_supplement.py",
            "evaluation/rigid_geometry.py",
            "sim/backend.py",
            "sim/rollout.py",
        )
    }


class RoutingTrainer(ContactTrainer):
    """E6 trainer whose model differs only in force routing.

    The parent builds E6's denoiser under the configured seed. Its weights are
    copied into the routed model, so both arms of a seed start from identical
    tensors and draw identical batches, sigmas and noise.
    """

    def __init__(self, data, geometry, config, *, device="cpu", future_frames=31):
        super().__init__(data, geometry, config, device=device, future_frames=future_frames)
        with torch.random.fork_rng(devices=[]):
            model = RoutedTrajectoryDiffusion(**routing_model_config(config, future_frames))
        model.load_state_dict(self.model.state_dict(), strict=True)
        self.model = model.to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=0.01
        )

    def save(self, path, provenance, *, elapsed_seconds=0, best_validation=None):
        state = {
            "trajectory_format_version": FORMAT_VERSION,
            "training_variant": VARIANT,
            "motion_contract": MOTION_CONTRACT,
            "config": asdict(self.config),
            "model_config": routing_model_config(self.config, self.model.future_frames),
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
            state["trajectory_format_version"] != FORMAT_VERSION
            or state["training_variant"] != VARIANT
            or state["motion_contract"] != MOTION_CONTRACT
            or asdict(RoutingConfig(**state["config"])) != asdict(self.config)
            or state["provenance"] != provenance
            or state["model_config"] != routing_model_config(self.config, self.model.future_frames)
        ):
            raise ValueError("Resume requires identical routing config, sources and data")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.step = state["step"]
        return state


def resume_probe(data, geometry, config, path, *, device):
    probe = replace(config, steps=4, width=16, layers=1, batch_size=2)
    trainer = RoutingTrainer(data, geometry, probe, device=device)
    trainer.train_step()
    trainer.train_step()
    identity = {"scope": "four unit-fixture updates; not model-quality evidence"}
    trainer.save(path, identity)
    expected = [trainer.train_step()["loss"], trainer.train_step()["loss"]]
    restored = RoutingTrainer(data, geometry, probe, device=device)
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
        raise ValueError("Routing optimizer continuation is not exact")
    return {
        "device": device,
        "maximum_abs_difference": maxima,
        "losses_equal": True,
        "scope": identity["scope"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "run-dir", "config", "supplement"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require_tc2()
    config = RoutingConfig(**json.loads(args.config.read_text()))
    config.validate()
    if args.run_dir.exists():
        parser.error("Preserve earlier attempts; use a fresh run directory")
    args.run_dir.mkdir(parents=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    data, provenance = prepare_trajectories(args.dataset, augment=config.augment)
    provenance.pop("preparation_seconds", None)
    data, provenance = add_training_supplement(
        args.dataset, args.supplement, data, provenance, augment=config.augment
    )
    geometry = prepare_contact_geometry(args.dataset, data)
    provenance["geometry_sha256"] = {
        split: {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in row.items()}
        for split, row in geometry.items()
    }
    provenance["source_revision"] = os.environ.get("C2F_SOURCE_REVISION", "unrecorded")
    provenance["source_sha256"] = source_identity()
    check = resume_probe(
        data, geometry, config, args.run_dir / "resume-unit-fixture.pt", device="cuda"
    )
    (args.run_dir / "resume-check.json").write_text(json.dumps(check, indent=2) + "\n")
    trainer = RoutingTrainer(data, geometry, config, device="cuda")
    with torch.no_grad():
        excess = max(
            float(
                canvas_excess(
                    trainer.data["train"]["motion"][start : start + 64],
                    trainer.data["train"]["features"][start : start + 64],
                    {
                        key: value[start : start + 64]
                        for key, value in trainer.geometry["train"].items()
                    },
                ).max()
                * 64
            )
            for start in range(0, len(trainer.data["train"]["motion"]), 64)
        )
    if config.boundary_weight > 0 and excess > config.boundary_margin_pixels + 1e-5:
        raise ValueError("Training-label canvas excess exceeds the declared raster margin")
    summary = {
        "scope": "E9 force-routing arm on the E6 canvas recipe; one seed; validation only",
        "config": asdict(config),
        "provenance": provenance,
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "oracle_training_canvas_excess_pixels": excess,
        "selection": "fixed-seed validation EDM loss, unchanged from E5/E6",
        "sampler": "unchanged E5 Heun; no projection, clamp or zero override",
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    best = None
    with (args.run_dir / "training.jsonl").open("a", buffering=1) as log:
        while trainer.step < config.steps:
            record = trainer.train_step()
            if trainer.step % 50 == 0:
                record["elapsed_seconds"] = time.perf_counter() - started
                log.write(json.dumps(record) + "\n")
            elapsed = time.perf_counter() - started
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
    summary.update(
        status="completed" if trainer.step == config.steps else "checkpointed_time_limit",
        steps_completed=trainer.step,
        elapsed_seconds=time.perf_counter() - started,
        best_validation_denoising_loss=best,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "steps": trainer.step}), flush=True)


if __name__ == "__main__":
    main()
