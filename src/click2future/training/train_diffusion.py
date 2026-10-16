"""Single-device pixel diffusion pilot with resumable, provenance-checked state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from click2future.data.loader import load_batch
from click2future.data.splits import load_split_manifest
from click2future.models.pilot_denoiser import PilotDenoiser

from .auxiliary import AuxiliaryLossConfig, auxiliary_loss
from .conditioning import training_inputs
from .diffusion import Diffusion
from .targets import build_auxiliary_targets


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    batch_size: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    ema_decay: float = 0.995
    seed: int = 7
    diffusion_steps: int = 1000
    base_channels: int = 32
    conditioning: str = "spatial"
    auxiliary: bool = False
    precision: str = "float32"
    limit_train: int | None = None
    checkpoint_every: int = 100
    collision_positive_weight: float = 1.0

    def validate(self):
        for name in ("steps", "batch_size", "checkpoint_every", "diffusion_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.limit_train is not None and (
            isinstance(self.limit_train, bool)
            or not isinstance(self.limit_train, int)
            or self.limit_train < 1
        ):
            raise ValueError("limit_train must be positive or null")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not math.isfinite(self.ema_decay)
            or not 0 <= self.ema_decay < 1
            or not math.isfinite(self.collision_positive_weight)
            or not 1 <= self.collision_positive_weight <= 20
        ):
            raise ValueError("Invalid optimizer, EMA or fixed training-label balance settings")


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_index(dataset: Path, limit: int | None) -> tuple[list[str], dict]:
    """Validate group membership and hash actual training inputs, never test arrays."""
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("manifest_hash"):
        raise ValueError("Dataset manifest hash mismatch")
    split_path = dataset / "splits.json"
    splits = load_split_manifest(split_path)["splits"]
    seen = set()
    training = []
    for record in manifest["samples"]:
        sample_id = record["sample_id"]
        if sample_id in seen or Path(sample_id).name != sample_id or not sample_id:
            raise ValueError("Duplicate or unsafe sample ID")
        seen.add(sample_id)
        if record["scene_group_id"] not in splits.get(record["split"], []):
            raise ValueError("Sample split disagrees with scene-group manifest")
        if record["split"] == "train":
            training.append(sample_id)
    ids = sorted(training)
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise ValueError("No training samples in the selected dataset")
    hashes = {name: _sha(dataset / name) for name in ("dataset_manifest.json", "splits.json")}
    for sample_id in ids:
        for extension in ("json", "npz"):
            name = f"{sample_id}.{extension}"
            hashes[name] = _sha(dataset / name)
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return ids, {
        "training_signature": signature,
        "manifest_hash": digest,
        "file_hashes": hashes,
        "training_sample_ids": ids,
        "scope": "train-only pilot; no held-out quality claim",
    }


def _rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume needs the original visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


class Trainer:
    def __init__(self, dataset: Path, config: TrainConfig, *, device="cpu"):
        config.validate()
        self.config = config
        self.dataset = Path(dataset)
        self.device = torch.device(device)
        if config.precision == "bfloat16" and (
            self.device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bfloat16 requires a supported CUDA allocation")
        self.sample_ids, self.provenance = training_index(self.dataset, config.limit_train)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.model = PilotDenoiser(
            base_channels=config.base_channels,
            conditioning=config.conditioning,
            auxiliary=config.auxiliary,
        ).to(self.device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        # Fixed horizon survives job boundaries; --until never changes the LR curve.
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.steps, eta_min=config.learning_rate * 0.1
        )
        self.diffusion = Diffusion(config.diffusion_steps, self.device)
        self.order_rng = torch.Generator().manual_seed(config.seed + 1)
        self.order = torch.randperm(len(self.sample_ids), generator=self.order_rng).tolist()
        self.cursor = 0
        self.step = 0

    def _next_ids(self):
        selected = []
        for _ in range(self.config.batch_size):
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.sample_ids), generator=self.order_rng).tolist()
                self.cursor = 0
            selected.append(self.sample_ids[self.order[self.cursor]])
            self.cursor += 1
        return selected

    def train_step(self):
        if self.step >= self.config.steps:
            raise ValueError("Configured training horizon reached")
        started = time.perf_counter()
        self.model.train()
        batch = load_batch(self.dataset, self._next_ids())
        clean, condition = training_inputs(batch, self.device)
        times = torch.randint(self.config.diffusion_steps, (len(clean),), device=self.device)
        noise = torch.randn_like(clean)
        noisy = self.diffusion.add_noise(clean, times, noise)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.precision == "bfloat16",
        ):
            output = self.model(noisy, times, **condition.kwargs())
            noise_loss = F.mse_loss(output.predicted_noise.float(), noise.float())
            loss = noise_loss
            auxiliary_record = {}
            if self.config.auxiliary:
                targets = build_auxiliary_targets(
                    torch.as_tensor(batch["positions"], device=self.device),
                    torch.as_tensor(batch["collision_onset"], device=self.device),
                    torch.as_tensor(batch["object_valid"], device=self.device),
                    torch.as_tensor(batch["instance_masks"][:, 0], device=self.device),
                    max_objects=self.model.max_objects,
                )
                aux = auxiliary_loss(
                    output,
                    targets,
                    times,
                    num_diffusion_steps=self.config.diffusion_steps,
                    config=AuxiliaryLossConfig(
                        collision_positive_weight=self.config.collision_positive_weight
                    ),
                )
                loss = loss + aux.total
                auxiliary_record = aux.record()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        with torch.no_grad():
            for moving, current in zip(self.ema.parameters(), self.model.parameters(), strict=True):
                moving.lerp_(current, 1 - self.config.ema_decay)
        self.step += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "step": self.step,
            "sample_ids": batch["sample_ids"],
            "loss": float(loss.detach()),
            "noise_loss": float(noise_loss.detach()),
            "gradient_norm_before_clip": float(norm),
            "learning_rate_next_step": self.optimizer.param_groups[0]["lr"],
            "seconds_including_load": time.perf_counter() - started,
            **auxiliary_record,
        }

    @torch.no_grad()
    def probe_loss(self, *, seed=1234):
        """Fixed training-batch noise diagnostic; not a validation/model-selection metric."""
        state = _rng_state()
        was_training = self.model.training
        self.model.eval()
        try:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            batch = load_batch(self.dataset, self.sample_ids[: self.config.batch_size])
            clean, condition = training_inputs(batch, self.device)
            times = torch.full((len(clean),), self.config.diffusion_steps // 2, device=self.device)
            noise = torch.randn_like(clean)
            predicted = self.model(
                self.diffusion.add_noise(clean, times, noise), times, **condition.kwargs()
            ).predicted_noise
            return float(F.mse_loss(predicted.float(), noise.float()))
        finally:
            self.model.train(was_training)
            _restore_rng(state)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "checkpoint_version": 1,
            "config": asdict(self.config),
            "provenance": self.provenance,
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "torch_version": str(torch.__version__),
            "device_type": self.device.type,
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng": _rng_state(),
            "order_rng": self.order_rng.get_state(),
            "order": self.order,
            "cursor": self.cursor,
            # bfloat16/float32 do not use gradient scaling.
            "gradient_scaler": None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path: Path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["checkpoint_version"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("Checkpoint version or immutable training config differs")
        if state["provenance"] != self.provenance:
            raise ValueError("Training data changed since checkpoint creation")
        if state["device_type"] != self.device.type or state["torch_version"] != str(
            torch.__version__
        ):
            raise ValueError("Exact resume requires the same device type and torch version")
        if not 0 <= state["step"] <= self.config.steps:
            raise ValueError("Checkpoint step outside training horizon")
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        self.order = state["order"]
        self.cursor = state["cursor"]
        self.order_rng.set_state(state["order_rng"])
        _restore_rng(state["rng"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="TrainConfig JSON")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--require-slurm", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--until", type=int, help="Stop at this absolute step; preserve configured horizon"
    )
    args = parser.parse_args(argv)
    if args.require_slurm and not os.environ.get("SLURM_JOB_ID"):
        parser.error("TC2 application execution requires an allocated Slurm job")
    config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    trainer = Trainer(args.dataset, config, device=args.device)
    if args.resume:
        trainer.resume(args.resume)
    stop = config.steps if args.until is None else args.until
    if not trainer.step < stop <= config.steps:
        parser.error("--until must exceed checkpoint step and not exceed configured steps")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "checkpoint.pt").exists() and not args.resume:
        parser.error("Run already has a checkpoint; resume it or choose a new run directory")
    before = trainer.probe_loss()
    log_path = args.run_dir / "training.jsonl"
    records = []
    for _ in range(trainer.step, stop):
        record = trainer.train_step()
        records.append(record)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if trainer.step % config.checkpoint_every == 0 or trainer.step == stop:
            trainer.save(args.run_dir / "checkpoint.pt")
        if trainer.step % 100 == 0 or trainer.step == stop:
            print(json.dumps(record), flush=True)
    summary = {
        "step": trainer.step,
        "config": asdict(config),
        "provenance": trainer.provenance,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        "probe_training_noise_mse_before": before,
        "probe_training_noise_mse_after": trainer.probe_loss(),
        "mean_step_seconds_excluding_first_10": (
            float(np.mean([r["seconds_including_load"] for r in records[10:]]))
            if len(records) > 10
            else None
        ),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(trainer.device) if args.device == "cuda" else None
        ),
        "scope": "Training-noise diagnostic; inspect samples and held-out action controls next",
    }
    (args.run_dir / f"summary-{stop}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
