"""TC2-only CVAE pilot training with scene splits, validation, and resumable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import assign_split, load_batch, load_split_manifest
from click2future.models.cvae import ConditionalVideoVAE

from .batch import prepare_video_batch
from .cvae_loss import beta_at_step, cvae_loss
from .pixel_loss import future_foreground, validate_foreground_weight


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_tc2():
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_PARTITION") != "MGPU-TC2"
        or not torch.cuda.is_available()
    ):
        raise ValueError(
            "Training requires a scheduled TC2 GPU allocation; CPU training is disabled"
        )
    if torch.cuda.device_count() != 1:
        raise ValueError("This pilot requires exactly one visible allocated GPU")


def dataset_index(root: Path):
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest.get("schema_version") not in {"dataset-0.1", "dataset-0.2", "dataset-0.3"}
        or manifest.get("manifest_hash") != digest
    ):
        raise ValueError("Dataset manifest hash/schema does not match")
    split_name = manifest["split_manifest"]
    if Path(split_name).name != split_name:
        raise ValueError("Split manifest must be inside the dataset folder")
    splits = load_split_manifest(root / split_name)
    rows, ids = manifest["samples"], set()
    selected = {"train": [], "validation": []}
    by_id = {}
    fingerprint = hashlib.sha256()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in ids or Path(sample_id).name != sample_id:
            raise ValueError("Dataset sample IDs must be unique filename components")
        ids.add(sample_id)
        if assign_split(row["scene_group_id"], splits) != row["split"]:
            raise ValueError("Dataset rows disagree with the scene-group split manifest")
        if row["split"] not in selected:
            continue  # Test arrays are never opened by the trainer.
        selected[row["split"]].append(sample_id)
        by_id[sample_id] = row
        for extension in (".npz", ".json"):
            filename = sample_id + extension
            fingerprint.update(filename.encode())
            fingerprint.update(bytes.fromhex(sha256(root / filename)))
    if len(rows) != manifest["sample_count"] or not all(selected.values()):
        raise ValueError(
            "The dataset needs a consistent sample count and nonempty train/validation"
        )
    return (
        selected,
        by_id,
        {
            "dataset_manifest_sha256": sha256(path),
            "split_manifest_sha256": sha256(root / split_name),
            "train_validation_content_sha256": fingerprint.hexdigest(),
        },
    )


def validate_config(config):
    data, model, train = (config[key] for key in ("data", "model", "training"))
    validate_foreground_weight(train.get("foreground_weight", 1.0))
    for key in ("seed", "batch_size", "max_steps", "checkpoint_every", "validation_batches"):
        value = train[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"Invalid training {key}")
    for key in ("learning_rate", "max_hours", "gradient_clip_norm"):
        if isinstance(train[key], bool) or not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"Invalid training {key}")
    if train["max_hours"] > 5.5:
        raise ValueError("Keep max_hours <= 5.5 to leave time within the six-hour TC2 limit")
    if not math.isfinite(train["beta"]) or train["beta"] < 0:
        raise ValueError("beta must be finite and nonnegative")
    beta_at_step(train["beta"], 0, train["max_steps"], train["beta_warmup_fraction"])
    if train["reconstruction"] not in {"l1", "mse"}:
        raise ValueError("Choose l1 or mse reconstruction")
    if (
        not isinstance(data["total_frames"], int)
        or data["total_frames"] < 2
        or model["future_frames"] != data["total_frames"] - 1
        or not isinstance(data["resolution"], int)
        or data["resolution"] < 16
        or data["resolution"] % 8
        or not math.isfinite(data["fps"])
        or data["fps"] != 8
        or data["force_duration"] != 0.25
        or not math.isfinite(data["force_scale"])
        or data["force_scale"] <= 0
    ):
        raise ValueError("Model frames, image dimensions, timing, and normalization must agree")


def model_batch(root, ids, rows, config, *, device, with_loss_mask=False):
    raw = load_batch(root, ids)
    d = config["data"]
    if raw["frames"].shape[1:] != (d["total_frames"], 3, d["resolution"], d["resolution"]):
        raise ValueError("Dataset clip shape differs from the run configuration")
    if not np.allclose(np.diff(raw["timestamps"], axis=1), 1 / d["fps"], rtol=0, atol=1e-8):
        raise ValueError("Dataset timing differs from the run configuration")
    for sample_id, scene in zip(ids, raw["scene_specs"], strict=True):
        if any(scene[key] != rows[sample_id][key] for key in ("scene_group_id", "condition_id")):
            raise ValueError("Saved rollout identity differs from the split index")
    batch = prepare_video_batch(raw, device=device, force_scale=d["force_scale"])
    if with_loss_mask:
        mask = (
            future_foreground(raw, device=device)
            if config["training"].get("foreground_weight", 1.0) != 1
            else None
        )
        return batch, mask
    return batch


def save_checkpoint(
    path, *, model, optimizer, state, config, provenance, train_generator, shuffle_generator
):
    payload = {
        "format_version": 1,
        "model_config": model.configuration(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "config": config,
        "provenance": provenance,
        "train_rng": train_generator.get_state(),
        "shuffle_rng": shuffle_generator.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(
    path, *, model, optimizer, config, provenance, train_generator, shuffle_generator
):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["format_version"] != 1
        or payload["model_config"] != model.configuration()
        or payload["config"] != config
        or payload["provenance"] != provenance
    ):
        raise ValueError(
            "Resume requires the same model, config, dataset content, and selected IDs"
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    train_generator.set_state(payload["train_rng"].cpu())
    shuffle_generator.set_state(payload["shuffle_rng"].cpu())
    torch.set_rng_state(payload["torch_rng"].cpu())
    return payload["state"]


@torch.no_grad()
def validate(model, root, ids, rows, config):
    t = config["training"]
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(t["seed"] + 100000)
    records, count = [], 0
    for start in range(
        0, min(len(ids), t["batch_size"] * t["validation_batches"]), t["batch_size"]
    ):
        selected = ids[start : start + t["batch_size"]]
        batch, loss_mask = model_batch(
            root, selected, rows, config, device=device, with_loss_mask=True
        )
        output = model(
            batch.first_frame,
            batch.selected_mask,
            batch.force_xy,
            batch.future,
            generator=generator,
        )
        loss = cvae_loss(
            output,
            batch.future,
            beta=t["beta"],
            reconstruction=t["reconstruction"],
            foreground=loss_mask,
            foreground_weight=t.get("foreground_weight", 1.0),
        ).record()
        records.append((len(selected), loss))
        count += len(selected)
    result = {
        key: sum(n * loss[key] for n, loss in records) / count
        for key in ("loss", "future_mse", "future_mae", "reconstruction_loss", "kl_mean_per_latent")
    }
    model.train()
    return {
        **result,
        "samples": count,
        "criterion": "fixed-seed posterior reconstruction plus beta KL",
        "foreground_weight": t.get("foreground_weight", 1.0),
        "prior_sample_quality": "not measured by this validation objective",
    }


def train(root, output, config, *, resume=None, overfit_samples=0):
    require_tc2()
    validate_config(config)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError("Use a new run folder or an explicit resume checkpoint")
    started = time.perf_counter()
    selected, rows, provenance = dataset_index(root)
    provenance["source_sha256"] = {
        path.name: sha256(path)
        for path in (
            Path(__file__),
            Path(__file__).with_name("batch.py"),
            Path(__file__).with_name("cvae_loss.py"),
            Path(__file__).with_name("pixel_loss.py"),
            Path(__file__).parents[1] / "models/cvae.py",
        )
    }
    if isinstance(overfit_samples, bool) or not 0 <= overfit_samples <= len(selected["train"]):
        raise ValueError("Overfit subset size must lie within the training split")
    if overfit_samples:
        selected["train"] = selected["train"][:overfit_samples]
    provenance["selected_ids"] = selected
    t = config["training"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(t["seed"])
    torch.set_num_threads(min(10, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ConditionalVideoVAE(**config["model"]).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=0)
    train_rng = torch.Generator(device="cuda").manual_seed(t["seed"] + 1)
    shuffle_rng = torch.Generator().manual_seed(t["seed"] + 2)
    state = {
        "step": 0,
        "cursor": 0,
        "epoch": 0,
        "order": [],
        "best_validation": None,
        "previous_training_seconds": 0.0,
    }
    if resume:
        state = restore_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            config=config,
            provenance=provenance,
            train_generator=train_rng,
            shuffle_generator=shuffle_rng,
        )
    inherited_seconds = state["previous_training_seconds"]
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    stopped = [False]
    previous_handlers = {}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    record = {
        "method": "E0_CVAE",
        "status": "running",
        "config": config,
        **provenance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "resume_checkpoint_sha256": sha256(resume) if resume else None,
        "inherited_training_seconds": inherited_seconds,
        "overfit_samples": overfit_samples,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def checkpoint(path):
        state["previous_training_seconds"] = inherited_seconds + time.perf_counter() - started
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            state=state,
            config=config,
            provenance=provenance,
            train_generator=train_rng,
            shuffle_generator=shuffle_rng,
        )

    try:
        with (output / "metrics.jsonl").open("a") as log:
            while (
                state["step"] < t["max_steps"]
                and not stopped[0]
                and (time.perf_counter() - started < t["max_hours"] * 3600)
            ):
                if state["cursor"] >= len(state["order"]):
                    state["order"] = torch.randperm(
                        len(selected["train"]), generator=shuffle_rng
                    ).tolist()
                    state["cursor"] = 0
                    state["epoch"] += 1
                indices = state["order"][state["cursor"] : state["cursor"] + t["batch_size"]]
                ids = [selected["train"][index] for index in indices]
                batch, loss_mask = model_batch(
                    root, ids, rows, config, device="cuda", with_loss_mask=True
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = model(
                    batch.first_frame,
                    batch.selected_mask,
                    batch.force_xy,
                    batch.future,
                    generator=train_rng,
                )
                beta = beta_at_step(
                    t["beta"], state["step"], t["max_steps"], t["beta_warmup_fraction"]
                )
                loss = cvae_loss(
                    prediction,
                    batch.future,
                    beta=beta,
                    reconstruction=t["reconstruction"],
                    foreground=loss_mask,
                    foreground_weight=t.get("foreground_weight", 1.0),
                )
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True
                )
                optimizer.step()
                state["cursor"] += len(indices)
                state["step"] += 1
                row = {"step": state["step"], "epoch": state["epoch"], **loss.record()}
                if state["step"] % t["checkpoint_every"] == 0 or state["step"] == t["max_steps"]:
                    row["validation"] = validate(model, root, selected["validation"], rows, config)
                    score = row["validation"]["loss"]
                    if state["best_validation"] is None or score < state["best_validation"]:
                        state["best_validation"] = score
                        checkpoint(output / "best.pt")
                    checkpoint(output / "latest.pt")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
        checkpoint(output / "latest.pt")
        torch.cuda.synchronize()
        record.update(
            status="completed" if state["step"] >= t["max_steps"] else "checkpointed",
            steps=state["step"],
            best_validation=state["best_validation"],
            current_job_seconds=time.perf_counter() - started,
            total_training_seconds=state["previous_training_seconds"],
            peak_tensor_bytes=torch.cuda.max_memory_allocated(),
            latest_checkpoint_sha256=sha256(output / "latest.pt"),
        )
        (output / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    except BaseException as error:
        record.update(
            status="failed", completed_steps=state["step"], error=f"{type(error).__name__}: {error}"
        )
        (output / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/cvae.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    args = parser.parse_args(argv)
    try:
        require_tc2()
        config = yaml.safe_load(args.config.read_text())
        for name in ("max_steps", "checkpoint_every"):
            if getattr(args, name) is not None:
                config["training"][name] = getattr(args, name)
        result = train(
            args.data_dir,
            args.output_dir,
            config,
            resume=args.resume,
            overfit_samples=args.overfit_samples,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"CVAE training stopped: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
