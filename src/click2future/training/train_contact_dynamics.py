"""D022 validation-only pilot: local learned impulses instead of whole-clip decoding."""

import argparse
import copy
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_rollout
from click2future.evaluation.passive_stationarity import passive_horizon, stationary_reference
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.contact_dynamics import CONTACT_DYNAMICS, ContactDynamics
from click2future.models.object_motion import observe, render_motion
from click2future.training.contact_geometry import observed_hulls
from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.world_data import load_cache

ROOT = Path(__file__).resolve().parents[3]
CONFIG = {
    "seed": 19,
    "steps": 8000,
    "batch": 128,
    "width": 128,
    "unroll": 4,
    "learning_rate": 0.0003,
    "checkpoint_every": 250,
    "max_seconds": 1800,
}


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def identity(cache):
    paths = [
        Path(__file__),
        *[
            ROOT / "src/click2future" / name
            for name in (
                "models/contact_dynamics.py",
                "models/object_motion.py",
                "models/world_motion.py",
                "training/contact_geometry.py",
                "training/world_data.py",
                "training/train_cvae.py",
            )
        ],
    ]
    return {
        "contract": CONTACT_DYNAMICS,
        "config": CONFIG,
        "cache_manifest_sha256": sha256(Path(cache) / "manifest.json"),
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }


class Trainer:
    def __init__(self, data, geometry, worlds, device="cuda"):
        self.device = device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(CONFIG["seed"])
            self.model = ContactDynamics(CONFIG["width"]).to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=CONFIG["learning_rate"])
        self.rng = torch.Generator(device=device).manual_seed(CONFIG["seed"])
        self.step = 0
        self.data, self.geometry, self.worlds, self.weights = {}, {}, {}, {}
        for split, row in data.items():
            motion = torch.as_tensor(row["motion"], device=device)
            angle = torch.atan2(motion[..., 2:3], 1 + 2 * motion[..., 3:4])
            # Training labels are temporally unwrapped before finite differences.
            unwrapped = np.unwrap(angle.cpu().numpy(), axis=1)
            state = torch.cat(
                (motion[..., :2] * 0.5, torch.as_tensor(unwrapped, device=device)), -1
            )
            state = torch.cat((torch.zeros_like(state[:, :1]), state), 1)
            self.data[split] = {
                "state": state,
                **{key: torch.as_tensor(row[key], device=device) for key in ("features", "force")},
            }
            self.geometry[split] = {
                k: torch.as_tensor(v, device=device) for k, v in geometry[split].items()
            }
            self.worlds[split] = torch.as_tensor(worlds[split], device=device)
            # Cache contact strata affect sampling only. No oracle contact enters the model.
            families = np.array(
                [
                    "zero"
                    if np.linalg.norm(row["force"][i]) < 1e-8
                    else "contact"
                    if example.get("contact", "collision" in example["sample_id"])
                    else "other"
                    for i, example in enumerate(row["examples"])
                ]
            )
            weights = np.zeros(len(families), np.float32)
            for name, mass in (("zero", 0.25), ("contact", 0.5), ("other", 0.25)):
                chosen = families == name
                if chosen.any():
                    weights[chosen] = mass / chosen.sum()
            self.weights[split] = torch.as_tensor(weights, device=device)

    def train_step(self):
        c, row = CONFIG, self.data["train"]
        indices = torch.multinomial(self.weights["train"], c["batch"], True, generator=self.rng)
        interval = torch.randint(
            0, 32 - c["unroll"], (c["batch"],), device=self.device, generator=self.rng
        )
        state = row["state"][indices, interval]
        previous = row["state"][indices, (interval - 1).clamp_min(0)]
        delta = state - previous
        features, force = row["features"][indices], row["force"][indices]
        geometry = {key: value[indices] for key, value in self.geometry["train"].items()}
        radius = geometry["hulls"].norm(dim=-1).amax(-1).clamp_min(1 / 64)
        scale = (
            torch.cat((torch.ones_like(radius[..., None]).expand(-1, -1, 2), radius[..., None]), -1)
            * 64
        )
        loss = state.new_zeros(())
        for offset in range(c["unroll"]):
            state, delta = self.model.advance(
                state,
                delta,
                features,
                force,
                self.worlds["train"][indices],
                geometry,
                interval + offset,
            )
            target = row["state"][indices, interval + offset + 1]
            target_delta = target - row["state"][indices, interval + offset]
            loss = (
                loss
                + (
                    ((state - target) * scale).square().mean()
                    + ((delta - target_delta) * scale).square().mean()
                )
                / c["unroll"]
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite contact-dynamics objective")
        rate = c["learning_rate"] * (
            0.03 + 0.97 * (1 + math.cos(math.pi * self.step / c["steps"])) / 2
        )
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(
                self.ema.parameters(), self.model.parameters(), strict=True
            ):
                averaged.lerp_(current, 0.005)
        self.step += 1
        return float(loss.detach())

    def save(self, path, provenance):
        state = {
            "identity": provenance,
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.get_state(),
        }
        temporary = Path(path).with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path, provenance):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["identity"] != provenance or not 0 <= state["step"] <= CONFIG["steps"]:
            raise ValueError("Resume requires identical source, config and cache")
        self.model.load_state_dict(state["model"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["rng"].cpu())
        self.step = state["step"]


@torch.inference_mode()
def evaluate(model, dataset, output):
    device = next(model.parameters()).device
    settings = yaml.safe_load((ROOT / "configs/evaluation/pixel.yaml").read_text())
    plan = prepare_plan(
        dataset, ROOT / "configs/evaluation/pixel.yaml", split="validation", metrics=("ssim",)
    )
    folder = output / "validation"
    folder.mkdir(exist_ok=False)
    write(folder / "plan.json", plan)
    reports, rows = [], []
    rest = {
        k: 0 for k in ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
    }
    passive = dict(rest)
    events = dict(tp=0, fp=0, fn=0)
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        motion = (
            model.sample(
                torch.as_tensor(obs.features[None], device=device),
                torch.as_tensor(ref["force_xy"][None], device=device),
                geometry={
                    k: torch.as_tensor(v[None], device=device)
                    for k, v in observed_hulls(obs).items()
                },
                seeds=plan["sampling_seeds"],
            )
            .cpu()
            .numpy()
        )
        if not np.isfinite(motion).all():
            raise ValueError("Nonfinite autonomous rollout")
        videos = np.stack([render_motion(obs, value) for value in motion])
        np.savez_compressed(folder / (case["sample_id"] + ".npz"), videos=videos, motion=motion)
        report, _ = evaluate_videos(
            ref,
            videos,
            tracker=TrackerConfig(**settings["tracker"]),
            max_gap_pixels=settings["max_gap_pixels"],
            event_tolerance=settings["event_tolerance_intervals"],
            motion_threshold=settings["motion_threshold"],
            motion_window=settings["motion_window"],
            motion_rule=settings["motion_rule"],
            run_id="D022-contact-dynamics",
        )
        reports.append(report)
        for sample in report["collision"]["per_sample"]:
            for key, field in (
                ("tp", "true_positive"),
                ("fp", "false_positive"),
                ("fn", "false_negative"),
            ):
                events[key] += sample["event"][field]
        if "-zero-" in case["sample_id"]:
            record = zero_horizon(ref, videos)
            for key in rest:
                rest[key] += record[key]
        if any(tag in case["sample_id"] for tag in ("-forward-", "-reverse-")) and all(
            stationary_reference(load_rollout(dataset, identifier))
            for identifier in case["reference_ids"]
        ):
            record = passive_horizon(ref, videos)
            for key in passive:
                passive[key] += record[key]
        geo = geometry_diagnostics(obs, motion)
        rows.append(
            {
                "sample_id": case["sample_id"],
                "penetrating_clips": geo["clips_with_penetration_over_one_pixel"],
                "penetrating_frames": sum(
                    r["penetration_over_one_pixel_frames"] for r in geo["clips"]
                ),
                "max_depth_pixels": max(r["maximum_hull_penetration_pixels"] for r in geo["clips"]),
            }
        )
        write(folder / (case["sample_id"] + ".json"), report)
    denominator = 2 * events["tp"] + events["fp"] + events["fn"]
    record = {
        "study_id": "D022",
        "scope": "original validation only; no fresh or historical test scoring",
        "quality": summarize_quality(reports),
        "zero_force": rest,
        "passive": passive,
        "penetrating_clips": sum(r["penetrating_clips"] for r in rows),
        "penetrating_frames": sum(r["penetrating_frames"] for r in rows),
        "contact_counts": events,
        "contact_f1": 2 * events["tp"] / denominator if denominator else None,
        "cases": rows,
        "test_splits_evaluated": [],
    }
    write(output / "validation-results.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    require_tc2()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    provenance = identity(args.cache)
    provenance["revision"] = os.environ["C2F_SOURCE_REVISION"]
    data, geometry, worlds, _ = load_cache(args.cache)
    args.output.mkdir(exist_ok=args.resume or args.evaluate_only, parents=True)
    trainer = Trainer(data, geometry, worlds)
    if args.resume or args.evaluate_only:
        trainer.resume(args.output / "latest.pt", provenance)
    else:
        # A real two-update CUDA continuation test, independent of the final run.
        trainer.train_step()
        trainer.save(args.output / "resume-probe.pt", provenance)
        expected_loss = trainer.train_step()
        expected = copy.deepcopy(trainer.model.state_dict())
        expected_ema = copy.deepcopy(trainer.ema.state_dict())
        trainer.resume(args.output / "resume-probe.pt", provenance)
        actual_loss = trainer.train_step()
        error = max(
            float((expected[k] - v).abs().max()) for k, v in trainer.model.state_dict().items()
        )
        ema_error = max(
            float((expected_ema[k] - v).abs().max()) for k, v in trainer.ema.state_dict().items()
        )
        write(
            args.output / "resume-check.json",
            {
                "model_max_error": error,
                "ema_max_error": ema_error,
                "loss_error": abs(expected_loss - actual_loss),
            },
        )
        if error or ema_error or expected_loss != actual_loss:
            raise ValueError("CUDA continuation is not exact")
        trainer = Trainer(data, geometry, worlds)
        trainer.save(args.output / "latest.pt", provenance)
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(True))
    started = time.monotonic()
    with (args.output / "training.jsonl").open("a", buffering=1) as stream:
        while not args.evaluate_only and trainer.step < CONFIG["steps"]:
            loss = trainer.train_step()
            if trainer.step % 50 == 0:
                stream.write(json.dumps({"step": trainer.step, "loss": loss}) + "\n")
            if trainer.step % CONFIG["checkpoint_every"] == 0 or stopping:
                trainer.save(args.output / "latest.pt", provenance)
            if stopping or time.monotonic() - started > CONFIG["max_seconds"]:
                break
    trainer.save(args.output / "latest.pt", provenance)
    summary = {
        "identity": provenance,
        "step": trainer.step,
        "status": "complete" if trainer.step == CONFIG["steps"] else "incomplete",
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "test_splits_evaluated": [],
    }
    write(
        args.output / ("evaluation-attempt.json" if args.evaluate_only else "summary.json"), summary
    )
    if trainer.step != CONFIG["steps"]:
        raise SystemExit("Incomplete run retained; resume explicitly")
    trainer.save(args.output / "final.pt", provenance)
    result = evaluate(trainer.ema, args.dataset, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
