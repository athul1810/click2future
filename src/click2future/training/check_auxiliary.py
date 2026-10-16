"""CPU correctness checks for loss wiring and synthetic head fitting, not E4 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from click2future.data import load_batch, save_rollout
from click2future.models.pilot_denoiser import DenoiserOutput, PilotDenoiser
from click2future.sim import make_scene, run_rollout

from .auxiliary import AuxiliaryLossConfig, auxiliary_loss, collision_balance
from .targets import build_auxiliary_targets


def fixture_batch():
    """Loader-level fixture: two/three objects, including a known positive contact."""
    with tempfile.TemporaryDirectory(prefix="click2future-aux-") as directory:
        for index, count in enumerate((2, 3)):
            scene = make_scene(80000 + index, object_count=count)
            if index == 0:
                for obj, x in zip(scene["objects"], (0.3, 0.45), strict=True):
                    obj.update(
                        shape="circle", geometry={"radius": 0.06}, position=[x, 0.5], angle=0.0
                    )
                    obj["hidden_physics"]["mass"] = 0.5
                scene["action"].update(selected_object_id=0, force_xy=[0.45, 0.0])
            save_rollout(
                run_rollout(scene, frame_count=4, resolution=64), directory, f"fixture-{index}"
            )
        batch = load_batch(directory, ["fixture-0", "fixture-1"])
    return batch


def run_checks(config: AuxiliaryLossConfig, *, max_objects=4, num_diffusion_steps=1000) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        batch = fixture_batch()
        target = build_auxiliary_targets(
            torch.from_numpy(batch["positions"]),
            torch.from_numpy(batch["collision_onset"]),
            torch.from_numpy(batch["object_valid"]),
            torch.from_numpy(batch["instance_masks"][:, 0]),
            max_objects=max_objects,
        )
        model = PilotDenoiser(8, auxiliary=True, max_objects=max_objects)
        frames = torch.from_numpy(batch["frames"])
        noisy = torch.randn(len(frames), 3, frames.shape[1] - 1, *frames.shape[-2:])
        timestep = torch.zeros(len(frames), dtype=torch.long)
        output = model(
            noisy,
            timestep,
            frames[:, 0],
            torch.from_numpy(batch["selected_mask"]),
            torch.from_numpy(batch["force_xy"]),
        )
        losses = auxiliary_loss(
            output, target, timestep, num_diffusion_steps=num_diffusion_steps, config=config
        )
        losses.total.backward()
        norms = {}
        for name in ("position_head", "collision_head", "bottleneck"):
            gradients = [p.grad for p in getattr(model, name).parameters()]
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                raise ValueError(f"Invalid auxiliary gradient in {name}")
            norms[name] = sum(float(g.abs().sum()) for g in gradients if g is not None)
        if config.trajectory_weight > 0 and norms["position_head"] <= 0:
            raise ValueError("Trajectory head did not receive a gradient")
        if config.collision_weight > 0 and norms["collision_head"] <= 0:
            raise ValueError("Collision head did not receive a gradient")
        # Optimize explicit prediction tensors only. No generator parameters are updated.
        position_variables = torch.nn.Parameter(torch.zeros_like(target.positions))
        contact_variables = torch.nn.Parameter(
            torch.zeros_like(target.collision_onset, dtype=torch.float32)
        )
        optimizer = torch.optim.Adam([position_variables, contact_variables], lr=0.15)
        history = []
        enabled = config.trajectory_weight > 0 or config.collision_weight > 0
        for step in range(61 if enabled else 1):
            optimizer.zero_grad(set_to_none=True)
            logits = (contact_variables + contact_variables.transpose(-1, -2)) / 2
            predictions = DenoiserOutput(
                torch.zeros_like(noisy), position_variables.sigmoid(), logits
            )
            result = auxiliary_loss(
                predictions,
                target,
                timestep,
                num_diffusion_steps=num_diffusion_steps,
                config=config,
            )
            history.append(float(result.total.detach()))
            if step == 60 or not enabled:
                break
            result.total.backward()
            optimizer.step()
        if enabled and (history[0] <= 0 or history[-1] >= history[0] * 0.2):
            raise ValueError("Synthetic prediction-tensor fitting did not reduce the loss enough")
    return {
        "status": "passed",
        "purpose": "CPU auxiliary-loss correctness; no trained E4 generator",
        "seed": 31,
        "torch": torch.__version__,
        "config": asdict(config),
        "max_objects": max_objects,
        "num_diffusion_steps": num_diffusion_steps,
        "fixture": {
            "visible_seeds": [80000, 80001],
            "object_counts": [2, 3],
            "total_frames": 4,
            "resolution": 64,
            "first_scene": "two radius-.06 circles at (.3,.5)/(.45,.5), mass .5, force (.45,0)",
            "canonical_to_scene": target.canonical_to_scene.tolist(),
        },
        "real_denoiser_check": {
            "backward_only": True,
            "base_channels": 8,
            "loss": losses.record(),
            "gradient_l1_norms": norms,
        },
        "fixture_collision_balance": collision_balance([target]),
        "synthetic_prediction_fit": {
            "status": "performed" if enabled else "disabled",
            "optimizer_updates": len(history) - 1,
            "optimizer": "Adam",
            "lr": 0.15,
            "measured_losses": history,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "final_to_initial_ratio": history[-1] / history[0] if history[0] else None,
            "generator_parameters_updated": False,
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).with_name("auxiliary.py"),
                Path(__file__).with_name("targets.py"),
            ]
        },
        "limitations": [
            "No diffusion schedule, dataset training, sampling, or quality comparison",
            "Fixture balance is not a production training-set estimate",
            "Loss weights/timestep gate are untuned proposals",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/auxiliary.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.config.resolve() == args.output.resolve():
        parser.error("Output must not overwrite the config")
    try:
        config = yaml.safe_load(args.config.read_text())
        report = run_checks(
            AuxiliaryLossConfig(**config["loss"]),
            max_objects=config["max_objects"],
            num_diffusion_steps=config["num_diffusion_steps"],
        )
        report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, yaml.YAMLError) as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
