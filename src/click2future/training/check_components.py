"""CPU correctness checks for E4 integration and the untrained E0 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_batch, save_rollout
from click2future.evaluation.heads import evaluate_heads, save_head_evaluation
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.video import evaluate_videos
from click2future.models.cvae import ConditionalVideoVAE
from click2future.models.pilot_denoiser import DenoiserOutput, PilotDenoiser
from click2future.sim import make_scene, run_rollout

from .auxiliary import AuxiliaryLossConfig
from .batch import prepare_video_batch
from .cvae_loss import cvae_loss
from .diffusion_step import diffusion_objective
from .targets import build_auxiliary_targets


def check(config_path: Path):
    with (
        torch.random.fork_rng(devices=[]),
        tempfile.TemporaryDirectory(prefix="c2f-components-") as tmp,
    ):
        torch.manual_seed(150000)
        scenes = [make_scene(150000 + i, object_count=2 + i) for i in range(2)]
        scenes[0]["scene_group_id"] = "component-known-collision"
        scenes[0]["condition_id"] = "component-known-collision-force-x"
        for obj, position in zip(scenes[0]["objects"], ([0.3, 0.5], [0.45, 0.5]), strict=True):
            obj.update(shape="circle", geometry={"radius": 0.06}, position=position, angle=0.0)
            obj["hidden_physics"]["mass"] = 0.5
        scenes[0]["action"].update(selected_object_id=0, force_xy=[0.45, 0.0])
        references = [run_rollout(scene, frame_count=4, resolution=64) for scene in scenes]
        for index, reference in enumerate(references):
            save_rollout(reference, tmp, f"fixture-{index}")
        raw = load_batch(tmp, ["fixture-0", "fixture-1"])
        batch = prepare_video_batch(raw)
        targets = build_auxiliary_targets(
            torch.from_numpy(raw["positions"]),
            torch.from_numpy(raw["collision_onset"]),
            torch.from_numpy(raw["object_valid"]),
            torch.from_numpy(raw["instance_masks"][:, 0]),
            max_objects=4,
        )
        if not targets.collision_onset[targets.pair_valid].any():
            raise RuntimeError("The joint-objective fixture must include a positive collision")
        denoiser = PilotDenoiser(base_channels=8, auxiliary=True)
        noise = torch.randn_like(batch.future)
        joint = diffusion_objective(
            denoiser,
            batch,
            0.8 * batch.future + 0.6 * noise,
            noise,
            torch.tensor([0, 2]),
            num_diffusion_steps=10,
            targets=targets,
            auxiliary_config=AuxiliaryLossConfig(),
        )
        joint.total.backward()
        gradients = {
            name: float(layer.weight.grad.abs().sum())
            for name, layer in {
                "noise_output": denoiser.output,
                "trajectory_head": denoiser.position_head,
                "collision_head": denoiser.collision_head,
                "shared_stem": denoiser.stem,
            }.items()
        }
        if not all(np.isfinite(value) and value > 0 for value in gradients.values()):
            raise RuntimeError("Joint objective did not reach all expected components")
        fixture_head = DenoiserOutput(
            joint.output.predicted_noise[:1].detach(),
            joint.output.positions[:1].detach(),
            joint.output.collision_logits[:1].detach(),
        )
        head_report, arrays = evaluate_heads(
            references[0],
            [fixture_head] * 4,
            timesteps=[0] * 4,
            num_diffusion_steps=10,
            max_timestep_fraction=0.25,
            collision_threshold=0.5,
            event_tolerance=1,
            run_id="untrained-teacher-forced-head-fixture",
        )
        head_report = save_head_evaluation(
            head_report,
            arrays,
            sample_id="fixture-0",
            archive_path=Path(tmp) / "heads.npz",
            report_path=Path(tmp) / "heads.json",
        )
        small = ConditionalVideoVAE(future_frames=3, base_channels=8, latent_dim=4)
        posterior = small(
            batch.first_frame,
            batch.selected_mask,
            batch.force_xy,
            batch.future,
            generator=torch.Generator().manual_seed(1),
        )
        loss = cvae_loss(posterior, batch.future)
        loss.total.backward()
        cvae_gradients = {
            name: sum(float(p.grad.abs().sum()) for p in layer.parameters() if p.grad is not None)
            for name, layer in {
                "prior": small.prior,
                "posterior": small.posterior,
                "condition_encoder": small.condition_encoder,
                "future_encoder": small.future_encoder,
                "decoder_output": small.output,
            }.items()
        }
        if not all(np.isfinite(x) and x > 0 for x in cvae_gradients.values()):
            raise RuntimeError("CVAE gradient check failed")
        full_reference = run_rollout(scenes[0], frame_count=32, resolution=64)
        full = ConditionalVideoVAE().eval()
        first = (
            torch.from_numpy(full_reference["frames"][:1].transpose(0, 3, 1, 2)).float() / 127.5 - 1
        )
        mask = torch.from_numpy(full_reference["selected_mask"][None, None])
        force = torch.from_numpy(full_reference["force_xy"][None]) / 0.5
        futures = full.sample(first, mask, force, seeds=[0, 1, 2, 3])[:, 0]
        rgb = ((futures.permute(0, 2, 3, 4, 1) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
        videos = np.concatenate(
            (np.broadcast_to(full_reference["frames"][:1], (4, 1, 64, 64, 3)), rgb), axis=1
        )
        config = yaml.safe_load(config_path.read_text())
        tracking, _ = evaluate_videos(
            full_reference,
            videos,
            tracker=TrackerConfig(**config["tracker"]),
            run_id="untrained-cvae-format-check",
            max_gap_pixels=config["max_gap_pixels"],
            motion_threshold=config["motion_threshold"],
            motion_window=config["motion_window"],
            motion_rule=config.get("motion_rule", "opposing"),
            event_tolerance=config["event_tolerance_intervals"],
        )
        trajectory = tracking["trajectory"]["all_objects"]
        return {
            "status": "passed",
            "purpose": "untrained CPU correctness only; no optimizer updates",
            "seed": 150000,
            "joint_loss": joint.record(),
            "e4_gradient_l1": gradients,
            "head_export": {
                "source": head_report["source"],
                "position_shape": list(arrays["predicted_positions"].shape),
                "fixture": "repeated teacher-forced outputs, not samples",
            },
            "cvae_loss": loss.record(),
            "cvae_gradient_l1": cvae_gradients,
            "default_cvae": {
                "parameters": sum(p.numel() for p in full.parameters()),
                "sample_seeds": [0, 1, 2, 3],
                "video_shape": list(videos.shape),
                "tracked_future_observations": trajectory["tracked_future_observations"],
                "expected_future_observations": trajectory["expected_future_observations"],
                "missing_track_rate": trajectory["missing_track_rate"],
                "quality": "untrained output; not a performance or uncertainty result",
            },
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__),
                    Path(__file__).with_name("batch.py"),
                    Path(__file__).with_name("diffusion_step.py"),
                    Path(__file__).with_name("cvae_loss.py"),
                    Path(__file__).parents[1] / "models/cvae.py",
                    Path(__file__).parents[1] / "evaluation/heads.py",
                ]
            },
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/pixel.yaml"))
    args = parser.parse_args(argv)
    result = check(args.config)
    # Store module-relative hashes so records work across checkouts.
    result["source_sha256"] = {
        "/".join(Path(path).parts[-3:]): value for path, value in result["source_sha256"].items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
