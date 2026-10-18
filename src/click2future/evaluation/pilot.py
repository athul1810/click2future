"""Run validation physics controls and separate head/CVAE diagnostics on trained pilots."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from click2future.data import load_batch, load_rollout
from click2future.models.cvae_flow import build_cvae
from click2future.training.batch import prepare_video_batch
from click2future.training.cvae_loss import cvae_loss
from click2future.training.multiscene_pilot import PROJECT, read_pilot
from click2future.training.pixel_loss import future_foreground
from click2future.training.sample_diffusion import load_generator, sample_videos
from click2future.training.train_cvae import require_tc2, sha256

from .heads import CaptureFinalHeads, evaluate_heads, save_head_evaluation
from .sample_suite import generate_samples
from .suite import _write, prepare_plan, run_suite
from .targets import prepare_target_plan
from .tracking import TRACKER_VERSION, TrackerConfig
from .video import evaluate_videos


def summarize_quality(reports, *, min_tracking=0.9, max_selected_ade=0.03125):
    """Keep every case in the gate, including undefined or incomplete trajectories."""
    if not reports:
        raise ValueError("Provide all validation case reports")
    expected = sum(r["trajectory"]["all_objects"]["expected_future_observations"] for r in reports)
    tracked = sum(r["trajectory"]["all_objects"]["tracked_future_observations"] for r in reports)
    selected = [r["trajectory"]["selected_object"] for r in reports]
    all_objects = [r["trajectory"]["all_objects"] for r in reports]
    selected_expected = sum(r["total_samples"] for r in selected)
    selected_complete = sum(r["complete_samples"] for r in selected)
    ades = [r["mean_ade_over_complete_samples"] for r in selected]
    complete = selected_complete == selected_expected
    return {
        "case_count": len(reports),
        "expected_observations": expected,
        "tracked_observations": tracked,
        "tracking_fraction": tracked / expected if expected else None,
        "complete_selected_clips": selected_complete,
        "expected_selected_clips": selected_expected,
        "complete_all_object_clips": sum(r["complete_samples"] for r in all_objects),
        "expected_all_object_clips": sum(r["total_samples"] for r in all_objects),
        "selected_ade_mean_cases_if_all_complete": float(np.mean(ades))
        if complete and all(a is not None for a in ades)
        else None,
        "passes_tracking_and_selected_trajectory_gate": bool(
            expected
            and tracked / expected >= min_tracking
            and complete
            and all(a is not None and a <= max_selected_ade for a in ades)
        ),
        "scope": (
            "Strict single-reference engineering check. Hidden-physics coverage and action "
            "response are reported separately; this is not a physical-validity guarantee."
        ),
    }


def score_heads(plan, run_spec, samples, output, *, device):
    """Replay exact RGB samples while capturing the actual final denoiser heads."""
    output = Path(output)
    output.mkdir()
    model, diffusion, _ = load_generator(Path(run_spec["checkpoint"]), device)
    config = json.loads(Path(run_spec["config"]).read_text())
    metrics = yaml.safe_load(Path(plan["config"]).read_text())
    records = []
    for case in plan["cases"]:
        ref = load_rollout(plan["dataset"], case["sample_id"])
        outputs, replayed = [], []
        for seed in plan["sampling_seeds"]:
            with CaptureFinalHeads(model) as captured:
                video = sample_videos(
                    model,
                    diffusion,
                    ref["frames"][0],
                    ref["selected_mask"],
                    ref["force_xy"],
                    seeds=(seed,),
                    future_frames=len(ref["frames"]) - 1,
                    sampling_steps=32,
                    force_limit=config["force_limit"],
                )
            outputs.append(captured.output)
            replayed.append(video[0])
        saved = Path(samples) / "pilot-E4" / (case["sample_id"] + ".npz")
        with np.load(saved, allow_pickle=False) as archive:
            if not np.array_equal(np.stack(replayed), archive["videos"]):
                raise ValueError("Head capture replay changed generated RGB pixels")
        report, arrays = evaluate_heads(
            ref,
            outputs,
            timesteps=[0] * len(outputs),
            num_diffusion_steps=diffusion.steps,
            max_timestep_fraction=0.25,
            collision_threshold=0.5,
            event_tolerance=1,
            run_id="pilot-E4-heads",
        )
        _, rgb = evaluate_videos(
            ref,
            np.stack(replayed),
            tracker=TrackerConfig(**metrics["tracker"]),
            max_gap_pixels=metrics["max_gap_pixels"],
            event_tolerance=metrics["event_tolerance_intervals"],
            motion_window=metrics["motion_window"],
            motion_threshold=metrics["motion_threshold"],
            motion_rule=metrics["motion_rule"],
            run_id="pilot-E4-rgb",
        )
        known = arrays["tracked"][:, 1:] & rgb["tracked"][:, 1:]
        distances = np.linalg.norm(
            arrays["predicted_positions"][:, 1:][known] - rgb["predicted_positions"][:, 1:][known],
            axis=-1,
        )
        pairs = np.triu(ref["object_valid"][:, None] & ref["object_valid"][None], k=1)
        joint = arrays["collision_observed"] & rgb["collision_observed"] & pairs[None, None]
        agreement = arrays["predicted_collision_onset"] == rgb["predicted_collision_onset"]
        report["head_vs_rgb"] = {
            "rgb_extractor_version": TRACKER_VERSION,
            "expected_future_object_observations": int(
                len(replayed) * (len(ref["frames"]) - 1) * ref["object_valid"].sum()
            ),
            "jointly_observed_future_objects": int(known.sum()),
            "mean_trajectory_distance_over_joint_observations": float(distances.mean())
            if len(distances)
            else None,
            "jointly_observed_pair_intervals": int(joint.sum()),
            "collision_interval_agreement_over_joint_observations": float(agreement[joint].mean())
            if joint.any()
            else None,
            "scope": "Agreement with approximate RGB extraction, not simulator ground truth.",
        }
        name = case["sample_id"]
        save_head_evaluation(
            report,
            arrays,
            sample_id=name,
            archive_path=output / (name + ".npz"),
            report_path=output / (name + ".json"),
        )
        records.append({"sample_id": name, "report": name + ".json"})
    _write(
        output / "manifest.json",
        {
            "status": "passed",
            "source": "auxiliary_head",
            "collision_threshold": 0.5,
            "threshold_status": "fixed pilot threshold, not a validation-selected final threshold",
            "rgb_replay": "pixel_exact_for_every_case_and_seed",
            "cases": records,
            "checkpoint_sha256": sha256(run_spec["checkpoint"]),
        },
    )


@torch.inference_mode()
def cvae_diagnostics(dataset, checkpoint, output, *, device):
    """Compare posterior/prior errors and latent activity on every train/validation clip."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = build_cvae(payload["model_config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    manifest = json.loads((Path(dataset) / "dataset_manifest.json").read_text())
    training = payload["config"]["training"]
    splits = {}
    for split in ("train", "validation"):
        rows, latent_kls = [], []
        for row in manifest["samples"]:
            if row["split"] != split:
                continue
            raw = load_batch(dataset, [row["sample_id"]])
            batch = prepare_video_batch(
                raw, device=device, force_scale=payload["config"]["data"]["force_scale"]
            )
            generator = torch.Generator(device=device).manual_seed(100000)
            posterior = model(
                batch.first_frame,
                batch.selected_mask,
                batch.force_xy,
                batch.future,
                generator=generator,
            )
            prior = model.sample(
                batch.first_frame, batch.selected_mask, batch.force_xy, seeds=[0, 1, 2, 3]
            )
            mask = future_foreground(
                raw, device=device, margin=training.get("foreground_margin", 0)
            )
            loss = cvae_loss(
                posterior,
                batch.future,
                beta=training["beta"],
                reconstruction=training["reconstruction"],
                foreground=mask,
                foreground_weight=training.get("foreground_weight", 1),
            ).record()
            qm, qv = posterior.posterior_mean, posterior.posterior_logvar
            pm, pv = posterior.prior_mean, posterior.prior_logvar
            latent_kls.append(
                (0.5 * (pv - qv + (qv - pv).exp() + (qm - pm).square() * (-pv).exp() - 1))
                .clamp_min(0)[0]
                .cpu()
                .numpy()
            )
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "posterior": loss,
                    "prior_future_mae": float((prior - batch.future[None]).abs().mean()),
                    "prior_pairwise_rgb_mae": float(
                        torch.stack(
                            [
                                (prior[a] - prior[b]).abs().mean()
                                for a, b in itertools.combinations(range(4), 2)
                            ]
                        ).mean()
                    ),
                }
            )
        kl = np.mean(latent_kls, axis=0)
        splits[split] = {
            "clips": len(rows),
            "posterior_future_mae": float(np.mean([r["posterior"]["future_mae"] for r in rows])),
            "prior_future_mae": float(np.mean([r["prior_future_mae"] for r in rows])),
            "prior_pairwise_rgb_mae": float(np.mean([r["prior_pairwise_rgb_mae"] for r in rows])),
            "kl_mean_per_latent": float(kl.mean()),
            "per_latent_kl": kl.tolist(),
            "latent_dimensions_kl_above_0_01": int((kl > 0.01).sum()),
            "per_clip": rows,
        }
    _write(
        output,
        {
            "status": "measured_diagnostic",
            "checkpoint_sha256": sha256(checkpoint),
            "training_steps": payload["state"]["step"],
            "untrained_fixture": payload["state"]["step"] == 0,
            "units": "RGB MAE in [-1,1]; frame 0 excluded",
            "sample_seeds": [0, 1, 2, 3],
            "posterior_seed": 100000,
            "splits": splits,
            "scope": (
                "Posterior uses future RGB for diagnosis only. Small KL or RGB spread alone "
                "does not establish posterior collapse or physical uncertainty."
            ),
        },
    )


def evaluate(root, output, *, device="cpu"):
    root, output = Path(root).resolve(), Path(output).resolve()
    if device == "cuda":
        require_tc2()
    if output.exists():
        raise ValueError("Use a new evaluation directory")
    pilot = read_pilot(root)
    if json.loads((root / "training.json").read_text())["status"] != "completed":
        raise ValueError("Complete the fixed training pilot before evaluating it")
    output.mkdir(parents=True)
    try:
        plan = prepare_plan(
            root / "data",
            PROJECT / "configs/evaluation/pixel.yaml",
            split="validation",
            metrics=pilot["specification"]["image_metrics"],
            protocol=PROJECT / "configs/evaluation/comparison.yaml",
            actions_config=PROJECT / "configs/evaluation/actions.yaml",
        )
        _write(output / "plan.json", plan)
        runs = [
            {
                "method": method,
                "run_id": "pilot-" + method,
                "checkpoint": str(
                    root / method / ("best.pt" if method == "E0" else "checkpoint.pt")
                ),
            }
            for method in ("E0", "E2", "E3", "E4")
        ]
        spec = {"sampling_steps": 32, "runs": runs}
        _write(output / "runs.json", spec)
        for name, path in (("validation", output / "plan.json"), ("selected-object", None)):
            if path is None:
                prepare_target_plan(
                    output / "plan.json",
                    output / "target-controls",
                    config=PROJECT / "configs/evaluation/actions.yaml",
                    force_magnitude=0.4,
                )
                path = output / "target-controls/plan.json"
            print(json.dumps({"sampling": name}), flush=True)
            generate_samples(
                path, output / "runs.json", output / (name + "-samples"), device=device
            )
            print(json.dumps({"scoring": name}), flush=True)
            run_suite(
                path, output / (name + "-samples/predictions.json"), output / (name + "-metrics")
            )
        print(json.dumps({"scoring": "E4 generated-sample heads"}), flush=True)
        score_heads(
            plan,
            {"checkpoint": runs[-1]["checkpoint"], "config": str(root / "configs/E4.json")},
            output / "validation-samples",
            output / "heads",
            device=device,
        )
        print(json.dumps({"scoring": "CVAE posterior/prior and latent activity"}), flush=True)
        cvae_diagnostics(
            root / "data",
            Path(runs[0]["checkpoint"]),
            output / "cvae-diagnostics.json",
            device=device,
        )
        manifest = json.loads((output / "validation-metrics/comparison.json").read_text())
        quality = {}
        for run in manifest["runs"]:
            reports = [
                json.loads((output / "validation-metrics" / r["trajectory"]).read_text())
                for r in run["reports"]
            ]
            quality[run["method"]] = summarize_quality(reports)
        record = {
            "status": "measured_validation_pilot",
            "extractor_version": TRACKER_VERSION,
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "quality_checks": quality,
            "dataset_manifest_sha256": pilot["dataset_manifest_sha256"],
            "test_splits_evaluated": [],
            "scope": (
                "One training seed per method; provisional extractor 2. CVAE has a different "
                "training budget. No final method ranking or test-based tuning."
            ),
        }
        _write(output / "pilot-results.json", record)
        return record
    except BaseException as error:
        _write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    evaluate(args.root, args.output, device=args.device)


if __name__ == "__main__":
    main()
