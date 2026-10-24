"""Measure where a trained CVAE carries the force signal on validation clips.

For each clip, flip the numeric force and decode with fixed latents to separate
the change through the spatial condition features from the change through the
prior mean. Compare both with prior seed spread and with the posterior-prior gap,
which shows how much future information the latent holds that the prior lacks.
Future masks only choose the region scored; they never enter the model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from click2future.data import load_batch
from click2future.models.cvae import _adaptive_mean, _statistics
from click2future.models.cvae_flow import build_cvae, warp_first_frame
from click2future.training.batch import prepare_video_batch
from click2future.training.pixel_loss import future_foreground
from click2future.training.train_cvae import require_tc2, sha256


def _region_mae(a, b, region):
    error = (a - b).abs().mean(dim=1, keepdim=True)
    return {
        "frame": error.mean().item(),
        "foreground": (error * region).sum().item() / max(region.sum().item(), 1.0),
    }


@torch.inference_mode()
def probe(dataset, checkpoint, *, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_cvae(payload["model_config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    flow = payload["model_config"].get("decoder_kind") == "flow"
    data = payload["config"]["data"]
    margin = payload["config"]["training"].get("foreground_margin", 0)
    manifest = json.loads((Path(dataset) / "dataset_manifest.json").read_text())
    names = (
        "force_total",
        "force_via_features",
        "force_via_prior",
        "prior_seed_spread",
        "posterior_minus_prior",
        "true_motion",
    )
    totals = {name: {"frame": 0.0, "foreground": 0.0} for name in names}
    prior_shift, clips = 0.0, 0
    for row in manifest["samples"]:
        if row["split"] != "validation":
            continue
        raw = load_batch(dataset, [row["sample_id"]])
        batch = prepare_video_batch(raw, device=device, force_scale=data["force_scale"])
        if not batch.force_xy.abs().sum():
            continue  # Zero-force clips have no reversed condition.
        region = future_foreground(raw, device=device, margin=margin).float()
        first, mask, force = batch.first_frame, batch.selected_mask, batch.force_xy

        def decode(features, latent):
            out = model._decode(features, latent)
            return warp_first_frame(first, out) if flow else out

        feat_pos, cond_pos = model._condition(first, mask, force)
        feat_neg, cond_neg = model._condition(first, mask, -force)
        mu_pos, logvar_pos = _statistics(model.prior(cond_pos))
        mu_neg, _ = _statistics(model.prior(cond_neg))
        future_features = model.future_encoder(batch.future)
        summary = _adaptive_mean(future_features, (4, 2, 2)).flatten(1)
        mu_post, _ = _statistics(model.posterior(torch.cat((cond_pos, summary), 1)))
        generator = torch.Generator(device=device).manual_seed(0)
        std = (0.5 * logvar_pos).exp()
        z0 = mu_pos + std * torch.randn(mu_pos.shape, device=device, generator=generator)
        z1 = mu_pos + std * torch.randn(mu_pos.shape, device=device, generator=generator)
        base = decode(feat_pos, mu_pos)
        still = first[:, :, None].expand_as(batch.future)
        pairs = {
            "force_total": (base, decode(feat_neg, mu_neg)),
            "force_via_features": (base, decode(feat_neg, mu_pos)),
            "force_via_prior": (base, decode(feat_pos, mu_neg)),
            "prior_seed_spread": (decode(feat_pos, z0), decode(feat_pos, z1)),
            "posterior_minus_prior": (decode(feat_pos, mu_post), base),
            "true_motion": (batch.future, still),
        }
        for name, (a, b) in pairs.items():
            for key, value in _region_mae(a, b, region).items():
                totals[name][key] += value
        prior_shift += ((mu_pos - mu_neg).norm() / (mu_post - mu_pos).norm().clamp_min(1e-8)).item()
        clips += 1
    if not clips:
        raise ValueError("No nonzero-force validation clips")
    return {
        "status": "measured_diagnostic",
        "validation_clips": clips,
        "units": "RGB MAE in [-1,1] over future frames; foreground = true future object masks",
        "mae": {n: {k: v / clips for k, v in t.items()} for n, t in totals.items()},
        "prior_mean_flip_over_posterior_gap": prior_shift / clips,
        "checkpoint_sha256": sha256(checkpoint),
        "training_steps": payload["state"]["step"],
        "model_config": payload["model_config"],
        "interpretation": (
            "force_* near zero relative to true_motion means the decoded video ignores "
            "the force; posterior_minus_prior large relative to force_via_prior means "
            "motion lives in the posterior latent that the prior cannot recover."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda":
        require_tc2()
    if args.output.exists():
        raise ValueError("Use a fresh output file")
    torch.use_deterministic_algorithms(True)
    record = probe(args.dataset, args.checkpoint.resolve(), device=args.device)
    record.update(
        source_revision=os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record["mae"], indent=2))


if __name__ == "__main__":
    main()
