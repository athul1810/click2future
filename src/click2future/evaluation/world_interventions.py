"""Exploratory prior-mass interventions with paired denoising noise; validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.models.object_motion import TrajectoryDiffusion, observe
from click2future.models.world_motion import BoundWorld
from click2future.training.sample_trajectory import load_trajectory, sampling_geometry
from click2future.training.train_cvae import sha256

from .passive_stationarity import stationary_reference
from .trajectory_audit import verify_inputs
from .uncertainty_strata import action_family


def mass_effects(light, heavy, force, selected):
    """Signed endpoint response to lighter mass, with all paired seeds retained."""
    light, heavy, force = map(np.asarray, (light, heavy, force))
    if (
        light.shape != heavy.shape
        or light.ndim != 4
        or light.shape[2:] != (2, 4)
        or force.shape != (2,)
        or not np.isfinite(force).all()
        or np.linalg.norm(force) <= 0
        or selected not in (0, 1)
        or not np.isfinite(light).all()
        or not np.isfinite(heavy).all()
    ):
        raise ValueError("Use paired finite motion samples and a nonzero force")
    unit = force / np.linalg.norm(force)
    projected = 0.5 * ((light[:, -1, selected, :2] - heavy[:, -1, selected, :2]) @ unit) * 64
    return {
        "samples": len(light),
        "lighter_further": int((projected > 0).sum()),
        "equal": int((projected == 0).sum()),
        "lighter_shorter": int((projected < 0).sum()),
        "light_minus_heavy_endpoint_pixels": projected.tolist(),
        "passive_maximum_excursion_pixels": {
            name: (
                np.linalg.norm(value[:, :, 1 - selected, :2] * 0.5, axis=-1).max(1) * 64
            ).tolist()
            for name, value in (("light", light), ("heavy", heavy))
        },
    }


def audit(plan_path, predictions, dataset, checkpoint, output, device="cpu"):
    plan, _, _, hashes = verify_inputs(plan_path, predictions, dataset, checkpoint)
    model, state = load_trajectory(checkpoint, device)
    if state["trajectory_format_version"] not in (3, 4, 5, 7):
        raise ValueError("Use a sampled-world checkpoint")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for case in plan["cases"]:
        if action_family(case) not in {"forward", "reverse"}:
            continue
        references = [load_rollout(dataset, sample) for sample in case["reference_ids"]]
        if not all(stationary_reference(ref) for ref in references):
            records.append({"sample_id": case["sample_id"], "eligible": False})
            continue
        reference = references[0]
        obs = observe(reference["frames"][0], reference["selected_mask"])
        features = torch.from_numpy(obs.features[None]).to(device)
        force = torch.from_numpy(reference["force_xy"][None].copy()).to(device)
        geometry = sampling_geometry(model, obs, device)
        samples = {}
        for kind, mass in (("light", 0.65), ("heavy", 1.85)):
            world = torch.tensor([[[1.25, 0.45, 0.2], [1.25, 0.45, 0.2]]], device=device)
            world[:, obs.selected_index, 0] = mass
            if geometry:
                from click2future.models.surface_motion import BoundSurface

                bound = BoundSurface(model, world, geometry["geometry"])
            else:
                bound = BoundWorld(model, world)
            samples[kind] = (
                (
                    model.decode_world(features, force, world)
                    .expand(len(plan["sampling_seeds"]), -1, -1, -1)
                    .clone()
                    if getattr(model, "world_prior_only", False)
                    else TrajectoryDiffusion.sample(
                        bound,
                        features,
                        force,
                        seeds=plan["sampling_seeds"],
                        steps=16,
                        sigma_min=model.sigma_min,
                        sigma_max=model.sigma_max,
                    )
                )
                .cpu()
                .numpy()
            )
        if not model.world_condition and not np.array_equal(samples["light"], samples["heavy"]):
            raise ValueError("Disabled world conditioning changed under a latent intervention")
        np.savez_compressed(output / (case["sample_id"] + ".npz"), **samples)
        records.append(
            {
                "sample_id": case["sample_id"],
                "eligible": True,
                **mass_effects(
                    samples["light"], samples["heavy"], reference["force_xy"], obs.selected_index
                ),
            }
        )
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("World-intervention input changed")
    summary = {
        key: sum(row[key] for row in records if row["eligible"])
        for key in ("samples", "lighter_further", "equal", "lighter_shorter")
    }
    result = {
        "scope": "D018 held-out diagnostic; no checkpoint selection"
        if plan.get("study_id") == "D018"
        else "post-test exploratory validation diagnostic; not checkpoint selection",
        "world_condition": state["config"]["world_condition"],
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sha256(Path(__file__)),
        "seed_pairing": (
            "identical deterministic repetitions under each fixed world; not independent samples"
            if getattr(model, "world_prior_only", False)
            else "identical diffusion noise; only selected prior mass changes"
        ),
        "prior_worlds": {
            "selected_mass": [0.65, 1.85],
            "other_mass": 1.25,
            "friction": 0.45,
            "restitution": 0.2,
        },
        "eligibility": "all references satisfy the existing stationary-passive contact-free rule",
        "records": records,
        "summary": summary,
        "test_splits_evaluated": ["D018:id_test"] if plan.get("study_id") == "D018" else [],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.device == "cuda":
        from click2future.training.train_cvae import require_tc2

        require_tc2()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print(
        json.dumps(
            audit(
                args.plan, args.predictions, args.dataset, args.checkpoint, args.output, args.device
            )["summary"]
        )
    )


if __name__ == "__main__":
    main()
