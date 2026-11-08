"""D027: rank calibration with independent hidden-physics draws for every case.

D025 reused the same four prior draws in every case, so even the exact simulator was
not uniform there. Rules fixed before scoring:
- Cases: every non-zero-force case of the fresh D026 ID plan.
- Draws: 24 per case, seeds 10_000_000 + 100 * case_index + k, distinct across cases.
- Quantity: the selected object's final travel distance from frame zero.
- Rank: number of the 24 samples below the reference, exact ties broken at random
  (seed 20261108). Bins of five ranks give five bins, uniform under calibration.
- Report bin counts, chi-square against uniform (4 degrees of freedom) and the fraction
  of references outside all 24 samples (expected 2/25).
- Generators: the true simulator with true geometry and sampled hidden physics (checks
  the test), the 0.5 px and calibrated hull baselines, and the neural model if given.
Cases in a scene group are correlated, so the effective sample size is smaller. This is
descriptive and changes no gate, scorer or historical result.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.suite import prepare_plan
from click2future.models.object_motion import observe
from click2future.models.physics_baseline import CALIBRATED_BEVEL_PIXELS, rollout_observation
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids

DRAWS = 24
SEED_BASE = 10_000_000
TIE_SEED = 20261108


def summarize(ranks):
    ranks = np.asarray(ranks)
    counts = np.bincount(ranks // 5, minlength=5)
    expected = len(ranks) / 5
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return {
        "cases": len(ranks),
        "bin_counts": counts.tolist(),
        "chi_square_4df": chi2,
        "p_value_uniform": math.exp(-chi2 / 2) * (1 + chi2 / 2),
        "outside_all_samples": float(np.mean((ranks == 0) | (ranks == DRAWS))),
        "mean_rank_over_draws": float(ranks.mean() / DRAWS),
    }


def calibrate(dataset, output, checkpoint=None, device="cpu"):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output file")
    config = Path(__file__).resolve().parents[1] / "configs/evaluation/pixel.yaml"
    plan = prepare_plan(dataset, config, split="id_test", metrics=("ssim",))
    model = None
    if checkpoint:
        from click2future.training.sample_trajectory import load_trajectory

        model, _ = load_trajectory(checkpoint)
        model = model.to(device).eval()
    names = ["true_simulator_sampled_world", "hull_0.5px", "hull_calibrated"]
    names += ["neural"] if model is not None else []
    ranks = {name: [] for name in names}
    rng = np.random.default_rng(TIE_SEED)
    rows = []
    keys = ("frame_count", "fps", "physics_hz", "resolution")
    for index, case in enumerate(plan["cases"]):
        ref = load_rollout(dataset, case["sample_id"])
        if "-zero-" in case["sample_id"] or not np.any(ref["force_xy"]):
            continue
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        sel = obs.selected_index
        positions = ref["positions"][:, order]
        target = float(np.linalg.norm(positions[-1, sel] - positions[0, sel]))
        seeds = [SEED_BASE + 100 * index + k for k in range(DRAWS)]
        values = {name: [] for name in names}
        for seed in seeds:
            world = sample_world(seed)[0].numpy()
            scene = copy.deepcopy(ref["scene_spec"])
            objects = {o["object_id"]: o for o in scene["objects"]}
            for slot, identifier in enumerate(order):
                objects[identifier]["hidden_physics"] = dict(
                    zip(
                        ("mass", "friction", "restitution"),
                        world[slot].astype(float).tolist(),
                        strict=True,
                    )
                )
            true = run_rollout(scene, **{key: ref["metadata"][key] for key in keys})
            p = true["positions"][:, order]
            values["true_simulator_sampled_world"].append(np.linalg.norm(p[-1, sel] - p[0, sel]))
            for name, bevel in (("hull_0.5px", 0.5), ("hull_calibrated", CALIBRATED_BEVEL_PIXELS)):
                motion = rollout_observation(obs, ref["force_xy"], world, bevel_pixels=bevel)
                values[name].append(np.linalg.norm(motion["motion"][-1, sel, :2] * 0.5))
        if model is not None:
            with torch.inference_mode():
                motion = (
                    model.sample(
                        torch.from_numpy(obs.features[None]).to(device),
                        torch.from_numpy(ref["force_xy"][None]).to(device),
                        seeds=seeds,
                        steps=1 if getattr(model, "world_prior_only", False) else 16,
                    )
                    .cpu()
                    .numpy()
                )
            values["neural"] = list(np.linalg.norm(motion[:, -1, sel, :2] * 0.5, axis=-1))
        row = {"sample_id": case["sample_id"], "scene_group_id": case["scene_group_id"]}
        row["reference"] = target
        for name in names:
            samples = np.asarray(values[name], dtype=float)
            below, ties = int(np.sum(samples < target)), int(np.sum(samples == target))
            rank = below + int(rng.integers(0, ties + 1))
            ranks[name].append(rank)
            row[name] = {"rank": rank, "median_sample": float(np.median(samples))}
        rows.append(row)
    result = {
        "study_id": "D027",
        "plan_sha256": plan["plan_sha256"],
        "draws_per_case": DRAWS,
        "seed_base": SEED_BASE,
        "quantity": "selected-object final travel distance, simulator position units",
        "generators": {name: summarize(values) for name, values in ranks.items()},
        "cases": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    r = calibrate(args.dataset, args.output, args.checkpoint, args.device)
    for name, g in r["generators"].items():
        p, outside = g["p_value_uniform"], g["outside_all_samples"]
        print(name, g["bin_counts"], f"p={p:.4f}", f"outside={outside:.2f}")
