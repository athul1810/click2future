"""D025: are the four sampled futures an honest spread around the hidden-physics reference?

Rules fixed before scoring:
- Cases: every case in the screen plan except zero-force cases, where all futures rest.
- Quantity: the selected object's final travel distance from frame zero, in the
  simulator position units used by the D023 same-world comparison.
- Rank: the number of the four samples below the reference value, with exact ties
  broken uniformly at random (seed 20261108). For a calibrated sampler the rank is
  uniform on 0-4, so each bin expects a fifth of the cases.
- Report rank counts, a chi-square statistic against uniform (4 degrees of freedom),
  the fraction of references outside all four samples (expected 0.4) and the fraction
  inside the sample range (expected 0.6).
- Generators: every model folder in the screen output, plus the true simulator with
  true geometry and the same four sampled hidden-physics draws as a calibration check.

Limits: cases in a scene group share worlds, and forward/reverse cases are mirrors, so
the effective sample size is below the case count. This is a descriptive, post-result
analysis of an opened corpus. It changes no gate, scorer or historical result.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.models.object_motion import observe
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids

TIE_SEED = 20261108


def chi2_sf_4(x):
    """Survival function of a chi-square variable with 4 degrees of freedom."""
    return math.exp(-x / 2) * (1 + x / 2)


def sampled_world_travel(ref, order, selected, seeds):
    keys = ("frame_count", "fps", "physics_hz", "resolution")
    values = []
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
        out = run_rollout(scene, **{key: ref["metadata"][key] for key in keys})
        positions = out["positions"][:, order]
        values.append(float(np.linalg.norm(positions[-1, selected] - positions[0, selected])))
    return np.asarray(values)


def summarize(ranks):
    ranks = np.asarray(ranks)
    n = len(ranks)
    counts = np.bincount(ranks, minlength=5)
    expected = n / 5
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    return {
        "cases": n,
        "rank_counts": counts.tolist(),
        "chi_square_4df": chi2,
        "p_value_uniform": chi2_sf_4(chi2),
        "outside_all_samples": float(np.mean((ranks == 0) | (ranks == 4))),
        "reference_below_all_samples": float(np.mean(ranks == 0)),
        "reference_above_all_samples": float(np.mean(ranks == 4)),
        "inside_sample_range": float(np.mean((ranks > 0) & (ranks < 4))),
    }


def calibrate(dataset, screen, output):
    dataset, screen, output = Path(dataset), Path(screen), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output file")
    plan = json.loads((screen / "plan.json").read_text())
    seeds = plan["sampling_seeds"]
    models = sorted(p.name for p in screen.iterdir() if p.is_dir())
    rng = np.random.default_rng(TIE_SEED)
    ranks = {name: [] for name in [*models, "true_simulator_sampled_world"]}
    rows = []
    for case in plan["cases"]:
        sample_id = case["sample_id"]
        ref = load_rollout(dataset, sample_id)
        if "-zero-" in sample_id or not np.any(ref["force_xy"]):
            continue
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        selected = obs.selected_index
        positions = ref["positions"][:, order]
        target = float(np.linalg.norm(positions[-1, selected] - positions[0, selected]))
        values = {}
        for name in models:
            with np.load(screen / name / (sample_id + ".npz")) as arrays:
                motion = arrays["motion"]
            if motion.shape[0] != len(seeds):
                raise ValueError("Every screen case must keep all sampled futures")
            delta = motion[:, -1, selected, :2] * 0.5
            values[name] = np.linalg.norm(delta, axis=-1)
        values["true_simulator_sampled_world"] = sampled_world_travel(ref, order, selected, seeds)
        row = {
            "sample_id": sample_id,
            "scene_group_id": case["scene_group_id"],
            "reference": target,
        }
        for name, samples in values.items():
            below, ties = int(np.sum(samples < target)), int(np.sum(samples == target))
            rank = below + int(rng.integers(0, ties + 1))
            ranks[name].append(rank)
            row[name] = {"samples": samples.tolist(), "rank": rank}
        rows.append(row)
    result = {
        "study_id": "D025",
        "split": plan["split"],
        "plan_sha256": plan.get("plan_sha256"),
        "quantity": "selected-object final travel distance, simulator position units",
        "tie_seed": TIE_SEED,
        "generators": {name: summarize(values) for name, values in ranks.items()},
        "cases": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = calibrate(args.dataset, args.screen, args.output)
    for name, g in r["generators"].items():
        p, outside = g["p_value_uniform"], g["outside_all_samples"]
        print(name, g["rank_counts"], f"p={p:.4f}", f"outside={outside:.2f}")
