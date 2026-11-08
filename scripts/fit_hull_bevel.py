"""D026 bevel fit on the original training and validation splits only.

For every non-zero-force clip and the four prior draws, compare the hull baseline with
the true simulator under the same hidden physics. Report same-world displacement ADE in
native pixels per action family for each candidate bevel. No test split is read.
"""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.models.object_motion import observe
from click2future.models.physics_baseline import rollout_observation
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids

BEVELS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)


def fit(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output file")
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    rows = [s for s in manifest["samples"] if s["split"] in ("train", "validation")]
    errors = defaultdict(lambda: defaultdict(list))
    keys = ("frame_count", "fps", "physics_hz", "resolution")
    for sample in rows:
        ref = load_rollout(dataset, sample["sample_id"])
        if not np.any(ref["force_xy"]):
            continue
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        sel = obs.selected_index
        for seed in range(4):
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
            delta = true["positions"][:, order] - true["positions"][:1, order]
            for bevel in BEVELS:
                motion = rollout_observation(obs, ref["force_xy"], world, bevel_pixels=bevel)
                predicted = motion["positions"] - motion["positions"][:1]
                ade = np.linalg.norm(predicted[1:] - delta[1:], axis=-1).mean(0) * 64
                errors[bevel][sample["action_variant"]].append((ade[sel], ade[1 - sel]))
    result = {"study_id": "D026-fit", "splits": ["train", "validation"], "bevels": {}}
    for bevel in BEVELS:
        families = {
            family: {
                "clips_times_draws": len(values),
                "selected_ade_pixels": float(np.mean([v[0] for v in values])),
                "other_ade_pixels": float(np.mean([v[1] for v in values])),
            }
            for family, values in sorted(errors[bevel].items())
        }
        every = [v for values in errors[bevel].values() for v in values]
        result["bevels"][f"{bevel:.2f}"] = {
            "selected_ade_pixels": float(np.mean([v[0] for v in every])),
            "other_ade_pixels": float(np.mean([v[1] for v in every])),
            "families": families,
        }
    best = min(result["bevels"], key=lambda b: result["bevels"][b]["selected_ade_pixels"])
    result["lowest_selected_ade_bevel"] = best
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = fit(args.dataset, args.output)
    for bevel, value in r["bevels"].items():
        print(bevel, round(value["selected_ade_pixels"], 4), round(value["other_ade_pixels"], 4))
    print("lowest", r["lowest_selected_ade_bevel"])
