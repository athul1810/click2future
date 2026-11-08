"""Exploratory same-world analysis of saved D021 outputs; primary scores stay unchanged."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.evaluation.suite import _write, read_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.video import evaluate_videos
from click2future.models.object_motion import observe
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids


def audit(screen_dir, output):
    screen_dir, output = Path(screen_dir), Path(output)
    if output.exists():
        raise ValueError("Retain previous analyses and use a new output file")
    plan = read_plan(screen_dir / "plan.json")
    settings = yaml.safe_load(Path(plan["config"]).read_text())
    totals = {
        name: {"ade": [], "momentum": [], "tp": 0, "fp": 0, "fn": 0}
        for name in ("physics_assisted", "neural", "rendered_oracle")
    }
    rows = []
    for case in plan["cases"]:
        ref = load_rollout(plan["dataset"], case["sample_id"])
        observation = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(observation, ref["selected_object_id"])
        saved = {}
        for name in ("physics_assisted", "neural"):
            with np.load(screen_dir / name / (case["sample_id"] + ".npz")) as arrays:
                saved[name] = {key: arrays[key].copy() for key in ("videos", "motion")}
        for index, seed in enumerate(plan["sampling_seeds"]):
            world = sample_world(seed)[0].numpy()
            scene = copy.deepcopy(ref["scene_spec"])
            objects = {obj["object_id"]: obj for obj in scene["objects"]}
            for slot, identifier in enumerate(order):
                objects[identifier]["hidden_physics"] = dict(
                    zip(
                        ("mass", "friction", "restitution"),
                        world[slot].astype(float).tolist(),
                        strict=True,
                    )
                )
            target = run_rollout(
                scene,
                **{
                    key: ref["metadata"][key]
                    for key in ("frame_count", "fps", "physics_hz", "resolution")
                },
            )
            if not np.array_equal(target["frames"][0], ref["frames"][0]):
                raise ValueError("Reference intervention changed the supplied visible image")
            true_delta = target["positions"][1:, order] - target["positions"][:1, order]
            momentum_eligible = not target["wall_contact_onset"].any()
            row = {
                "sample_id": case["sample_id"],
                "sampling_seed": seed,
                "momentum_eligible": bool(momentum_eligible),
                "models": {},
            }
            for name in totals:
                if name == "rendered_oracle":
                    video, delta = target["frames"], true_delta
                else:
                    video = saved[name]["videos"][index]
                    delta = saved[name]["motion"][index, ..., :2] * 0.5
                ade = float(
                    np.linalg.norm(
                        delta[:, observation.selected_index]
                        - true_delta[:, observation.selected_index],
                        axis=-1,
                    ).mean()
                )
                totals[name]["ade"].append(ade)
                report, _ = evaluate_videos(
                    target,
                    video[None],
                    tracker=TrackerConfig(**settings["tracker"]),
                    max_gap_pixels=settings["max_gap_pixels"],
                    event_tolerance=settings["event_tolerance_intervals"],
                    motion_threshold=settings["motion_threshold"],
                    motion_window=settings["motion_window"],
                    motion_rule=settings["motion_rule"],
                    run_id=name,
                )
                event = report["collision"]["per_sample"][0]["event"]
                for short, key in (
                    ("tp", "true_positive"),
                    ("fp", "false_positive"),
                    ("fn", "false_negative"),
                ):
                    totals[name][short] += event[key]
                value = {"relative_displacement_ade": ade, "contact_event": event}
                if momentum_eligible:
                    displacement = np.concatenate((np.zeros((1, 2, 2)), delta.astype(float)))
                    velocity = np.diff(displacement, axis=0) * 8
                    momentum = (velocity * world[None, :, :1]).sum(1)
                    expected = ref["force_xy"] * ref["force_duration"]
                    error = float(np.linalg.norm(momentum[2:] - expected, axis=-1).mean())
                    totals[name]["momentum"].append(error)
                    value["post_pulse_momentum_error"] = error
                row["models"][name] = value
            rows.append(row)
    summary = {}
    for name, values in totals.items():
        denominator = 2 * values["tp"] + values["fp"] + values["fn"]
        summary[name] = {
            "clips": len(values["ade"]),
            "matched_world_displacement_ade": float(np.mean(values["ade"])),
            "contact_event_f1": 2 * values["tp"] / denominator if denominator else None,
            "contact_tp_fp_fn": [values[key] for key in ("tp", "fp", "fn")],
            "wall_free_momentum_clips": len(values["momentum"]),
            "post_pulse_momentum_error": float(np.mean(values["momentum"])),
        }
    result = {
        "scope": (
            "post-D021 exploratory same-world reference analysis; saved predictions are unchanged"
        ),
        "reference_world": (
            "the exact sampled prior draw, with original geometry used only for evaluation"
        ),
        "primary_scores_replaced": False,
        "limitation": (
            "This supplementary analysis was specified after the primary D021 results. "
            "It does not remove inferred-geometry error or calibrate a posterior. "
            "The RGB oracle control exposes contact-estimator error."
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary,
        "cases": rows,
    }
    _write(output, result)
    print(json.dumps(summary), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.screen, args.output)


if __name__ == "__main__":
    main()
