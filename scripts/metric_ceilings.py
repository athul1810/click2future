"""D030: what the unchanged scorers give for the true simulator on a fresh corpus.

Two controls, rendered by the simulator and scored exactly like the models:
- true_world: true geometry and true hidden physics, as a scorer reference control.
- sampled_world: true geometry with the four prior draws (seeds of the plan), as a
  prior-sampling control. Neither is a mathematical bound on other generators.
Reports selected ADE in normalized units and native pixels, the strict gate and RGB
contact F1. Descriptive only; no model, gate, scorer or historical result changes.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.suite import prepare_plan
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.video import evaluate_videos
from click2future.models.object_motion import observe
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids


def ceilings(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output file")
    config = Path(__file__).resolve().parents[1] / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(config.read_text())
    plan = prepare_plan(dataset, config, split="id_test", metrics=("ssim",))
    keys = ("frame_count", "fps", "physics_hz", "resolution")
    reports = {"true_world": [], "sampled_world": []}
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        videos = {name: [] for name in reports}
        for seed in plan["sampling_seeds"]:
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
            meta = {key: ref["metadata"][key] for key in keys}
            videos["sampled_world"].append(run_rollout(scene, **meta)["frames"])
            true = run_rollout(copy.deepcopy(ref["scene_spec"]), **meta)["frames"]
            videos["true_world"].append(true)
        for name, clips in videos.items():
            report, _ = evaluate_videos(
                ref,
                np.stack(clips),
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id="D030-" + name,
            )
            reports[name].append(report)
    result = {"study_id": "D030", "plan_sha256": plan["plan_sha256"], "controls": {}}
    for name, values in reports.items():
        events = {"tp": 0, "fp": 0, "fn": 0}
        for report in values:
            for sample in report["collision"]["per_sample"]:
                events["tp"] += sample["event"]["true_positive"]
                events["fp"] += sample["event"]["false_positive"]
                events["fn"] += sample["event"]["false_negative"]
        den = 2 * events["tp"] + events["fp"] + events["fn"]
        quality = summarize_quality(values)
        ade = quality["selected_ade_mean_cases_if_all_complete"]
        result["controls"][name] = {
            "quality": quality,
            "selected_ade_native_pixels": None if ade is None else ade * 64,
            "contact_counts": events,
            "contact_f1": 2 * events["tp"] / den if den else None,
        }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name, value in ceilings(args.dataset, args.output)["controls"].items():
        q = value["quality"]
        print(
            name,
            "selADE",
            q["selected_ade_mean_cases_if_all_complete"],
            "px",
            value["selected_ade_native_pixels"],
            "F1",
            value["contact_f1"],
            "gate",
            q["passes_tracking_and_selected_trajectory_gate"],
        )
