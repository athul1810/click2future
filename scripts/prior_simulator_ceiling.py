"""D024 prior-sampling control, scored by the unchanged single-reference gate.

Each case keeps its true scene geometry and visible frame zero. Only the hidden mass,
friction and restitution are replaced by the same four prior draws that the
physics-assisted baselines use. The renders are scored with the unchanged RGB tracker
and the original single-reference gate. A control with the true hidden physics checks
the tracker. These are descriptive simulator controls, not bounds on other methods.
No historical result changes.
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

GATE_ADE = 0.03125


def simulate(ref, scene):
    keys = ("frame_count", "fps", "physics_hz", "resolution")
    out = run_rollout(scene, **{key: ref["metadata"][key] for key in keys})
    if not np.array_equal(out["frames"][0], ref["frames"][0]):
        raise ValueError("Changing hidden physics changed the visible condition")
    return out["frames"]


def ceiling(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Choose a fresh output file")
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/evaluation/pixel.yaml"
    settings = yaml.safe_load(config.read_text())
    plan = prepare_plan(dataset, config, split="id_test", metrics=("ssim",))
    reports = {"sampled_world": [], "true_world": []}
    cases = []
    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        videos = {"sampled_world": [], "true_world": []}
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
            videos["sampled_world"].append(simulate(ref, scene))
            videos["true_world"].append(simulate(ref, copy.deepcopy(ref["scene_spec"])))
        row = {"sample_id": case["sample_id"], "scene_group_id": case["scene_group_id"]}
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
                run_id="D024-" + name,
            )
            reports[name].append(report)
            row[name + "_selected_ade"] = report["trajectory"]["selected_object"][
                "mean_ade_over_complete_samples"
            ]
        cases.append(row)
    result = {"study_id": "D024", "split": "fresh_id_test", "gate_selected_ade": GATE_ADE}
    for name, values in reports.items():
        quality = summarize_quality(values)
        ades = [r[name + "_selected_ade"] for r in cases]
        result[name] = {
            "quality": quality,
            "cases_over_gate": sum(a is None or a > GATE_ADE for a in ades),
            "case_count": len(ades),
        }
    result["cases"] = cases
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    r = ceiling(args.dataset, args.output)
    for name in ("sampled_world", "true_world"):
        q = r[name]["quality"]
        print(
            name,
            r[name]["cases_over_gate"],
            "/",
            r[name]["case_count"],
            "over gate; mean ADE",
            q["selected_ade_mean_cases_if_all_complete"],
            "gate",
            q["passes_tracking_and_selected_trajectory_gate"],
        )
