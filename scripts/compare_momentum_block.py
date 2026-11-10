"""Compare the optional momentum/contact blocks on frozen trajectory samples."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout
from click2future.evaluation.passive_stationarity import passive_horizon, stationary_reference
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.evaluation.tracking import TrackerConfig
from click2future.evaluation.uncertainty_strata import zero_horizon
from click2future.evaluation.video import evaluate_videos
from click2future.models.contact_projection import PROJECTION_CONTRACT, project_motion
from click2future.models.momentum_block import MOMENTUM_BLOCK_CONTRACT, apply_momentum_block
from click2future.models.object_motion import observe, render_motion
from click2future.models.world_motion import sample_world
from click2future.sim import run_rollout
from click2future.training.trajectory_data import canonical_ids

MODES = ("raw", "momentum", "contact", "combined")
ADE_MARGIN = 0.0005


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_archive(folder, sample_id):
    candidates = [folder / f"{sample_id}-poses.npz", folder / f"{sample_id}.npz"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"Require one frozen pose archive for {sample_id}")
    return matches[0]


def transform(obs, motion, world, force, mode):
    if mode in {"momentum", "combined"}:
        motion = apply_momentum_block(obs, motion, world[:, 0], force)
    if mode in {"contact", "combined"}:
        motion = project_motion(obs, motion, world[:, 0])
    return motion


def momentum_error(motion, masses, force, duration=0.25, fps=8):
    positions = np.concatenate((np.zeros((1, 2, 2)), 0.5 * motion[..., :2]))
    velocity = np.diff(positions, axis=0) * fps
    start = int(np.ceil(duration * fps))
    total = (masses[:, None] * velocity[start:]).sum(1)
    return float(np.linalg.norm(total - np.asarray(force) * duration, axis=-1).mean())


def empty_totals():
    counts = {
        key: 0
        for key in (
            "total_samples",
            "complete_samples",
            "missing_samples",
            "drifting_samples",
        )
    }
    return {
        mode: {
            "reports": [],
            "selected_ade": [],
            "same_world_ade": [],
            "momentum_error": [],
            "penetrating_clips": 0,
            "boundary_clips": 0,
            "events": {"tp": 0, "fp": 0, "fn": 0},
            "zero_force": counts.copy(),
            "passive": counts.copy(),
        }
        for mode in MODES
    }


def compare(dataset, plan_path, poses_dir, checkpoint, output):
    dataset, poses_dir, output = map(Path, (dataset, poses_dir, output))
    if output.exists():
        raise ValueError("Retain the previous comparison and choose a fresh output")
    plan = json.loads(Path(plan_path).read_text())
    if plan["split"] != "id_test" or len(plan["sampling_seeds"]) != 4:
        raise ValueError("Use the frozen four-sample ID plan")
    settings = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/evaluation/pixel.yaml").read_text()
    )
    totals = empty_totals()

    for case in plan["cases"]:
        ref = load_rollout(dataset, case["sample_id"])
        obs = observe(ref["frames"][0], ref["selected_mask"])
        order = canonical_ids(obs, ref["selected_object_id"])
        with np.load(pose_archive(poses_dir, case["sample_id"])) as arrays:
            raw = arrays["motion"]
        if raw.shape != (4, 31, 2, 4):
            raise ValueError("Frozen pose samples have the wrong shape")
        true_delta = ref["positions"][1:, order] - ref["positions"][:1, order]
        videos = {mode: [] for mode in MODES}
        motions = {mode: [] for mode in MODES}

        for index, seed in enumerate(plan["sampling_seeds"]):
            world = sample_world(seed)[0].numpy()
            scene = copy.deepcopy(ref["scene_spec"])
            objects = {obj["object_id"]: obj for obj in scene["objects"]}
            for slot, object_id in enumerate(order):
                objects[object_id]["hidden_physics"] = dict(
                    zip(("mass", "friction", "restitution"), world[slot].tolist(), strict=True)
                )
            target = run_rollout(
                scene,
                **{
                    name: ref["metadata"][name]
                    for name in ("frame_count", "fps", "physics_hz", "resolution")
                },
            )
            same_world = target["positions"][1:, order] - target["positions"][:1, order]
            for mode in MODES:
                motion = transform(obs, raw[index], world, ref["force_xy"], mode)
                motions[mode].append(motion)
                videos[mode].append(render_motion(obs, motion))
                delta = 0.5 * motion[..., :2]
                selected = obs.selected_index
                totals[mode]["selected_ade"].append(
                    float(
                        np.linalg.norm(
                            delta[:, selected] - true_delta[:, selected], axis=-1
                        ).mean()
                    )
                )
                totals[mode]["same_world_ade"].append(
                    float(
                        np.linalg.norm(
                            delta[:, selected] - same_world[:, selected], axis=-1
                        ).mean()
                    )
                )
                if "-wall-" not in case["sample_id"]:
                    totals[mode]["momentum_error"].append(
                        momentum_error(motion, world[:, 0], ref["force_xy"])
                    )

        for mode in MODES:
            motion = np.stack(motions[mode])
            video = np.stack(videos[mode])
            report, _ = evaluate_videos(
                ref,
                video,
                tracker=TrackerConfig(**settings["tracker"]),
                max_gap_pixels=settings["max_gap_pixels"],
                event_tolerance=settings["event_tolerance_intervals"],
                motion_threshold=settings["motion_threshold"],
                motion_window=settings["motion_window"],
                motion_rule=settings["motion_rule"],
                run_id="momentum-block-" + mode,
            )
            totals[mode]["reports"].append(report)
            for sample in report["collision"]["per_sample"]:
                for key, field in (
                    ("tp", "true_positive"),
                    ("fp", "false_positive"),
                    ("fn", "false_negative"),
                ):
                    totals[mode]["events"][key] += sample["event"][field]
            geometry = geometry_diagnostics(obs, motion)
            totals[mode]["penetrating_clips"] += geometry["clips_with_penetration_over_one_pixel"]
            totals[mode]["boundary_clips"] += sum(
                row["boundary_over_one_pixel_object_frames"] > 0 for row in geometry["clips"]
            )
            if "-zero-" in case["sample_id"]:
                values = zero_horizon(ref, video)
                for key in totals[mode]["zero_force"]:
                    totals[mode]["zero_force"][key] += values[key]
            if any(tag in case["sample_id"] for tag in ("-forward-", "-reverse-")) and all(
                stationary_reference(load_rollout(dataset, sample_id))
                for sample_id in case["reference_ids"]
            ):
                values = passive_horizon(ref, video)
                for key in totals[mode]["passive"]:
                    totals[mode]["passive"][key] += values[key]

    results = {}
    for mode, values in totals.items():
        events = values["events"]
        denominator = 2 * events["tp"] + events["fp"] + events["fn"]
        results[mode] = {
            "quality": summarize_quality(values["reports"]),
            "pose_selected_ade": float(np.mean(values["selected_ade"])),
            "same_world_selected_ade": float(np.mean(values["same_world_ade"])),
            "post_force_momentum_error": float(np.mean(values["momentum_error"])),
            "penetrating_clips": values["penetrating_clips"],
            "boundary_clips": values["boundary_clips"],
            "zero_force": values["zero_force"],
            "passive": values["passive"],
            "contact_f1": 2 * events["tp"] / denominator if denominator else None,
        }

    def screen(candidate, control):
        return bool(
            candidate["post_force_momentum_error"] < control["post_force_momentum_error"]
            and candidate["pose_selected_ade"] <= control["pose_selected_ade"] + ADE_MARGIN
            and candidate["same_world_selected_ade"]
            <= control["same_world_selected_ade"] + ADE_MARGIN
            and candidate["penetrating_clips"] <= control["penetrating_clips"]
            and candidate["boundary_clips"] <= control["boundary_clips"]
            and candidate["zero_force"]["drifting_samples"]
            <= control["zero_force"]["drifting_samples"]
            and candidate["passive"]["drifting_samples"]
            <= control["passive"]["drifting_samples"]
        )

    result = {
        "scope": "exploratory rescore of frozen ID poses; no training or new model samples",
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "plan_sha256": sha256(plan_path),
        "checkpoint_sha256": sha256(checkpoint),
        "momentum_contract": MOMENTUM_BLOCK_CONTRACT,
        "contact_projection_contract": PROJECTION_CONTRACT,
        "modes": results,
        "momentum_screen": screen(results["momentum"], results["raw"]),
        "combined_screen": screen(results["combined"], results["contact"]),
    }
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--poses-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.dataset, args.plan, args.poses_dir, args.checkpoint, args.output)
