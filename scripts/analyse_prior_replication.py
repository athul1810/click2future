"""Verify completed D018 artifacts and retain per-scene aggregates for every run."""

import argparse
import hashlib
import json
from pathlib import Path

from click2future.evaluation.prior_decision import analyse
from click2future.evaluation.suite import read_plan
from click2future.evaluation.trajectory_audit import verify_inputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    record = read(root / "replication.json")
    decision = analyse(root)
    if decision != read(root / "decision.json"):
        raise ValueError("Saved decision differs from the registered rules")
    frozen = read(root / "frozen-runs.json")["runs"]
    common = read_plan(root / "plan.json")
    selection = read_plan(root / "target-controls/plan.json")
    if common["split"] != "id_test" or len(common["cases"]) != 60:
        raise ValueError("D018 requires every declared ID condition")
    groups = sorted({c["scene_group_id"] for c in common["cases"]})
    if len(groups) != 10 or len(selection["cases"]) != 20:
        raise ValueError("Scene/selection counts differ")
    results = {}
    for row in frozen:
        name = row["name"]
        folder = root / name
        checkpoint = Path(row["checkpoint"])
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed: " + name)
        plan, _, _, _ = verify_inputs(
            folder / "plan.json",
            folder / "validation-samples/predictions.json",
            common["dataset"],
            checkpoint,
        )
        if plan != common:
            raise ValueError("Primary plan differs: " + name)
        geometry = read(folder / "geometry-audit/geometry.json")
        collision = read(folder / "collision.json")
        if not geometry["pixel_exact_pose_replay"] or len(geometry["cases"]) != 60:
            raise ValueError("Pose replay is incomplete: " + name)
        scores = read(folder / "validation-metrics/comparison.json")
        scene_rows = {
            g: {
                "conditions": 0,
                "samples": 0,
                "penetrating_clips": 0,
                "penetrating_frames": 0,
                "complete_selected_clips": 0,
                "selected_ade": [],
                "expected_contact_timelines": 0,
                "complete_contact_timelines": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
            }
            for g in groups
        }
        for case in geometry["cases"]:
            if not case["pixel_exact_pose_replay"]:
                raise ValueError("A saved geometry replay differs")
            target = scene_rows[case["scene_group_id"]]
            value = case["learned"]
            target["conditions"] += 1
            target["samples"] += value["samples"]
            target["penetrating_clips"] += value["clips_with_penetration_over_one_pixel"]
            target["penetrating_frames"] += value["penetration_over_one_pixel_frames"]
        for entry in scores["runs"][0]["reports"]:
            report = read(folder / "validation-metrics" / entry["trajectory"])
            target = scene_rows[report["scene_group_id"]]
            selected = report["trajectory"]["selected_object"]
            target["complete_selected_clips"] += selected["complete_samples"]
            target["selected_ade"].append(selected["mean_ade_over_complete_samples"])
        for case in collision["cases"]:
            target = scene_rows[case["scene_group_id"]]
            for sample in case["samples"]:
                value = sample["reference_comparisons"][0]
                target["expected_contact_timelines"] += 1
                if value["complete"]:
                    target["complete_contact_timelines"] += 1
                    for short, key in [
                        ("tp", "true_positive"),
                        ("fp", "false_positive"),
                        ("fn", "false_negative"),
                    ]:
                        target[short] += value["event"][key]
        for target in scene_rows.values():
            if (
                target["conditions"] != 6
                or target["samples"] != 24
                or len(target["selected_ade"]) != 6
            ):
                raise ValueError("A declared scene is incomplete")
            ades = target.pop("selected_ade")
            target["selected_ade_if_all_complete"] = (
                sum(ades) / len(ades)
                if target["complete_selected_clips"] == 24 and all(v is not None for v in ades)
                else None
            )
            denominator = 2 * target["tp"] + target["fp"] + target["fn"]
            target["contact_f1_over_complete"] = (
                2 * target["tp"] / denominator if denominator else None
            )
        geometry_summary = geometry["summary"]["learned"]
        if (
            sum(r["penetrating_clips"] for r in scene_rows.values())
            != geometry_summary["clips_with_penetration_over_one_pixel"]
        ):
            raise ValueError("Scene penetration totals do not reproduce the aggregate")
        if (
            sum(r["penetrating_frames"] for r in scene_rows.values())
            != geometry_summary["penetration_over_one_pixel_frames"]
        ):
            raise ValueError("Scene penetration-frame totals differ")
        events = collision["model_events_against_first_reference"]
        for short, key in [
            ("tp", "true_positive"),
            ("fp", "false_positive"),
            ("fn", "false_negative"),
        ]:
            if sum(r[short] for r in scene_rows.values()) != events[key]:
                raise ValueError("Scene event counts differ")
        sampling = read(folder / "validation-samples/sampling.json")
        expected_steps = 1 if name.startswith("direct") else 16
        if (
            sampling["sampling_steps"] != expected_steps
            or sampling["network_evaluations_per_sample"] != 2 * expected_steps - 1
        ):
            raise ValueError("Sampler budget differs")
        results[name] = {
            "checkpoint_sha256": row["checkpoint_sha256"],
            "scene_groups": scene_rows,
            "sampling": {
                k: sampling[k]
                for k in ["sampling_steps", "network_evaluations_per_sample", "seconds", "device"]
            },
            "pixel_exact_primary_pose_replays": 240,
            "quality": read(folder / "quality.json"),
        }
        print(json.dumps({"verified": name, "primary_clips": 240, "groups": 10}), flush=True)
    return {
        "status": "verified_completed_D018",
        "decision": decision,
        "scope": (
            "all three refinement seeds conditional on one fixed teacher; "
            "ten ID scenes, no OOD or external SOTA claim"
        ),
        "evaluation_source_revision": record["source_revision"],
        "analysis_source_sha256": sha(__file__),
        "primary_pose_replay_clips": 1440,
        "models": results,
        "original_failed_attempt_retained": (
            "results; missing original-corpus files, before scoring"
        ),
        "inference_corrections": False,
        "test_splits_evaluated": ["D018:id_test"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Keep previous analyses and use a new output file")
    args.output.write_text(
        json.dumps(verify(args.root), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
