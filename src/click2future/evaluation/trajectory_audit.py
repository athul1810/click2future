"""Replay a frozen E5 validation ensemble and audit its uncomposited geometry."""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.models.object_motion import encode_motion, observe, render_motion
from click2future.training.sample_trajectory import load_trajectory, sampling_geometry
from click2future.training.train_cvae import require_tc2, sha256
from click2future.training.trajectory_data import canonical_ids, replay_angles

from .rigid_geometry import GEOMETRY_VERSION, geometry_diagnostics
from .suite import _digest, _seeds, _write


def verify_inputs(plan_path, predictions_path, dataset, checkpoint):
    """Verify historical identities while explicitly relocating local artifacts.

    This is a new geometry audit, not a rerun of the historical evaluator. Its
    original source hashes stay recorded; they are not replaced by current ones.
    All original dataset/reference hashes and the entire case list must match.
    """
    plan_path, predictions_path, dataset, checkpoint = map(
        Path, (plan_path, predictions_path, dataset, checkpoint)
    )
    plan = json.loads(plan_path.read_text())
    if (
        plan["schema_version"] != 1
        or _digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]
        or plan["split"] != "validation"
    ):
        raise ValueError("Provide an unchanged validation plan; final tests stay reserved")
    predictions = json.loads(predictions_path.read_text())
    if (
        predictions["source"] != "generated_video"
        or predictions["evaluation_plan_sha256"] != plan["plan_sha256"]
        or _seeds(predictions["sampling_seeds"]) != _seeds(plan["sampling_seeds"])
        or len(predictions["runs"]) != 1
    ):
        raise ValueError("Predictions must identify the original frozen ensemble")
    run = predictions["runs"][0]
    sampling_path = predictions_path.with_name("sampling.json")
    sampling = json.loads(sampling_path.read_text())
    direct = run["method"] == "E11" and run.get("world_prior_only") is True
    expected_steps, expected_nfe = (1, 1) if direct else (16, 31)
    if (
        sampling["sampling_steps"] != expected_steps
        or sampling["network_evaluations_per_sample"] != expected_nfe
        or any(
            sampling[key] != predictions[key]
            for key in ("source", "runs", "sampling_seeds", "evaluation_plan_sha256")
        )
    ):
        raise ValueError("Audit replay requires the recorded 16-step E5 sampler")
    expected = {case["sample_id"] for case in plan["cases"]}
    if (
        run["method"] not in {"E5", "E6", "E7", "E8", "E9", "E10", "E11"}
        or run["checkpoint_sha256"] != sha256(checkpoint)
        or run.get("untrained_fixture") is not False
        or set(run["videos"]) != expected
        or set(run["video_sha256"]) != expected
    ):
        raise ValueError("Provide the complete trained E5 run and its exact checkpoint")
    hashes, paths = {}, {}
    for key, filename in (("dataset_manifest", "dataset_manifest.json"), ("splits", "splits.json")):
        hashes[dataset / filename] = plan["input_sha256"][key]
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    validation = {row["sample_id"] for row in manifest["samples"] if row["split"] == "validation"}
    references = {name for case in plan["cases"] for name in case["reference_ids"]}
    if references != validation or not expected or len(expected) != len(plan["cases"]):
        raise ValueError("The audit must retain every frozen validation reference and condition")
    groups = defaultdict(list)
    for row in manifest["samples"]:
        if row["split"] == "validation":
            groups[(row["scene_group_id"], row["condition_id"])].append(row["sample_id"])
    if len(plan["cases"]) != len(groups) or any(
        case["reference_ids"] != sorted(groups[(case["scene_group_id"], case["condition_id"])])
        or not case["reference_ids"]
        or case["sample_id"] != case["reference_ids"][0]
        for case in plan["cases"]
    ):
        raise ValueError("Validation cases must match the original grouped references")
    for name in references:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe reference name")
        for suffix in (".npz", ".json"):
            hashes[dataset / (name + suffix)] = plan["input_sha256"][name + suffix]
    for name, filename in run["videos"].items():
        path = predictions_path.parent / filename
        if (
            Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or "\\" in filename
            or path.is_symlink()
            or not path.resolve().is_relative_to(predictions_path.parent.resolve())
        ):
            raise ValueError("Prediction files must stay in their original directory")
        paths[name], hashes[path] = path, run["video_sha256"][name]
    hashes.update(
        {
            plan_path: sha256(plan_path),
            predictions_path: sha256(predictions_path),
            checkpoint: run["checkpoint_sha256"],
            sampling_path: sha256(sampling_path),
        }
    )
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Historical audit input changed: " + path.name)
    return plan, run, paths, hashes


def summarize(records, family):
    keys = (
        "samples",
        "expected_pair_frames",
        "expected_object_frames",
        "clips_with_overlap",
        "clips_with_overlap_over_ten_percent",
        "overlap_frames",
        "overlap_over_ten_percent_frames",
        "clips_outside_interior",
        "outside_interior_object_frames",
        "clips_with_penetration_over_one_pixel",
        "penetration_over_one_pixel_frames",
        "clips_with_boundary_over_one_pixel",
        "boundary_over_one_pixel_object_frames",
    )
    result = {key: sum(row[family][key] for row in records) for key in keys}
    result["condition_count"] = len(records)
    result["scene_groups"] = len({row["scene_group_id"] for row in records})
    result["aggregation"] = "descriptive counts over all declared clips; no independent-clip CI"
    return result


def audit(plan_path, predictions_path, dataset, checkpoint, output, *, device="cpu"):
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise ValueError("Use a fresh audit directory; preserve earlier attempts")
    plan, run, paths, hashes = verify_inputs(plan_path, predictions_path, dataset, checkpoint)
    model, state = load_trajectory(checkpoint, device)
    if run["method"] != {1: "E5", 2: "E6", 3: "E7", 4: "E8", 5: "E9", 6: "E10", 7: "E11"}.get(
        state["trajectory_format_version"]
    ):
        raise ValueError("Checkpoint training variant and audit method disagree")
    if run["method"] == "E11" and run.get("world_prior_only") != model.world_prior_only:
        raise ValueError("Checkpoint prior-only arm and saved sampler disagree")
    replay_steps = 1 if getattr(model, "world_prior_only", False) else 16
    if (
        state["step"] != run["training_steps"]
        or state["config"]["seed"] != run["training_seed"]
        or state["provenance"]["dataset_manifest_sha256"]
        != plan["input_sha256"]["dataset_manifest"]
        or state["provenance"]["split_manifest_sha256"] != plan["input_sha256"]["splits"]
    ):
        raise ValueError("Checkpoint metadata and frozen data disagree")
    output.mkdir(parents=True)
    records = []
    try:
        for case in plan["cases"]:
            reference = load_rollout(dataset, case["sample_id"])
            observation = observe(reference["frames"][0], reference["selected_mask"])
            motion = (
                model.sample(
                    torch.from_numpy(observation.features[None]).to(device),
                    torch.from_numpy(reference["force_xy"][None].copy()).to(device),
                    seeds=plan["sampling_seeds"],
                    steps=replay_steps,
                    sigma_min=model.sigma_min,
                    sigma_max=model.sigma_max,
                    **sampling_geometry(model, observation, device),
                )
                .cpu()
                .numpy()
            )
            with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
                saved = archive["videos"]
            rendered = np.stack([render_motion(observation, item) for item in motion])
            if saved.dtype != np.uint8 or not np.array_equal(rendered, saved):
                raise ValueError(
                    "Pose replay must reproduce every saved RGB pixel: " + case["sample_id"]
                )
            oracle = []
            for name in case["reference_ids"]:
                ref = load_rollout(dataset, name)
                if not np.array_equal(ref["frames"][0], observation.first):
                    raise ValueError("Reference futures must share the observed image")
                order = canonical_ids(observation, ref["selected_object_id"])
                oracle.append(
                    encode_motion(ref["positions"][:, order], replay_angles(ref)[:, order])
                )
            row = {
                **case,
                "pixel_exact_pose_replay": True,
                "learned": geometry_diagnostics(observation, motion),
                "oracle_control": geometry_diagnostics(observation, np.stack(oracle)),
                "static_control": geometry_diagnostics(observation, np.zeros_like(motion)),
            }
            np.savez_compressed(output / (case["sample_id"] + "-poses.npz"), motion=motion)
            records.append(row)
            print(json.dumps({"case": case["sample_id"], "pixel_exact": True}), flush=True)
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise ValueError("Audit input changed during evaluation: " + path.name)
        report = {
            "status": "measured_geometry_audit",
            "version": GEOMETRY_VERSION,
            "method": run["method"],
            "scope": "validation-only pose/silhouette diagnostics; not generated-RGB tracking",
            "sampling_steps": replay_steps,
            "sampling_seeds": plan["sampling_seeds"],
            "checkpoint_sha256": sha256(checkpoint),
            "historical_plan_sha256": plan["plan_sha256"],
            "historical_evaluator_source_sha256": plan["evaluator_source_sha256"],
            "device": device,
            "torch": str(torch.__version__),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_revision": os.environ.get("C2F_SOURCE_REVISION", "unrecorded"),
            "source_sha256": {
                "trajectory_audit.py": sha256(Path(__file__)),
                "rigid_geometry.py": sha256(Path(__file__).with_name("rigid_geometry.py")),
            },
            "summary": {
                family: summarize(records, family)
                for family in ("learned", "oracle_control", "static_control")
            },
            "cases": records,
            "test_splits_evaluated": [],
            "pixel_exact_pose_replay": True,
            "interpretation": (
                "Oracle motion calibrates raster bias only. Static copying is a geometry control, "
                "not an action baseline success. No trajectory, tracker or renderer is corrected."
            ),
        }
        _write(output / "geometry.json", report)
        return report
    except Exception as error:
        _write(output / "failure.json", {"status": "failed", "error": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.device == "cuda":
        require_tc2()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = audit(
        args.plan, args.predictions, args.dataset, args.checkpoint, args.output, device=args.device
    )
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
