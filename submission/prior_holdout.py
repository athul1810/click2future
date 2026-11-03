"""D018 inference-only ID replication; all weights are frozen before data generation."""

import argparse
import json
import os
import runpy
import subprocess
import time
from pathlib import Path

import torch

from click2future.data.generalization import (
    generate_generalization_dataset,
    validate_generalization_dataset,
)
from click2future.evaluation import (
    action_horizon,
    collision_audit,
    passive_stationarity,
    wall_clipping,
)
from click2future.evaluation.pilot import summarize_quality
from click2future.evaluation.prior_decision import analyse
from click2future.evaluation.prior_protocol import COUNTS, SEED, STUDY, registered_holdout
from click2future.evaluation.suite import _digest, _write, prepare_plan, read_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan
from click2future.evaluation.trajectory_pilot import extended_audit, sample_plan
from click2future.evaluation.world_interventions import audit as world_audit
from click2future.submission import verify_extracted_tree
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.train_cvae import sha256
from click2future.training.train_prior import prior_sources

PROJECT = Path(__file__).resolve().parents[1]
NAMES = {f"{arm}-seed{seed}" for seed in (19, 20, 21) for arm in ("direct", "diffusion")}


def source_identity(project):
    """Use the checkout itself, never a parent repository containing an extraction."""
    project = Path(project).resolve()
    if (project / ".git").exists():
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip()
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=project, text=True
        ).strip():
            raise ValueError("Commit the source before freezing or opening D018")
        return {"kind": "git", "revision": revision}
    receipt = verify_extracted_tree(project)
    return {
        "kind": "source_archive",
        "revision": "archive-manifest-sha256:" + receipt["manifest_sha256"],
        **receipt,
    }


def freeze_runs(path):
    rows = json.loads(Path(path).read_text())["runs"]
    if len(rows) != 6 or {r["name"] for r in rows} != NAMES:
        raise ValueError("D018 retains both arms of all three declared refinement seeds")
    result = []
    original = json.loads((PROJECT / "configs/training/prior_direct.json").read_text())
    teacher = None
    for row in sorted(rows, key=lambda r: r["name"]):
        checkpoint = Path(row["checkpoint"]).resolve()
        arm, seed = row["name"].split("-seed")
        model, state = load_trajectory(checkpoint)
        expected = {**original, "seed": int(seed), "world_prior_only": arm == "direct"}
        if (
            state["trajectory_format_version"] != 7
            or state["step"] != 8000
            or state["config"] != expected
            or state["provenance"]["source_sha256"] != prior_sources()
        ):
            raise ValueError("A registered checkpoint or training configuration differs")
        identity = state["provenance"]["initialization"]["checkpoint_sha256"]
        if teacher is not None and identity != teacher:
            raise ValueError("Refinement teachers differ")
        teacher = identity
        result.append(
            {
                "name": row["name"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "config": state["config"],
                "teacher_sha256": identity,
                "training_source_revision": state["provenance"]["source_revision"],
            }
        )
        del model
    return result


def prepare(dataset, training, output):
    if not dataset.exists():
        generate_generalization_dataset(dataset, seed=SEED, group_counts=COUNTS)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    if manifest["seed"] != SEED or manifest["group_counts"] != COUNTS:
        raise ValueError("D018 corpus differs from its registration")
    validate_generalization_dataset(dataset)
    plan = prepare_plan(
        dataset,
        PROJECT / "configs/evaluation/pixel.yaml",
        split="id_test",
        metrics=("ssim",),
        protocol=PROJECT / "configs/evaluation/comparison.yaml",
        actions_config=PROJECT / "configs/evaluation/actions.yaml",
        training_corpus=training,
    )
    plan.pop("plan_sha256")
    plan["study_id"] = STUDY
    plan["plan_sha256"] = _digest(plan)
    if not registered_holdout(plan):
        raise ValueError("D018 plan identity differs")
    _write(output / "plan.json", plan)
    prepare_target_plan(
        output / "plan.json",
        output / "target-controls",
        config=PROJECT / "configs/evaluation/actions.yaml",
        force_magnitude=0.4,
    )
    if not registered_holdout(read_plan(output / "target-controls/plan.json")):
        raise ValueError("D018 selection scope differs")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "dataset", "training-data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Retain attempts and use a fresh output directory")
    identity = source_identity(PROJECT)
    revision = identity["revision"]
    os.environ["C2F_SOURCE_REVISION"] = revision
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # CPU checkpoint validation precedes every new-scene read/generation.
    runs = freeze_runs(args.runs)
    args.output.mkdir(parents=True)
    _write(args.output / "frozen-runs.json", {"decision": STUDY, "runs": runs})
    record = {
        "decision": STUDY,
        "status": "preparing",
        "source_revision": revision,
        "source_identity": identity,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch": str(torch.__version__),
        "driver_sha256": sha256(Path(__file__)),
        "runs_spec_sha256": sha256(args.runs),
        "scope": "inference only; fixed-teacher ID scene replication",
        "declared_test_splits": ["D018:id_test"],
        "test_splits_evaluated": [],
        "runs": {},
        "phases": [],
    }
    started = time.perf_counter()

    def save():
        record["elapsed_seconds"] = time.perf_counter() - started
        _write(args.output / "replication.json", record)

    def phase(name, function, *values, **kwargs):
        row = {"name": name, "started_seconds": time.perf_counter() - started}
        record["phases"].append(row)
        save()
        print(json.dumps({"phase": name}), flush=True)
        value = function(*values, **kwargs)
        row["completed_seconds"] = time.perf_counter() - started
        save()
        return value

    try:
        plan = prepare(args.dataset, args.training_data, args.output)
        record.update(
            status="running",
            plan_sha256=plan["plan_sha256"],
            dataset_manifest_sha256=sha256(args.dataset / "dataset_manifest.json"),
        )
        save()
        response = runpy.run_path(str(PROJECT / "submission/trajectory_response.py"))["diagnose"]
        for run in runs:
            name, checkpoint = run["name"], Path(run["checkpoint"])
            folder = args.output / name
            folder.mkdir()
            # File names retain the existing diagnostic layout; plan metadata says ID test.
            _write(folder / "plan.json", plan)
            steps = 1 if run["config"]["world_prior_only"] else 16
            for split, path in (
                ("validation", args.output / "plan.json"),
                ("selected-object", args.output / "target-controls/plan.json"),
            ):
                samples = folder / (split + "-samples")
                phase(
                    name + "/sample-" + split,
                    sample_plan,
                    path,
                    checkpoint,
                    samples,
                    device=args.device,
                    steps=steps,
                    method="E11",
                    run_id="E11-" + name,
                )
                phase(
                    name + "/score-" + split,
                    run_suite,
                    path,
                    samples / "predictions.json",
                    folder / (split + "-metrics"),
                )
                if split == "validation":
                    record["test_splits_evaluated"] = ["D018:id_test"]
                    save()
            phase(
                name + "/geometry-rest-strata",
                extended_audit,
                folder,
                args.dataset,
                checkpoint,
                args.device,
            )
            predictions = folder / "validation-samples/predictions.json"
            common = (folder / "plan.json", predictions, args.dataset, checkpoint)
            for label, function, values in (
                ("passive", passive_stationarity.audit, common),
                ("clipping", wall_clipping.audit, (*common, folder / "geometry-audit")),
                (
                    "collision",
                    collision_audit.audit,
                    (*common, folder / "geometry-audit", PROJECT / "configs/evaluation/pixel.yaml"),
                ),
            ):
                result = phase(name + "/" + label, function, *values)
                _write(folder / (label + ".json"), result)
            _write(
                folder / "action-horizons.json",
                phase(
                    name + "/horizons", action_horizon.evaluate, folder / "plan.json", predictions
                ),
            )
            phase(
                name + "/world",
                world_audit,
                *common,
                folder / "world-interventions",
                device=args.device,
            )
            phase(
                name + "/response",
                response,
                *common,
                folder / "response-diagnostic",
                device=args.device,
            )
            comparison = json.loads((folder / "validation-metrics/comparison.json").read_text())
            reports = [
                json.loads((folder / "validation-metrics" / r["trajectory"]).read_text())
                for r in comparison["runs"][0]["reports"]
            ]
            quality = summarize_quality(reports)
            _write(folder / "quality.json", quality)
            record["runs"][name] = {**run, "quality": quality}
            if sha256(checkpoint) != run["checkpoint_sha256"]:
                raise ValueError("A frozen checkpoint changed")
            save()
        if sha256(args.runs) != record["runs_spec_sha256"]:
            raise ValueError("Frozen run registry changed")
        if source_identity(PROJECT) != identity:
            raise ValueError("Evaluation source revision changed")
        record["status"] = "completed"
        save()
        _write(args.output / "decision.json", phase("registered-decision", analyse, args.output))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
