"""Full-horizon passive-object rest, supplementing the original early action checks."""

import argparse
import json
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.training.train_cvae import sha256

from .suite import _write
from .tracking import TRACKER_VERSION, TrackerConfig, track_video
from .trajectory_audit import verify_inputs
from .uncertainty_strata import action_family

PASSIVE_METRIC_VERSION = "2"


def stationary_reference(reference):
    if len(reference["object_valid"]) != 2 or not reference["object_valid"].all():
        return False
    passive = 1 - reference["selected_object_id"]
    return (
        np.allclose(
            reference["positions"][:, passive],
            reference["positions"][:1, passive],
            atol=1e-8,
            rtol=0,
        )
        and not reference["collision_onset"][:, passive].any()
        and not reference["wall_contact_onset"][:, passive].any()
        and np.array_equal(
            reference["instance_masks"][:, passive],
            np.repeat(reference["instance_masks"][:1, passive], len(reference["frames"]), axis=0),
        )
    )


def passive_horizon(reference, videos):
    if not stationary_reference(reference):
        raise ValueError("Passive rest requires a stationary, contact-free reference object")
    videos = np.asarray(videos)
    if (
        videos.dtype != np.uint8
        or videos.ndim != 5
        or not len(videos)
        or videos.shape[1:] != reference["frames"].shape
        or not np.all(videos[:, 0] == reference["frames"][0])
    ):
        raise ValueError("Provide complete native videos with exact frame zero")
    passive = 1 - reference["selected_object_id"]
    threshold = 1 / videos.shape[3]
    samples = []
    for index, video in enumerate(videos):
        track = track_video(
            video, reference["instance_masks"][0], reference["object_valid"], TrackerConfig()
        )
        complete = bool(track.tracked[:, passive].all())
        excursion = (
            float(
                np.linalg.norm(
                    track.positions[:, passive] - track.positions[:1, passive], axis=-1
                ).max()
            )
            if complete
            else None
        )
        samples.append(
            {
                "sample_index": index,
                "complete": complete,
                "max_excursion": excursion,
                "drifting": excursion >= threshold if complete else None,
            }
        )
    return {
        "passive_metric_version": PASSIVE_METRIC_VERSION,
        "motion_rule": "maximum excursion >= motion_threshold",
        "total_samples": len(samples),
        "complete_samples": sum(row["complete"] for row in samples),
        "missing_samples": sum(not row["complete"] for row in samples),
        "drifting_samples": sum(row["drifting"] is True for row in samples),
        "samples": samples,
        "passive_object_id": passive,
        "motion_threshold": threshold,
        "end_frame": len(reference["frames"]) - 1,
        "extractor_version": TRACKER_VERSION,
    }


def audit(plan_path, predictions_path, dataset, checkpoint):
    plan, run, paths, hashes = verify_inputs(plan_path, predictions_path, dataset, checkpoint)
    rows = []
    for case in plan["cases"]:
        if action_family(case) not in {"forward", "reverse"}:
            continue
        references = [load_rollout(dataset, name) for name in case["reference_ids"]]
        eligible = all(stationary_reference(ref) for ref in references)
        row = {
            **case,
            "eligible": eligible,
            "exclusion_reason": None
            if eligible
            else "at least one physical reference moves or contacts the passive object",
        }
        if eligible:
            with np.load(paths[case["sample_id"]], allow_pickle=False) as archive:
                videos = archive["videos"]
            row["model"] = passive_horizon(references[0], videos)
        rows.append(row)
    expected = 2 * len({case["scene_group_id"] for case in plan["cases"]})
    if len(rows) != expected:
        raise ValueError("Retain both declared noncontact families from every validation scene")
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError("Passive-stationarity input changed during scoring")
    return {
        "passive_metric_version": PASSIVE_METRIC_VERSION,
        "motion_rule": "maximum excursion >= motion_threshold",
        "scope": (
            "supplementary full-horizon passive rest in forward/reverse families; "
            "not original early-window success"
        ),
        "method": run["method"],
        "cases": rows,
        "expected_cases": expected,
        "eligible_cases": sum(row["eligible"] for row in rows),
        "expected_samples": expected * len(plan["sampling_seeds"]),
        "summary": {
            key: sum(row["model"][key] for row in rows if row["eligible"])
            for key in ("total_samples", "complete_samples", "missing_samples", "drifting_samples")
        },
        "reference_rule": (
            "passive pose/mask stationary and contact-free in every hidden-physics reference"
        ),
        "plan_sha256": plan["plan_sha256"],
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sha256(Path(__file__)),
        "test_splits_evaluated": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "dataset", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve previous audits; use a fresh output")
    report = audit(args.plan, args.predictions, args.dataset, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
