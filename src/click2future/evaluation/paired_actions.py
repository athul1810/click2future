"""Freeze reference-only action eligibility and score paired generated videos."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from click2future.data import load_rollout, validate_rollout

from .actions import action_metrics, reversal_metrics
from .aggregate import _estimate, _integer, _mean
from .controls import KINDS, evaluate_controls, validate_controls
from .tracking import TrackerConfig


def action_settings(path):
    config = yaml.safe_load(Path(path).read_text())
    _integer(config["end_frame"], "end_frame", 1)
    threshold = config["motion_threshold"]
    if isinstance(threshold, bool) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Action motion_threshold must be finite and positive")
    tracker = TrackerConfig(**config["tracker"])
    tracker.validate()
    if not np.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("Action fps must be finite and positive")
    return {**config, "tracker": tracker}


def _objects(ref):
    return json.dumps(ref["scene_spec"]["objects"], sort_keys=True)


def _visible_objects(ref):
    return [
        {key: value for key, value in obj.items() if key != "hidden_physics"}
        for obj in ref["scene_spec"]["objects"]
    ]


def prepare_action_groups(plan, config_path):
    """Keep the fixed window only if ALL hidden-physics references avoid contact.

    The zero action has one reference because these scenes start at rest. Its
    physics must match one of the paired forward/reverse references. Every
    nonzero hidden-physics variant must have its exact opposite-force partner.
    """
    settings = action_settings(config_path)
    end = settings["end_frame"]
    groups = defaultdict(dict)
    for case in plan["cases"]:
        ref = load_rollout(plan["dataset"], case["sample_id"])
        kind = ref["scene_spec"].get("action_variant")
        if kind in KINDS:
            group = groups[case["scene_group_id"]]
            if kind in group:
                raise ValueError("Action controls require one condition per action and scene group")
            group[kind] = case
    expected = {case["scene_group_id"] for case in plan["cases"]}
    if set(groups) != expected or any(set(group) != set(KINDS) for group in groups.values()):
        raise ValueError("Action controls require forward, reverse and zero in every scene group")
    result = []
    for group_id, cases in sorted(groups.items()):
        refs = {
            kind: [load_rollout(plan["dataset"], name) for name in case["reference_ids"]]
            for kind, case in cases.items()
        }
        base = refs["forward"][0]
        forces = {kind: refs[kind][0]["force_xy"] for kind in KINDS}
        if (
            not np.any(forces["forward"])
            or not np.array_equal(forces["reverse"], -forces["forward"])
            or np.any(forces["zero"])
        ):
            raise ValueError("Action controls require nonzero, reversed and zero forces")
        contact = False
        for kind, variants in refs.items():
            for ref in variants:
                validate_rollout(ref)
                if (
                    end >= len(ref["frames"])
                    or list(ref["frames"].shape[1:3]) != list(settings["resolution_hw"])
                    or not np.allclose(
                        np.diff(ref["timestamps"]), 1 / settings["fps"], rtol=0, atol=1e-8
                    )
                    or not np.array_equal(ref["timestamps"], base["timestamps"])
                    or not np.array_equal(ref["frames"][0], base["frames"][0])
                    or not np.array_equal(ref["force_xy"], forces[kind])
                    or ref["selected_object_id"] != base["selected_object_id"]
                    or ref["force_duration"] != base["force_duration"]
                    or _visible_objects(ref) != _visible_objects(base)
                ):
                    raise ValueError("Action reference appearance, target or timing differs")
                if any(
                    any(obj["initial_velocity"]) or obj["initial_angular_velocity"] != 0
                    for obj in ref["scene_spec"]["objects"]
                ):
                    raise ValueError("Action controls must start at rest")
                contact |= bool(
                    ref["collision_onset"][:end].any() or ref["wall_contact_onset"][:end].any()
                )
        forward = {_objects(ref): ref for ref in refs["forward"]}
        reverse = {_objects(ref): ref for ref in refs["reverse"]}
        if (
            len(forward) != len(refs["forward"])
            or len(reverse) != len(refs["reverse"])
            or set(forward) != set(reverse)
        ):
            raise ValueError("Forward/reverse hidden-physics references must match exactly")
        if len(refs["zero"]) != 1 or _objects(refs["zero"][0]) not in forward:
            raise ValueError("Zero reference physics must match a forward/reverse pair")
        if not contact:
            key = _objects(refs["zero"][0])
            validate_controls(
                {"forward": forward[key], "reverse": reverse[key], "zero": refs["zero"][0]}, end
            )
        group_record = {
            "scene_group_id": group_id,
            "eligible": not contact,
            "exclusion_reason": "reference_contact_within_fixed_window" if contact else None,
            "sample_ids": {kind: cases[kind]["sample_id"] for kind in KINDS},
            "reference_ids": {kind: cases[kind]["reference_ids"] for kind in KINDS},
            "scoring_reference_ids": {
                kind: next(
                    name
                    for name, ref in zip(cases[kind]["reference_ids"], refs[kind], strict=True)
                    if _objects(ref) == _objects(refs["zero"][0])
                )
                for kind in KINDS
            },
        }
        if not contact:
            # Calibration only: labels and rasterized references never condition a model.
            ordered = {
                "forward": [forward[key] for key in sorted(forward)],
                "reverse": [reverse[key] for key in sorted(forward)],
                "zero": refs["zero"],
            }
            positions = {
                kind: np.stack([ref["positions"] for ref in variants])
                for kind, variants in ordered.items()
            }
            tracked = {
                kind: np.ones(value.shape[:-1], dtype=bool) for kind, value in positions.items()
            }
            common = dict(
                selected_object=base["selected_object_id"],
                end_frame=end,
                motion_threshold=settings["motion_threshold"],
            )
            label_scores = {
                kind: action_metrics(
                    value, tracked[kind], base["object_valid"], force_xy=forces[kind], **common
                )
                for kind, value in positions.items()
            }
            label_scores["reversal"] = reversal_metrics(
                positions["forward"],
                positions["reverse"],
                tracked["forward"],
                tracked["reverse"],
                base["object_valid"],
                force_xy=forces["forward"],
                **common,
            )
            zero_key = _objects(refs["zero"][0])
            matched = {
                "forward": forward[zero_key],
                "reverse": reverse[zero_key],
                "zero": refs["zero"][0],
            }
            raster_videos = {
                kind: np.stack([ref["frames"] for ref in variants])
                for kind, variants in ordered.items()
                if kind != "zero"
            }
            raster_videos["zero"] = np.repeat(refs["zero"][0]["frames"][None], len(forward), axis=0)
            group_record.update(
                reference_label_scores=label_scores,
                reference_rgb_scores=evaluate_controls(
                    matched,
                    raster_videos,
                    source="simulator_control",
                    run_id="reference-calibration",
                    **settings,
                )["scores"],
            )
        result.append(group_record)
    return result


def evaluate_action_groups(plan, paths, *, source, run_id):
    settings = action_settings(plan["files"]["actions_config"])
    records = []
    for group in plan["action_groups"]:
        record = dict(group)
        if group["eligible"]:
            refs = {
                kind: load_rollout(plan["dataset"], name)
                for kind, name in group["scoring_reference_ids"].items()
            }
            videos, valid = {}, {}
            for kind, name in group["sample_ids"].items():
                with np.load(paths[name], allow_pickle=False) as archive:
                    videos[kind] = archive["videos"]
                    valid[kind] = archive["frame_valid"] if "frame_valid" in archive else None
            report = evaluate_controls(
                refs, videos, frame_valid=valid, source=source, run_id=run_id, **settings
            )
            record["scores"] = report["scores"]
        records.append(record)
    return records


def summarize_actions(records, *, bootstrap=2000, confidence=0.95, seed=0):
    """Average samples within a scene, then training runs, then scene groups.

    Bootstrap only whole scene groups. Rates condition on complete tracks;
    missing rates, stationary counts and denominators are retained alongside.
    """
    _integer(bootstrap, "bootstrap", 1)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    group_ids = [group["scene_group_id"] for group in records[0]["groups"] if group["eligible"]]
    draws = np.random.default_rng(seed).integers(
        0, max(1, len(group_ids)), (bootstrap, len(group_ids))
    )
    methods = []
    for method in sorted({record["method"] for record in records}):
        runs = [record for record in records if record["method"] == method]
        values, counts = defaultdict(lambda: defaultdict(list)), defaultdict(int)
        for run in runs:
            for group in run["groups"]:
                if not group["eligible"]:
                    continue
                group_id = group["scene_group_id"]
                for kind, score in group["scores"].items():
                    for key, value in score.items():
                        if key.endswith(("_samples", "_pairs")):
                            counts[f"{kind}.{key}"] += value
                        elif key.startswith(
                            ("mean_", "drift_rate_", "stationary_rate_", "correct_force_")
                        ):
                            values[f"{kind}.{key}"][group_id].append(value)
                    total = score["total_pairs"] if kind == "reversal" else score["total_samples"]
                    missing = (
                        score["missing_pairs"] if kind == "reversal" else score["missing_samples"]
                    )
                    values[f"{kind}.missing_rate"][group_id].append(missing / total)
        methods.append(
            {
                "method": method,
                "runs": len(runs),
                "counts": dict(counts),
                "metrics": {
                    key: _estimate([_mean(groups[name]) for name in group_ids], draws, confidence)
                    for key, groups in sorted(values.items())
                },
            }
        )
    return {
        "methods": methods,
        "bootstrap": {
            "unit": "scene_group_id",
            "replicates": bootstrap,
            "confidence": confidence,
            "seed": seed,
        },
    }


def actions_csv(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "method",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
            "scored_scene_groups",
            "expected_scene_groups",
            "source",
        ]
    )
    for row in report["summary"]["methods"] + report["reference_calibration"]["summary"]["methods"]:
        for key, metric in row["metrics"].items():
            writer.writerow(
                [
                    row["method"],
                    key,
                    metric["estimate"],
                    *(metric["ci"] or [None, None]),
                    metric["scored_scene_groups"],
                    metric["expected_scene_groups"],
                    "simulator_reference"
                    if row["method"].startswith("reference-")
                    else report["source"],
                ]
            )
    return stream.getvalue()
