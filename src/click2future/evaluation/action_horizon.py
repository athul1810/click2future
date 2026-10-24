"""Action scores beyond the fixed early window, and reversal without zero force.

This supplements, never replaces, the original 0.5-second action and
selected-object controls. The unchanged scorers run at the configured window
and at the final frame. A horizon is scored only when every hidden-physics
reference avoids contact throughout it. Groups without a zero-force condition,
such as force-OOD, keep their forward/reverse scores and omit only zero force.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from click2future.data import load_rollout, validate_rollout

from .actions import action_metrics, reversal_metrics
from .paired_actions import _objects, _visible_objects, action_settings, summarize_actions
from .suite import _digest, _evaluator_sources, _seeds, _sha, _write
from .targets import KINDS as TARGET_KINDS
from .targets import _score_rgb, target_metrics
from .tracking import TRACKER_VERSION, track_video

ACTION_HORIZON_VERSION = "1"
ACTION_KINDS = ("forward", "reverse", "zero")
HORIZONS = ("window", "full")
SCORER_SOURCES = ("action_horizon.py", "actions.py", "targets.py", "tracking.py")
DEFAULT_ACTIONS = Path(__file__).resolve().parents[3] / "configs/evaluation/actions.yaml"


def contact_free(references, end_frame):
    """Apply the original window rule: no collision or wall onset before end_frame."""
    return not any(
        ref["collision_onset"][:end_frame].any() or ref["wall_contact_onset"][:end_frame].any()
        for ref in references
    )


def _horizons(base, settings):
    window, final = settings["end_frame"], len(base["frames"]) - 1
    if not 1 <= window < final:
        raise ValueError("The action window must end before the final frame")
    return {"window": window, "full": final}


def _horizon(base, end_frame, references):
    eligible = contact_free(references, end_frame)
    return {
        "end_frame": end_frame,
        "end_time_seconds": float(base["timestamps"][end_frame]),
        "eligible": eligible,
        "exclusion_reason": None if eligible else "reference_contact_within_horizon",
    }


def _track(ref, clips, frame_valid, tracker, end_frame):
    clips = np.asarray(clips)
    if (
        clips.dtype != np.uint8
        or clips.ndim != 5
        or not len(clips)
        or clips.shape[1:] != ref["frames"].shape
        or not np.all(clips[:, 0] == ref["frames"][0])
    ):
        raise ValueError("Videos must be uint8 [K,T,H,W,3] matching references with exact frame 0")
    known = np.ones(clips.shape[:2], dtype=bool) if frame_valid is None else np.asarray(frame_valid)
    if known.dtype != np.bool_ or known.shape != clips.shape[:2] or not known[:, 0].all():
        raise ValueError("frame_valid must be boolean [K,T] with valid first frames")
    # Truncate before tracking, exactly as the original window controls do.
    tracks = [
        track_video(clip[: end_frame + 1], ref["instance_masks"][0], ref["object_valid"], tracker)
        for clip in clips
    ]
    positions = np.stack([track.positions for track in tracks])
    tracked = np.stack([track.tracked for track in tracks]) & known[:, : end_frame + 1, None]
    positions[~tracked] = np.nan
    return positions, tracked


def score_actions(references, videos, *, settings, end_frame, frame_valid=None):
    """Score forward/reverse and optional zero-force videos with the original rules."""
    if len({len(videos[kind]) for kind in references}) != 1:
        raise ValueError("All actions need the same paired sample seeds")
    common = dict(end_frame=end_frame, motion_threshold=settings["motion_threshold"])
    arrays, scores = {}, {}
    for kind, ref in references.items():
        valid = None if frame_valid is None else frame_valid.get(kind)
        arrays[kind] = _track(ref, videos[kind], valid, settings["tracker"], end_frame)
        scores[kind] = action_metrics(
            *arrays[kind],
            ref["object_valid"],
            selected_object=ref["selected_object_id"],
            force_xy=ref["force_xy"],
            **common,
        )
    base = references["forward"]
    scores["reversal"] = reversal_metrics(
        arrays["forward"][0],
        arrays["reverse"][0],
        arrays["forward"][1],
        arrays["reverse"][1],
        base["object_valid"],
        selected_object=base["selected_object_id"],
        force_xy=base["force_xy"],
        **common,
    )
    return scores


def _label_actions(ordered, base, end_frame, settings):
    positions = {
        kind: np.stack([ref["positions"] for ref in refs]) for kind, refs in ordered.items()
    }
    tracked = {kind: np.ones(value.shape[:-1], dtype=bool) for kind, value in positions.items()}
    common = dict(
        selected_object=base["selected_object_id"],
        end_frame=end_frame,
        motion_threshold=settings["motion_threshold"],
    )
    scores = {
        kind: action_metrics(
            positions[kind],
            tracked[kind],
            base["object_valid"],
            force_xy=ordered[kind][0]["force_xy"],
            **common,
        )
        for kind in ordered
    }
    scores["reversal"] = reversal_metrics(
        positions["forward"],
        positions["reverse"],
        tracked["forward"],
        tracked["reverse"],
        base["object_valid"],
        force_xy=base["force_xy"],
        **common,
    )
    return scores


def prepare_action_groups(plan, settings):
    """Group forward/reverse/zero cases; zero force is optional, unlike the original suite."""
    cases = defaultdict(dict)
    for case in plan["cases"]:
        kind = load_rollout(plan["dataset"], case["sample_id"])["scene_spec"].get("action_variant")
        if kind in ACTION_KINDS:
            if kind in cases[case["scene_group_id"]]:
                raise ValueError("Expected one condition per action and scene group")
            cases[case["scene_group_id"]][kind] = case
    groups = []
    for group_id, group_cases in sorted(cases.items()):
        kinds = [kind for kind in ACTION_KINDS if kind in group_cases]
        record = {
            "scene_group_id": group_id,
            "actions": kinds,
            "sample_ids": {kind: group_cases[kind]["sample_id"] for kind in kinds},
            "horizons": {},
        }
        if not {"forward", "reverse"} <= set(kinds):
            groups.append({**record, "exclusion_reason": "forward_or_reverse_missing"})
            continue
        refs = {
            kind: [
                load_rollout(plan["dataset"], name) for name in group_cases[kind]["reference_ids"]
            ]
            for kind in kinds
        }
        base = refs["forward"][0]
        forces = {kind: refs[kind][0]["force_xy"] for kind in kinds}
        if (
            not np.any(forces["forward"])
            or not np.array_equal(forces["reverse"], -forces["forward"])
            or ("zero" in kinds and np.any(forces["zero"]))
        ):
            raise ValueError("Action groups require a nonzero force, its reverse and zero force")
        for kind, variants in refs.items():
            for ref in variants:
                validate_rollout(ref)
                if (
                    not np.array_equal(ref["timestamps"], base["timestamps"])
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
        physics = {kind: {_objects(ref): ref for ref in refs[kind]} for kind in kinds}
        if any(len(physics[kind]) != len(refs[kind]) for kind in kinds) or set(
            physics["forward"]
        ) != set(physics["reverse"]):
            raise ValueError("Forward/reverse hidden-physics references must match exactly")
        if "zero" in kinds:
            if len(refs["zero"]) != 1 or _objects(refs["zero"][0]) not in physics["forward"]:
                raise ValueError("Zero reference physics must match a forward/reverse pair")
            key = _objects(refs["zero"][0])
        else:
            # Without zero force, score against the first listed forward physics variant.
            key = _objects(refs["forward"][0])
        names = {
            kind: dict(
                zip(map(_objects, refs[kind]), group_cases[kind]["reference_ids"], strict=True)
            )
            for kind in kinds
        }
        record["scoring_reference_ids"] = {kind: names[kind][key] for kind in kinds}
        scoring = {kind: physics[kind][key] for kind in kinds}
        ordered = {
            kind: [physics[kind][p] for p in sorted(physics["forward"])]
            for kind in ("forward", "reverse")
        }
        if "zero" in kinds:
            ordered["zero"] = refs["zero"]
        everything = [ref for variants in refs.values() for ref in variants]
        for name, end in _horizons(base, settings).items():
            entry = _horizon(base, end, everything)
            if entry["eligible"]:
                # Calibration only: labels and reference renders never condition a model.
                videos = {
                    kind: np.stack([ref["frames"] for ref in ordered[kind]]) for kind in kinds
                }
                if "zero" in kinds:
                    videos["zero"] = np.repeat(videos["zero"], len(ordered["forward"]), axis=0)
                entry.update(
                    reference_label_scores=_label_actions(ordered, base, end, settings),
                    reference_rgb_scores=score_actions(
                        scoring, videos, settings=settings, end_frame=end
                    ),
                )
            record["horizons"][name] = entry
        groups.append(record)
    return groups


def prepare_selection_groups(plan, settings):
    """Re-score frozen selected-object pairs at both horizons."""
    groups = []
    for group in plan.get("target_groups", []):
        refs = {
            kind: [load_rollout(plan["dataset"], name) for name in group["reference_ids"][kind]]
            for kind in TARGET_KINDS
        }
        base = refs["first"][0]
        record = {
            "scene_group_id": group["scene_group_id"],
            "sample_ids": group["sample_ids"],
            "selected_object_ids": group["selected_object_ids"],
            "horizons": {},
        }
        for name, end in _horizons(base, settings).items():
            entry = _horizon(base, end, [ref for variants in refs.values() for ref in variants])
            if entry["eligible"]:
                positions = [
                    np.stack([ref["positions"] for ref in refs[kind]]) for kind in TARGET_KINDS
                ]
                entry.update(
                    reference_label_scores={
                        "selection": target_metrics(
                            *positions,
                            *(np.ones(value.shape[:-1], dtype=bool) for value in positions),
                            base["object_valid"],
                            selected_object_ids=group["selected_object_ids"],
                            force_xy=group["force_xy"],
                            end_frame=end,
                            motion_threshold=settings["motion_threshold"],
                        )
                    },
                    reference_rgb_scores={
                        "selection": _score_rgb(
                            {kind: refs[kind][0] for kind in TARGET_KINDS},
                            {
                                kind: np.stack([ref["frames"] for ref in refs[kind]])
                                for kind in TARGET_KINDS
                            },
                            {**settings, "end_frame": end},
                        )
                    },
                )
            record["horizons"][name] = entry
        groups.append(record)
    return groups


def _relocate(path, relocate):
    for old, new in relocate:
        if path == old or path.startswith(old.rstrip("/") + "/"):
            return str(Path(new) / Path(path).relative_to(old))
    return path


def verify_inputs(plan_path, predictions_path, actions_config=None, relocate=()):
    """Check plan, configuration and video identities before scoring anything.

    Plans prepared before this module existed keep their original evaluator hashes;
    plans prepared afterwards must still match the current evaluator exactly.
    Relocation maps recorded path prefixes, such as TC2 folders, to local copies
    after the plan hash is verified; every relocated file must keep its hash.
    """
    plan_path, predictions_path = Path(plan_path), Path(predictions_path)
    plan = json.loads(plan_path.read_text())
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan["schema_version"] != 1 or _digest(payload) != plan["plan_sha256"]:
        raise ValueError("Evaluation plan changed; prepare a new plan explicitly")
    if relocate:
        plan = {
            **plan,
            "dataset": _relocate(plan["dataset"], relocate),
            "config": _relocate(plan["config"], relocate),
            "files": {key: _relocate(path, relocate) for key, path in plan["files"].items()},
        }
    sources = plan["evaluator_source_sha256"]
    if Path(__file__).name in sources and sources != _evaluator_sources():
        raise ValueError("Evaluator source changed after preparing the plan")
    for key, path in plan["files"].items():
        if _sha(path) != plan["input_sha256"][key]:
            raise ValueError(f"Frozen evaluation input changed: {key}")
    frozen = plan["files"].get("actions_config")
    actions_config = Path(actions_config or frozen or DEFAULT_ACTIONS)
    if frozen and _sha(actions_config) != plan["input_sha256"]["actions_config"]:
        raise ValueError("Use the plan's frozen action configuration")
    seeds = _seeds(plan["sampling_seeds"])
    predictions = json.loads(predictions_path.read_text())
    if (
        predictions["source"] not in {"generated_video", "simulator_control"}
        or _seeds(predictions["sampling_seeds"]) != seeds
        or predictions.get("evaluation_plan_sha256", plan["plan_sha256"]) != plan["plan_sha256"]
        or not predictions["runs"]
    ):
        raise ValueError("Predictions must match the plan's identity and sampling seeds")
    expected = {case["sample_id"] for case in plan["cases"]}
    runs = []
    for run in predictions["runs"]:
        if set(run["videos"]) != expected:
            raise ValueError("Every run must cover every frozen case exactly once")
        paths = {
            key: (predictions_path.parent / value).resolve() for key, value in run["videos"].items()
        }
        hashes = {key: _sha(path) for key, path in paths.items()}
        if "video_sha256" in run and run["video_sha256"] != hashes:
            raise ValueError("Video changed after sampling")
        runs.append((run, paths, hashes))
    return plan, actions_config, runs


def _load(path, count):
    with np.load(path, allow_pickle=False) as archive:
        videos = archive["videos"]
        valid = archive["frame_valid"] if "frame_valid" in archive else None
    if len(videos) != count:
        raise ValueError("Prediction K differs from the frozen sampling seeds")
    return videos, valid


def _score_run(plan, families, paths, settings):
    count = len(plan["sampling_seeds"])
    scored = {}
    for family, groups in families.items():
        for horizon in HORIZONS:
            rows = []
            for group in groups:
                entry = group["horizons"].get(horizon)
                row = {
                    "scene_group_id": group["scene_group_id"],
                    "eligible": bool(entry and entry["eligible"]),
                }
                if row["eligible"]:
                    loaded = {
                        kind: _load(paths[name], count)
                        for kind, name in group["sample_ids"].items()
                    }
                    videos = {kind: value[0] for kind, value in loaded.items()}
                    valid = {kind: value[1] for kind, value in loaded.items()}
                    if family == "actions":
                        refs = {
                            kind: load_rollout(plan["dataset"], name)
                            for kind, name in group["scoring_reference_ids"].items()
                        }
                        row["scores"] = score_actions(
                            refs,
                            videos,
                            settings=settings,
                            end_frame=entry["end_frame"],
                            frame_valid=valid,
                        )
                    else:
                        refs = {
                            kind: load_rollout(plan["dataset"], group["sample_ids"][kind])
                            for kind in TARGET_KINDS
                        }
                        row["scores"] = {
                            "selection": _score_rgb(
                                refs, videos, {**settings, "end_frame": entry["end_frame"]}, valid
                            )
                        }
                rows.append(row)
            scored.setdefault(family, {})[horizon] = rows
    return scored


def _reference_records(groups, horizon):
    return [
        {
            "method": method,
            "groups": [
                {
                    "scene_group_id": group["scene_group_id"],
                    "eligible": group["horizons"].get(horizon, {}).get("eligible", False),
                    "scores": group["horizons"].get(horizon, {}).get(key),
                }
                for group in groups
            ],
        }
        for method, key in (
            ("reference-labels", "reference_label_scores"),
            ("reference-rgb", "reference_rgb_scores"),
        )
    ]


def evaluate(plan_path, predictions_path, *, actions_config=None, bootstrap=None, relocate=()):
    relocate = [tuple(pair) for pair in relocate]
    plan, actions_config, runs = verify_inputs(
        plan_path, predictions_path, actions_config, relocate
    )
    settings = action_settings(actions_config)
    # Same scene-group bootstrap as the original action report.
    protocol = plan["comparison_protocol"].get("bootstrap", {})
    uncertainty = dict(
        bootstrap=protocol.get("replicates", 2000) if bootstrap is None else bootstrap,
        confidence=protocol.get("confidence", 0.95),
        seed=protocol.get("seed", 0),
    )
    families = {
        "actions": prepare_action_groups(plan, settings),
        "selection": prepare_selection_groups(plan, settings),
    }
    families = {family: groups for family, groups in families.items() if groups}
    if not families:
        raise ValueError("The plan has no forward/reverse or selected-object groups")
    rows = []
    for run, paths, hashes in runs:
        rows.append(
            {
                "method": run["method"],
                "run_id": run["run_id"],
                "training_seed": run.get("training_seed"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "groups": _score_run(plan, families, paths, settings),
            }
        )
        if any(_sha(path) != hashes[key] for key, path in paths.items()):
            raise ValueError("A video changed during scoring")
    summary, calibration = {}, {}
    for family, groups in families.items():
        for horizon in HORIZONS:
            records = [
                {
                    "method": row["method"],
                    "run_id": row["run_id"],
                    "groups": row["groups"][family][horizon],
                }
                for row in rows
            ]
            summary.setdefault(family, {})[horizon] = summarize_actions(records, **uncertainty)
            calibration.setdefault(family, {})[horizon] = summarize_actions(
                _reference_records(groups, horizon), **uncertainty
            )
    return {
        "action_horizon_version": ACTION_HORIZON_VERSION,
        "scope": (
            "supplementary scores at the original window and the final frame; the original "
            "fixed-window action and selected-object results stay unchanged and reported"
        ),
        "motion_rule": (
            "unchanged scorers: moving needs displacement >= motion_threshold, correct signs "
            "need projections >= motion_threshold and drift is maximum excursion >= threshold"
        ),
        "eligibility_rule": (
            "a horizon is scored only if no hidden-physics reference has a collision or wall "
            "onset before its end frame"
        ),
        "zero_force": "scored when the split contains it; otherwise omitted, reversal still scored",
        "split": plan["split"],
        "plan_sha256": plan["plan_sha256"],
        "relocated_path_prefixes": [list(pair) for pair in relocate],
        "predictions_sha256": _sha(predictions_path),
        "actions_config_sha256": _sha(actions_config),
        "motion_threshold": settings["motion_threshold"],
        "extractor_version": TRACKER_VERSION,
        "scorer_source_sha256": {
            name: _sha(Path(__file__).parent / name) for name in SCORER_SOURCES
        },
        "groups": families,
        "runs": rows,
        "summary": summary,
        "reference_calibration": calibration,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--actions-config", type=Path, help="Defaults to the plan's frozen action config"
    )
    parser.add_argument(
        "--relocate",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Read files recorded under OLD from NEW; hashes are still checked",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve previous reports; use a fresh output")
    relocate = [item.partition("=")[::2] for item in args.relocate]
    if any(not old or not new for old, new in relocate):
        parser.error("Use --relocate OLD=NEW")
    report = evaluate(
        args.plan, args.predictions, actions_config=args.actions_config, relocate=relocate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    print(
        json.dumps(
            {
                family: {
                    horizon: sum(
                        group["horizons"].get(horizon, {}).get("eligible", False)
                        for group in groups
                    )
                    for horizon in HORIZONS
                }
                for family, groups in report["groups"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
