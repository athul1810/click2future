"""Full-horizon action scores beside the fixed window, and reversal without zero force."""

import json
from pathlib import Path

import numpy as np
import pytest

from click2future.data import (
    generate_controlled_dataset,
    generate_generalization_dataset,
    load_rollout,
)
from click2future.evaluation import action_horizon
from click2future.evaluation.suite import prepare_plan, run_suite
from click2future.evaluation.targets import prepare_target_plan

pytest.importorskip("skimage")
CONFIGS = Path(__file__).resolve().parents[1] / "configs/evaluation"
SIGN = "reversal.correct_force_sign_rate_over_complete"


@pytest.fixture(scope="module")
def controlled(tmp_path_factory):
    root = tmp_path_factory.mktemp("horizon-controlled")
    generate_controlled_dataset(root, group_count=4, seed=197000)
    return root


@pytest.fixture(scope="module")
def generalization(tmp_path_factory):
    root = tmp_path_factory.mktemp("horizon-generalization")
    generate_generalization_dataset(root, seed=879321)
    return root


def write_plan(data, folder, split, *, actions=True):
    plan = prepare_plan(
        data,
        CONFIGS / "pixel.yaml",
        split=split,
        metrics=("ssim",),
        protocol=CONFIGS / "comparison.yaml",
        actions_config=CONFIGS / "actions.yaml" if actions else None,
    )
    path = folder / f"{split}-plan.json"
    path.write_text(json.dumps(plan))
    return plan, path


def write_predictions(plan, folder, modes):
    """Oracle copies the reference, static holds frame 0, late starts moving after frame 7."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "simulator_control", "sampling_seeds": plan["sampling_seeds"], "runs": []}
    for mode in modes:
        run = {"run_id": mode, "method": mode, "videos": {}}
        for index, case in enumerate(plan["cases"]):
            frames = load_rollout(plan["dataset"], case["sample_id"])["frames"]
            if mode == "static":
                frames = np.repeat(frames[:1], len(frames), axis=0)
            elif mode == "late":
                frames = np.concatenate([np.repeat(frames[:1], 8, axis=0), frames[1:-7]])
            path = folder / f"{mode}-{index}.npz"
            np.savez_compressed(path, videos=np.repeat(frames[None], 4, axis=0))
            run["videos"][case["sample_id"]] = path.name
        manifest["runs"].append(run)
    path = folder / "predictions.json"
    path.write_text(json.dumps(manifest))
    return path


def estimates(summary, key):
    return {row["method"]: row["metrics"][key]["estimate"] for row in summary["methods"]}


def test_window_matches_the_original_suite_and_full_horizon_finds_late_motion(controlled, tmp_path):
    plan, plan_path = write_plan(controlled, tmp_path, "validation")
    predictions = write_predictions(plan, tmp_path / "videos", ("oracle", "static", "late"))
    run_suite(plan_path, predictions, tmp_path / "suite", bootstrap=20)
    original = json.loads((tmp_path / "suite/actions.json").read_text())
    report = action_horizon.evaluate(plan_path, predictions, bootstrap=20)

    window, full = report["summary"]["actions"]["window"], report["summary"]["actions"]["full"]
    assert window == original["summary"]
    assert (
        report["reference_calibration"]["actions"]["window"]
        == original["reference_calibration"]["summary"]
    )
    (group,) = report["groups"]["actions"]
    assert group["actions"] == ["forward", "reverse", "zero"]
    assert group["horizons"]["full"]["end_frame"] == 31
    assert group["horizons"]["full"]["eligible"]
    assert estimates(window, SIGN) == {"late": 0.0, "oracle": 1.0, "static": 0.0}
    assert estimates(full, SIGN) == {"late": 1.0, "oracle": 1.0, "static": 0.0}
    labels = report["reference_calibration"]["actions"]
    assert estimates(labels["window"], SIGN)["reference-labels"] < 1
    assert estimates(labels["full"], SIGN)["reference-labels"] == 1
    assert estimates(full, "zero.drift_rate_over_complete")["oracle"] == 0


def test_force_ood_scores_reversal_without_zero_force(generalization, tmp_path):
    plan, plan_path = write_plan(generalization, tmp_path, "force_ood", actions=False)
    assert "action_groups" not in plan
    predictions = write_predictions(plan, tmp_path / "videos", ("oracle", "late"))
    report = action_horizon.evaluate(plan_path, predictions, bootstrap=20)

    (group,) = report["groups"]["actions"]
    assert group["actions"] == ["forward", "reverse"]
    assert set(group["scoring_reference_ids"]) == {"forward", "reverse"}
    window = report["summary"]["actions"]["window"]
    assert estimates(window, SIGN) == {"late": 0.0, "oracle": 1.0}
    assert not any(key.startswith("zero.") for key in window["methods"][0]["metrics"])
    # High OOD forces reach a wall within the clip, so that horizon is not scored.
    assert group["horizons"]["full"]["exclusion_reason"] == "reference_contact_within_horizon"
    assert report["summary"]["actions"]["full"]["methods"][0]["metrics"] == {}


def test_selected_object_pairs_are_scored_at_each_eligible_horizon(controlled, tmp_path):
    _, parent = write_plan(controlled, tmp_path, "validation")
    prepare_target_plan(
        parent, tmp_path / "targets", config=CONFIGS / "actions.yaml", force_magnitude=0.4
    )
    plan_path = tmp_path / "targets/plan.json"
    plan = json.loads(plan_path.read_text())
    predictions = write_predictions(plan, tmp_path / "videos", ("oracle", "late"))
    report = action_horizon.evaluate(plan_path, predictions, bootstrap=20)

    assert "actions" not in report["groups"]
    (group,) = report["groups"]["selection"]
    window = report["summary"]["selection"]["window"]
    target = "selection.correct_target_rate_over_complete"
    assert estimates(window, target) == {"late": 0.0, "oracle": 1.0}
    assert group["horizons"]["full"]["eligible"] is False


def test_any_reference_contact_excludes_the_horizon():
    still = {
        "collision_onset": np.zeros((32, 2, 2), bool),
        "wall_contact_onset": np.zeros((32, 2), bool),
    }
    late = {**still, "wall_contact_onset": still["wall_contact_onset"].copy()}
    late["wall_contact_onset"][20, 0] = True
    assert action_horizon.contact_free([still, late], 4)
    assert action_horizon.contact_free([still, late], 20)
    assert not action_horizon.contact_free([still, late], 31)


def test_changed_videos_and_existing_outputs_are_refused(controlled, tmp_path):
    plan, plan_path = write_plan(controlled, tmp_path, "validation")
    predictions = write_predictions(plan, tmp_path / "videos", ("oracle",))
    manifest = json.loads(predictions.read_text())
    run = manifest["runs"][0]
    run["video_sha256"] = {key: "0" * 64 for key in run["videos"]}
    changed = tmp_path / "videos/changed.json"
    changed.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Video changed"):
        action_horizon.evaluate(plan_path, changed)
    output = tmp_path / "existing.json"
    output.write_text("{}")
    with pytest.raises(SystemExit):
        action_horizon.main(
            ["--plan", str(plan_path), "--predictions", str(predictions), "--output", str(output)]
        )
