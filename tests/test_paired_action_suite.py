"""Action adherence from real paired videos, reference gating and missing RGB."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from click2future.data import generate_controlled_dataset, load_rollout
from click2future.evaluation import paired_actions
from click2future.evaluation.suite import prepare_plan, run_suite

pytest.importorskip("skimage")
CONFIGS = Path(__file__).resolve().parents[1] / "configs/evaluation"


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-action-data")
    generate_controlled_dataset(root, group_count=4, seed=197000)
    return root


@pytest.fixture
def plan(data, tmp_path):
    actions = tmp_path / "actions.yaml"
    actions.write_bytes((CONFIGS / "actions.yaml").read_bytes())
    return prepare_plan(
        data,
        CONFIGS / "pixel.yaml",
        split="validation",
        metrics=("ssim",),
        protocol=CONFIGS / "comparison.yaml",
        actions_config=actions,
    )


def predictions(plan, folder, modes=("oracle", "static", "ignored")):
    manifest = {"source": "simulator_control", "sampling_seeds": plan["sampling_seeds"], "runs": []}
    for mode in modes:
        run = {"run_id": mode, "method": mode, "videos": {}}
        for index, case in enumerate(plan["cases"]):
            ref = load_rollout(plan["dataset"], case["sample_id"])
            frames = ref["frames"]
            if mode == "static":
                frames = np.repeat(frames[:1], len(frames), axis=0)
            elif mode == "ignored" and ref["scene_spec"]["action_variant"] == "reverse":
                frames = load_rollout(
                    plan["dataset"], plan["action_groups"][0]["sample_ids"]["forward"]
                )["frames"]
            path = folder / f"{mode}-{index}.npz"
            np.savez_compressed(path, videos=np.repeat(frames[None], 4, axis=0))
            run["videos"][case["sample_id"]] = path.name
        manifest["runs"].append(run)
    path = folder / "predictions.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def run(plan, tmp_path, modes=("oracle", "static", "ignored")):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    prediction_path, _ = predictions(plan, tmp_path, modes)
    state = np.random.get_state()
    output = tmp_path / "results"
    summary = run_suite(plan_path, prediction_path, output, bootstrap=20)
    assert np.array_equal(state[1], np.random.get_state()[1])
    assert summary["action_report"] == "actions.json"
    return json.loads((output / "actions.json").read_text())


def test_native_suite_distinguishes_reversal_stationary_and_ignored_action(plan, tmp_path):
    result = run(plan, tmp_path)
    assert result["expected_scene_groups"] == result["eligible_scene_groups"] == 1
    rows = {row["method"]: row for row in result["summary"]["methods"]}
    sign = "reversal.correct_force_sign_rate_over_complete"
    assert rows["oracle"]["metrics"][sign]["estimate"] == 1
    assert rows["static"]["metrics"][sign]["estimate"] == 0
    assert rows["ignored"]["metrics"][sign]["estimate"] == 0
    assert rows["ignored"]["metrics"]["reversal.mean_reversal_error_degrees"]["estimate"] == 180
    assert rows["static"]["counts"]["reversal.stationary_pairs"] == 4
    assert rows["oracle"]["metrics"]["zero.mean_max_drift"]["estimate"] == 0
    assert rows["oracle"]["metrics"][sign]["ci"] is None
    assert "not measured" in result["selected_object_control"]
    assert (tmp_path / "results/actions.csv").read_text().startswith("method,metric,")
    assert result["reference_calibration"]["summary"]["methods"]


def test_low_force_reference_stationarity_is_calibrated_without_changing_threshold(tmp_path):
    data = tmp_path / "data"
    generate_controlled_dataset(data, group_count=20, seed=219000)
    prepared = prepare_plan(
        data,
        CONFIGS / "pixel.yaml",
        split="validation",
        metrics=("ssim",),
        actions_config=CONFIGS / "actions.yaml",
    )
    assert len(prepared["action_groups"]) == 2
    assert all(group["eligible"] for group in prepared["action_groups"])
    assert (
        sum(
            group["reference_rgb_scores"]["reversal"]["stationary_pairs"]
            for group in prepared["action_groups"]
        )
        > 0
    )
    assert all(
        group["reference_label_scores"]["reversal"]["motion_threshold"] == 1 / 64
        for group in prepared["action_groups"]
    )


@pytest.mark.parametrize("missing_frame,expected", [(2, 1), (7, 0)])
def test_missing_rgb_in_fixed_window_remains_missing_not_stationary(
    plan, tmp_path, missing_frame, expected
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    path, manifest = predictions(plan, tmp_path, ("oracle",))
    forward = plan["action_groups"][0]["sample_ids"]["forward"]
    video = tmp_path / manifest["runs"][0]["videos"][forward]
    with np.load(video) as archive:
        clips = archive["videos"]
    valid = np.ones(clips.shape[:2], dtype=bool)
    valid[0, missing_frame] = False
    np.savez_compressed(video, videos=clips, frame_valid=valid)
    run_suite(plan_path, path, tmp_path / "results", bootstrap=20)
    result = json.loads((tmp_path / "results/actions.json").read_text())
    scores = result["runs"][0]["groups"][0]["scores"]
    assert scores["forward"]["missing_samples"] == expected
    assert scores["reversal"]["missing_pairs"] == expected
    assert scores["forward"]["stationary_samples"] == 0


def test_contact_in_any_hidden_reference_excludes_group_before_predictions(plan, monkeypatch):
    original = paired_actions.load_rollout
    last = plan["action_groups"][0]["reference_ids"]["reverse"][-1]

    def contact(root, name):
        ref = copy.deepcopy(original(root, name))
        if name == last:
            ref["collision_onset"][2, 0, 1] = ref["collision_onset"][2, 1, 0] = True
        return ref

    monkeypatch.setattr(paired_actions, "load_rollout", contact)
    groups = paired_actions.prepare_action_groups(plan, plan["files"]["actions_config"])
    assert len(groups) == 1 and not groups[0]["eligible"]
    assert groups[0]["exclusion_reason"] == "reference_contact_within_fixed_window"


def test_hidden_physics_mismatch_cannot_be_called_a_paired_control(plan, monkeypatch):
    original = paired_actions.load_rollout
    last = plan["action_groups"][0]["reference_ids"]["reverse"][-1]

    def changed(root, name):
        ref = copy.deepcopy(original(root, name))
        if name == last:
            ref["scene_spec"]["objects"][0]["hidden_physics"]["mass"] *= 1.1
        return ref

    monkeypatch.setattr(paired_actions, "load_rollout", changed)
    with pytest.raises(ValueError, match="hidden-physics references must match"):
        paired_actions.prepare_action_groups(plan, plan["files"]["actions_config"])


def test_changed_action_config_invalidates_plan_before_outputs(plan, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    path, _ = predictions(plan, tmp_path, ("oracle",))
    Path(plan["files"]["actions_config"]).write_text("end_frame: 8\n")
    with pytest.raises(ValueError, match="Frozen evaluation input changed: actions_config"):
        run_suite(plan_path, path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_missing_action_rejected_instead_of_silently_dropping_a_group(plan):
    changed = copy.deepcopy(plan)
    zero = plan["action_groups"][0]["sample_ids"]["zero"]
    changed["cases"] = [case for case in plan["cases"] if case["sample_id"] != zero]
    with pytest.raises(ValueError, match="every scene group"):
        paired_actions.prepare_action_groups(changed, plan["files"]["actions_config"])


def test_all_excluded_keep_denominators(plan, tmp_path):
    report = run(plan, tmp_path, ("static",))
    runs = report["runs"]
    runs[0]["groups"][0].update(eligible=False, scores={})
    summary = paired_actions.summarize_actions(runs, bootstrap=20)
    assert summary["methods"][0]["metrics"] == {}


def test_completely_missing_control_is_never_reported_as_correct(plan, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    path, manifest = predictions(plan, tmp_path, ("oracle",))
    for kind in ("forward", "reverse", "zero"):
        name = plan["action_groups"][0]["sample_ids"][kind]
        video = tmp_path / manifest["runs"][0]["videos"][name]
        with np.load(video) as archive:
            clips = archive["videos"]
        valid = np.ones(clips.shape[:2], dtype=bool)
        valid[:, 1:] = False
        np.savez_compressed(video, videos=clips, frame_valid=valid)
    run_suite(plan_path, path, tmp_path / "results", bootstrap=20)
    report = json.loads((tmp_path / "results/actions.json").read_text())
    result = report["summary"]["methods"][0]
    assert result["metrics"]["reversal.correct_force_sign_rate_over_complete"]["estimate"] is None
    assert result["metrics"]["reversal.missing_rate"]["estimate"] == 1
    assert result["metrics"]["zero.mean_max_drift"]["estimate"] is None
    assert result["counts"]["reversal.complete_pairs"] == 0
    assert result["counts"]["reversal.missing_pairs"] == 4


def test_scene_bootstrap_pairs_methods_and_does_not_treat_samples_as_groups(plan, tmp_path):
    report = run(plan, tmp_path, ("oracle", "static"))
    runs = copy.deepcopy(report["runs"])
    for record in runs:
        second = copy.deepcopy(record["groups"][0])
        second["scene_group_id"] = "second-independent-scene"
        record["groups"].append(second)
    first = paired_actions.summarize_actions(runs, bootstrap=50, seed=42)
    assert first == paired_actions.summarize_actions(runs, bootstrap=50, seed=42)
    for method in first["methods"]:
        metric = method["metrics"]["reversal.correct_force_sign_rate_over_complete"]
        expected = 1 if method["method"] == "oracle" else 0
        assert metric["estimate"] == expected and metric["ci"] == [expected, expected]
        assert metric["scored_scene_groups"] == metric["expected_scene_groups"] == 2


def test_predictions_bound_to_another_plan_fail_before_outputs(plan, tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    path, manifest = predictions(plan, tmp_path, ("oracle",))
    manifest["evaluation_plan_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different evaluation plan"):
        run_suite(plan_path, path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_changed_video_during_evaluation_cannot_publish_success(plan, tmp_path, monkeypatch):
    from click2future.evaluation import suite

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    path, manifest = predictions(plan, tmp_path, ("oracle",))
    original = suite.evaluate_action_groups

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / next(iter(manifest["runs"][0]["videos"].values()))).write_bytes(b"changed")
        return result

    monkeypatch.setattr(suite, "evaluate_action_groups", changed)
    with pytest.raises(ValueError, match="changed during evaluation"):
        run_suite(plan_path, path, tmp_path / "results", bootstrap=20)
    assert (tmp_path / "results/failure.json").is_file()
    assert not (tmp_path / "results/summary.json").exists()
