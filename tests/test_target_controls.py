"""Selected-object intervention isolation, missing outputs and checkpoint wiring."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from click2future.data import generate_controlled_dataset, load_rollout
from click2future.evaluation import targets
from click2future.evaluation.sample_suite import generate_samples
from click2future.evaluation.suite import _sha, prepare_plan, read_plan, run_suite
from click2future.models.cvae import ConditionalVideoVAE
from click2future.sim.replay import replay_saved_rollout
from click2future.training.train_cvae import dataset_index, save_checkpoint
from click2future.training.train_diffusion import TrainConfig, Trainer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def metric_inputs():
    first = np.broadcast_to(np.array([[0.4, 0.4], [0.4, 0.7]]), (4, 5, 2, 2)).copy()
    second = first.copy()
    first[:, :, 0, 0] += np.linspace(0, 0.1, 5)
    second[:, :, 1, 0] += np.linspace(0, 0.1, 5)
    return first, second, np.ones((4, 5, 2), bool), np.ones((4, 5, 2), bool)


def score(*args):
    return targets.target_metrics(
        *args,
        [True, True],
        selected_object_ids=[0, 1],
        force_xy=[0.15, 0],
        end_frame=4,
        motion_threshold=1 / 64,
    )


@pytest.mark.parametrize("mode", ["oracle", "static", "ignored", "both", "wrong_direction"])
def test_metric_distinguishes_click_following_from_other_motion(mode):
    first, second, known_first, known_second = metric_inputs()
    if mode == "static":
        first[:] = first[:, :1]
        second[:] = second[:, :1]
    elif mode == "ignored":
        second = first.copy()
    elif mode == "both":
        first[:, :, 1, 0] += np.linspace(0, 0.1, 5)
        second = first.copy()
    elif mode == "wrong_direction":
        first[:, :, 0, 0] -= 2 * np.linspace(0, 0.1, 5)
        second[:, :, 1, 0] -= 2 * np.linspace(0, 0.1, 5)
    result = score(first, second, known_first, known_second)
    assert result["correct_target_rate_over_complete"] == (1 if mode == "oracle" else 0)
    assert result["correct_contrast_rate_over_complete"] == (1 if mode == "oracle" else 0)
    assert result["stationary_pairs"] == (4 if mode in {"static", "ignored"} else 0)
    assert result["missing_pairs"] == 0


def test_missing_any_object_cannot_pass_as_stationary_or_correct():
    first, second, a, b = metric_inputs()
    b[:, 2, 0] = False  # Nonselected identity is needed to establish absence of motion.
    second[:, 2, 0] = np.nan
    result = score(first, second, a, b)
    assert result["missing_pairs"] == 4 and result["complete_pairs"] == 0
    assert result["stationary_pairs"] == 0
    assert result["correct_target_rate_over_complete"] is None


def test_nonselected_excursion_counts_even_if_object_returns_to_origin():
    first, second, a, b = metric_inputs()
    first[:, 2, 1, 0] += 0.1
    result = score(first, second, a, b)
    assert result["correct_target_rate_over_complete"] == 0
    assert result["drift_rate_over_complete"] == 1


def test_padding_cannot_change_target_score():
    first, second, a, b = metric_inputs()
    padded = [
        np.pad(value, ((0, 0), (0, 0), (0, 1), (0, 0)), constant_values=np.nan)
        for value in (first, second)
    ]
    known = [np.pad(value, ((0, 0), (0, 0), (0, 1))) for value in (a, b)]
    result = targets.target_metrics(
        *padded,
        *known,
        np.array([True, True, False]),
        selected_object_ids=[0, 1],
        force_xy=[0.15, 0],
        end_frame=4,
        motion_threshold=1 / 64,
    )
    assert result == score(first, second, a, b)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("target-corpus")
    generate_controlled_dataset(root, group_count=4, seed=197000)
    return root


@pytest.fixture
def prepared(corpus, tmp_path):
    base = prepare_plan(
        corpus,
        CONFIGS / "evaluation/pixel.yaml",
        split="validation",
        metrics=("ssim",),
        protocol=CONFIGS / "evaluation/comparison.yaml",
    )
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(base))
    folder = tmp_path / "targets"
    result = targets.prepare_target_plan(parent, folder, config=CONFIGS / "evaluation/actions.yaml")
    return base, result, folder / "plan.json"


def predictions(plan, folder, modes=("oracle", "static", "ignored"), missing=None):
    manifest = {
        "source": "simulator_control",
        "sampling_seeds": plan["sampling_seeds"],
        "runs": [],
        "evaluation_plan_sha256": plan["plan_sha256"],
    }
    for mode in modes:
        run = {"run_id": mode, "method": mode, "videos": {}}
        for group in plan["target_groups"]:
            for kind in targets.KINDS:
                name = group["sample_ids"][kind]
                names = group["reference_ids"]["first" if mode == "ignored" else kind]
                videos = np.stack([load_rollout(plan["dataset"], ref)["frames"] for ref in names])
                assert len(videos) == 4
                if mode == "static":
                    videos[:] = videos[:, :1]
                valid = np.ones(videos.shape[:2], bool)
                if missing is not None and kind == "second":
                    valid[0, missing] = False
                path = folder / f"{mode}-{name}.npz"
                np.savez_compressed(path, videos=videos, frame_valid=valid)
                run["videos"][name] = path.name
        manifest["runs"].append(run)
    path = folder / "predictions.json"
    path.write_text(json.dumps(manifest))
    return path


def test_derived_cases_only_change_clicked_object_and_preserve_original_corpus(prepared):
    base, plan, path = prepared
    assert read_plan(path) == plan
    assert len(plan["cases"]) == 2 and len(plan["target_groups"]) == 1
    assert plan["split"] == "validation-selected-object"
    assert "action_groups" not in plan
    assert plan["input_sha256"]["dataset_manifest"] == base["input_sha256"]["dataset_manifest"]
    assert plan["input_sha256"]["splits"] == base["input_sha256"]["splits"]
    assert Path(plan["dataset"]) != Path(base["dataset"])
    assert plan["intervention"]["training_corpus_unchanged"]
    for group in plan["target_groups"]:
        assert group["eligible"]
        for first, second in zip(
            *[group["reference_ids"][kind] for kind in targets.KINDS], strict=True
        ):
            a, b = [load_rollout(plan["dataset"], name) for name in (first, second)]
            assert a["selected_object_id"] != b["selected_object_id"]
            np.testing.assert_array_equal(a["frames"][0], b["frames"][0])
            np.testing.assert_array_equal(a["instance_masks"][0], b["instance_masks"][0])
            np.testing.assert_array_equal(a["force_xy"], b["force_xy"])
            assert a["force_duration"] == b["force_duration"]
            assert a["scene_spec"]["objects"] == b["scene_spec"]["objects"]
            assert replay_saved_rollout(plan["dataset"], first, atol=0, rtol=0)["ok"]
            assert replay_saved_rollout(plan["dataset"], second, atol=0, rtol=0)["ok"]
    for key, value in base["input_sha256"].items():
        assert _sha(base["files"][key]) == value


def test_native_controls_use_same_videos_and_reference_calibration(prepared, tmp_path):
    _, plan, path = prepared
    output = tmp_path / "metrics"
    summary = run_suite(path, predictions(plan, tmp_path), output, bootstrap=20)
    assert summary["target_report"] == "targets.json"
    report = json.loads((output / "targets.json").read_text())
    rows = {row["method"]: row for row in report["summary"]["methods"]}
    reference = {
        row["method"]: row for row in report["reference_calibration"]["summary"]["methods"]
    }
    key = "selection.correct_target_rate_over_complete"
    assert rows["oracle"]["metrics"][key] == reference["reference-rgb"]["metrics"][key]
    assert rows["oracle"]["metrics"][key]["estimate"] > 0
    assert rows["static"]["metrics"][key]["estimate"] == 0
    assert rows["ignored"]["metrics"][key]["estimate"] == 0
    assert rows["static"]["counts"]["selection.stationary_pairs"] == 4
    assert rows["oracle"]["metrics"][key]["ci"] is None
    assert report["eligible_scene_groups"] == report["expected_scene_groups"] == 1
    assert (output / "targets.csv").read_text().startswith("method,metric,")
    assert len(summary["provenance"]) == 3 * 3 * len(plan["cases"])


@pytest.mark.parametrize("missing,expected", [(2, 1), (7, 0)])
def test_missing_rgb_only_within_fixed_window_counts_missing(prepared, tmp_path, missing, expected):
    _, plan, path = prepared
    output = tmp_path / "metrics"
    run_suite(path, predictions(plan, tmp_path, ("oracle",), missing), output, bootstrap=10)
    report = json.loads((output / "targets.json").read_text())
    assert report["runs"][0]["groups"][0]["scores"]["selection"]["missing_pairs"] == expected


def test_contact_in_one_hidden_variant_excludes_whole_group(corpus, tmp_path, monkeypatch):
    parent = tmp_path / "parent.json"
    base = prepare_plan(
        corpus, CONFIGS / "evaluation/pixel.yaml", split="validation", metrics=("ssim",)
    )
    parent.write_text(json.dumps(base))
    original = targets.run_rollout
    calls = 0

    def contact(*args, **kwargs):
        nonlocal calls
        calls += 1
        ref = original(*args, **kwargs)
        if calls == 8:
            ref["collision_onset"][2, 0, 1] = ref["collision_onset"][2, 1, 0] = True
        return ref

    monkeypatch.setattr(targets, "run_rollout", contact)
    plan = targets.prepare_target_plan(
        parent, tmp_path / "controls", config=CONFIGS / "evaluation/actions.yaml"
    )
    group = plan["target_groups"][0]
    assert (
        not group["eligible"]
        and group["exclusion_reason"] == "reference_contact_within_fixed_window"
    )
    assert len(plan["cases"]) == 2 and "reference_rgb_scores" not in group
    run_suite(
        tmp_path / "controls/plan.json",
        predictions(plan, tmp_path),
        tmp_path / "metrics",
        bootstrap=10,
    )
    result = json.loads((tmp_path / "metrics/targets.json").read_text())
    assert result["eligible_scene_groups"] == 0
    assert result["summary"]["methods"][0]["metrics"] == {}


@pytest.mark.parametrize("input_kind", ["derived", "parent", "original"])
def test_frozen_intervention_binds_generated_parent_and_corpus_inputs(
    prepared, tmp_path, input_kind
):
    _, plan, path = prepared
    predictions_path = predictions(plan, tmp_path, ("static",))
    key = {
        "derived": plan["cases"][0]["sample_id"] + ".json",
        "parent": "parent_plan",
        "original": "dataset_manifest",
    }[input_kind]
    changed = Path(plan["files"][key])
    original = changed.read_bytes()
    try:
        changed.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="Frozen evaluation input changed"):
            run_suite(path, predictions_path, tmp_path / "metrics")
        assert not (tmp_path / "metrics").exists()
    finally:
        changed.write_bytes(original)


def test_no_forward_condition_and_recursive_derivation_are_rejected(prepared, tmp_path):
    _, _, path = prepared
    with pytest.raises(ValueError, match="ordinary held-out"):
        targets.prepare_target_plan(
            path, tmp_path / "nested", config=CONFIGS / "evaluation/actions.yaml"
        )
    changed = read_plan(path)
    changed.pop("intervention")
    from click2future.evaluation.suite import _digest

    changed.pop("plan_sha256")
    changed["plan_sha256"] = _digest(changed)
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="needs a forward condition"):
        targets.prepare_target_plan(
            changed_path, tmp_path / "invalid", config=CONFIGS / "evaluation/actions.yaml"
        )


@pytest.mark.parametrize("magnitude", [True, 0, -0.1, float("nan"), float("inf")])
def test_invalid_shared_force_override_is_rejected_before_writes(prepared, tmp_path, magnitude):
    base, _, path = prepared
    with pytest.raises(ValueError, match="force_magnitude"):
        targets.prepare_target_plan(
            path.parent.parent / "parent.json",
            tmp_path / "invalid-force",
            force_magnitude=magnitude,
        )
    assert not (tmp_path / "invalid-force").exists()


def test_shared_force_override_is_frozen_and_does_not_change_source(prepared, tmp_path):
    base, _, path = prepared
    plan = targets.prepare_target_plan(
        path.parent.parent / "parent.json",
        tmp_path / "stronger",
        config=CONFIGS / "evaluation/actions.yaml",
        force_magnitude=0.4,
    )
    assert plan["intervention"]["force_magnitude_override"] == 0.4
    for case in plan["cases"]:
        ref = load_rollout(plan["dataset"], case["sample_id"])
        np.testing.assert_array_equal(ref["force_xy"], np.array([0.4, 0], np.float32))
    for key, value in base["input_sha256"].items():
        assert _sha(base["files"][key]) == value


def test_original_checkpoint_provenance_accepts_held_out_interventions(prepared, tmp_path):
    base, plan, path = prepared
    runs = []
    for method, conditioning in (("E1", "first_frame"), ("E2", "numeric"), ("E3", "spatial")):
        trainer = Trainer(
            Path(base["dataset"]),
            TrainConfig(
                base_channels=8,
                diffusion_steps=10,
                conditioning=conditioning,
                seed=3,
                limit_train=1,
                force_limit=1.1,
            ),
        )
        checkpoint = tmp_path / f"{method}.pt"
        trainer.save(checkpoint)
        runs.append(
            {"method": method, "run_id": "untrained-" + method, "checkpoint": str(checkpoint)}
        )
    config = yaml.safe_load((CONFIGS / "training/cvae.yaml").read_text())
    config["model"].update(base_channels=8, latent_dim=4)
    with torch.random.fork_rng(devices=[]):
        model = ConditionalVideoVAE(**config["model"])
    _, _, provenance = dataset_index(Path(base["dataset"]))
    checkpoint = tmp_path / "E0.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        state={"step": 0},
        config=config,
        provenance=provenance,
        train_generator=torch.Generator().manual_seed(4),
        shuffle_generator=torch.Generator().manual_seed(5),
    )
    runs.insert(0, {"method": "E0", "run_id": "untrained-E0", "checkpoint": str(checkpoint)})
    spec = tmp_path / "runs.json"
    spec.write_text(json.dumps({"sampling_steps": 4, "runs": runs}))
    rng = torch.get_rng_state().clone()
    record = generate_samples(path, spec, tmp_path / "samples", allow_untrained=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(record["runs"]) == 4 and all(run["untrained_fixture"] for run in record["runs"])
    # E1 deliberately cannot see the click. Identical paired seeds must give identical videos.
    first, second = plan["target_groups"][0]["sample_ids"].values()
    e1 = next(run for run in record["runs"] if run["method"] == "E1")
    with (
        np.load(tmp_path / "samples" / e1["videos"][first]) as a,
        np.load(tmp_path / "samples" / e1["videos"][second]) as b,
    ):
        np.testing.assert_array_equal(a["videos"], b["videos"])
    summary = run_suite(
        path, tmp_path / "samples/predictions.json", tmp_path / "metrics", bootstrap=10
    )
    assert len(summary["untrained_fixture_runs"]) == 4
    assert summary["target_report"] == "targets.json"
