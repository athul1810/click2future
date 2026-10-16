"""Known action responses, stationary outputs, missing tracks, and paired controls."""

import copy
import json

import numpy as np
import pytest

from click2future.evaluation.actions import action_metrics, reversal_metrics
from click2future.evaluation.controls import (
    evaluate_controls,
    main,
    simulator_controls,
    validate_controls,
)
from click2future.evaluation.tracking import TrackerConfig


def tracks(dx=0.04):
    p = np.full((1, 5, 2, 2), 0.5)
    p[0, :, 0, 0] += np.linspace(0, dx, 5)
    return p, np.ones(p.shape[:-1], bool)


def score(p, known, force=(1, 0)):
    return action_metrics(
        p,
        known,
        np.ones(2, bool),
        selected_object=0,
        force_xy=force,
        end_frame=4,
        motion_threshold=0.01,
    )


@pytest.mark.parametrize("dx,angle", [(0.04, 0), (-0.04, 180)])
def test_action_direction(dx, angle):
    assert score(*tracks(dx))["mean_direction_error_degrees"] == pytest.approx(angle)


def test_perpendicular_motion_and_stationary_have_different_statuses():
    p, known = tracks()
    p = p[..., ::-1]
    assert score(p, known)["mean_direction_error_degrees"] == pytest.approx(90)
    stationary = score(*tracks(0))
    assert stationary["stationary_samples"] == 1
    assert stationary["moving_samples"] == 0
    assert stationary["mean_direction_error_degrees"] is None


def test_missing_track_never_becomes_stationary_or_good_direction():
    p, known = tracks()
    known[0, 2, 0] = False
    p[0, 2, 0] = np.nan
    result = score(p, known)
    assert result["missing_samples"] == 1
    assert result["stationary_samples"] == 0
    assert result["mean_direction_error_degrees"] is None


def test_zero_force_counts_nonselected_out_and_back_drift():
    p, known = tracks(0)
    p[0, 2, 1, 0] += 0.1
    result = score(p, known, (0, 0))
    assert result["mean_max_drift"] == pytest.approx(0.1)
    assert result["drifting_samples"] == 1
    known[0, 2, 1] = False
    result = score(p, known, (0, 0))
    assert result["missing_samples"] == 1
    assert result["mean_max_drift"] is None


def paired(dx):
    p, k = tracks(0.04)
    q, reversed_known = tracks(dx)
    return reversal_metrics(
        p,
        q,
        k,
        reversed_known,
        np.ones(2, bool),
        selected_object=0,
        force_xy=[1, 0],
        end_frame=4,
        motion_threshold=0.01,
    )


def test_reversal_catches_ignored_action_and_stationary_samples():
    correct = paired(-0.06)
    assert correct["mean_reversal_error_degrees"] == 0
    assert correct["correct_force_sign_rate_over_complete"] == 1
    ignored = paired(0.04)
    assert ignored["mean_reversal_error_degrees"] == 180
    assert ignored["correct_force_sign_rate_over_complete"] == 0
    stationary = paired(0)
    assert stationary["mean_reversal_error_degrees"] is None
    assert stationary["stationary_pairs"] == 1
    assert stationary["correct_force_sign_rate_over_complete"] == 0


def test_reversal_rejects_different_initial_states():
    p, k = tracks(0.04)
    q = p.copy() + 0.1
    with pytest.raises(ValueError, match="identical initial"):
        reversal_metrics(
            p,
            q,
            k,
            k,
            np.ones(2, bool),
            selected_object=0,
            force_xy=[1, 0],
            end_frame=4,
            motion_threshold=0.01,
        )


@pytest.mark.parametrize(
    "options",
    [
        dict(end_frame=0),
        dict(end_frame=5),
        dict(motion_threshold=0),
        dict(selected_object=2),
        dict(force_xy=[np.nan, 0]),
    ],
)
def test_action_invalid_inputs(options):
    p, known = tracks()
    settings = dict(selected_object=0, force_xy=[1, 0], end_frame=4, motion_threshold=0.01)
    with pytest.raises(ValueError):
        action_metrics(p, known, np.ones(2, bool), **(settings | options))


def test_reference_controls_reject_contact_and_changed_hidden_physics():
    refs = simulator_controls(50000)
    validate_controls(refs, 4)
    bad = copy.deepcopy(refs)
    bad["reverse"]["scene_spec"]["objects"][0]["hidden_physics"]["mass"] *= 2
    with pytest.raises(ValueError, match="share objects"):
        validate_controls(bad, 4)
    bad = copy.deepcopy(refs)
    bad["zero"]["collision_onset"][0, 0, 1] = True
    bad["zero"]["collision_onset"][0, 1, 0] = True
    with pytest.raises(ValueError, match="contact-free"):
        validate_controls(bad, 4)


def test_simulator_control_pipeline_and_static_video_failure():
    refs = simulator_controls(50000)
    videos = {kind: ref["frames"][None] for kind, ref in refs.items()}
    settings = dict(
        tracker=TrackerConfig(),
        end_frame=4,
        motion_threshold=1 / 64,
        source="simulator_control",
        run_id="test",
    )
    result = evaluate_controls(refs, videos, **settings)
    assert result["scores"]["zero"]["mean_max_drift"] == 0
    assert result["scores"]["reversal"]["correct_force_sign_rate_over_complete"] == 1
    static = {kind: np.repeat(clip[:, :1], 5, axis=1) for kind, clip in videos.items()}
    result = evaluate_controls(refs, static, **(settings | {"source": "generated_video"}))
    assert result["scores"]["forward"]["stationary_samples"] == 1
    assert result["scores"]["reversal"]["stationary_pairs"] == 1


def test_simulator_control_cli(tmp_path):
    output = tmp_path / "controls.json"
    assert main(["--simulator-check", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result["records"]) == 16
    assert all(row["source"] == "simulator_control" for row in result["records"])


def test_generated_control_cli_scores_video_and_records_seeds(tmp_path):
    from click2future.data import save_rollout

    refs = simulator_controls(50001)
    for kind, ref in refs.items():
        save_rollout(ref, tmp_path, kind)
    videos = tmp_path / "samples.npz"
    np.savez_compressed(videos, **{kind: ref["frames"][None] for kind, ref in refs.items()})
    output = tmp_path / "report.json"
    args = [
        "--reference-dir",
        str(tmp_path),
        "--forward-id",
        "forward",
        "--reverse-id",
        "reverse",
        "--zero-id",
        "zero",
        "--videos",
        str(videos),
        "--sample-seeds",
        "123",
        "--run-id",
        "cli-fixture",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    report = json.loads(output.read_text())
    assert report["source"] == "generated_video"
    assert report["sample_seeds"] == [123]
    assert len(report["input_sha256"]) == 8
    assert report["scores"]["reversal"]["correct_force_sign_rate_over_complete"] == 1
    # Missing a control array produces a failed record, not partial scores.
    np.savez_compressed(videos, forward=refs["forward"]["frames"][None])
    assert main(args) == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_control_cli_protects_input_files(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("keep this input")
    with pytest.raises(SystemExit):
        main(["--simulator-check", "--config", str(config), "--output", str(config)])
    assert config.read_text() == "keep this input"


def test_padding_and_overflow_are_not_hidden():
    p, known = tracks()
    p[:, :, 1] = np.nan
    result = action_metrics(
        p,
        known,
        np.array([True, False]),
        selected_object=0,
        force_xy=[0, 0],
        end_frame=4,
        motion_threshold=0.01,
    )
    assert result["complete_samples"] == 1
    p, known = tracks()
    p[0, 0, 0] = -1e308
    p[0, 4, 0] = 1e308
    with pytest.raises(ValueError, match="overflowed"):
        score(p, known)
