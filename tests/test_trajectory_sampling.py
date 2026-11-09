"""Observable-only sampler parity and strict checkpoint/input boundaries."""

import json
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from click2future.demo.models import CheckpointModel
from click2future.evaluation import trajectory_pilot
from click2future.models.object_motion import observe
from click2future.sim import run_rollout
from click2future.training.sample_trajectory import (
    load_trajectory,
    sample_trajectory_videos,
)
from click2future.training.sample_trajectory import (
    main as sample_main,
)
from click2future.training.train_cvae import sha256
from click2future.training.train_trajectory import TrajectoryConfig, TrajectoryTrainer


@pytest.fixture
def fixture_checkpoint(tmp_path):
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    ref = run_rollout(scene, frame_count=32, resolution=64, fps=8)
    obs = observe(ref["frames"][0], ref["selected_mask"])
    data = {
        split: {
            "features": np.repeat(obs.features[None], 2, 0),
            "force": np.repeat(ref["force_xy"][None], 2, 0),
            "motion": np.zeros((2, 31, 2, 4), np.float32),
        }
        for split in ("train", "validation")
    }
    trainer = TrajectoryTrainer(data, TrajectoryConfig(steps=2, width=16, layers=1, batch_size=2))
    trainer.train_step()  # Unit fixture only; never a model-quality result.
    path = tmp_path / "fixture.pt"
    trainer.save(
        path,
        {
            "fixture": "unit test only",
            "source_sha256": {
                "object_motion.py": sha256(Path("src/click2future/models/object_motion.py"))
            },
        },
    )
    return path, ref


def test_public_sampler_and_demo_match_and_isolate_seeds(fixture_checkpoint):
    path, ref = fixture_checkpoint
    rng = torch.get_rng_state().clone()
    model, _ = load_trajectory(path)
    assert torch.equal(rng, torch.get_rng_state())
    condition = ref["frames"][0], ref["selected_mask"], ref["force_xy"]
    expected = sample_trajectory_videos(model, *condition, seeds=[7, 11], sampling_steps=3)
    reordered = sample_trajectory_videos(model, *condition, seeds=[11, 7], sampling_steps=3)
    np.testing.assert_array_equal(reordered[::-1], expected)
    np.testing.assert_array_equal(expected[:, 0], np.repeat(condition[0][None], 2, 0))
    adapter = CheckpointModel("E5-unit", path, trajectory_steps=3)
    actual = adapter.predict(
        *condition, seeds=[7, 11], future_frames=31, cancelled=threading.Event()
    )
    np.testing.assert_array_equal(actual, expected)
    assert torch.equal(rng, torch.get_rng_state())
    assert adapter.metadata["kind"] == "trajectory_diffusion"
    assert adapter.metadata["network_evaluations_per_sample"] == 5
    assert "training only" in adapter.metadata["supervision"]
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        adapter.predict(*condition, seeds=[7], future_frames=31, cancelled=stop)
    with pytest.raises(ValueError, match="Force magnitude"):
        adapter.predict(
            condition[0],
            condition[1],
            np.array([0.7, 0]),
            seeds=[7],
            future_frames=31,
            cancelled=threading.Event(),
        )


def test_physics_selection_cli_retains_candidate_provenance(fixture_checkpoint, tmp_path):
    path, ref = fixture_checkpoint
    condition = tmp_path / "condition.npz"
    output = tmp_path / "selected.npz"
    np.savez_compressed(
        condition,
        frames=ref["frames"],
        selected_mask=ref["selected_mask"],
        force_xy=ref["force_xy"],
    )
    sample_main(
        [
            "--checkpoint",
            str(path),
            "--condition",
            str(condition),
            "--output",
            str(output),
            "--sampling-steps",
            "3",
            "--candidate-count",
            "4",
            "--output-count",
            "2",
            "--physics-select",
        ]
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["videos"].shape == (2, 32, 64, 64, 3)
        assert archive["candidate_motion"].shape == (4, 31, 2, 4)
        np.testing.assert_array_equal(archive["candidate_seeds"], np.arange(4))
        selected = archive["selected_indices"].tolist()
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert len(selected) == len(set(selected)) == 2
    assert metadata["seeds"] == [metadata["candidate_seeds"][index] for index in selected]
    assert metadata["physics_selection"]["candidate_count"] == 4


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [-1]])
def test_public_sampler_rejects_invalid_ensembles(fixture_checkpoint, seeds):
    path, ref = fixture_checkpoint
    model, _ = load_trajectory(path)
    with pytest.raises(ValueError, match="distinct"):
        sample_trajectory_videos(
            model, ref["frames"][0], ref["selected_mask"], ref["force_xy"], seeds=seeds
        )


def test_checkpoint_contract_source_and_action_bounds_are_enforced(fixture_checkpoint, tmp_path):
    path, ref = fixture_checkpoint
    state = torch.load(path, weights_only=True)
    state["provenance"]["source_sha256"]["object_motion.py"] = "0" * 64
    broken = tmp_path / "changed-source.pt"
    torch.save(state, broken)
    with pytest.raises(ValueError, match="source differs"):
        load_trajectory(broken)
    model, _ = load_trajectory(path)
    with pytest.raises(ValueError, match="Force"):
        sample_trajectory_videos(model, ref["frames"][0], ref["selected_mask"], [10.0, 0])
    state = torch.load(path, weights_only=True)
    state["motion_contract"]["fps"] = 9
    torch.save(state, broken)
    with pytest.raises(ValueError, match="Unsupported"):
        load_trajectory(broken)


@pytest.mark.parametrize(
    "split", ["train", "id_test", "shape_ood", "texture_ood", "force_ood", "physics_test"]
)
def test_pilot_rejects_nonvalidation_before_loading_model(tmp_path, monkeypatch, split):
    monkeypatch.setattr(trajectory_pilot, "read_plan", lambda _: {"split": split})
    with pytest.raises(ValueError, match="final-test splits"):
        trajectory_pilot.sample_plan("plan.json", tmp_path / "missing.pt", tmp_path / "samples")
    assert not (tmp_path / "samples").exists()
