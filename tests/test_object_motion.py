"""Motion coordinates, observable-only extraction, EDM math and exact continuation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from click2future.models.object_motion import (
    TrajectoryDiffusion,
    encode_motion,
    observe,
    render_motion,
)
from click2future.sim import run_rollout
from click2future.training.train_trajectory import TrajectoryConfig, TrajectoryTrainer, main
from click2future.training.trajectory_data import canonical_ids, replay_angles


def visible_fixture():
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    return run_rollout(scene, frame_count=4, resolution=64, fps=8)


def test_observed_identity_order_is_independent_of_selection():
    ref = visible_fixture()
    first = observe(ref["frames"][0], ref["instance_masks"][0, 0])
    second = observe(ref["frames"][0], ref["instance_masks"][0, 1])
    np.testing.assert_array_equal(first.masks, second.masks)
    np.testing.assert_array_equal(first.features[:, :-1], second.features[:, :-1])
    assert first.selected_index != second.selected_index
    for obs, selected in ((first, 0), (second, 1)):
        np.testing.assert_array_equal(
            obs.masks, ref["instance_masks"][0, canonical_ids(obs, selected)]
        )
        assert obs.features[obs.selected_index, -1] == 1


def test_renderer_identity_positive_axes_and_right_angle_rotation():
    first = np.full((64, 64, 3), 210, dtype=np.uint8)
    first[[0, -1]] = 70
    first[:, [0, -1]] = 70
    first[20:26, 10:24] = [220, 30, 30]
    first[42:50, 40:48] = [30, 120, 230]
    mask = np.zeros((64, 64), bool)
    mask[20:26, 10:24] = True
    obs = observe(first, mask)
    motion = np.zeros((3, 2, 4), dtype=np.float32)
    identity = render_motion(obs, motion)
    np.testing.assert_array_equal(identity, np.broadcast_to(first, identity.shape))
    for channel in (0, 1):
        moved = motion.copy()
        moved[:, obs.selected_index, channel] = 0.25
        image = render_motion(obs, moved)[1]
        y, x = np.nonzero(np.all(image == [220, 30, 30], -1))
        initial_y, initial_x = np.nonzero(mask)
        delta = np.array([x.mean() - initial_x.mean(), y.mean() - initial_y.mean()])
        np.testing.assert_allclose(delta, [8, 0] if channel == 0 else [0, 8], atol=1e-6)
    rotated = motion.copy()
    rotated[:, obs.selected_index, 2:] = [1, -0.5]
    y, x = np.nonzero(np.all(render_motion(obs, rotated)[1] == [220, 30, 30], -1))
    assert np.ptp(y) > np.ptp(x)


def test_angular_label_replay_matches_frozen_trajectory_and_uses_training_only_inputs():
    ref = visible_fixture()
    angles = replay_angles(ref)
    assert angles.shape == (4, 2)
    motion = encode_motion(ref["positions"], angles)
    assert motion.shape == (3, 2, 4) and np.isfinite(motion).all()
    changed = dict(ref, positions=ref["positions"] + 0.1)
    with pytest.raises(ValueError, match="frozen trajectory"):
        replay_angles(changed)
    # The observable entry point takes no reference dictionary or future arguments.
    ref["frames"][1:] = 0
    ref["scene_spec"]["objects"][0]["hidden_physics"]["mass"] *= 2
    observed = observe(ref["frames"][0], ref["selected_mask"])
    assert observed.features.shape == (2, 11)


def inputs(frames=3):
    ref = visible_fixture()
    obs = observe(ref["frames"][0], ref["selected_mask"])
    return (
        torch.zeros(1, frames, 2, 4),
        torch.from_numpy(obs.features[None]),
        torch.tensor([[0.2, -0.1]]),
    )


def test_edm_skip_equation_and_object_permutation_equivariance():
    noisy, features, force = inputs()
    noisy = torch.randn_like(noisy)
    model = TrajectoryDiffusion(future_frames=3, width=16, layers=1).eval()
    sigma = torch.tensor([0.7])
    expected = noisy * 0.25**2 / (0.7**2 + 0.25**2)
    torch.testing.assert_close(model(noisy, sigma, features, force), expected)
    with torch.no_grad():
        model.output.weight.normal_(0, 0.1)
    original = model(noisy, sigma, features, force)
    swapped = model(noisy.flip(2), sigma, features.flip(1), force)
    torch.testing.assert_close(swapped, original.flip(2), atol=2e-6, rtol=1e-6)
    changed = model(noisy, sigma, features, -force)
    assert not torch.equal(original, changed)


def test_heun_oracle_recovers_clean_motion_and_preserves_rng():
    _, features, force = inputs()

    class Oracle(TrajectoryDiffusion):
        def forward(self, noisy, sigma, features, force):
            return torch.full_like(noisy, 0.125)

    oracle = Oracle(future_frames=3, width=16, layers=1).train()
    state = torch.get_rng_state().clone()
    samples = oracle.sample(features, force, seeds=[3, 4], steps=4)
    torch.testing.assert_close(samples, torch.full_like(samples, 0.125), atol=1e-6, rtol=0)
    assert oracle.training and torch.equal(state, torch.get_rng_state())


def test_trajectory_optimizer_resume_is_exact(tmp_path):
    noisy, features, force = inputs()
    data = {
        split: {
            "features": features.repeat(4, 1, 1).numpy(),
            "force": force.repeat(4, 1).numpy(),
            "motion": (noisy + 0.01).repeat(4, 1, 1, 1).numpy(),
        }
        for split in ("train", "validation")
    }
    config = TrajectoryConfig(steps=4, width=16, layers=1, batch_size=2)
    trainer = TrajectoryTrainer(data, config, future_frames=3)
    trainer.train_step()
    trainer.train_step()
    path = tmp_path / "checkpoint.pt"
    trainer.save(path, {"fixture": "unit test; not a quality result"})
    expected = [trainer.train_step(), trainer.train_step()]
    continued = TrajectoryTrainer(data, config, future_frames=3)
    continued.resume(path, {"fixture": "unit test; not a quality result"})
    actual = [continued.train_step(), continued.train_step()]
    assert actual == expected
    assert all(
        torch.equal(v, continued.model.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    )
    assert all(
        torch.equal(v, continued.ema.state_dict()[k]) for k, v in trainer.ema.state_dict().items()
    )
    assert trainer.validate() == trainer.validate()
    with pytest.raises(ValueError, match="identical"):
        continued.resume(path, {"fixture": "changed"})


@pytest.mark.parametrize(
    "mutation", [{"sigma_data": 0}, {"augment": "yes"}, {"max_seconds": 22000}, {"steps": True}]
)
def test_invalid_training_config_fails(mutation):
    with pytest.raises(ValueError):
        replace(TrajectoryConfig(), **mutation).validate()


def test_cli_cannot_train_outside_allocation(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="scheduled TC2"):
        main(
            [
                "--dataset",
                str(tmp_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--config",
                str(tmp_path / "missing.json"),
            ]
        )
    assert not (tmp_path / "run").exists()
