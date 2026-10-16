"""Compatibility with B's trainer branch; active once its modules are present."""

import numpy as np
import pytest
import torch

from click2future.data import load_batch, save_rollout
from click2future.evaluation.heads import CaptureFinalHeads, evaluate_heads
from click2future.models.pilot_denoiser import PilotDenoiser
from click2future.sim import make_scene, run_rollout
from click2future.training.batch import prepare_video_batch


@pytest.fixture
def handoff_modules():
    return (
        pytest.importorskip("click2future.training.conditioning"),
        pytest.importorskip("click2future.training.diffusion"),
    )


def test_native_normalization_matches_b_boundary(handoff_modules, tmp_path):
    conditioning, _ = handoff_modules
    ref = run_rollout(make_scene(160000), frame_count=4, resolution=64)
    save_rollout(ref, tmp_path, "fixture")
    raw = load_batch(tmp_path, ["fixture"])
    ours = prepare_video_batch(raw)
    clean, condition = conditioning.training_inputs(raw, "cpu")
    assert torch.equal(ours.future, clean)
    for field in ("first_frame", "selected_mask", "force_xy"):
        assert torch.equal(getattr(ours, field), getattr(condition, field))


def test_real_ddim_capture_preserves_samples_and_exports_actual_final_heads(handoff_modules):
    conditioning, diffusion = handoff_modules
    ref = run_rollout(make_scene(160001), frame_count=4, resolution=64)
    condition = conditioning.observable_condition(
        torch.from_numpy(ref["frames"][:1].transpose(0, 3, 1, 2)).float() / 255,
        torch.from_numpy(ref["selected_mask"][None, None]),
        torch.from_numpy(ref["force_xy"][None]),
    )
    model = PilotDenoiser(base_channels=8, auxiliary=True).eval()
    sampler = diffusion.Diffusion(10)
    state = torch.get_rng_state().clone()
    baseline = sampler.sample(model, condition, future_frames=3, sampling_steps=3, seed=7)
    with CaptureFinalHeads(model) as captured:
        samples = sampler.sample(model, condition, future_frames=3, sampling_steps=3, seed=7)
    assert torch.equal(samples, baseline) and torch.equal(state, torch.get_rng_state())
    assert not model._forward_hooks
    assert captured.output.positions.grad_fn is None
    report, arrays = evaluate_heads(
        ref,
        [captured.output],
        timesteps=[0],
        num_diffusion_steps=10,
        max_timestep_fraction=0.25,
        collision_threshold=0.5,
        event_tolerance=1,
        run_id="untrained-ddim-compatibility",
    )
    assert report["source"] == "auxiliary_head"
    assert arrays["predicted_positions"].shape == (1, 4, 2, 2)
    assert np.isfinite(arrays["predicted_positions"]).all()
