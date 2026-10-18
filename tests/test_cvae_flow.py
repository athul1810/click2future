"""Check warp coordinates, gradients, observable-only sampling and checkpoint guards."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from click2future.data import generate_dataset
from click2future.models.cvae_flow import FlowVideoVAE, build_cvae, warp_first_frame
from click2future.training.cvae_loss import cvae_loss
from click2future.training.multiscene_pilot import PROJECT
from click2future.training.sample_cvae import sample_checkpoint
from click2future.training.train_cvae import sha256


def test_warp_matches_bilinear_border_values_and_flow_gradients():
    rng = torch.Generator().manual_seed(95)
    first = torch.rand((2, 3, 7, 9), generator=rng) * 2 - 1
    flow = (torch.rand((2, 2, 3, 7, 9), generator=rng) * 0.4 - 0.2).requires_grad_()
    copy = flow.detach().clone().requires_grad_()
    y, x = torch.meshgrid(
        (torch.arange(7) + 0.5) * 2 / 7 - 1, (torch.arange(9) + 0.5) * 2 / 9 - 1, indexing="ij"
    )
    grid = torch.stack((x, y), -1)[None, None] - copy.permute(0, 2, 3, 4, 1)
    expected = F.grid_sample(
        first.repeat_interleave(3, 0),
        grid.reshape(6, 7, 9, 2),
        padding_mode="border",
        align_corners=False,
    )
    expected = expected.reshape(2, 3, 3, 7, 9).permute(0, 2, 1, 3, 4)
    actual = warp_first_frame(first, flow)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(flow.grad, copy.grad, atol=2e-6, rtol=0)


def test_zero_flow_copies_pixels_and_positive_flow_moves_content_right():
    first = torch.zeros(1, 3, 8, 8)
    first[:, :, 2:4, 2:4] = 1
    zero = torch.zeros(1, 2, 3, 8, 8)
    assert torch.equal(warp_first_frame(first, zero), first[:, :, None].expand(-1, -1, 3, -1, -1))
    zero[:, 0] = 0.25
    moved = warp_first_frame(first, zero)
    torch.testing.assert_close(moved[:, :, :, 2:4, 3:5], torch.ones(1, 3, 3, 2, 2))
    assert not moved[:, :, :, :, :3].any()


def test_flow_model_starts_static_has_output_gradients_and_prior_ignores_future():
    torch.manual_seed(96)
    model = FlowVideoVAE(future_frames=3, base_channels=8, latent_dim=4)
    first = torch.rand(1, 3, 16, 16) * 2 - 1
    mask, force = torch.ones(1, 1, 16, 16), torch.tensor([[0.5, 0.0]])
    future = first[:, :, None].expand(-1, -1, 3, -1, -1).roll(1, -1)
    prediction = model(first, mask, force, future)
    cvae_loss(prediction, future).total.backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.output.weight.grad).all()
    model.eval()

    def forbidden(*_):
        raise AssertionError("Future encoder was called during prior sampling")

    handle = model.future_encoder.register_forward_hook(forbidden)
    try:
        samples = model.sample(first, mask, force, seeds=[0, 1, 2, 3])
    finally:
        handle.remove()
    assert torch.equal(samples, first[None, :, :, None].expand(4, -1, -1, 3, -1, -1))
    assert build_cvae(model.configuration()).configuration() == model.configuration()


def test_flow_checkpoint_round_trip_and_source_protection(tmp_path):
    import yaml

    data = tmp_path / "data"
    generated = generate_dataset(
        data,
        group_count=1,
        variants_per_group=1,
        seed=950000,
        config={"frame_count": 4, "resolution": 16, "fps": 8},
    )
    name = generated["manifest"]["samples"][0]["sample_id"]
    config = yaml.safe_load((PROJECT / "configs/training/cvae_flow_pilot.yaml").read_text())
    config["data"].update(resolution=16, total_frames=4)
    config["model"].update(future_frames=3, base_channels=8, latent_dim=4)
    model = build_cvae(config["model"])
    payload = dict(
        format_version=1,
        model_config=model.configuration(),
        config=config,
        model=model.state_dict(),
        state={"step": 0},
        provenance={
            "source_sha256": {
                n: sha256(PROJECT / "src/click2future/models" / n)
                for n in ("cvae.py", "cvae_flow.py")
            }
        },
    )
    checkpoint = tmp_path / "flow.pt"
    torch.save(payload, checkpoint)
    output = tmp_path / "samples"
    sample_checkpoint(
        checkpoint, data, [name], output, seeds=[0, 1], device="cpu", run_id="fixture"
    )
    with np.load(output / (name + ".npz")) as saved:
        assert np.all(saved["videos"] == saved["videos"][:, :1])
    assert json.loads((output / "samples.json").read_text())["training_steps"] == 0
    del payload["provenance"]["source_sha256"]["cvae_flow.py"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="flow decoder source"):
        sample_checkpoint(checkpoint, data, [name], tmp_path / "bad", seeds=[0], run_id="bad")
    assert not (tmp_path / "bad").exists()


def test_legacy_factory_stays_rgb_and_rejects_unknown_decoder():
    from click2future.models.cvae import ConditionalVideoVAE

    assert (
        type(build_cvae(dict(future_frames=3, base_channels=8, latent_dim=4)))
        is ConditionalVideoVAE
    )
    with pytest.raises(ValueError, match="decoder_kind"):
        build_cvae(dict(future_frames=3, base_channels=8, latent_dim=4, decoder_kind="other"))
