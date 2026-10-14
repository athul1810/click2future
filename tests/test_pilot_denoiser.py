"""Test information parity, auxiliary semantics, and gradients of the architecture pilot."""

import pytest
import torch

from click2future.models.pilot_denoiser import PilotDenoiser, _positions_from_heatmaps


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    future = torch.randn(1, 3, 2, 16, 16, requires_grad=True)
    first = torch.randn(1, 3, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:8, 4:8] = 1
    return future, torch.tensor([500]), first, mask, torch.tensor([[0.5, -0.25]])


def test_forward_backward_and_auxiliary_outputs(inputs):
    model = PilotDenoiser(8, auxiliary=True, max_objects=2)
    output = model(*inputs)
    assert output.predicted_noise.shape == inputs[0].shape
    assert output.positions.shape == (1, 2, 2, 2)
    assert output.collision_logits.shape == (1, 2, 2, 2)
    torch.testing.assert_close(output.collision_logits, output.collision_logits.transpose(-1, -2))
    loss = output.predicted_noise.square().mean() + output.positions.square().mean()
    loss = loss + output.collision_logits[..., 0, 1].square().mean()
    loss.backward()
    assert torch.isfinite(inputs[0].grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_first_frame_mode_cannot_use_mask_or_action(inputs):
    model = PilotDenoiser(8, conditioning="first_frame").eval()
    with torch.no_grad():
        reference = model(*inputs).predicted_noise
        modified = (*inputs[:3], torch.ones_like(inputs[3]), -inputs[4])
        torch.testing.assert_close(reference, model(*modified).predicted_noise, rtol=0, atol=0)


def test_numeric_and_spatial_modes_share_capacity_and_zero_force_behavior(inputs):
    numeric = PilotDenoiser(8, conditioning="numeric").eval()
    spatial = PilotDenoiser(8, conditioning="spatial").eval()
    spatial.load_state_dict(numeric.state_dict())
    assert sum(p.numel() for p in numeric.parameters()) == sum(
        p.numel() for p in spatial.parameters()
    )
    zero_action = (*inputs[:4], torch.zeros_like(inputs[4]))
    with torch.no_grad():
        torch.testing.assert_close(
            numeric(*zero_action).predicted_noise,
            spatial(*zero_action).predicted_noise,
            rtol=0,
            atol=0,
        )
        assert not torch.allclose(
            numeric(*inputs).predicted_noise, spatial(*inputs).predicted_noise
        )


def test_invalid_frame_shape_is_rejected(inputs):
    model = PilotDenoiser(8)
    with pytest.raises(ValueError, match="first_frame"):
        model(inputs[0], inputs[1], inputs[2][:, :, :-1], inputs[3], inputs[4])


def test_position_heatmaps_preserve_absolute_location_and_xy_order():
    logits = torch.full((1, 2, 1, 3, 3), -100.0)
    logits[0, 0, 0, 0, 2] = 100.0
    logits[0, 1, 0, 2, 0] = 100.0
    expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    torch.testing.assert_close(_positions_from_heatmaps(logits), expected, rtol=0, atol=0)


def test_optimizer_and_rng_resume_matches_the_next_update(inputs, tmp_path):
    model = PilotDenoiser(8, conditioning="numeric")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def update(network, optim):
        optim.zero_grad(set_to_none=True)
        target = torch.randn_like(inputs[0])
        loss = (network(*inputs).predicted_noise - target).square().mean()
        loss.backward()
        optim.step()

    update(model, optimizer)
    path = tmp_path / "resume.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        },
        path,
    )
    update(model, optimizer)
    expected_rng = torch.get_rng_state().clone()
    resumed = PilotDenoiser(8, conditioning="numeric")
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    checkpoint = torch.load(path, weights_only=True)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    update(resumed, resumed_optimizer)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("shape", [(0, 3, 2, 16, 16), (1, 3, 2, 0, 16)])
def test_empty_batch_or_spatial_dimensions_are_rejected(inputs, shape):
    model = PilotDenoiser(8)
    with pytest.raises(ValueError, match="Nonempty batch"):
        model(torch.empty(shape), *inputs[1:])
