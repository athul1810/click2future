"""CVAE posterior/prior separation, analytic loss, and seeded sampling."""

import pytest
import torch

from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.cvae_loss import beta_at_step, cvae_loss, diagonal_gaussian_kl


@pytest.fixture
def inputs():
    first = torch.zeros(2, 3, 16, 16)
    masks = torch.zeros(2, 1, 16, 16)
    masks[:, :, 3:7, 5:9] = 1
    force = torch.tensor([[0.4, -0.2], [0.0, 0.5]])
    future = torch.zeros(2, 3, 3, 16, 16)
    return first, masks, force, future


def test_real_backward_reaches_prior_posterior_condition_and_decoder(inputs):
    model = ConditionalVideoVAE(future_frames=3, base_channels=8, latent_dim=4)
    output = model(*inputs, generator=torch.Generator().manual_seed(5))
    assert output.reconstruction.shape == inputs[-1].shape
    loss = cvae_loss(output, inputs[-1])
    loss.total.backward()
    for block in (model.prior, model.posterior, model.condition_encoder, model.future_encoder):
        gradients = [p.grad for p in block.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(x).all() for x in gradients)
        assert sum(float(x.abs().sum()) for x in gradients) > 0
    assert torch.isfinite(model.output.weight.grad).all()
    assert loss.record()["frame_zero_scored"] is False


def test_prior_only_samples_are_seeded_and_do_not_consume_global_rng(inputs, monkeypatch):
    model = ConditionalVideoVAE(future_frames=3, base_channels=8, latent_dim=4).eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference read future/posterior")

    monkeypatch.setattr(model.future_encoder, "forward", forbidden)
    monkeypatch.setattr(model.posterior, "forward", forbidden)
    rng = torch.get_rng_state().clone()
    samples = model.sample(*inputs[:3], seeds=[1, 2, 3, 4])
    assert samples.shape == (4, 2, 3, 3, 16, 16)
    assert torch.equal(samples, model.sample(*inputs[:3], seeds=[1, 2, 3, 4]))
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(samples[0], samples[1])
    assert samples.min() >= -1 and samples.max() <= 1
    changed = model.sample(inputs[0], inputs[1], -inputs[2], seeds=[1])
    assert not torch.equal(samples[:1], changed)


def test_gaussian_kl_matches_closed_form_and_beta_zero(inputs):
    zero, one = torch.zeros(2, 4), torch.ones(2, 4)
    assert torch.equal(diagonal_gaussian_kl(zero, zero, zero, zero), torch.zeros(2))
    assert torch.equal(diagonal_gaussian_kl(one, zero, zero, zero), torch.full((2,), 0.5))
    model = ConditionalVideoVAE(future_frames=3, base_channels=8)
    output = model(*inputs)
    loss = cvae_loss(output, inputs[-1], beta=0)
    assert torch.equal(loss.total, loss.reconstruction)
    with pytest.raises(ValueError):
        cvae_loss(output, inputs[-1], beta=-1)


def test_sampling_requires_eval_and_unique_seeds(inputs):
    model = ConditionalVideoVAE(future_frames=3, base_channels=8)
    with pytest.raises(ValueError, match="eval"):
        model.sample(*inputs[:3], seeds=[0])
    model.eval()
    with pytest.raises(ValueError):
        model.sample(*inputs[:3], seeds=[0, 0])
    bad_mask = inputs[1] * 0
    with pytest.raises(ValueError):
        model.sample(inputs[0], bad_mask, inputs[2], seeds=[0])


def test_beta_warmup_is_a_function_of_resumable_step():
    assert beta_at_step(0.001, 0, 500, 0.2) == 0
    assert beta_at_step(0.001, 50, 500, 0.2) == 0.0005
    assert beta_at_step(0.001, 100, 500, 0.2) == 0.001
    assert beta_at_step(0.001, 300, 500, 0.2) == 0.001
    assert beta_at_step(0.001, 0, 500, 0) == 0.001
