"""Foreground balance, loss gradients and future-label inference boundaries."""

import copy

import pytest
import torch

from click2future.models.cvae import CVAEOutput
from click2future.training.batch import prepare_video_batch
from click2future.training.cvae_loss import cvae_loss
from click2future.training.pixel_loss import future_foreground, pixel_mean
from click2future.training.train_diffusion import TrainConfig


def test_weighted_mean_matches_hand_calculation_and_gradient():
    errors = torch.tensor([1.0, 3.0, 2.0, 4.0]).reshape(1, 2, 1, 1, 2).requires_grad_()
    foreground = torch.tensor([True, False]).reshape(1, 1, 1, 1, 2)
    result = pixel_mean(errors, foreground=foreground, foreground_weight=3)
    assert result.item() == 2.0  # (3*1 + 3 + 3*2 + 4) / 8
    result.sum().backward()
    torch.testing.assert_close(errors.grad.flatten(), torch.tensor([3, 1, 3, 1]) / 8)


def test_clip_weights_do_not_depend_on_foreground_area():
    errors = torch.stack((torch.full((3, 2, 2, 2), 2.0), torch.full((3, 2, 2, 2), 5.0)))
    masks = torch.zeros(2, 1, 2, 2, 2, dtype=torch.bool)
    masks[1] = True
    assert torch.equal(
        pixel_mean(errors, foreground=masks, foreground_weight=64), torch.tensor([2.0, 5.0])
    )
    assert torch.equal(pixel_mean(errors), errors.flatten(1).mean(1))


@pytest.mark.parametrize("weight", [True, 0, -1, 65, float("nan"), float("inf"), "16"])
def test_invalid_loss_weight_rejects(weight):
    with pytest.raises(ValueError, match="foreground_weight"):
        pixel_mean(torch.ones(1, 3, 1, 2, 2), foreground_weight=weight)
    with pytest.raises(ValueError, match="foreground_weight"):
        TrainConfig(foreground_weight=weight).validate()


def test_weighted_loss_requires_matching_boolean_future_mask():
    errors = torch.ones(1, 3, 2, 2, 2)
    with pytest.raises(ValueError, match="explicit future"):
        pixel_mean(errors, foreground_weight=16)
    for bad in (torch.ones(1, 1, 2, 2, 2), torch.ones(1, 1, 3, 2, 2, dtype=torch.bool)):
        with pytest.raises(ValueError, match="Foreground"):
            pixel_mean(errors, foreground=bad, foreground_weight=16)


def loader_batch():
    return {
        "frames": torch.zeros(1, 3, 3, 8, 8),
        "selected_mask": torch.ones(1, 1, 8, 8),
        "force_xy": torch.zeros(1, 2),
        "force_duration": torch.full((1,), 0.25),
        "timestamps": torch.arange(3, dtype=torch.float64)[None] / 8,
        "instance_masks": torch.zeros(1, 3, 2, 8, 8, dtype=torch.bool),
        "object_valid": torch.tensor([[True, False]]),
    }


def test_frame_zero_and_padding_do_not_enter_future_loss_mask():
    raw = loader_batch()
    raw["instance_masks"][:, 0, 0] = True
    raw["instance_masks"][:, 1, 0, 2, 3] = True
    mask = future_foreground(raw, device="cpu")
    assert mask.shape == (1, 1, 2, 8, 8) and mask.sum() == 1
    assert mask[0, 0, 0, 2, 3]
    raw["instance_masks"][:, 2, 1, 2, 3] = True
    with pytest.raises(ValueError, match="valid object slots"):
        future_foreground(raw, device="cpu")


def test_training_only_masks_cannot_change_observable_conditioning():
    raw = loader_batch()
    baseline = prepare_video_batch(raw)
    changed = copy.deepcopy(raw)
    changed["instance_masks"][:, 1:, 0] = True
    weighted = prepare_video_batch(changed)
    assert set(vars(baseline)) == {"first_frame", "selected_mask", "force_xy", "future"}
    assert future_foreground(changed, device="cpu").all()
    for name in ("first_frame", "selected_mask", "force_xy", "future"):
        assert torch.equal(getattr(baseline, name), getattr(weighted, name))
    del raw["instance_masks"], raw["object_valid"]
    assert torch.equal(prepare_video_batch(raw).first_frame, baseline.first_frame)


@pytest.mark.parametrize("kind", ["l1", "mse"])
def test_cvae_logs_raw_errors_separately_from_weighted_objective(kind):
    prediction = torch.tensor([1.0, 2.0]).expand(1, 3, 1, 1, 2).clone().requires_grad_()
    zero = torch.zeros(1, 4)
    output = CVAEOutput(prediction, zero, zero, zero, zero)
    mask = torch.tensor([True, False]).reshape(1, 1, 1, 1, 2)
    loss = cvae_loss(
        output,
        torch.zeros_like(prediction),
        beta=0,
        reconstruction=kind,
        foreground=mask,
        foreground_weight=3,
    )
    assert loss.total.item() == (1.25 if kind == "l1" else 1.75)
    assert loss.record()["future_mae"] == 1.5 and loss.record()["future_mse"] == 2.5
    loss.total.backward()
    assert prediction.grad[..., 0].sum() > 0
