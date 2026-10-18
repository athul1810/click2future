"""Flow supervision has explicit units and cannot change observable conditioning."""

import copy

import numpy as np
import pytest
import torch
import yaml

from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.batch import VideoBatch
from click2future.training.cvae_loss import cvae_loss
from click2future.training.cvae_motion import flow_motion_loss, objective
from click2future.training.multiscene_pilot import PROJECT
from click2future.training.train_cvae import validate_config


def labels():
    masks = np.zeros((1, 3, 3, 8, 8), bool)
    for frame in range(3):
        masks[0, frame, 0, 1:3, 1 + frame : 3 + frame] = True
        masks[0, frame, 1, 5:7, 5:7] = True
    positions = np.full((1, 3, 3, 2), np.nan, np.float32)
    positions[0, :, 0] = [[0.25, 0.25], [0.375, 0.25], [0.5, 0.25]]
    positions[0, :, 1] = [0.75, 0.75]
    return dict(
        instance_masks=masks, positions=positions, object_valid=np.array([[True, True, False]])
    )


def test_displacement_supervision_matches_flow_units_and_ignores_padding():
    raw = labels()
    flow = torch.zeros(1, 2, 2, 8, 8)
    for frame, amount in enumerate((0.25, 0.5)):
        flow[0, 0, frame][raw["instance_masks"][0, frame + 1, 0]] = amount
    loss, mae = flow_motion_loss(flow, raw)
    assert loss.item() == 0 and mae.item() == 0
    zero = torch.zeros_like(flow, requires_grad=True)
    loss, _ = flow_motion_loss(zero, raw)
    loss.backward()
    assert zero.grad[:, 0].sum() < 0
    assert not zero.grad[:, 1].any()
    invalid = copy.deepcopy(raw)
    invalid["positions"][0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        flow_motion_loss(flow, invalid)


def test_zero_motion_weight_keeps_legacy_objective_exact_and_removes_failed_hook():
    torch.manual_seed(98)
    model = ConditionalVideoVAE(future_frames=3, base_channels=8, latent_dim=4)
    batch = VideoBatch(
        torch.rand(1, 3, 16, 16) * 2 - 1,
        torch.ones(1, 1, 16, 16),
        torch.zeros(1, 2),
        torch.rand(1, 3, 3, 16, 16) * 2 - 1,
    )
    expected = cvae_loss(
        model(
            batch.first_frame,
            batch.selected_mask,
            batch.force_xy,
            batch.future,
            generator=torch.Generator().manual_seed(0),
        ),
        batch.future,
    )
    actual = objective(
        model,
        batch,
        generator=torch.Generator().manual_seed(0),
        beta=0.001,
        reconstruction="l1",
        foreground=None,
        foreground_weight=1,
    )
    assert torch.equal(actual.total, expected.total)
    with pytest.raises(ValueError, match="matching"):
        objective(
            model,
            batch,
            generator=None,
            beta=0.001,
            reconstruction="l1",
            foreground=None,
            foreground_weight=1,
            motion_weight=1,
            raw=labels(),
        )
    assert not model.output._forward_hooks


def test_motion_config_requires_flow_and_valid_nonnegative_weight():
    config = yaml.safe_load((PROJECT / "configs/training/cvae_flow_motion_pilot.yaml").read_text())
    validate_config(config)
    config["training"]["flow_motion_weight"] = -1
    with pytest.raises(ValueError, match="flow_motion_weight"):
        validate_config(config)
    config["training"]["flow_motion_weight"] = 1
    del config["model"]["decoder_kind"]
    with pytest.raises(ValueError, match="flow_motion_weight"):
        validate_config(config)
