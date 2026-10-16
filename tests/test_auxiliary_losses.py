"""Object matching, masked reductions, class balance, timestep gates and real-head gradients."""

import copy
import math
from dataclasses import replace

import pytest
import torch

from click2future.models.pilot_denoiser import DenoiserOutput, PilotDenoiser
from click2future.training import (
    AuxiliaryLossConfig,
    auxiliary_loss,
    build_auxiliary_targets,
    canonical_object_order,
    collision_balance,
    to_scene_order,
)


def raw_batch(counts=(2, 3), frames=4):
    batch, objects = len(counts), max(counts)
    positions = torch.full((batch, frames, objects, 2), torch.nan)
    masks = torch.zeros(batch, objects, 32, 32, dtype=torch.bool)
    valid = torch.zeros(batch, objects, dtype=torch.bool)
    collisions = torch.zeros(batch, frames - 1, objects, objects, dtype=torch.bool)
    for b, count in enumerate(counts):
        valid[b, :count] = True
        for obj in range(count):
            x = 22 - obj * 7
            masks[b, obj, 12:15, x : x + 3] = True
            positions[b, :, obj] = torch.tensor([(x + 1.5) / 32, 13.5 / 32])
            positions[b, :, obj, 0] += torch.arange(frames) * 0.01
        if count >= 3:
            collisions[b, 0, 0, 2] = collisions[b, 0, 2, 0] = True
    return positions, collisions, valid, masks


def targets(counts=(2, 3), **kwargs):
    return build_auxiliary_targets(*raw_batch(counts), max_objects=4, **kwargs)


def output_for(target, *, offset=0.0, dtype=torch.float32):
    positions = (target.positions + offset).to(dtype).detach().requires_grad_()
    logits = torch.zeros_like(target.collision_onset, dtype=dtype, requires_grad=True)
    noise = torch.zeros(len(positions), 3, positions.shape[1], 4, 4, requires_grad=True)
    return DenoiserOutput(noise, positions, logits)


def loss(output, target, config=None, timesteps=None):
    if timesteps is None:
        timesteps = torch.zeros(len(target.positions), dtype=torch.long)
    return auxiliary_loss(output, target, timesteps, num_diffusion_steps=101, config=config)


def test_frame_zero_matching_reorders_both_collision_axes_and_padding():
    original = raw_batch()
    target = build_auxiliary_targets(*original, max_objects=4)
    assert target.canonical_to_scene.tolist() == [[1, 0, -1, -1], [2, 1, 0, -1]]
    torch.testing.assert_close(target.positions[1, :, 0], original[0][1, 1:, 2])
    assert target.collision_onset[1, 0, 0, 2]
    assert target.collision_onset[1, 0, 2, 0]
    assert target.pair_valid[1, 0].sum() == 3
    assert not target.pair_valid.diagonal(dim1=-2, dim2=-1).any()
    changed = list(copy.deepcopy(original))
    changed[0][:, 1:] = 1 - changed[0][:, 1:]
    other = build_auxiliary_targets(*changed, max_objects=4)
    assert torch.equal(target.canonical_to_scene, other.canonical_to_scene)


def test_relabelled_simulator_ids_produce_identical_canonical_targets():
    positions, collisions, valid, masks = raw_batch((3,))
    base = build_auxiliary_targets(positions, collisions, valid, masks, max_objects=4)
    permutation = torch.tensor([2, 0, 1])
    changed = build_auxiliary_targets(
        positions[:, :, permutation],
        collisions[:, :, permutation][:, :, :, permutation],
        valid[:, permutation],
        masks[:, permutation],
        max_objects=4,
    )
    torch.testing.assert_close(changed.positions, base.positions)
    assert torch.equal(changed.collision_onset, base.collision_onset)
    assert torch.equal(changed.pair_valid, base.pair_valid)


def test_equal_x_uses_y_to_order_first_frame_masks():
    masks = torch.zeros(1, 2, 8, 8, dtype=torch.bool)
    masks[0, 0, 5:7, 2:4] = True
    masks[0, 1, 1:3, 2:4] = True
    assert canonical_object_order(
        masks, torch.ones(1, 2, dtype=torch.bool), max_objects=2
    ).tolist() == [[1, 0]]


def test_future_masks_require_both_interval_endpoints():
    valid = torch.tensor([[False, True, True], [True, True, True]])
    target = targets(future_valid=valid)
    assert not target.position_valid[0, 0].any()
    assert not target.pair_valid[0, :2].any()
    assert target.pair_valid[0, 2].sum() == 1


def test_restore_scene_order_keeps_fixed_ids_and_padded_nan_slots():
    raw = raw_batch()
    target = build_auxiliary_targets(*raw, max_objects=4)
    restored, logits, valid = to_scene_order(
        target.positions, target.collision_onset.float(), target.canonical_to_scene, scene_objects=3
    )
    torch.testing.assert_close(
        restored[valid[:, None].expand(-1, 3, -1)], raw[0][:, 1:][valid[:, None].expand(-1, 3, -1)]
    )
    assert torch.equal(valid, raw[2])
    assert torch.equal(logits[1], raw[1][1].float())
    assert torch.isnan(restored[0, :, 2]).all()


def test_analytic_losses_and_equal_clip_weighting():
    target = targets()
    output = output_for(target, offset=0.1)
    output.positions.data[0] = target.positions[0]
    result = loss(output, target)
    # Smooth L1(.1, beta=.05) = .075; two clips receive equal weight despite different N.
    assert result.trajectory.item() == pytest.approx(0.075 / 2)
    assert result.collision.item() == pytest.approx(math.log(2))
    assert result.total.item() == pytest.approx(0.075 / 2 + 0.1 * math.log(2))
    assert result.statistics["position_coordinates"] == 30
    assert result.statistics["collision_pairs"] == 12


def test_padding_nan_lower_triangle_and_missing_frames_have_zero_gradient():
    target = targets(future_valid=torch.tensor([[True, False, True], [True, True, True]]))
    output = output_for(target, offset=0.02)
    output.positions.data[~target.position_valid] = torch.nan
    output.collision_logits.data[~target.pair_valid] = torch.nan
    result = loss(output, target)
    result.total.backward()
    assert torch.isfinite(output.positions.grad).all()
    assert torch.isfinite(output.collision_logits.grad).all()
    assert not output.positions.grad[~target.position_valid].any()
    assert not output.collision_logits.grad[~target.pair_valid].any()
    assert output.positions.grad[target.position_valid].abs().sum() > 0


def test_timestep_boundary_and_high_noise_clips():
    target = targets()
    output = output_for(target, offset=0.1)
    result = loss(output, target, timesteps=torch.tensor([25, 26]))
    assert result.statistics["eligible_timestep_clips"] == 1
    assert result.statistics["timestep_cutoff_inclusive"] == 25
    assert result.trajectory.item() == pytest.approx(0.075)
    result.total.backward()
    assert not output.positions.grad[1].any()
    assert not output.collision_logits.grad[1].any()
    output = output_for(target)
    result = loss(output, target, timesteps=torch.tensor([100, 100]))
    assert result.total.item() == 0
    result.total.backward()
    assert not output.positions.grad.any()
    assert not output.collision_logits.grad.any()


def test_component_ablation_and_no_auxiliary_outputs():
    target = targets()
    output = output_for(target, offset=0.1)
    result = loss(output, target, AuxiliaryLossConfig(collision_weight=0))
    assert result.total.item() == pytest.approx(0.075)
    disabled = AuxiliaryLossConfig(trajectory_weight=0, collision_weight=0)
    empty = DenoiserOutput(torch.ones(2, 3, 3, 4, 4, requires_grad=True))
    result = loss(empty, target, disabled)
    result.total.backward()
    assert result.total.item() == 0
    assert not empty.predicted_noise.grad.any()


def test_single_object_all_masked_frames_and_balance_degenerate_cases():
    target = targets((1,), future_valid=torch.zeros(1, 3, dtype=torch.bool))
    result = loss(output_for(target), target)
    assert result.total.item() == 0
    assert collision_balance([target])["status"] == "no_valid_pairs"
    negative = targets((2,))
    assert collision_balance([negative])["status"] == "no_positive_events"
    assert collision_balance([negative])["positive_weight"] == 1


def test_collision_balance_training_counts_cap_and_positive_bce_weight():
    target = targets()
    balance = collision_balance([target], max_positive_weight=5)
    assert balance["positive_count"] == 1
    assert balance["negative_count"] == 11
    assert balance["positive_weight"] == 5
    result = loss(output_for(target), target, AuxiliaryLossConfig(collision_positive_weight=5))
    assert result.collision.item() == pytest.approx((math.log(2) + 13 / 9 * math.log(2)) / 2)


def test_bfloat16_heads_use_float32_reductions():
    target = targets()
    output = output_for(target, offset=0.01, dtype=torch.bfloat16)
    result = loss(output, target)
    assert result.total.dtype == torch.float32
    result.total.backward()
    assert torch.isfinite(output.positions.grad).all()
    assert torch.isfinite(output.collision_logits.grad).all()


def test_actual_denoiser_has_finite_nonzero_auxiliary_gradients():
    torch.manual_seed(31)
    target = targets((2,))
    model = PilotDenoiser(8, auxiliary=True, max_objects=4)
    output = model(
        torch.randn(1, 3, 3, 16, 16),
        torch.tensor([2]),
        torch.randn(1, 3, 16, 16),
        torch.ones(1, 1, 16, 16),
        torch.tensor([[0.3, 0.0]]),
    )
    result = loss(output, target)
    result.total.backward()
    for module in (model.position_head, model.collision_head, model.bottleneck):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


@pytest.mark.parametrize(
    "config",
    [
        AuxiliaryLossConfig(trajectory_weight=-1),
        AuxiliaryLossConfig(smooth_l1_beta=0),
        AuxiliaryLossConfig(max_timestep_fraction=2),
        AuxiliaryLossConfig(collision_positive_weight=0.5),
    ],
)
def test_invalid_configs_fail(config):
    with pytest.raises(ValueError):
        config.validate()


def test_bad_targets_timesteps_and_capacity_fail():
    raw = raw_batch()
    with pytest.raises(ValueError, match="max_objects"):
        build_auxiliary_targets(*raw, max_objects=1)
    target = targets()
    with pytest.raises(ValueError, match="integer"):
        auxiliary_loss(
            output_for(target), target, torch.tensor([0.1, 0.2]), num_diffusion_steps=101
        )
    with pytest.raises(ValueError, match="timestep"):
        loss(output_for(target), target, timesteps=torch.tensor([0, 101]))
    bad = replace(target, pair_valid=torch.ones_like(target.pair_valid))
    with pytest.raises(ValueError, match="unordered"):
        loss(output_for(target), bad)


def test_identical_mask_centres_are_rejected_without_hidden_id_tiebreak():
    masks = torch.zeros(1, 2, 8, 8, dtype=torch.bool)
    masks[0, 0, 2, 2] = masks[0, 0, 4, 4] = True
    masks[0, 1, 2, 4] = masks[0, 1, 4, 2] = True
    with pytest.raises(ValueError, match="distinct"):
        canonical_object_order(masks, torch.ones(1, 2, dtype=torch.bool), max_objects=2)


def test_training_labels_are_detached_and_all_positive_weight_stays_finite():
    raw = list(raw_batch())
    raw[0].requires_grad_()
    target = build_auxiliary_targets(*raw, max_objects=4)
    result = loss(output_for(target, offset=0.1), target)
    result.total.backward()
    assert raw[0].grad is None
    one = targets((2,))
    positive = replace(one, collision_onset=one.pair_valid | one.pair_valid.transpose(-1, -2))
    balance = collision_balance([positive])
    assert balance["status"] == "no_negative_intervals"
    assert balance["positive_weight"] == 1


def test_auxiliary_check_cli(tmp_path):
    import json

    from click2future.training.check_auxiliary import main

    output = tmp_path / "aux-check.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["real_denoiser_check"]["gradient_l1_norms"]["position_head"] > 0
    assert report["real_denoiser_check"]["gradient_l1_norms"]["collision_head"] > 0
    assert report["synthetic_prediction_fit"]["generator_parameters_updated"] is False
    assert report["synthetic_prediction_fit"]["final_to_initial_ratio"] < 0.2
    assert report["synthetic_prediction_fit"]["optimizer_updates"] == 60
    assert report["fixture_collision_balance"]["positive_count"] > 0


def test_disabled_check_does_not_fit_prediction_tensors():
    from click2future.training.check_auxiliary import run_checks

    result = run_checks(AuxiliaryLossConfig(trajectory_weight=0, collision_weight=0))
    assert result["synthetic_prediction_fit"]["status"] == "disabled"
    assert result["synthetic_prediction_fit"]["optimizer_updates"] == 0
    assert result["synthetic_prediction_fit"]["final_to_initial_ratio"] is None
