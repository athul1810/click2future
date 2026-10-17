"""Loader-to-objective gradients and head exports without oracle conditioning."""

import copy
import json

import numpy as np
import pytest
import torch

from click2future.data import load_batch, save_rollout
from click2future.evaluation.__main__ import main as evaluate_archive
from click2future.evaluation.heads import CaptureFinalHeads, evaluate_heads, save_head_evaluation
from click2future.models.pilot_denoiser import DenoiserOutput, PilotDenoiser
from click2future.sim import make_scene, run_rollout
from click2future.training.auxiliary import AuxiliaryLossConfig
from click2future.training.batch import prepare_video_batch
from click2future.training.diffusion_step import diffusion_objective
from click2future.training.targets import build_auxiliary_targets, canonical_object_order


@pytest.fixture
def loader_batch(tmp_path):
    refs = [
        run_rollout(make_scene(130000 + i, object_count=2 + i), frame_count=4, resolution=16)
        for i in range(2)
    ]
    for index, ref in enumerate(refs):
        save_rollout(ref, tmp_path, f"fixture-{index}")
    return load_batch(tmp_path, ["fixture-0", "fixture-1"]), refs


def test_adapter_normalizes_axes_and_preserves_ood_force_without_labels(loader_batch):
    raw, _ = loader_batch
    raw["force_xy"][0] = [1.0, -0.75]
    batch = prepare_video_batch(raw)
    assert batch.future.shape == (2, 3, 3, 16, 16)
    assert torch.equal(batch.first_frame, torch.from_numpy(raw["frames"][:, 0]) * 2 - 1)
    assert batch.force_xy[0].tolist() == [2.0, -1.5]
    altered = {
        **raw,
        "positions": "unavailable",
        "collision_onset": "unavailable",
        "scene_specs": "hidden physics must not be read",
    }
    other = prepare_video_batch(altered)
    assert torch.equal(batch.future, other.future)
    assert set(vars(batch)) == {"first_frame", "selected_mask", "force_xy", "future"}


@pytest.mark.parametrize("bad", ["rgb", "mask", "force", "scale"])
def test_adapter_refuses_bad_normalization(loader_batch, bad):
    raw, _ = loader_batch
    raw = copy.deepcopy(raw)
    kwargs = {}
    if bad == "rgb":
        raw["frames"][0, 0, 0, 0, 0] = 1.1
    elif bad == "mask":
        raw["selected_mask"][0] = False
    elif bad == "force":
        raw["force_xy"][0, 0] = np.nan
    else:
        kwargs["force_scale"] = 0
    with pytest.raises(ValueError):
        prepare_video_batch(raw, **kwargs)


def test_joint_real_model_gradients_and_e3_reduction(loader_batch):
    raw, _ = loader_batch
    batch = prepare_video_batch(raw)
    targets = build_auxiliary_targets(
        torch.from_numpy(raw["positions"]),
        torch.from_numpy(raw["collision_onset"]),
        torch.from_numpy(raw["object_valid"]),
        torch.from_numpy(raw["instance_masks"][:, 0]),
        max_objects=4,
    )
    model = PilotDenoiser(base_channels=8, auxiliary=True)
    noise = torch.randn_like(batch.future)
    noisy = 0.8 * batch.future + 0.6 * noise
    timestep = torch.tensor([0, 2])
    loss = diffusion_objective(
        model,
        batch,
        noisy,
        noise,
        timestep,
        num_diffusion_steps=10,
        targets=targets,
        auxiliary_config=AuxiliaryLossConfig(),
    )
    loss.total.backward()
    for layer in (model.output, model.position_head, model.collision_head, model.stem):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum() > 0
    assert loss.record()["prediction_target"] == "epsilon"
    only_noise = diffusion_objective(model, batch, noisy, noise, timestep, num_diffusion_steps=10)
    assert only_noise.auxiliary is None
    assert torch.equal(only_noise.total, only_noise.noise)
    with pytest.raises(ValueError, match="explicit targets"):
        diffusion_objective(
            model,
            batch,
            noisy,
            noise,
            timestep,
            num_diffusion_steps=10,
            auxiliary_config=AuxiliaryLossConfig(),
        )


def test_head_export_restores_ids_and_round_trips_existing_metrics(loader_batch, tmp_path):
    _, refs = loader_batch
    ref = refs[0]
    order = canonical_object_order(
        torch.from_numpy(ref["instance_masks"][:1]),
        torch.from_numpy(ref["object_valid"][None]),
        max_objects=4,
    )[0]
    positions = torch.full((1, 3, 4, 2), torch.nan)
    logits = torch.full((1, 3, 4, 4), torch.nan)
    for slot, obj in enumerate(order.tolist()):
        if obj < 0:
            continue
        positions[0, :, slot] = torch.from_numpy(ref["positions"][1:, obj])
        for other, target in enumerate(order.tolist()):
            if target >= 0:
                logits[0, :, slot, other] = torch.where(
                    torch.from_numpy(ref["collision_onset"][:, obj, target]), 20.0, -20.0
                )
    output = DenoiserOutput(torch.zeros((1, 3, 3, 16, 16)), positions, logits)
    kwargs = dict(
        timesteps=[0] * 4,
        num_diffusion_steps=1000,
        max_timestep_fraction=0.25,
        collision_threshold=0.5,
        event_tolerance=1,
        run_id="head-fixture",
    )
    report, arrays = evaluate_heads(ref, [output] * 4, **kwargs)
    assert report["source"] == "auxiliary_head"
    assert report["trajectory"]["all_objects"]["best_of_k"]["min_ade"] == 0
    archive, path = tmp_path / "heads.npz", tmp_path / "heads.json"
    save_head_evaluation(
        report, arrays, sample_id="fixture-0", archive_path=archive, report_path=path
    )
    assert (
        evaluate_archive(
            [
                "--input",
                str(archive),
                "--output",
                str(tmp_path / "rescored.json"),
                "--source",
                "auxiliary_head",
                "--run-id",
                "head-fixture",
                "--sample-id",
                "fixture-0",
                "--scene-group-id",
                ref["scene_spec"]["scene_group_id"],
                "--condition-id",
                ref["scene_spec"]["condition_id"],
                "--selected-object",
                str(ref["selected_object_id"]),
                "--event-tolerance",
                "1",
            ]
        )
        == 0
    )
    other = json.loads((tmp_path / "rescored.json").read_text())
    assert other["trajectory"] == report["trajectory"]
    with pytest.raises(ValueError, match="supervised timestep"):
        evaluate_heads(ref, [output] * 4, **{**kwargs, "timesteps": [800] * 4})


def test_head_capture_does_not_replace_outputs_and_cleans_up_on_error(loader_batch):
    raw, _ = loader_batch
    batch = prepare_video_batch(raw)
    model = PilotDenoiser(base_channels=8, auxiliary=True).eval()
    noisy = torch.zeros_like(batch.future)
    with CaptureFinalHeads(model) as captured:
        for step in (9, 4, 0):
            output = model(
                noisy,
                torch.full((2,), step),
                batch.first_frame,
                batch.selected_mask,
                batch.force_xy,
            )
    assert torch.equal(captured.output.positions, output.positions.detach())
    assert not model._forward_hooks and captured.output.positions.grad_fn is None
    with pytest.raises(RuntimeError), CaptureFinalHeads(model):
        raise RuntimeError("sampler failed")
    assert not model._forward_hooks
