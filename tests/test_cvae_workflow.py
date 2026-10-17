"""Dataset split checks, untrained checkpoint transport, and inference isolation."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
import yaml

from click2future.data import generate_controlled_dataset, generate_dataset, load_rollout
from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.sample_cvae import sample_checkpoint
from click2future.training.train_cvae import (
    dataset_index,
    main,
    model_batch,
    restore_checkpoint,
    save_checkpoint,
    validate_config,
)


def configuration():
    from pathlib import Path

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/training/cvae.yaml").read_text()
    )
    config["data"].update(total_frames=4, resolution=16)
    config["model"].update(future_frames=3, base_channels=8, latent_dim=4)
    return config


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    generate_dataset(
        root,
        group_count=10,
        variants_per_group=1,
        seed=140000,
        config={"frame_count": 4, "resolution": 16, "fps": 8},
    )
    return root


def test_split_index_never_opens_test_arrays_and_detects_manifest_corruption(dataset):
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    for row in manifest["samples"]:
        if row["split"] == "test":
            (dataset / (row["sample_id"] + ".npz")).unlink()
    selected, rows, provenance = dataset_index(dataset)
    assert len(selected["train"]) == 8 and len(selected["validation"]) == 1
    assert provenance["train_validation_content_sha256"]
    batch = model_batch(dataset, selected["train"][:2], rows, configuration(), device="cpu")
    assert batch.future.shape == (2, 3, 3, 16, 16)
    manifest["samples"][0]["split"] = "validation"
    (dataset / "dataset_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash/schema"):
        dataset_index(dataset)


def test_cli_refuses_training_without_tc2_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["--data-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "run")])
    assert error.value.code == 1 and not (tmp_path / "run").exists()


def test_config_cannot_exceed_six_hour_job_or_mismatch_future():
    config = configuration()
    validate_config(config)
    config["training"]["max_hours"] = 6.1
    with pytest.raises(ValueError):
        validate_config(config)
    config = configuration()
    config["model"]["future_frames"] = 4
    with pytest.raises(ValueError):
        validate_config(config)


def make_checkpoint(path, config=None):
    config = configuration() if config is None else config
    model = ConditionalVideoVAE(**config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    train_rng, shuffle_rng = torch.Generator().manual_seed(3), torch.Generator().manual_seed(5)
    state = {
        "step": 0,
        "order": [2, 0, 1],
        "cursor": 1,
        "epoch": 1,
        "best_validation": None,
        "previous_training_seconds": 0.0,
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        state=state,
        config=config,
        provenance={"fixture": "untrained checkpoint transport"},
        train_generator=train_rng,
        shuffle_generator=shuffle_rng,
    )
    return model, optimizer, train_rng, shuffle_rng, state, config


def test_checkpoint_restores_weights_sampler_cursor_rng_and_refuses_changed_data(tmp_path):
    path = tmp_path / "fixture.pt"
    model, optimizer, train_rng, shuffle_rng, state, config = make_checkpoint(path)
    # Synthetic optimizer moments test serialization without updating generator weights.
    parameter = next(model.parameters())
    optimizer.state[parameter] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.full_like(parameter, 0.25),
        "exp_avg_sq": torch.full_like(parameter, 0.5),
    }
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        state=state,
        config=config,
        provenance={"fixture": "untrained checkpoint transport"},
        train_generator=train_rng,
        shuffle_generator=shuffle_rng,
    )
    weights = copy.deepcopy(model.state_dict())
    expected_train, expected_shuffle = (
        torch.randn(3, generator=train_rng),
        torch.randperm(3, generator=shuffle_rng),
    )
    with torch.no_grad():
        next(model.parameters()).add_(1)
    optimizer.state.clear()
    kwargs = dict(
        model=model,
        optimizer=optimizer,
        config=config,
        provenance={"fixture": "untrained checkpoint transport"},
        train_generator=train_rng,
        shuffle_generator=shuffle_rng,
    )
    assert restore_checkpoint(path, **kwargs) == state
    assert all(torch.equal(value, weights[key]) for key, value in model.state_dict().items())
    assert torch.equal(optimizer.state[parameter]["exp_avg"], torch.full_like(parameter, 0.25))
    assert torch.equal(torch.randn(3, generator=train_rng), expected_train)
    assert torch.equal(torch.randperm(3, generator=shuffle_rng), expected_shuffle)
    with pytest.raises(ValueError, match="same model, config, dataset"):
        restore_checkpoint(path, **{**kwargs, "provenance": {"fixture": "different data"}})


def test_prior_sampling_exports_evaluation_format_and_ignores_future_rgb(
    dataset, tmp_path, monkeypatch
):
    path = tmp_path / "fixture.pt"
    make_checkpoint(path)
    ref_id = json.loads((dataset / "dataset_manifest.json").read_text())["samples"][0]["sample_id"]
    kwargs = dict(
        checkpoint=path,
        data_dir=dataset,
        sample_ids=[ref_id],
        seeds=[0, 1, 2, 3],
        run_id="untrained-fixture",
    )
    rng = torch.get_rng_state().clone()
    first = sample_checkpoint(output_dir=tmp_path / "first", **kwargs)
    assert torch.equal(rng, torch.get_rng_state())
    assert first["training_steps"] == 0
    with np.load(tmp_path / "first" / (ref_id + ".npz")) as archive:
        videos = archive["videos"]
    assert videos.shape == (4, 4, 16, 16, 3) and videos.dtype == np.uint8
    ref = load_rollout(dataset, ref_id)
    assert np.all(videos[:, 0] == ref["frames"][0])
    changed = copy.deepcopy(ref)
    changed["frames"][1:] = 255 - changed["frames"][1:]
    changed["scene_spec"]["objects"][0]["hidden_physics"]["mass"] *= 1.1
    monkeypatch.setattr("click2future.training.sample_cvae.load_rollout", lambda *_: changed)
    sample_checkpoint(output_dir=tmp_path / "second", **kwargs)
    with np.load(tmp_path / "second" / (ref_id + ".npz")) as archive:
        assert np.array_equal(videos, archive["videos"])


def test_valid_hash_cannot_override_scene_group_split(dataset):
    path = dataset / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["split"] = (
        "test" if manifest["samples"][0]["split"] != "test" else "train"
    )
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="scene-group split"):
        dataset_index(dataset)


def test_controlled_dataset_native_wall_condition_reaches_cvae_sampling(tmp_path):
    root = tmp_path / "controlled"
    result = generate_controlled_dataset(root, group_count=10, seed=194000)
    for row in result["manifest"]["samples"]:
        if row["split"] == "test":
            (root / (row["sample_id"] + ".npz")).unlink()
    selected, rows, _ = dataset_index(root)
    assert len(selected["train"]) == 8 * 21
    assert len(selected["validation"]) == 21
    config = configuration()
    config["data"].update(total_frames=32, resolution=64)
    config["model"]["future_frames"] = 31
    validate_config(config)
    wall_id = next(
        name for name in selected["validation"] if rows[name]["action_variant"] == "wall"
    )
    batch = model_batch(root, [wall_id], rows, config, device="cpu")
    assert batch.future.shape == (1, 3, 31, 64, 64)
    assert batch.force_xy.norm().item() == pytest.approx(2.2)
    checkpoint = tmp_path / "untrained.pt"
    make_checkpoint(checkpoint, config)
    output_dir = tmp_path / "sample"
    record = sample_checkpoint(
        checkpoint=checkpoint,
        data_dir=root,
        sample_ids=[wall_id],
        seeds=[0, 1, 2, 3],
        run_id="untrained-controlled-fixture",
        output_dir=output_dir,
    )
    assert record["training_steps"] == 0
    with np.load(output_dir / (wall_id + ".npz"), allow_pickle=False) as archive:
        videos = archive["videos"]
    assert videos.shape == (4, 32, 64, 64, 3)
    ref = load_rollout(root, wall_id)
    np.testing.assert_array_equal(videos[:, 0], np.broadcast_to(ref["frames"][0], (4, 64, 64, 3)))
