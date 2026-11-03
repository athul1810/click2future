"""E10 force routing and separately generated held-out corpus plans (D016/D017)."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from test_contact_training import fixture_data

from click2future.data.generalization import generate_generalization_dataset
from click2future.evaluation.suite import _write, prepare_plan, read_plan
from click2future.evaluation.trajectory_pilot import sample_plan
from click2future.models.object_motion import TrajectoryDiffusion
from click2future.models.routed_motion import RoutedTrajectoryDiffusion
from click2future.training.sample_trajectory import load_trajectory
from click2future.training.train_contact import ContactConfig, ContactTrainer
from click2future.training.train_routing import (
    RoutingConfig,
    RoutingTrainer,
    resume_probe,
    source_identity,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs/evaluation"


def inputs(force):
    generator = torch.Generator().manual_seed(5)
    features = torch.randn((2, 2, 11), generator=generator)
    features[..., -1] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    noisy = torch.randn((2, 3, 2, 4), generator=generator)
    return noisy, torch.tensor([0.3, 2.0]), features, torch.tensor(force)


def test_selected_routing_is_exactly_e6_and_broadcast_changes_only_the_force_path():
    torch.manual_seed(0)
    base = TrajectoryDiffusion(future_frames=3, width=16, layers=1)
    torch.nn.init.normal_(base.output.weight)
    models = {}
    for routing in ("selected", "broadcast"):
        models[routing] = RoutedTrajectoryDiffusion(
            force_routing=routing, future_frames=3, width=16, layers=1
        )
        models[routing].load_state_dict(base.state_dict(), strict=True)
    pushed = inputs([[0.25, 0.0], [0.0, -0.4]])
    assert torch.equal(models["selected"](*pushed), base(*pushed))
    assert not torch.allclose(models["broadcast"](*pushed), base(*pushed))
    still = inputs([[0.0, 0.0], [0.0, 0.0]])
    assert torch.equal(models["broadcast"](*still), base(*still))
    assert sum(p.numel() for p in models["broadcast"].parameters()) == sum(
        p.numel() for p in base.parameters()
    )
    with pytest.raises(ValueError, match="force_routing"):
        RoutedTrajectoryDiffusion(force_routing="numeric", future_frames=3, width=16, layers=1)


def test_selected_arm_trains_exactly_like_e6_and_both_arms_start_identically():
    data, geometry, _ = fixture_data()
    config = RoutingConfig(steps=4, width=16, layers=1, batch_size=3)
    contact = {key: value for key, value in asdict(config).items() if key != "broadcast_force"}
    e6 = ContactTrainer(data, geometry, ContactConfig(**contact))
    selected = RoutingTrainer(data, geometry, config)
    broadcast = RoutingTrainer(
        data, geometry, RoutingConfig(**{**contact, "broadcast_force": True})
    )
    for trainer in (selected, broadcast):
        for name, value in e6.model.state_dict().items():
            assert torch.equal(value, trainer.model.state_dict()[name])
    for _ in range(3):
        assert e6.train_step()["loss"] == selected.train_step()["loss"]
        broadcast.train_step()
    for name, value in e6.model.state_dict().items():
        assert torch.equal(value, selected.model.state_dict()[name])
    assert any(
        not torch.equal(value, broadcast.model.state_dict()[name])
        for name, value in e6.model.state_dict().items()
    )


def test_routing_checkpoints_resume_exactly_and_refuse_changed_model_source(tmp_path):
    data, geometry, _ = fixture_data()
    config = RoutingConfig(steps=4, width=16, layers=1, batch_size=3, broadcast_force=True)
    assert resume_probe(data, geometry, config, tmp_path / "resume.pt", device="cpu")[
        "losses_equal"
    ]
    trainer = RoutingTrainer(data, geometry, config)
    trainer.train_step()
    trainer.save(tmp_path / "routing.pt", {"source_sha256": source_identity()})
    model, state = load_trajectory(tmp_path / "routing.pt")
    assert isinstance(model, RoutedTrajectoryDiffusion) and model.force_routing == "broadcast"
    assert state["trajectory_format_version"] == 6 and model.force_limit == config.force_limit
    changed = {**source_identity(), "models/routed_motion.py": "0" * 64}
    trainer.save(tmp_path / "changed.pt", {"source_sha256": changed})
    with pytest.raises(ValueError, match="routed_motion"):
        load_trajectory(tmp_path / "changed.pt")


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    roots = {name: tmp_path_factory.mktemp(name) for name in ("training", "held_out", "copy")}
    generate_generalization_dataset(roots["training"], seed=301000)
    generate_generalization_dataset(roots["held_out"], seed=302000)
    generate_generalization_dataset(roots["copy"], seed=301000)
    return roots


def plan_for(dataset, split="id_test", training_corpus=None):
    return prepare_plan(
        dataset,
        CONFIGS / "pixel.yaml",
        split=split,
        metrics=("ssim",),
        protocol=CONFIGS / "comparison.yaml",
        training_corpus=training_corpus,
    )


def test_separate_held_out_plan_binds_the_training_corpus_and_rejects_shared_scenes(corpora):
    plan = plan_for(corpora["held_out"], training_corpus=corpora["training"])
    record = plan["training_corpus"]
    assert record["dataset_manifest"] == plan["input_sha256"]["training_dataset_manifest"]
    assert record["splits"] == plan["input_sha256"]["training_splits"]
    assert record["dataset_manifest"] != plan["input_sha256"]["dataset_manifest"]
    assert record["checked_held_out_clips"] == sum(len(c["reference_ids"]) for c in plan["cases"])
    with pytest.raises(ValueError, match="must differ"):
        plan_for(corpora["training"], training_corpus=corpora["training"])
    with pytest.raises(ValueError, match="also appears in the training corpus"):
        plan_for(corpora["copy"], training_corpus=corpora["training"])
    assert "training_corpus" not in plan_for(corpora["held_out"])


def test_pose_sampler_opens_test_splits_only_on_a_separate_corpus(corpora, tmp_path):
    reserved, separate = tmp_path / "reserved.json", tmp_path / "separate.json"
    _write(reserved, plan_for(corpora["training"]))
    _write(separate, plan_for(corpora["held_out"], training_corpus=corpora["training"]))
    read_plan(separate)
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="final-test splits"):
        sample_plan(reserved, missing, tmp_path / "a", method="E10")
    with pytest.raises(FileNotFoundError):
        sample_plan(separate, missing, tmp_path / "b", method="E10")
