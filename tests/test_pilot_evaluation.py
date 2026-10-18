"""Keep failed cases in pilot gates and verify real sampler/head transport."""

import copy
import json
from dataclasses import asdict

import numpy as np
import torch
import yaml

from click2future.data import generate_dataset, load_rollout
from click2future.evaluation.pilot import cvae_diagnostics, score_heads, summarize_quality
from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.multiscene_pilot import PROJECT
from click2future.training.sample_diffusion import load_generator, sample_videos
from click2future.training.train_diffusion import TrainConfig, Trainer


def test_incomplete_case_cannot_disappear_from_pilot_gate():
    objects = dict(
        expected_future_observations=20,
        tracked_future_observations=20,
        total_samples=4,
        complete_samples=4,
        mean_ade_over_complete_samples=0.01,
    )
    good = {"trajectory": {"all_objects": objects, "selected_object": objects.copy()}}
    assert summarize_quality([good])["passes_tracking_and_selected_trajectory_gate"]
    failed = copy.deepcopy(good)
    failed["trajectory"]["all_objects"].update(tracked_future_observations=0, complete_samples=0)
    failed["trajectory"]["selected_object"].update(
        complete_samples=0, mean_ade_over_complete_samples=None
    )
    result = summarize_quality([good, failed])
    assert not result["passes_tracking_and_selected_trajectory_gate"]
    assert result["tracking_fraction"] == 0.5
    assert result["expected_selected_clips"] == 8
    assert result["complete_selected_clips"] == 4
    assert result["selected_ade_mean_cases_if_all_complete"] is None


def test_pilot_scores_actual_auxiliary_sampler_calls_without_changing_rgb(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(
        dataset,
        group_count=8,
        variants_per_group=1,
        seed=920000,
        config={"frame_count": 4, "resolution": 32, "fps": 8},
    )
    config = TrainConfig(steps=1, base_channels=8, diffusion_steps=32, auxiliary=True)
    trainer = Trainer(dataset, config)
    trainer.train_step()
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save(checkpoint)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    sample_id = trainer.sample_ids[0]
    reference = load_rollout(dataset, sample_id)
    plan = {
        "dataset": str(dataset),
        "cases": [{"sample_id": sample_id}],
        "sampling_seeds": [0, 1, 2, 3],
        "config": str(PROJECT / "configs/evaluation/pixel.yaml"),
    }
    model, diffusion, _ = load_generator(checkpoint)
    videos = sample_videos(
        model,
        diffusion,
        reference["frames"][0],
        reference["selected_mask"],
        reference["force_xy"],
        future_frames=3,
        sampling_steps=32,
        seeds=plan["sampling_seeds"],
    )
    samples = tmp_path / "samples"
    (samples / "pilot-E4").mkdir(parents=True)
    np.savez_compressed(samples / "pilot-E4" / (sample_id + ".npz"), videos=videos)
    output = tmp_path / "heads"
    score_heads(
        plan,
        {"checkpoint": str(checkpoint), "config": str(config_path)},
        samples,
        output,
        device="cpu",
    )
    record = json.loads((output / "manifest.json").read_text())
    assert record["rgb_replay"] == "pixel_exact_for_every_case_and_seed"
    report = json.loads((output / (sample_id + ".json")).read_text())
    assert report["source"] == "auxiliary_head"
    assert report["settings"]["head_timesteps"] == [0, 0, 0, 0]
    comparison = report["head_vs_rgb"]
    assert comparison["expected_future_object_observations"] == 24
    assert comparison["jointly_observed_future_objects"] <= 24
    if comparison["jointly_observed_future_objects"] == 0:
        assert comparison["mean_trajectory_distance_over_joint_observations"] is None


def test_cvae_diagnostics_exclude_test_clips_and_label_untrained_fixture(tmp_path):
    dataset = tmp_path / "data"
    generate_dataset(
        dataset,
        group_count=10,
        variants_per_group=1,
        seed=930000,
        config={"frame_count": 4, "resolution": 16, "fps": 8},
    )
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    for row in manifest["samples"]:
        if row["split"] == "test":
            (dataset / row["arrays"]).unlink()
    config = yaml.safe_load((PROJECT / "configs/training/cvae_multiscene_pilot.yaml").read_text())
    config["data"].update(resolution=16, total_frames=4)
    config["model"].update(future_frames=3, base_channels=8, latent_dim=4)
    model = ConditionalVideoVAE(**config["model"])
    checkpoint = tmp_path / "cvae.pt"
    torch.save(
        dict(
            model_config=config["model"], config=config, model=model.state_dict(), state={"step": 0}
        ),
        checkpoint,
    )
    output = tmp_path / "diagnostics.json"
    before = torch.get_rng_state().clone()
    cvae_diagnostics(dataset, checkpoint, output, device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    record = json.loads(output.read_text())
    assert record["untrained_fixture"]
    assert record["splits"]["train"]["clips"] == 8
    assert record["splits"]["validation"]["clips"] == 1
    assert len(record["splits"]["validation"]["per_latent_kl"]) == 4
