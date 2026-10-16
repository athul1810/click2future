"""Image metrics on analytic controls, missing frames, and real pretrained LPIPS."""

import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

pytest.importorskip("skimage")

from click2future.data import save_rollout  # noqa: E402
from click2future.evaluation.images import image_quality, main  # noqa: E402
from click2future.evaluation.perceptual import LPIPSScorer  # noqa: E402
from click2future.sim import make_scene, run_rollout  # noqa: E402


def images():
    reference = np.random.default_rng(4).integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
    return reference[None].copy(), reference


def test_exact_ssim_and_first_frame_exclusion():
    videos, reference = images()
    videos[:, 0] = 0
    result = image_quality(videos, reference, metrics=("ssim",))
    assert result["frame_zero_scored"] is False
    assert result["expected_future_frames"] == 3
    assert result["mean_over_complete_samples"]["ssim"] == 1
    assert result["per_sample"][0]["per_future_frame"]["ssim"] == [1, 1, 1]
    assert "lpips" not in result["preprocessing"]


def test_uniform_ssim_matches_analytic_luminance_term():
    reference = np.full((2, 64, 64, 3), 100, np.uint8)
    predicted = np.full((1, 2, 64, 64, 3), 120, np.uint8)
    result = image_quality(predicted, reference, metrics=("ssim",))
    expected = (2 * 100 * 120 + (255 * 0.01) ** 2) / (100**2 + 120**2 + (255 * 0.01) ** 2)
    assert result["mean_over_complete_samples"]["ssim"] == pytest.approx(expected, abs=1e-10)


def test_shift_noise_and_frozen_outputs_have_lower_ssim_than_exact():
    videos, reference = images()
    for bad in (np.roll(videos, 4, axis=3), 255 - videos, np.repeat(videos[:, :1], 4, axis=1)):
        result = image_quality(bad, reference, metrics=("ssim",))
        assert result["mean_over_complete_samples"]["ssim"] < 0.9


def test_missing_frames_preserve_counts_and_exclude_incomplete_clip_means():
    videos, reference = images()
    videos = np.repeat(videos, 2, axis=0)
    valid = np.ones((2, 4), bool)
    valid[1, 2] = False
    valid[:, 0] = False  # Frame 0 is unscored even in the observation mask.
    result = image_quality(videos, reference, metrics=("ssim",), frame_valid=valid)
    assert result["complete_samples"] == 1
    assert result["scored_future_frames"] == 5
    assert result["missing_future_frames"] == 1
    assert result["mean_over_complete_samples"]["ssim"] == 1
    assert result["per_sample"][1]["mean"]["ssim"] is None
    assert result["per_sample"][1]["per_future_frame"]["ssim"][1] is None
    valid[:, 1:] = False
    result = image_quality(videos, reference, metrics=("ssim",), frame_valid=valid)
    assert result["mean_over_complete_samples"]["ssim"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", ["float", "shape", "channels", "one_frame", "mask", "small"])
def test_invalid_image_inputs_are_rejected(change):
    videos, reference = images()
    mask = None
    if change == "float":
        videos = videos.astype(np.float32) / 255
    elif change == "shape":
        videos = videos[:, :2]
    elif change == "channels":
        videos, reference = videos[..., :1], reference[..., :1]
    elif change == "one_frame":
        videos, reference = videos[:, :1], reference[:1]
    elif change == "mask":
        mask = np.ones((1, 4), np.uint8)
    else:
        videos, reference = videos[:, :, :8, :8], reference[:, :8, :8]
    with pytest.raises(ValueError):
        image_quality(videos, reference, metrics=("ssim",), frame_valid=mask)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(metrics=()),
        dict(metrics=("ssim", "ssim")),
        dict(metrics=("wrong",)),
        dict(batch_size=0),
    ],
)
def test_invalid_metric_requests_are_rejected(kwargs):
    videos, reference = images()
    with pytest.raises(ValueError):
        image_quality(videos, reference, **kwargs)


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("lpips")
    try:
        return LPIPSScorer()
    except RuntimeError as error:
        pytest.skip(f"Pretrained evaluation weights not cached: {error}")


def test_real_lpips_exact_zero_and_distortion_positive(scorer):
    videos, reference = images()
    scores = scorer.score(videos[0], reference)
    np.testing.assert_allclose(scores, 0, atol=1e-8)
    assert (scorer.score(255 - videos[0], reference) > 0).all()
    assert not scorer._model.training
    assert all(not parameter.requires_grad for parameter in scorer._model.parameters())
    assert scorer.metadata["backbone_sha256"].startswith("7be5be79")
    assert len(scorer.metadata["calibration_sha256"]) == 64


def test_lpips_matches_official_api_and_is_invariant_to_batch_size(scorer):
    import torch

    videos, reference = images()
    videos[:, 1:] = np.roll(videos[:, 1:], 2, axis=3)
    first = image_quality(videos, reference, scorer=scorer, batch_size=1)
    second = image_quality(videos, reference, scorer=scorer, batch_size=3)
    np.testing.assert_allclose(
        first["per_sample"][0]["per_future_frame"]["lpips"],
        second["per_sample"][0]["per_future_frame"]["lpips"],
        atol=1e-6,
    )
    predicted = torch.from_numpy(videos[0, 1:].copy()).permute(0, 3, 1, 2).float() / 255
    target = torch.from_numpy(reference[1:].copy()).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        official = scorer._model(predicted, target, normalize=True).flatten().numpy()
    np.testing.assert_allclose(
        official, first["per_sample"][0]["per_future_frame"]["lpips"], rtol=1e-5, atol=1e-7
    )


def test_lpips_constructor_preserves_cpu_rng(scorer):
    import torch

    before = torch.get_rng_state().clone()
    LPIPSScorer()
    assert torch.equal(torch.get_rng_state(), before)


def test_uncached_lpips_does_not_download_implicitly(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lpips")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="not cached"):
        LPIPSScorer()
    assert not list(tmp_path.iterdir())


def test_corrupt_lpips_weights_are_rejected_before_loading(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")
    pytest.importorskip("lpips")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    filename = Path(urlparse(models.AlexNet_Weights.IMAGENET1K_V1.url).path).name
    weights = tmp_path / "checkpoints" / filename
    weights.parent.mkdir()
    weights.write_bytes(b"corrupted checkpoint")
    with pytest.raises(ValueError, match="checksum"):
        LPIPSScorer()


def cli_fixture(tmp_path):
    ref = run_rollout(make_scene(31), frame_count=3)
    save_rollout(ref, tmp_path, "ref")
    videos = tmp_path / "videos.npz"
    np.savez_compressed(videos, videos=ref["frames"][None])
    output = tmp_path / "scores.json"
    args = [
        "--reference-dir",
        str(tmp_path),
        "--sample-id",
        "ref",
        "--videos",
        str(videos),
        "--source",
        "simulator_control",
        "--run-id",
        "test",
        "--output",
        str(output),
    ]
    return args, output, videos, ref


def test_ssim_only_cli_needs_no_lpips_model(tmp_path, monkeypatch):
    from click2future.evaluation import images as module

    def forbidden(**kwargs):
        pytest.fail("SSIM-only mode tried to load LPIPS")

    monkeypatch.setattr(module, "LPIPSScorer", forbidden)
    args, output, _, _ = cli_fixture(tmp_path)
    assert main(args + ["--metrics", "ssim"]) == 0
    report = json.loads(output.read_text())
    assert report["source"] == "simulator_control"
    assert report["image_quality"]["mean_over_complete_samples"]["ssim"] == 1
    assert len(report["input_sha256"]) == 3


def test_cli_real_lpips_and_missing_frame_export(tmp_path, scorer):
    args, output, path, ref = cli_fixture(tmp_path)
    np.savez_compressed(path, videos=ref["frames"][None], frame_valid=[[True, True, False]])
    assert main(args) == 0
    report = json.loads(output.read_text())["image_quality"]
    assert report["per_sample"][0]["mean"] == {"ssim": None, "lpips": None}
    assert report["per_sample"][0]["per_future_frame"]["lpips"] == [0, None]


def test_cli_corrupt_input_and_input_preservation(tmp_path):
    args, output, videos, _ = cli_fixture(tmp_path)
    videos.write_bytes(b"not a video archive")
    assert main(args + ["--metrics", "ssim"]) == 1
    assert json.loads(output.read_text())["status"] == "failed"
    with pytest.raises(SystemExit):
        main(args + ["--output", str(videos)])
    assert videos.read_bytes() == b"not a video archive"


def test_simulator_image_controls_without_lpips(tmp_path):
    from click2future.evaluation.image_controls import main as control_main

    output = tmp_path / "image-controls.json"
    assert control_main(["--metrics", "ssim", "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["scene_groups"] == 4
    assert result["violations"] == []
    assert "lpips" not in result["preprocessing"]
    for row in result["records"]:
        assert row["scores"]["exact"]["mean_over_complete_samples"]["ssim"] == 1
        assert row["scores"]["exact"]["expected_future_frames"] == 15
        assert row["scores"]["frozen_first_frame"]["mean_over_complete_samples"]["ssim"] < 1
