"""The upload archive must preserve the original EMA bytes and provenance."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "prepare_streamforce_checkpoint",
    Path(__file__).resolve().parents[1] / "scripts/prepare_streamforce_checkpoint.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ema_archive_preserves_all_bytes_and_leaves_source_unchanged(tmp_path):
    source, output = tmp_path / "original.pt", tmp_path / "ema.pt"
    ema = {"weight": torch.tensor([0.0, -0.0, float("nan"), 1.0]), "scalar": torch.tensor(2)}
    base = torch.arange(12).reshape(3, 4)
    ema.update(base=base, view=base[1:, 1:].T)
    torch.save({"generator": {"unrelated": torch.zeros(5)}, "generator_ema": ema}, source)
    original = module.digest(source)
    result = module.prepare(source, output, original)
    assert result["status"] == "passed_exact_ema_extraction"
    assert result["ema_tensors"] == 4
    assert result["omitted"] == ["generator"]
    assert result["output_sha256"] == module.digest(output)
    assert module.digest(source) == original
    loaded = torch.load(output, weights_only=True)["generator_ema"]
    assert torch.equal(loaded["view"], ema["view"])
    assert loaded["view"].storage_offset() == ema["view"].storage_offset()
    assert (
        loaded["base"].untyped_storage().data_ptr() == loaded["view"].untyped_storage().data_ptr()
    )
    with pytest.raises(FileExistsError):
        module.prepare(source, output, original)


def test_rejects_wrong_source_identity_or_missing_ema(tmp_path):
    source, output = tmp_path / "original.pt", tmp_path / "ema.pt"
    torch.save({"generator": {}}, source)
    with pytest.raises(ValueError, match="SHA-256"):
        module.prepare(source, output, "0" * 64)
    with pytest.raises(ValueError, match="EMA"):
        module.prepare(source, output, module.digest(source))
    assert not output.exists()
