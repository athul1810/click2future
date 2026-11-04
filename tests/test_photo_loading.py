"""Strict low-memory checkpoint assignment, separate from model quality."""

import importlib.util
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / "cluster/streamforce_runtime.py"
spec = importlib.util.spec_from_file_location("streamforce_runtime", path)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_meta_assignment_preserves_the_published_tensor_cast_and_forward():
    original = torch.nn.Linear(3, 2)
    state = original.state_dict()
    with torch.device("meta"):
        target = torch.nn.Linear(3, 2)
    assert runtime.assign_state(target, state, device="cpu", dtype=torch.bfloat16) == 2
    assert not any(p.is_meta for p in target.parameters())
    reference = original.to(torch.bfloat16)
    inputs = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    assert torch.equal(reference(inputs), target(inputs))


def test_loading_rejects_partial_wrong_shape_and_ambiguous_wrappers():
    module = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        runtime.assign_state(module, {}, device="cpu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Invalid"):
        runtime.assign_state(
            module,
            {"weight": torch.zeros(2, 4), "bias": torch.zeros(2)},
            device="cpu",
            dtype=torch.bfloat16,
        )
    with pytest.raises(ValueError, match="collide"):
        runtime.clean_state({"weight": torch.zeros(1), "_orig_mod.weight": torch.zeros(1)})
