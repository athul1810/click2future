"""Late passive movement and missing frames must survive full-horizon scoring."""

import json
from pathlib import Path

import numpy as np
import pytest

from click2future.evaluation.passive_stationarity import passive_horizon, stationary_reference
from click2future.sim import run_rollout


def reference():
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    return run_rollout(scene, frame_count=32, resolution=64, fps=8)


def test_late_passive_drift_and_missing_frame_are_separate_failures():
    ref = reference()
    assert stationary_reference(ref)
    videos = np.repeat(ref["frames"][None], 3, axis=0)
    passive = 1 - ref["selected_object_id"]
    mask = ref["instance_masks"][0, passive]
    moved = np.roll(mask, 3, axis=1)
    videos[1, -1, mask] = ref["frames"][0, 1, 1]
    videos[1, -1, moved] = ref["frames"][0, mask]
    videos[2, 20, mask] = ref["frames"][0, 1, 1]
    result = passive_horizon(ref, videos)
    assert result["total_samples"] == 3
    assert result["complete_samples"] == 2 and result["missing_samples"] == 1
    assert result["drifting_samples"] == 1
    assert result["samples"][1]["max_excursion"] == pytest.approx(3 / 64)
    assert result["samples"][2]["drifting"] is None


def test_nonstationary_reference_is_ineligible():
    ref = reference()
    ref["positions"][-1, 1 - ref["selected_object_id"], 0] += 0.1
    assert not stationary_reference(ref)
    with pytest.raises(ValueError, match="stationary"):
        passive_horizon(ref, ref["frames"][None])
