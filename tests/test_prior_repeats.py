"""Repeat recipes cannot silently change the screen or the paired control."""

import json
import runpy
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "seed",
        "steps",
        "world_prior_only",
        "contact_transition_weight",
        "normal_velocity_weight",
        "dynamics_residual_weight",
        "collision_impulse_weight",
    ],
)
def test_repeat_recipes_keep_the_registered_budget_and_pairs(tmp_path, changed):
    shutil.copytree(ROOT / "configs/training", tmp_path / "configs/training")
    p = tmp_path / "configs/training/prior_repeats/direct-seed20.json"
    value = json.loads(p.read_text())
    if changed:
        value[changed] = False if changed == "world_prior_only" else value.get(changed, 0) + 1
        p.write_text(json.dumps(value))
    driver = runpy.run_path(str(ROOT / "submission/prior_repeats.py"))
    if changed:
        with pytest.raises(ValueError, match="only the refinement seed"):
            driver["recipes"](tmp_path)
    else:
        assert set(driver["recipes"](tmp_path)) == {
            "direct-seed20",
            "diffusion-seed20",
            "direct-seed21",
            "diffusion-seed21",
        }


def test_screen_comparison_normalizes_only_missing_default_fields():
    driver = runpy.run_path(str(ROOT / "submission/prior_repeats.py"))
    normalize = driver["normalized_config"]
    legacy = json.loads((ROOT / "configs/training/prior_repeats/direct-seed20.json").read_text())
    explicit = {
        **legacy,
        "contact_transition_weight": 0.0,
        "normal_velocity_weight": 0.0,
        "dynamics_residual_weight": 0.0,
        "collision_impulse_weight": 0.0,
    }
    assert normalize(legacy) == normalize(explicit)
    assert normalize(legacy) != normalize({**explicit, "normal_velocity_weight": 4.0})
