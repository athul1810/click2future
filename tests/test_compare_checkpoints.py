"""D031-D033 rules: scale selection, projection acceptance and the energy-score checks."""

import copy
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/compare_checkpoints.py"))


def summary(pen=4, bnd=6, ade=0.025, same=0.0073, energy=0.80, drift=0, missing=0):
    return {
        "quality": {
            "complete_all_object_clips": 96,
            "selected_ade_mean_cases_if_all_complete": ade,
        },
        "penetrating_clips": pen,
        "boundary_clips": bnd,
        "matched_world_ade": same,
        "zero_force": {"drifting_samples": drift, "missing_samples": missing},
        "passive": {"drifting_samples": 0, "missing_samples": 0},
        "energy": {"selected": energy},
    }


def test_eligibility_needs_less_penetration_within_every_margin():
    eligible = MODULE["eligible"]
    control = summary()
    assert eligible(summary(pen=2), control)
    assert not eligible(summary(pen=4), control)
    assert not eligible(summary(pen=2, ade=0.0256), control)
    assert not eligible(summary(pen=2, same=0.0079), control)
    assert not eligible(summary(pen=2, energy=0.84), control)
    assert not eligible(summary(pen=2, drift=1), control)
    assert not eligible(summary(pen=2, missing=1), control)
    fewer = summary(pen=2)
    fewer["quality"]["complete_all_object_clips"] = 95
    assert not eligible(fewer, control)
    assert eligible(summary(pen=2, ade=0.0254, same=0.0077, energy=0.83), control)


def test_selection_prefers_fewest_penetrations_then_boundary_then_smaller_scale():
    select = MODULE["select_scale"]
    models = {
        "c": summary(),
        "x16": summary(pen=2, bnd=3),
        "x128": summary(pen=1, bnd=5),
        "x1024": summary(pen=1, bnd=5),
    }
    assert select(models, "c", ["x16", "x128", "x1024"]) == "x128"
    models["x1024"]["boundary_clips"] = 4
    assert select(models, "c", ["x16", "x128", "x1024"]) == "x1024"
    models = {"c": summary(), "x16": summary(pen=5)}
    assert select(models, "c", ["x16"]) is None


def test_projection_rule_requires_zero_violations_and_margins():
    holds = MODULE["projection_holds"]
    raw = summary()
    assert holds(summary(pen=0, bnd=0, ade=0.0254, energy=0.83), raw)
    assert not holds(summary(pen=1, bnd=0), raw)
    assert not holds(summary(pen=0, bnd=1), raw)
    assert not holds(summary(pen=0, bnd=0, ade=0.0256), raw)
    assert not holds(summary(pen=0, bnd=0, energy=0.833), raw)
    assert not holds(summary(pen=0, bnd=0, drift=1), raw)


def test_learned_geometry_rule_uses_the_same_world_interval():
    holds = MODULE["learned_geometry_holds"]
    control = summary()
    assert holds(summary(pen=1, bnd=6), control, [-0.0001, 0.0004])
    assert not holds(summary(pen=1, bnd=6), control, [-0.0001, 0.0006])
    assert not holds(summary(pen=1, bnd=7), control, [-0.0001, 0.0004])
    assert not holds(summary(pen=4, bnd=6), control, [-0.0001, 0.0004])


def test_metric_check_flags_a_generator_that_beats_the_prior_matched_simulator():
    check = MODULE["metric_check"]
    models = {
        "simulator_sampled_world": summary(energy=0.72),
        "oracle_first_reference": summary(energy=1.36),
        "neural": summary(energy=0.80),
    }
    intervals = {
        "oracle_first_reference_minus_simulator_energy_selected": [0.3, 0.9],
        "neural_minus_simulator_energy_selected": [0.04, 0.13],
    }
    assert check(models, intervals)["passes"]
    bad = copy.deepcopy(intervals)
    bad["neural_minus_simulator_energy_selected"] = [-0.2, -0.01]
    result = check(models, bad)
    assert not result["passes"]
    assert result["significantly_better_than_simulator"] == ["neural"]
    models["oracle_first_reference"]["energy"]["selected"] = 0.70
    assert not check(models, intervals)["oracle_worse_than_simulator"]


def test_named_checkpoints_reject_duplicates_controls_and_projection_suffix():
    parse = MODULE["parse_named"]
    assert parse(["a=x.pt", "b=y.pt"]) == {"a": Path("x.pt"), "b": Path("y.pt")}
    for bad in (["a=x.pt", "a=y.pt"], ["simulator_sampled_world=x.pt"], ["a+b=x.pt"], ["=x"]):
        with pytest.raises(ValueError):
            parse(bad)


def test_families_list_the_control_first_and_scales_in_increasing_order():
    families = MODULE["FAMILIES"]
    for control, candidates in families.values():
        assert control not in candidates
        scales = [int(name.rsplit("x", 1)[1]) for name in candidates]
        assert scales == sorted(scales)
    root = Path(__file__).resolve().parents[1] / "configs/training"
    for scale in (16, 128, 1024):
        assert (root / f"prior_rest_anchored_geometry_x{scale}.json").is_file()
    for scale in (128, 1024):
        assert (root / f"prior_direct_contact_dynamics_x{scale}.json").is_file()
