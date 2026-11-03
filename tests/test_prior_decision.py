"""A contact improvement cannot conceal rest regressions or missing coverage."""

import copy

import pytest

from click2future.evaluation.prior_decision import compare


@pytest.mark.parametrize(
    "regression", [None, "rest", "missing_coverage", "undefined_ade", "timing"]
)
def test_prior_decision_keeps_each_registered_safeguard(regression):
    response = {
        h: {
            "selection": {
                "model": {
                    "expected_eligible_samples": 40,
                    "correct_targets": 40,
                    "missing_samples": 0,
                    "nonselected_drift": 0,
                }
            },
            "force": {
                "model": {"expected_eligible_samples": 40, "increases": 40, "missing_samples": 0}
            },
        }
        for h in ("window", "full")
    }
    control = {
        "cases": 60,
        "complete": 240,
        "penetrating": 10,
        "maximum_depth": 2.0,
        "longest_penetration": 12,
        "ade": 0.02,
        "contact_f1": 0.4,
        "zero": {"total_samples": 40, "missing_samples": 0, "drifting_samples": 0},
        "passive": {"total_samples": 80, "missing_samples": 0, "drifting_samples": 0},
        "response": response,
        "reversal": {h: {"rate": 1.0, "missing": 0} for h in response},
        "coverage": {
            f: {"estimate": 0.02, "scored_scene_groups": 10, "expected_scene_groups": 10}
            for f in ("forward", "reverse", "zero", "wall", "collision_0_to_1", "collision_1_to_0")
        },
    }
    direct = copy.deepcopy(control)
    direct["penetrating"] = 1
    if regression == "rest":
        direct["zero"]["drifting_samples"] = 1
    elif regression == "missing_coverage":
        direct["coverage"]["wall"]["scored_scene_groups"] = 9
    elif regression == "undefined_ade":
        direct["ade"] = None
    elif regression == "timing":
        direct["contact_f1"] = 0.39
    result = compare(direct, control)
    assert result["passes_all_registered_gates"] is (regression is None)
