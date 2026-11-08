"""Known-depth populations must survive floating-point bisection boundaries."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("toward", [-np.inf, np.inf])
def test_control_populations_include_every_nominal_boundary_trial(toward):
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/penetration_control.py")
    )
    rows = [
        {
            "target_depth": depth,
            "true_depth": float(np.nextafter(depth, toward)),
            "measured_depth": 1.0 if depth == 1.5 else depth + 0.2,
            "mask_overlap_pixels": 0,
            "shapes": ["circle", "rectangle"],
        }
        for depth in module["DEPTHS"]
    ]
    result = module["summarize"](rows, ["known-layout"], 0.2, 1, 42, False)
    assert result["threshold_group_counts"] == {
        "target_at_most_one_pixel": {"trials": 5, "flagged": 1},
        "target_at_least_one_and_half_pixel": {"trials": 5, "missed": 1},
        "fit_target_at_least_half_pixel": 9,
    }
    assert result["false_flag_rate_true_at_most_one_pixel"] == pytest.approx(0.2)
    assert result["miss_rate_true_at_least_one_and_half_pixel"] == pytest.approx(0.2)
