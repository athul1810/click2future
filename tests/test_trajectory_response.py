"""Paired force response keeps missing, stationary and backwards cases visible."""

import importlib.util
from pathlib import Path

import numpy as np


def module():
    spec = importlib.util.spec_from_file_location(
        "trajectory_response", Path("submission/trajectory_response.py")
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_force_scaling_preserves_pairing_and_missing_denominators():
    low = np.zeros((4, 32, 2, 2), np.float64)
    high = low.copy()
    low[:, 31, 1, 0] = [0.1, 0.1, 0, 0.1]
    high[:, 31, 1, 0] = [0.2, 0.05, 0.2, 0.2]
    known = np.ones((4, 32, 2), bool)
    missing = known.copy()
    missing[3, 20, 1] = False
    result = module().scaling((low, known), (high, missing), np.array([1.0, 0.0]), 1, end_frame=31)
    assert (
        result["total_samples"] == 4
        and result["complete_samples"] == 3
        and result["missing_samples"] == 1
    )
    assert result["per_sample"][0]["high_to_low_ratio"] == 2
    assert result["per_sample"][0]["increases_by_at_least_one_pixel"]
    assert result["per_sample"][1]["decreases_by_at_least_one_pixel"]
    assert result["per_sample"][2]["high_to_low_ratio"] is None
    assert not result["per_sample"][3]["complete"]
    early = module().scaling((low, known), (high, missing), np.array([1.0, 0.0]), 1, end_frame=4)
    assert early["complete_samples"] == 4
    assert all(not r["increases_by_at_least_one_pixel"] for r in early["per_sample"])


def test_signed_projection_tracks_force_instead_of_distance():
    low = np.zeros((1, 32, 2, 2), np.float64)
    high = low.copy()
    low[:, 31, 0, 0] = -0.1
    high[:, 31, 0, 0] = -0.2
    known = np.ones((1, 32, 2), bool)
    correct = module().scaling((low, known), (high, known), np.array([-1.0, 0.0]), 0, end_frame=31)
    wrong = module().scaling((low, known), (high, known), np.array([1.0, 0.0]), 0, end_frame=31)
    assert correct["per_sample"][0]["increases_by_at_least_one_pixel"]
    assert wrong["per_sample"][0]["decreases_by_at_least_one_pixel"]
    assert wrong["per_sample"][0]["high_to_low_ratio"] is None
