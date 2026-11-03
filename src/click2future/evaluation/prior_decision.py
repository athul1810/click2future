"""Predeclared D018 comparisons; no pooled gain can offset a per-seed regression."""

import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def collect(folder):
    folder = Path(folder)
    quality = read(folder / "quality.json")
    collision = read(folder / "collision.json")
    geometry = read(folder / "geometry-audit/geometry.json")["summary"]["learned"]
    zero = read(folder / "contact-results.json")["zero_force_full_horizon"]["summary"]["model"]
    passive = read(folder / "passive.json")["summary"]
    horizons = read(folder / "action-horizons.json")["summary"]["actions"]
    strata = read(folder / "uncertainty-strata.json")["action_strata"]
    return {
        "cases": quality["case_count"],
        "complete": quality["complete_all_object_clips"],
        "ade": quality["selected_ade_mean_cases_if_all_complete"],
        "penetrating": geometry["clips_with_penetration_over_one_pixel"],
        "penetrating_frames": geometry["penetration_over_one_pixel_frames"],
        "maximum_depth": max(
            r["maximum_depth_pixels"] for r in collision["action_families"].values()
        ),
        "longest_penetration": max(
            r["longest_consecutive_penetration_frames"]
            for r in collision["action_families"].values()
        ),
        "contact_f1": collision["model_events_against_first_reference"]["f1"],
        "zero": zero,
        "passive": passive,
        "response": read(folder / "response-diagnostic/response.json")["summary"],
        "reversal": {
            horizon: {
                "rate": value["methods"][0]["metrics"][
                    "reversal.correct_force_sign_rate_over_complete"
                ]["estimate"],
                "missing": value["methods"][0]["counts"]["reversal.missing_pairs"],
            }
            for horizon, value in horizons.items()
        },
        "coverage": {
            family: value["methods"][0]["metrics"][
                "selected_object.mean_reference_to_nearest_sample"
            ]
            for family, value in strata.items()
        },
    }


def compare(direct, control):
    checks = []

    def check(name, passed, d, c):
        checks.append({"name": name, "passed": bool(passed), "direct": d, "control": c})

    def numeric(name, d, c, *, higher=False, strict=False):
        valid = d is not None and c is not None and math.isfinite(d) and math.isfinite(c)
        passed = valid and (
            (d > c if strict else d >= c) if higher else (d < c if strict else d <= c)
        )
        check(name, passed, d, c)

    check(
        "all_declared_ID_cases",
        direct["cases"] == control["cases"] == 60,
        direct["cases"],
        control["cases"],
    )
    numeric("complete_clips", direct["complete"], control["complete"], higher=True)
    numeric("fewer_penetrating_clips", direct["penetrating"], control["penetrating"], strict=True)
    for name in ("maximum_depth", "longest_penetration", "ade"):
        numeric(name, direct[name], control[name])
    numeric("contact_f1", direct["contact_f1"], control["contact_f1"], higher=True)
    check(
        "all_zero_force_samples",
        direct["zero"]["total_samples"] == control["zero"]["total_samples"] == 40,
        direct["zero"]["total_samples"],
        control["zero"]["total_samples"],
    )
    for family in ("zero", "passive"):
        check(
            family + "_same_eligible_denominator",
            direct[family]["total_samples"] == control[family]["total_samples"],
            direct[family]["total_samples"],
            control[family]["total_samples"],
        )
        for name in ("missing_samples", "drifting_samples"):
            numeric(family + "_" + name, direct[family][name], control[family][name])
    for horizon in ("window", "full"):
        d, c = (v["response"][horizon] for v in (direct, control))
        for family, metric in (("selection", "correct_targets"), ("force", "increases")):
            a, b = d[family]["model"], c[family]["model"]
            check(
                horizon + "_" + family + "_same_eligible_denominator",
                a["expected_eligible_samples"] == b["expected_eligible_samples"],
                a["expected_eligible_samples"],
                b["expected_eligible_samples"],
            )
            numeric(horizon + "_" + family, a[metric], b[metric], higher=True)
            numeric(horizon + "_" + family + "_missing", a["missing_samples"], b["missing_samples"])
        numeric(
            horizon + "_selection_passive_drift",
            d["selection"]["model"]["nonselected_drift"],
            c["selection"]["model"]["nonselected_drift"],
        )
        numeric(
            horizon + "_reversal",
            direct["reversal"][horizon]["rate"],
            control["reversal"][horizon]["rate"],
            higher=True,
        )
        numeric(
            horizon + "_reversal_missing",
            direct["reversal"][horizon]["missing"],
            control["reversal"][horizon]["missing"],
        )
    families = {"forward", "reverse", "zero", "wall", "collision_0_to_1", "collision_1_to_0"}
    check(
        "all_uncertainty_families",
        set(direct["coverage"]) == set(control["coverage"]) == families,
        sorted(direct["coverage"]),
        sorted(control["coverage"]),
    )
    for family in sorted(families):
        a, b = (v["coverage"].get(family) for v in (direct, control))
        complete = (
            a is not None
            and b is not None
            and all(r["scored_scene_groups"] == r["expected_scene_groups"] == 10 for r in (a, b))
        )
        check(family + "_coverage_complete", complete, a, b)
        numeric(
            family + "_reference_coverage",
            None if a is None else a["estimate"],
            None if b is None else b["estimate"],
        )
    return {
        "passes_all_registered_gates": all(r["passed"] for r in checks),
        "failed_checks": [r["name"] for r in checks if not r["passed"]],
        "checks": checks,
    }


def analyse(root):
    root = Path(root)
    replication = read(root / "replication.json")
    if replication["status"] != "completed" or set(replication["runs"]) != {
        f"{arm}-seed{seed}" for seed in (19, 20, 21) for arm in ("direct", "diffusion")
    }:
        raise ValueError("All six frozen held-out runs must finish before comparison")
    rows = {}
    for seed in (19, 20, 21):
        d, c = (collect(root / f"{arm}-seed{seed}") for arm in ("direct", "diffusion"))
        rows[str(seed)] = {"direct": d, "control": c, "decision": compare(d, c)}
    return {
        "decision": "D018",
        "scope": "ID generalization conditional on one fixed teacher; not external SOTA",
        "all_seeds_pass_registered_gates": all(
            r["decision"]["passes_all_registered_gates"] for r in rows.values()
        ),
        "seeds": rows,
        "test_splits_evaluated": ["D018:id_test"],
    }
