"""The held-out scope cannot silently turn into D017 or the training corpus."""

import copy
import json

import pytest

from click2future.evaluation.prior_protocol import COUNTS, registered_holdout


@pytest.mark.parametrize("changed", [None, "split", "seed", "counts", "same_corpus", "training"])
def test_registered_prior_holdout_scope(tmp_path, changed):
    # Metadata-only fixture. No samples or predictive-quality claim are made here.
    manifest = {"seed": 830000, "group_counts": copy.deepcopy(COUNTS)}
    training = {"seed": 730000, "samples": [{}] * 668}
    plan = {
        "study_id": "D018",
        "split": "id_test",
        "input_sha256": {"dataset_manifest": "new"},
        "training_corpus": {
            "dataset_manifest": "old",
            "checked_training_clips": 668,
            "shared_frame_zero_or_visible_seed": 0,
        },
        "files": {
            "dataset_manifest": str(tmp_path / "new.json"),
            "training_dataset_manifest": str(tmp_path / "old.json"),
        },
    }
    if changed == "split":
        plan["split"] = "shape_ood"
    elif changed == "seed":
        manifest["seed"] = 790000
    elif changed == "counts":
        manifest["group_counts"]["id_test"] = 1
    elif changed == "same_corpus":
        plan["training_corpus"]["dataset_manifest"] = "new"
    elif changed == "training":
        training["samples"] = [{}]
    (tmp_path / "new.json").write_text(json.dumps(manifest))
    (tmp_path / "old.json").write_text(json.dumps(training))
    if changed:
        with pytest.raises(ValueError):
            registered_holdout(plan)
    else:
        assert registered_holdout(plan)
        plan["split"] = "id_test-selected-object"
        assert registered_holdout(plan)
        plan.pop("study_id")
        assert registered_holdout(plan) is False
