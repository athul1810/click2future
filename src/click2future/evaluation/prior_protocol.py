"""Narrow D018 scope guard; no scoring formula or threshold changes."""

import json
from pathlib import Path

STUDY = "D018"
SEED = 830000
COUNTS = {
    "train": 1,
    "validation": 1,
    "id_test": 10,
    "shape_ood": 1,
    "texture_ood": 1,
    "force_ood": 1,
    "physics_test": 1,
}


def registered_holdout(plan):
    if plan.get("study_id") != STUDY:
        return False
    if plan["split"] not in {"id_test", "id_test-selected-object"}:
        raise ValueError("D018 opens only its ID-test conditions and derived selection controls")
    corpus = plan.get("training_corpus", {})
    if (
        corpus.get("checked_training_clips") != 668
        or corpus.get("shared_frame_zero_or_visible_seed") != 0
        or corpus.get("dataset_manifest") == plan["input_sha256"]["dataset_manifest"]
    ):
        raise ValueError("D018 requires the distinct checked original training corpus")
    manifest = json.loads(Path(plan["files"]["dataset_manifest"]).read_text())
    if manifest["seed"] != SEED or manifest["group_counts"] != COUNTS:
        raise ValueError("D018 held-out generator identity differs")
    training = json.loads(Path(plan["files"]["training_dataset_manifest"]).read_text())
    if training["seed"] != 730000 or len(training["samples"]) != 668:
        raise ValueError("D018 requires the original dataset-0.3 corpus")
    return True
