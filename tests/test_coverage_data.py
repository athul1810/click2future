"""Pairing, contamination and mutation checks for the coverage intervention."""

import copy
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_action_supplement import corpus as corpus

from click2future.data import load_rollout
from click2future.data.generalization import _digest
from click2future.training import coverage_data as module
from click2future.training.trajectory_data import replay_angles
from click2future.training.world_data import load_cache, prepare_cache


@pytest.fixture(scope="module")
def paired(corpus, tmp_path_factory):
    parent, _, _ = corpus
    root = tmp_path_factory.mktemp("coverage") / "paired"
    recipe = {"seed": 920000, "pairs": 2, "directions": 4, "magnitudes": [0.25]}
    record = module.generate(parent, root, recipe)
    return parent, root, record


def test_pairs_match_actions_physics_and_preserve_all_visible_profiles(paired):
    parent, root, record = paired
    assert module.validate(parent, root) == record
    assert record["sample_count_per_arm"] == 68
    visible = {}
    for row in record["samples"]:
        clips = [load_rollout(root / arm, row["sample_id"]) for arm in module.ARMS]
        assert [o["hidden_physics"] for o in clips[0]["scene_spec"]["objects"]] == [
            o["hidden_physics"] for o in clips[1]["scene_spec"]["objects"]
        ]
        np.testing.assert_array_equal(clips[0]["force_xy"], clips[1]["force_xy"])
        assert clips[0]["selected_object_id"] == clips[1]["selected_object_id"]
        np.testing.assert_array_equal(
            clips[0]["frames"][0], load_rollout(parent, row["parent_sample_id"])["frames"][0]
        )
        for arm, clip in zip(module.ARMS, clips, strict=True):
            image = clip["frames"][0]
            key = (arm, row["pair"])
            if key in visible:
                np.testing.assert_array_equal(image, visible[key])
            visible[key] = image
        assert not np.array_equal(clips[0]["frames"][0], clips[1]["frames"][0])
    assert (
        len(
            {
                hashlib.sha256(image.tobytes()).hexdigest()
                for (arm, _), image in visible.items()
                if arm == "broad"
            }
        )
        == 2
    )
    assert replay_angles(clips[1]).shape == (32, 2)


def test_only_parent_training_templates_are_opened(paired, tmp_path, monkeypatch):
    parent, _, record = paired
    original = module.source_rollout
    calls = []

    def checked(root, row):
        assert Path(root) == parent and row["split"] == "train" and row["physics_variant"] == 0
        calls.append(row["sample_id"])
        return original(root, row)

    monkeypatch.setattr(module, "source_rollout", checked)
    recipe = {**record["recipe"], "pairs": 1, "directions": 2}
    module.generate(parent, tmp_path / "new", recipe)
    module.validate(parent, tmp_path / "new")
    assert calls


@pytest.mark.parametrize("kind", ["membership", "coverage", "sampling", "binding", "bytes"])
def test_mutations_fail_even_with_rehashed_manifest(paired, tmp_path, kind):
    parent, root, _ = paired
    target = tmp_path / "changed"
    shutil.copytree(root, target)
    record = module.read(target / "coverage.json")
    if kind == "membership":
        record["samples"][0]["split"] = "validation"
    elif kind == "coverage":
        record["samples"][-1] = copy.deepcopy(record["samples"][0])
    elif kind == "sampling":
        record["samples"][0]["sampling_contact"] = not record["samples"][0]["sampling_contact"]
    elif kind == "binding":
        record["parent_manifest_sha256"] = "0" * 64
    else:
        path = target / "broad" / (record["samples"][0]["sample_id"] + ".npz")
        path.write_bytes(path.read_bytes() + b"changed")
    record["manifest_hash"] = _digest({k: v for k, v in record.items() if k != "manifest_hash"})
    module.write(target / "coverage.json", record)
    with pytest.raises(ValueError):
        module.validate(parent, target)


def test_append_preserves_validation_and_pair_sampling(corpus, paired, tmp_path):
    parent, extra, _ = corpus
    _, coverage, _ = paired
    base = tmp_path / "base"
    prepare_cache(parent, extra, base)
    original, geometry, worlds, _ = load_cache(base)
    caches = {}
    for arm in module.ARMS:
        target = tmp_path / arm
        module.append_cache(parent, base, coverage, target, arm)
        data, shapes, labels, identity = load_cache(target)
        for key in ("motion", "force", "features"):
            np.testing.assert_array_equal(data["validation"][key], original["validation"][key])
            np.testing.assert_array_equal(
                data["train"][key][: len(original["train"][key])], original["train"][key]
            )
        np.testing.assert_array_equal(labels["validation"], worlds["validation"])
        for key in geometry["validation"]:
            np.testing.assert_array_equal(shapes["validation"][key], geometry["validation"][key])
        assert identity["coverage_addition"]["examples_added"] == 68 * 8
        caches[arm] = data
    np.testing.assert_array_equal(
        caches["narrow"]["train"]["force"], caches["broad"]["train"]["force"]
    )
    assert [r.get("contact") for r in caches["narrow"]["train"]["examples"]] == [
        r.get("contact") for r in caches["broad"]["train"]["examples"]
    ]


def test_overlapping_visible_seed_is_rejected(paired, tmp_path):
    parent, _, record = paired
    recipe = {**record["recipe"], "seed": module.read(parent / "dataset_manifest.json")["seed"]}
    with pytest.raises(ValueError, match="overlap"):
        module.generate(parent, tmp_path / "leak", recipe)


def test_recipe_roundoff_is_limited_to_derived_vectors_and_does_not_relax_labels():
    actual = {
        "action": {"force_xy": [0.1767766952966369, 0.0]},
        "objects": [{"position": [0.5, 0.45], "hidden_physics": {"mass": 1.0}}],
    }
    expected = copy.deepcopy(actual)
    expected["action"]["force_xy"][0] = float(np.nextafter(actual["action"]["force_xy"][0], 0))
    expected["objects"][0]["position"][0] = float(np.nextafter(0.5, 1))
    assert module.scene_recipe_equal(actual, expected)
    for value in (0.5 + 1e-12, float("nan"), True):
        changed = copy.deepcopy(actual)
        changed["objects"][0]["position"][0] = value
        assert not module.scene_recipe_equal(changed, actual)
    changed = copy.deepcopy(actual)
    changed["objects"][0]["hidden_physics"]["mass"] = float(np.nextafter(1.0, 2))
    assert not module.scene_recipe_equal(changed, actual)
