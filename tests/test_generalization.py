"""Legacy-preserving textures, physical triangles, and disjoint OOD datasets."""

import copy
import json
import math

import numpy as np
import pytest

from click2future.cli import main
from click2future.data import (
    generate_generalization_dataset,
    load_rollout,
    load_split_manifest,
    validate_generalization_dataset,
)
from click2future.sim import build_space, make_scene, render_frame, run_rollout, snapshot
from click2future.sim.generalization import (
    SPLITS,
    make_generalization_scene,
    make_textured_scene,
    texture_scene,
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("generalization")
    result = generate_generalization_dataset(root, seed=201000)
    return root, result["manifest"]


@pytest.mark.parametrize("objects", [2, 3, 4])
def test_random_textures_preserve_masks_physics_and_initial_condition(objects):
    first = make_textured_scene(33, object_count=objects, physics_seed=1)
    other = make_textured_scene(33, object_count=objects, physics_seed=2)
    assert [o["appearance"] for o in first["objects"]] == [
        o["appearance"] for o in other["objects"]
    ]
    plain = copy.deepcopy(first)
    for obj in plain["objects"]:
        obj["appearance"]["texture"] = "solid"
    state = snapshot(build_space(first))
    rgb, masks = render_frame(first, state)
    plain_rgb, plain_masks = render_frame(plain, state)
    np.testing.assert_array_equal(masks, plain_masks)
    np.testing.assert_array_equal(rgb[~masks.any(0)], plain_rgb[~masks.any(0)])
    assert not np.array_equal(rgb, plain_rgb)
    first["action"]["force_xy"] = [0.0, 0.0]
    rollout = run_rollout(first, frame_count=4)
    np.testing.assert_array_equal(rollout["frames"], np.repeat(rgb[None], 4, axis=0))


def test_stripes_rotate_with_object_instead_of_staying_on_the_table():
    scene = make_scene(55)
    obj = scene["objects"][0]
    obj.update(
        shape="rectangle", geometry={"width": 0.25, "height": 0.25}, position=[0.5, 0.5], angle=0
    )
    obj["appearance"].update(texture="stripes", color=[100, 80, 60])
    scene["objects"][1]["position"] = [0.85, 0.85]
    original, _ = render_frame(scene, snapshot(build_space(scene)))
    obj["angle"] = math.pi / 2
    rotated, _ = render_frame(scene, snapshot(build_space(scene)))
    a, b = original[26:38, 26:38], rotated[26:38, 26:38]
    assert np.all(a == a[:1]) and not np.all(a == a[:, :1])
    assert np.all(b == b[:, :1]) and not np.all(b == b[:1])


def test_triangle_geometry_participates_in_physics_and_collision_labels():
    scene = make_generalization_scene(
        50, split="shape_ood", action_variant="forward", physics_seed=1
    )
    for index, obj in enumerate(scene["objects"]):
        obj.update(position=[0.3 + index * 0.25, 0.5], angle=0, geometry={"radius": 0.07})
        obj["hidden_physics"].update(mass=1.0, restitution=0.2)
    scene["action"]["force_xy"] = [1.1, 0.0]
    rollout = run_rollout(scene, frame_count=32)
    assert rollout["collision_onset"].any()
    assert rollout["positions"][-1, 1, 0] > rollout["positions"][0, 1, 0]
    assert rollout["metadata"]["sim_version"] == "sim-0.2"
    assert rollout["metadata"]["render_version"] == "render-0.2"


def test_all_splits_are_disjoint_and_each_condition_has_identical_observables(dataset):
    root, manifest = dataset
    assert manifest["schema_version"] == "dataset-0.3" and manifest["sample_count"] == 166
    splits = load_split_manifest(root / "splits.json")["splits"]
    assert set(splits) == set(SPLITS)
    groups = [group for values in splits.values() for group in values]
    assert len(groups) == len(set(groups)) == 7
    initial, conditions = {}, {}
    for row in manifest["samples"]:
        assert row["scene_group_id"] in splits[row["split"]]
        clip = load_rollout(root, row["sample_id"])
        frame = clip["frames"][0]
        group, condition = row["scene_group_id"], row["condition_id"]
        if group in initial:
            np.testing.assert_array_equal(initial[group], frame)
        initial[group] = frame
        observable = (clip["selected_mask"], clip["force_xy"])
        if condition in conditions:
            for expected, actual in zip(conditions[condition], observable, strict=True):
                np.testing.assert_array_equal(expected, actual)
        conditions[condition] = observable
    counts = {}
    for row in manifest["samples"]:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    assert counts == dict.fromkeys(SPLITS[:5], 21) | {"force_ood": 20, "physics_test": 41}


@pytest.mark.parametrize("split", ["shape_ood", "texture_ood", "force_ood"])
def test_only_the_declared_factor_changes(split):
    args = dict(action_variant="forward", physics_seed=100)
    baseline = make_generalization_scene(100, split="id_test", **args)
    shifted = make_generalization_scene(100, split=split, **args)
    restored = copy.deepcopy(shifted)
    if split == "shape_ood":
        assert all(o["shape"] == "triangle" for o in shifted["objects"])
        for obj, old in zip(restored["objects"], baseline["objects"], strict=True):
            obj.update(shape=old["shape"], geometry=old["geometry"])
    elif split == "texture_ood":
        assert all(o["appearance"]["texture"] == "checker" for o in shifted["objects"])
        for obj, old in zip(restored["objects"], baseline["objects"], strict=True):
            obj["appearance"] = old["appearance"]
    else:
        assert np.linalg.norm(shifted["action"]["force_xy"]) > 1.1
        restored["action"] = baseline["action"]
    assert restored == baseline


def test_complete_native_data_replays_exactly_and_cli_reports_validation(dataset, capsys):
    root, _ = dataset
    report = validate_generalization_dataset(root, replay=True)
    assert report["ok"] and report["replayed_samples"] == 166
    assert main(["validate-generalization-dataset", "--input-dir", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]


def test_training_index_excludes_all_held_out_splits_and_ood_force_is_not_clipped(dataset):
    from click2future.data import load_batch
    from click2future.training.conditioning import training_inputs
    from click2future.training.train_diffusion import training_index

    root, manifest = dataset
    selected, _ = training_index(root, None)
    assert set(selected) == {r["sample_id"] for r in manifest["samples"] if r["split"] == "train"}
    name = next(r["sample_id"] for r in manifest["samples"] if r["split"] == "force_ood")
    batch = load_batch(root, [name])
    _, condition = training_inputs(batch, "cpu", force_limit=1.5)
    np.testing.assert_allclose(condition.force_xy, batch["force_xy"] / 0.5)
    assert condition.force_xy.norm().item() == pytest.approx(3.0)


def test_frozen_data_is_not_overwritten_and_corrupted_bytes_fail(dataset, tmp_path):
    root, manifest = dataset
    with pytest.raises(ValueError, match="never overwritten"):
        generate_generalization_dataset(root)
    # Copy just the manifest files: missing or changed clips cannot pass validation.
    for name in ("dataset_manifest.json", "splits.json"):
        (tmp_path / name).write_bytes((root / name).read_bytes())
    first = manifest["samples"][0]
    (tmp_path / first["arrays"]).write_bytes(b"corrupted")
    result = validate_generalization_dataset(tmp_path)
    assert not result["ok"] and "hash changed" in result["errors"][0]


@pytest.mark.parametrize("bad", [0, -1, True])
def test_invalid_group_counts_fail_before_output_is_created(tmp_path, bad):
    root = tmp_path / "missing"
    counts = dict.fromkeys(SPLITS, 1) | {"train": bad}
    with pytest.raises(ValueError, match="group count"):
        generate_generalization_dataset(root, group_counts=counts)
    assert not root.exists()


def test_unknown_texture_is_rejected():
    with pytest.raises(ValueError, match="textures"):
        texture_scene(make_scene(0), textures=("unknown",))
