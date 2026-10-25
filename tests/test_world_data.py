"""Privileged physical labels follow object identity under every square transform."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from click2future.models.object_motion import observe
from click2future.sim import run_rollout
from click2future.training import contact_geometry, world_data


def test_world_labels_follow_selected_identity_and_geometry_matches_legacy(monkeypatch):
    scene = json.loads(Path("tests/fixtures/tiny_scene.json").read_text())
    references = {}
    for selected in (0, 1):
        current = copy.deepcopy(scene)
        current["action"]["selected_object_id"] = selected
        references[str(selected)] = run_rollout(current, frame_count=32, fps=8, resolution=64)
    monkeypatch.setattr(world_data, "load_rollout", lambda root, name: references[name])
    monkeypatch.setattr(contact_geometry, "load_rollout", lambda root, name: references[name])
    examples, features, expected = [], [], []
    for selected, ref in references.items():
        for turns in range(4):
            for mirror in (False, True):
                image, mask = (
                    np.rot90(ref["frames"][0], turns),
                    np.rot90(ref["selected_mask"], turns),
                )
                if mirror:
                    image, mask = np.fliplr(image), np.fliplr(mask)
                obs = observe(image, mask)
                order = (
                    [int(selected), 1 - int(selected)]
                    if obs.selected_index == 0
                    else [1 - int(selected), int(selected)]
                )
                expected.append(
                    [
                        [
                            scene["objects"][i]["hidden_physics"][name]
                            for name in ("mass", "friction", "restitution")
                        ]
                        for i in order
                    ]
                )
                features.append(obs.features)
                examples.append({"sample_id": selected, "quarter_turns": turns, "mirror": mirror})
    data = {"train": {"features": np.stack(features), "examples": examples}}
    geometry, worlds = world_data.world_geometry("unused", data)
    np.testing.assert_array_equal(worlds["train"], np.asarray(expected, np.float32))
    old = contact_geometry.prepare_contact_geometry("unused", data)
    for key, value in old["train"].items():
        np.testing.assert_array_equal(value, geometry["train"][key])
    data["test"] = data["train"]
    with pytest.raises(ValueError, match="train/validation"):
        world_data.world_geometry("unused", data)
