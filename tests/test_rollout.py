import copy
import json
import unittest
from pathlib import Path

import numpy as np

from click2future.sim import make_scene, render_rgb, run_rollout


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_scene.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class RolloutTests(unittest.TestCase):
    def test_rollout_shapes_match_contract(self) -> None:
        scene = make_scene(41, object_count=3)
        rollout = run_rollout(scene)

        self.assertEqual(rollout["frames"].shape, (16, 64, 64, 3))
        self.assertEqual(rollout["instance_masks"].shape, (16, 3, 64, 64))
        self.assertEqual(rollout["positions"].shape, (16, 3, 2))
        self.assertEqual(rollout["timestamps"].shape, (16,))
        self.assertEqual(rollout["collision_onset"].shape, (15, 3, 3))
        self.assertEqual(rollout["wall_contact_onset"].shape, (15, 3, 4))
        self.assertEqual(rollout["action_map"].shape, (2, 64, 64))
        self.assertEqual(rollout["frames"].dtype, np.uint8)
        self.assertEqual(rollout["instance_masks"].dtype, np.bool_)

    def test_first_frame_is_pre_action_render(self) -> None:
        scene = make_scene(42)
        rollout = run_rollout(scene)
        expected = render_rgb(
            scene,
            {
                "objects": [
                    {
                        "object_id": obj["object_id"],
                        "position": obj["position"],
                        "angle": obj["angle"],
                    }
                    for obj in scene["objects"]
                ]
            },
        )
        np.testing.assert_array_equal(rollout["frames"][0], expected)

    def test_zero_force_produces_stationary_positions(self) -> None:
        scene = make_scene(43)
        scene["action"]["force_xy"] = [0.0, 0.0]
        rollout = run_rollout(scene)
        expected = np.broadcast_to(rollout["positions"][0], rollout["positions"].shape)
        np.testing.assert_allclose(rollout["positions"], expected, atol=1e-6)

    def test_repeated_rollout_is_identical(self) -> None:
        scene = make_scene(44)
        first = run_rollout(scene)
        second = run_rollout(copy.deepcopy(scene))
        for field in ("frames", "instance_masks", "positions", "collision_onset", "wall_contact_onset"):
            np.testing.assert_array_equal(first[field], second[field])

    def test_reversing_force_reverses_initial_displacement(self) -> None:
        forward_scene = _load_fixture()
        reverse_scene = copy.deepcopy(forward_scene)
        reverse_scene["action"]["force_xy"] = [-0.3, 0.0]

        forward = run_rollout(forward_scene, frame_count=4)
        reverse = run_rollout(reverse_scene, frame_count=4)
        selected = int(forward["selected_object_id"])
        forward_dx = forward["positions"][-1, selected, 0] - forward["positions"][0, selected, 0]
        reverse_dx = reverse["positions"][-1, selected, 0] - reverse["positions"][0, selected, 0]

        self.assertGreater(float(forward_dx), 0.0)
        self.assertLess(float(reverse_dx), 0.0)

    def test_known_scene_records_right_wall_contact(self) -> None:
        scene = _load_fixture()
        scene["objects"][0]["position"] = [0.9, 0.5]
        scene["objects"][1]["position"] = [0.3, 0.75]
        scene["action"]["force_xy"] = [1.0, 0.0]

        rollout = run_rollout(scene, frame_count=16)

        self.assertTrue(np.any(rollout["wall_contact_onset"][:, 0, 1]))


if __name__ == "__main__":
    unittest.main()
