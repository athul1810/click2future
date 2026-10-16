import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from click2future.data import load_rollout, save_rollout, validate_rollout
from click2future.sim import make_scene, run_rollout


class DataTests(unittest.TestCase):
    def test_rollout_round_trip(self) -> None:
        rollout = run_rollout(make_scene(51))
        with tempfile.TemporaryDirectory() as directory:
            paths = save_rollout(rollout, directory, "smoke_0001")
            loaded = load_rollout(directory, "smoke_0001")

            self.assertTrue(Path(paths["arrays"]).exists())
            self.assertTrue(Path(paths["metadata"]).exists())
            for field in (
                "frames",
                "instance_masks",
                "positions",
                "timestamps",
                "collision_onset",
                "wall_contact_onset",
            ):
                np.testing.assert_array_equal(rollout[field], loaded[field])
            self.assertEqual(rollout["scene_spec"], loaded["scene_spec"])

    def test_validator_rejects_wrong_mask_shape(self) -> None:
        rollout = run_rollout(make_scene(52))
        rollout["selected_mask"] = np.zeros((8, 8), dtype=bool)
        with self.assertRaises(ValueError):
            validate_rollout(rollout)

    def test_validator_rejects_non_finite_values(self) -> None:
        rollout = run_rollout(make_scene(53))
        corruptions = (
            ("positions", (1, 0, 0)),
            ("timestamps", (1,)),
            ("force_xy", (0,)),
            ("action_map", (0, 0, 0)),
        )
        for field, index in corruptions:
            with self.subTest(field=field):
                corrupted = copy.deepcopy(rollout)
                corrupted[field][index] = np.nan
                with self.assertRaises(ValueError):
                    validate_rollout(corrupted)

        corrupted = copy.deepcopy(rollout)
        corrupted["force_duration"] = float("nan")
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

    def test_validator_rejects_invalid_timestamps_and_positions(self) -> None:
        rollout = run_rollout(make_scene(54))

        corrupted = copy.deepcopy(rollout)
        corrupted["timestamps"][0] = 1.0
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

        corrupted = copy.deepcopy(rollout)
        corrupted["timestamps"][1] = corrupted["timestamps"][0]
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

        corrupted = copy.deepcopy(rollout)
        corrupted["positions"][1, 0, 0] = 1.1
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

    def test_validator_rejects_invalid_collision_labels(self) -> None:
        rollout = run_rollout(make_scene(55))

        corrupted = copy.deepcopy(rollout)
        corrupted["collision_onset"][0, 0, 0] = True
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

        corrupted = copy.deepcopy(rollout)
        corrupted["collision_onset"][0, 0, 1] = True
        corrupted["collision_onset"][0, 1, 0] = False
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

    def test_validator_rejects_invalid_selected_slot_and_action_map(self) -> None:
        rollout = run_rollout(make_scene(56))
        selected_object_id = int(rollout["selected_object_id"])

        corrupted = copy.deepcopy(rollout)
        corrupted["object_valid"][selected_object_id] = False
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

        corrupted = copy.deepcopy(rollout)
        corrupted["action_map"].fill(123.0)
        with self.assertRaises(ValueError):
            validate_rollout(corrupted)

    def test_validator_rejects_non_unit_world(self) -> None:
        rollout = run_rollout(make_scene(57))
        rollout["scene_spec"]["world_size"] = [2.0, 2.0]

        with self.assertRaises(ValueError):
            validate_rollout(rollout)


if __name__ == "__main__":
    unittest.main()
