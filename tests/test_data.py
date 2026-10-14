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


if __name__ == "__main__":
    unittest.main()
