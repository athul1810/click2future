import tempfile
import unittest

import numpy as np

from click2future.data import load_batch, load_sample, save_rollout
from click2future.sim import make_scene, run_rollout


class LoaderTests(unittest.TestCase):
    def test_sample_loader_uses_normalized_channel_first_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_rollout(run_rollout(make_scene(81)), directory, "sample_a")
            sample = load_sample(directory, "sample_a")

        self.assertEqual(sample["frames"].shape, (16, 3, 64, 64))
        self.assertEqual(sample["frames"].dtype, np.float32)
        self.assertGreaterEqual(float(sample["frames"].min()), 0.0)
        self.assertLessEqual(float(sample["frames"].max()), 1.0)
        self.assertEqual(sample["selected_mask"].shape, (1, 64, 64))

    def test_batch_loader_pads_object_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_rollout(run_rollout(make_scene(82, object_count=2)), directory, "sample_a")
            save_rollout(run_rollout(make_scene(83, object_count=3)), directory, "sample_b")
            batch = load_batch(directory, ["sample_a", "sample_b"])

        self.assertEqual(batch["frames"].shape, (2, 16, 3, 64, 64))
        self.assertEqual(batch["instance_masks"].shape, (2, 16, 3, 64, 64))
        self.assertEqual(batch["positions"].shape, (2, 16, 3, 2))
        np.testing.assert_array_equal(batch["object_valid"][0], [True, True, False])
        np.testing.assert_array_equal(batch["positions"][0, :, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
