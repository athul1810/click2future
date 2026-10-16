import tempfile
import unittest

import numpy as np

from click2future.data import (
    generate_controlled_dataset,
    load_rollout,
    validate_controlled_dataset,
)


class ControlledDatasetTests(unittest.TestCase):
    def test_generation_produces_the_controlled_scene_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate_controlled_dataset(
                directory,
                group_count=1,
                seed=30,
                config={"resolution": 32},
            )

            self.assertTrue(result["validation"]["ok"])
            self.assertEqual(result["manifest"]["sample_count"], 21)
            self.assertEqual(result["validation"]["object_collision_clips"], 8)
            self.assertEqual(result["validation"]["wall_contact_clips"], 4)

            records = {
                (record["action_variant"], record["physics_variant"]): record
                for record in result["manifest"]["samples"]
            }
            forward = load_rollout(directory, records[("forward", 2)]["sample_id"])
            reverse = load_rollout(directory, records[("reverse", 2)]["sample_id"])
            collision_0_to_1 = load_rollout(
                directory, records[("collision_0_to_1", 2)]["sample_id"]
            )
            collision_1_to_0 = load_rollout(
                directory, records[("collision_1_to_0", 2)]["sample_id"]
            )
            wall = load_rollout(directory, records[("wall", 2)]["sample_id"])
            np.testing.assert_array_equal(forward["frames"][0], reverse["frames"][0])
            np.testing.assert_array_equal(forward["force_xy"], -reverse["force_xy"])
            self.assertEqual(
                [obj["hidden_physics"] for obj in forward["scene_spec"]["objects"]],
                [obj["hidden_physics"] for obj in collision_0_to_1["scene_spec"]["objects"]],
            )
            self.assertTrue(collision_0_to_1["collision_onset"].any())
            self.assertTrue(collision_1_to_0["collision_onset"].any())
            self.assertEqual(collision_0_to_1["selected_object_id"], 0)
            self.assertEqual(collision_1_to_0["selected_object_id"], 1)
            self.assertTrue(wall["wall_contact_onset"].any())
            self.assertFalse(wall["collision_onset"].any())

    def test_validation_rejects_a_missing_required_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate_controlled_dataset(
                directory,
                group_count=1,
                seed=40,
                config={"resolution": 32},
            )
            record = next(
                row
                for row in result["manifest"]["samples"]
                if row["action_variant"] == "collision_0_to_1"
            )
            array_path = result["manifest_path"].parent / record["arrays"]
            with np.load(array_path, allow_pickle=False) as saved:
                arrays = {name: saved[name] for name in saved.files}
            arrays["collision_onset"] = np.zeros_like(arrays["collision_onset"])
            np.savez_compressed(array_path, **arrays)

            report = validate_controlled_dataset(directory)

            self.assertFalse(report["ok"])
            self.assertTrue(any("did not collide" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
