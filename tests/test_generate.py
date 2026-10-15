import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from click2future.data import generate_dataset, load_rollout


class DatasetGenerationTests(unittest.TestCase):
    def test_variants_share_visible_condition_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = generate_dataset(
                directory,
                group_count=2,
                variants_per_group=2,
                seed=70,
            )
            manifest = result["manifest"]
            self.assertEqual(manifest["sample_count"], 4)
            self.assertTrue(Path(result["manifest_path"]).exists())
            self.assertTrue(Path(result["split_path"]).exists())

            by_group: dict[str, list[dict[str, object]]] = {}
            for record in manifest["samples"]:
                by_group.setdefault(record["scene_group_id"], []).append(record)

            for records in by_group.values():
                self.assertEqual(len({record["split"] for record in records}), 1)
                rollouts = []
                hidden_physics = []
                for record in records:
                    rollout = load_rollout(directory, record["sample_id"])
                    rollouts.append(rollout)
                    hidden_physics.append(
                        [obj["hidden_physics"] for obj in rollout["scene_spec"]["objects"]]
                    )
                first, second = rollouts
                np.testing.assert_array_equal(first["frames"][0], second["frames"][0])
                np.testing.assert_array_equal(first["selected_mask"], second["selected_mask"])
                np.testing.assert_array_equal(first["force_xy"], second["force_xy"])
                np.testing.assert_array_equal(first["action_map"], second["action_map"])
                self.assertNotEqual(hidden_physics[0], hidden_physics[1])

            saved_manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["manifest_hash"], manifest["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
