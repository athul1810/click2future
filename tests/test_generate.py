import json
import tempfile
import unittest
from pathlib import Path

from click2future.data import generate_dataset, load_rollout
from click2future.sim import visible_scene


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
                scenes = []
                hidden_physics = []
                for record in records:
                    rollout = load_rollout(directory, record["sample_id"])
                    scenes.append(visible_scene(rollout["scene_spec"]))
                    hidden_physics.append(
                        [obj["hidden_physics"] for obj in rollout["scene_spec"]["objects"]]
                    )
                self.assertEqual(scenes[0], scenes[1])
                self.assertNotEqual(hidden_physics[0], hidden_physics[1])

            saved_manifest = json.loads(
                Path(result["manifest_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["manifest_hash"], manifest["manifest_hash"])


if __name__ == "__main__":
    unittest.main()
