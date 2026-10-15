import unittest

from click2future.data import assign_split, build_split_manifest, validate_split_manifest


class SplitTests(unittest.TestCase):
    def test_split_manifest_is_deterministic_and_disjoint(self) -> None:
        groups = [f"scene-{index:03d}" for index in range(10)]
        first = build_split_manifest(groups, seed=7)
        second = build_split_manifest(groups, seed=7)

        self.assertEqual(first, second)
        validate_split_manifest(first)
        flattened = [group for values in first["splits"].values() for group in values]
        self.assertEqual(sorted(flattened), sorted(groups))

    def test_assign_split_uses_scene_group_id(self) -> None:
        manifest = build_split_manifest(["scene-a", "scene-b", "scene-c"], seed=2)
        for group in ("scene-a", "scene-b", "scene-c"):
            self.assertIn(assign_split(group, manifest), manifest["splits"])

    def test_small_manifest_uses_each_split_when_possible(self) -> None:
        manifest = build_split_manifest(["scene-a", "scene-b", "scene-c"], seed=2)
        self.assertEqual(
            sorted(len(groups) for groups in manifest["splits"].values()),
            [1, 1, 1],
        )

    def test_duplicate_scene_group_is_rejected(self) -> None:
        manifest = build_split_manifest(["scene-a", "scene-b"], seed=3)
        manifest["splits"]["validation"].append(manifest["splits"]["train"][0])
        with self.assertRaises(ValueError):
            validate_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
