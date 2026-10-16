import json
import unittest

from click2future.sim import make_scene, visible_scene


class SceneTests(unittest.TestCase):
    def test_same_seed_is_reproducible(self) -> None:
        self.assertEqual(make_scene(7), make_scene(7))

    def test_hidden_physics_variants_share_visible_condition(self) -> None:
        first = make_scene(11, physics_seed=1, action_seed=99)
        second = make_scene(11, physics_seed=2, action_seed=99)

        self.assertEqual(visible_scene(first), visible_scene(second))
        self.assertNotEqual(
            [obj["hidden_physics"] for obj in first["objects"]],
            [obj["hidden_physics"] for obj in second["objects"]],
        )

    def test_scene_is_json_serializable(self) -> None:
        scene = make_scene(3)
        self.assertEqual(json.loads(json.dumps(scene)), scene)

    def test_object_count_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            make_scene(3, object_count=1)

    def test_non_unit_world_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_scene(3, world_size=(2.0, 2.0))

    def test_objects_are_inside_world(self) -> None:
        scene = make_scene(5, object_count=4)
        for obj in scene["objects"]:
            x, y = obj["position"]
            self.assertGreater(x, 0.0)
            self.assertLess(x, 1.0)
            self.assertGreater(y, 0.0)
            self.assertLess(y, 1.0)


if __name__ == "__main__":
    unittest.main()
