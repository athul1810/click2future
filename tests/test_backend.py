import unittest

from click2future.sim import (
    apply_force,
    build_space,
    consume_step_contacts,
    make_scene,
    snapshot,
    step,
)


class BackendTests(unittest.TestCase):
    def test_initial_snapshot_matches_scene(self) -> None:
        scene = make_scene(4)
        state = build_space(scene)
        actual = snapshot(state)

        for expected, observed in zip(scene["objects"], actual["objects"]):
            self.assertEqual(expected["object_id"], observed["object_id"])
            self.assertAlmostEqual(expected["position"][0], observed["position"][0])
            self.assertAlmostEqual(expected["position"][1], observed["position"][1])
            self.assertAlmostEqual(expected["angle"], observed["angle"])

    def test_force_moves_selected_object_in_force_direction(self) -> None:
        scene = make_scene(8, action_seed=123)
        state = build_space(scene)
        selected = scene["action"]["selected_object_id"]
        force = tuple(scene["action"]["force_xy"])
        before = snapshot(state)["objects"][selected]["position"]

        for _ in range(60):
            apply_force(state, selected, force)
            step(state)

        after = snapshot(state)["objects"][selected]["position"]
        displacement = (after[0] - before[0], after[1] - before[1])
        dot = displacement[0] * force[0] + displacement[1] * force[1]
        self.assertGreater(dot, 0.0)

    def test_zero_force_keeps_initially_stationary_objects_still(self) -> None:
        scene = make_scene(12)
        state = build_space(scene)
        before = snapshot(state)

        for _ in range(60):
            step(state)

        after = snapshot(state)
        for expected, observed in zip(before["objects"], after["objects"]):
            self.assertAlmostEqual(expected["position"][0], observed["position"][0], places=6)
            self.assertAlmostEqual(expected["position"][1], observed["position"][1], places=6)

    def test_contact_events_are_available(self) -> None:
        scene = make_scene(20)
        state = build_space(scene)
        first, second = sorted(state["bodies"])
        state["bodies"][first].position = (0.4, 0.5)
        state["bodies"][second].position = (0.6, 0.5)
        state["bodies"][first].velocity = (1.0, 0.0)
        state["bodies"][second].velocity = (-1.0, 0.0)
        state["space"].reindex_shapes_for_body(state["bodies"][first])
        state["space"].reindex_shapes_for_body(state["bodies"][second])

        for _ in range(30):
            step(state)
            events = consume_step_contacts(state)
            if events["object_object"]:
                break
        else:
            self.fail("expected an object-object contact event")


if __name__ == "__main__":
    unittest.main()
