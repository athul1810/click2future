import unittest

import numpy as np

from click2future.sim import build_space, make_scene, render_frame, render_instance_masks, render_rgb, snapshot
from click2future.sim.render import DEFAULT_BORDER


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = make_scene(31, object_count=2)
        self.state = build_space(self.scene)
        self.current = snapshot(self.state)

    def test_rgb_shape_and_dtype(self) -> None:
        frame = render_rgb(self.scene, self.current, resolution=64)
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_masks_shape_and_dtype(self) -> None:
        masks = render_instance_masks(self.scene, self.current, resolution=64)
        self.assertEqual(masks.shape, (2, 64, 64))
        self.assertEqual(masks.dtype, np.bool_)
        self.assertTrue(np.all(masks.reshape(2, -1).sum(axis=1) > 0))

    def test_render_frame_is_reproducible(self) -> None:
        first_rgb, first_masks = render_frame(self.scene, self.current)
        second_rgb, second_masks = render_frame(self.scene, self.current)
        np.testing.assert_array_equal(first_rgb, second_rgb)
        np.testing.assert_array_equal(first_masks, second_masks)

    def test_rgb_frame_has_visible_table_border(self) -> None:
        frame = render_rgb(self.scene, self.current)
        self.assertTrue(np.all(frame[0, 0] == DEFAULT_BORDER))
        self.assertTrue(np.all(frame[-1, -1] == DEFAULT_BORDER))


if __name__ == "__main__":
    unittest.main()
