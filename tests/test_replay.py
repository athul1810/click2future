import tempfile
import unittest

from click2future.data import save_rollout
from click2future.sim import (
    compare_rollouts,
    make_scene,
    replay_saved_rollout,
    run_rollout,
)


class ReplayTests(unittest.TestCase):
    def test_saved_rollout_replays_identically(self) -> None:
        rollout = run_rollout(make_scene(61))
        with tempfile.TemporaryDirectory() as directory:
            save_rollout(rollout, directory, "sample")
            report = replay_saved_rollout(directory, "sample")
        self.assertTrue(report["ok"], report)

    def test_changed_force_fails_comparison(self) -> None:
        scene = make_scene(62)
        expected = run_rollout(scene)
        scene["action"]["force_xy"][0] *= -1.0
        actual = run_rollout(scene)
        report = compare_rollouts(expected, actual)
        self.assertFalse(report["ok"])
        self.assertFalse(report["fields"]["frames"]["ok"])


if __name__ == "__main__":
    unittest.main()
