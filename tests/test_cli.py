import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_generate_then_validate_with_replay(self) -> None:
        environment = os.environ.copy()
        source_path = str(REPOSITORY_ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else source_path + os.pathsep + existing_pythonpath
        )

        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "click2future.cli",
                    "generate-smoke",
                    "--output-dir",
                    directory,
                    "--count",
                    "1",
                    "--seed",
                    "91",
                    "--config",
                    "configs/simulator/smoke.toml",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "click2future.cli",
                    "validate-sample",
                    "--input-dir",
                    directory,
                    "--sample-id",
                    "sample_00000091",
                    "--replay",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        report = json.loads(completed.stdout)
        self.assertTrue(report["ok"])
        self.assertTrue(report["replay"]["ok"])

    def test_generate_and_validate_controlled_dataset(self) -> None:
        environment = os.environ.copy()
        source_path = str(REPOSITORY_ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else source_path + os.pathsep + existing_pythonpath
        )

        with tempfile.TemporaryDirectory() as directory:
            generated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "click2future.cli",
                    "generate-controlled-dataset",
                    "--output-dir",
                    directory,
                    "--group-count",
                    "1",
                    "--config",
                    "configs/simulator/dataset_0_2.toml",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            validated = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "click2future.cli",
                    "validate-controlled-dataset",
                    "--input-dir",
                    directory,
                    "--replay",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )

        generation_report = json.loads(generated.stdout)
        validation_report = json.loads(validated.stdout)
        self.assertEqual(generation_report["sample_count"], 21)
        self.assertTrue(validation_report["ok"])
        self.assertEqual(validation_report["replayed_samples"], 21)


if __name__ == "__main__":
    unittest.main()
