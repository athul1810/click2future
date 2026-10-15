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


if __name__ == "__main__":
    unittest.main()
