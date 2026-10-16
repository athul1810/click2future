"""Public simulator and data imports must not depend on initialization order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("first, second", [("sim", "data"), ("data", "sim")])
def test_public_import_order(first, second):
    result = subprocess.run(
        [sys.executable, "-c", f"import click2future.{first}; import click2future.{second}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
