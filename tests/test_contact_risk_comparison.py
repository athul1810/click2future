"""The predeclared comparison cannot silently change other training settings."""

import json
import runpy
from pathlib import Path

import pytest


def test_matched_risk_recipes_reject_another_changed_factor(tmp_path):
    paired = runpy.run_path("submission/contact_risk_comparison.py")["paired_configs"]
    original = paired(Path.cwd())
    folder = tmp_path / "configs/training"
    folder.mkdir(parents=True)
    for path in original.values():
        (folder / path.name).write_bytes(path.read_bytes())
    paired(tmp_path)
    target = folder / "contact_risk_tail.json"
    value = json.loads(target.read_text())
    value["learning_rate"] *= 2
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="may differ only"):
        paired(tmp_path)
