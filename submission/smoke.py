"""Check extracted source with temporary generated data and no saved model."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from click2future.data import generate_dataset, load_batch
    from click2future.sim import replay_saved_rollout

    with tempfile.TemporaryDirectory(prefix="click2future-source-check-") as temporary:
        result = generate_dataset(
            temporary,
            group_count=1,
            variants_per_group=2,
            seed=91000,
            object_count=2,
            config={"frame_count": 4, "resolution": 64, "fps": 8, "physics_hz": 240},
        )
        sample_ids = [item["sample_id"] for item in result["manifest"]["samples"]]
        for sample_id in sample_ids:
            report = replay_saved_rollout(temporary, sample_id, atol=1e-6, rtol=1e-6)
            if not report["ok"]:
                raise RuntimeError(f"Replay failed: {sample_id}")
        batch = load_batch(temporary, sample_ids)
        if batch["frames"].shape != (2, 4, 3, 64, 64):
            raise RuntimeError("Extracted loader returned the wrong batch shape")
    for name, module in tuple(sys.modules.items()):
        if name.startswith("click2future") and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT / "src"):
                raise RuntimeError(f"Project import came from outside the archive: {name}")
    print(
        json.dumps(
            {
                "status": "passed",
                "generated_rollouts": 2,
                "replay": "passed",
                "batch_shape": list(batch["frames"].shape),
                "source_root": str(ROOT),
            }
        )
    )


if __name__ == "__main__":
    main()
