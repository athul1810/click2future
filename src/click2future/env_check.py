"""Check installed dependencies, artifact IO, and CPU/CUDA tensor execution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path


def _tensor_check(torch, device: str, folder: Path) -> None:
    """Exercise forward/backward, an optimizer step, and checkpoint reload."""
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(2, 3, 16, 16, device=device)
    before = model.weight.detach().clone()
    loss = model(inputs).square().mean()
    loss.backward()
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Non-finite loss on {device}")
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Missing or non-finite gradients on {device}")
    optimizer.step()
    if torch.equal(before, model.weight):
        raise RuntimeError(f"Optimizer did not update weights on {device}")
    checkpoint = folder / f"{device}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored[name], rtol=0, atol=0)
    if device == "cuda":
        torch.cuda.synchronize()


def check_environment(device: str = "auto") -> dict:
    """Return a serializable report; propagate any failed check to the caller."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("This development environment requires Python 3.12")

    import numpy as np
    import torch
    import yaml
    from PIL import Image

    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable; check the PyTorch build and GPU driver")
    selected_device = "cuda" if device == "cuda" or (device == "auto" and cuda_available) else "cpu"
    checks = {}
    with tempfile.TemporaryDirectory(prefix="click2future-check-") as temporary:
        folder = Path(temporary)
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        Image.fromarray(pixels).save(folder / "image.png")
        with Image.open(folder / "image.png") as image:
            np.testing.assert_array_equal(np.asarray(image), pixels)
        checks["png_roundtrip"] = "passed"

        np.savez_compressed(folder / "sample.npz", frames=pixels)
        with np.load(folder / "sample.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["frames"], pixels)
        checks["npz_roundtrip"] = "passed"

        config = {"seed": 0, "artifacts": {"outputs": "outputs"}}
        if yaml.safe_load(yaml.safe_dump(config)) != config:
            raise RuntimeError("YAML roundtrip failed")
        checks["yaml_roundtrip"] = "passed"
        np.testing.assert_array_equal(torch.from_numpy(pixels).numpy(), pixels)
        checks["numpy_torch_bridge"] = "passed"

        _tensor_check(torch, "cpu", folder)
        checks["cpu_forward_backward_checkpoint"] = "passed"
        if selected_device == "cuda":
            _tensor_check(torch, "cuda", folder)
            checks["cuda_forward_backward_checkpoint"] = "passed"

    return {
        "status": "passed",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("click2future", "numpy", "pillow", "pyyaml", "torch")
        },
        "device": selected_device,
        "cuda_available": cuda_available,
        "torch_cuda_runtime": torch.version.cuda,
        "gpus": [
            {
                "name": torch.cuda.get_device_properties(index).name,
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path")
    args = parser.parse_args(argv)
    try:
        report = check_environment(args.device)
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
