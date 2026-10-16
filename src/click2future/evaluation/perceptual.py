"""Frozen, calibrated LPIPS-AlexNet with explicit weight availability and provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LPIPSScorer:
    """LPIPS 0.1 with ImageNet AlexNet and learned calibration, never random features.

    Weights stay in the normal Torch cache and installed LPIPS package, outside Git.
    Offline evaluation is the default. A caller must explicitly allow the first download.
    """

    def __init__(self, *, device: str = "cpu", allow_download: bool = False):
        try:
            import lpips
            import torch
            from torchvision.models import AlexNet_Weights
        except ImportError as error:
            raise RuntimeError(
                "Install the image-metrics extra with the cpu or cu128 extra"
            ) from error
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Choose cpu or an available cuda device")
        weights = AlexNet_Weights.IMAGENET1K_V1
        filename = Path(urlparse(weights.url).path).name
        backbone_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not backbone_path.is_file():
            if not allow_download:
                raise RuntimeError(
                    "LPIPS AlexNet weights are not cached; use --download-weights once"
                )
            weights.get_state_dict(progress=False, check_hash=True)
        backbone_hash = _sha256(backbone_path)
        expected_prefix = filename.rsplit("-", 1)[1].split(".")[0]
        if not backbone_hash.startswith(expected_prefix):
            raise ValueError("Cached AlexNet weights failed the official checksum prefix")
        calibration_path = Path(lpips.__file__).parent / "weights" / "v0.1" / "alex.pth"
        calibration_hash = _sha256(calibration_path)
        # Constructing layers should not advance the experiment's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            self._model = lpips.LPIPS(
                net="alex",
                version="0.1",
                pretrained=True,
                pnet_rand=False,
                pnet_tune=False,
                spatial=False,
                eval_mode=True,
                model_path=str(calibration_path),
                verbose=False,
            )
        self._model = self._model.float().eval().requires_grad_(False).to(device)
        self.device = device
        self.metadata = {
            "implementation": "lpips.LPIPS",
            "package_version": importlib.metadata.version("lpips"),
            "torch_version": torch.__version__,
            "torchvision_version": importlib.metadata.version("torchvision"),
            "backbone": "alex",
            "backbone_weights": "AlexNet_Weights.IMAGENET1K_V1",
            "calibration_version": "0.1",
            "backbone_sha256": backbone_hash,
            "calibration_sha256": calibration_hash,
            "input": "uint8 RGB -> float32 NCHW / 127.5 - 1; normalize=False",
            "resize": "none",
            "spatial": False,
            "device": device,
            "mode": "eval; frozen parameters; inference_mode",
        }

    def score(self, predicted, reference) -> np.ndarray:
        import torch

        predicted, reference = np.asarray(predicted), np.asarray(reference)
        if (
            predicted.dtype != np.uint8
            or reference.dtype != np.uint8
            or predicted.ndim != 4
            or predicted.shape != reference.shape
            or predicted.shape[-1] != 3
            or len(predicted) < 1
            or min(predicted.shape[1:3]) < 64
        ):
            raise ValueError("LPIPS batches must be matching uint8 [B,H,W,3], with H,W >= 64")

        def tensor(array):
            channels_first = np.ascontiguousarray(array.transpose(0, 3, 1, 2))
            return torch.from_numpy(channels_first).to(self.device, dtype=torch.float32) / 127.5 - 1

        with torch.inference_mode():
            scores = self._model(tensor(predicted), tensor(reference), normalize=False)
        scores = scores.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if scores.shape != (len(predicted),) or not np.isfinite(scores).all():
            raise ValueError("LPIPS returned invalid scores")
        return scores
