"""Fetch the published inference artifacts into an isolated external checkout."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    import gdown
    from huggingface_hub import HfApi, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    model_id = "Wan-AI/Wan2.2-TI2V-5B"
    info = HfApi().model_info(model_id)
    base = root / "wan_models/Wan2.2-TI2V-5B"
    # The released student contains the full generator. Its strict state loader
    # avoids loading base diffusion weights that would immediately be overwritten.
    snapshot_download(
        model_id,
        revision=info.sha,
        local_dir=base,
        allow_patterns=[
            "config.json",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2_VAE.pth",
            "google/umt5-xxl/*",
        ],
        max_workers=2,
    )
    checkpoint = root / "checkpoints/streamforce_v1_ckpt.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.is_file():
        downloaded = gdown.download(
            id="1ZaOEVMzOxAtcadX2c8hBF6wA7BrtUTOi",
            output=str(checkpoint),
            quiet=False,
            resume=True,
            use_cookies=False,
        )
        if downloaded is None:
            raise RuntimeError("The published student checkpoint could not be downloaded")
    paths = [path for path in base.rglob("*") if path.is_file() and ".cache" not in path.parts] + [
        checkpoint
    ]
    record = {
        "source": "published StreamForce student; external pretrained model",
        "base_model": model_id,
        "base_revision": info.sha,
        "checkpoint_reference": "official repository Google Drive release",
        "files": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(paths)
        },
        "scope": "download identities; not an inference or quality result",
    }
    (root / "ARTIFACTS.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"downloaded_files": len(paths), "base_revision": info.sha}))


if __name__ == "__main__":
    main()
