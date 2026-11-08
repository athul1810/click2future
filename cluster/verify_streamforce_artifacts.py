"""Verify staged official base files and the exact derived EMA upload."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path, *, git_blob=False):
    value = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    source = json.loads((root / "SOURCE.json").read_text())
    if source["revision"] != "57a15e95eca25bbbb32d6ac6322ba3e12fd45083":
        raise ValueError("Unexpected external source revision")
    files = {}
    for name, expected in source["files_sha256"].items():
        path = root / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"External source changed: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual}
    model_id = "Wan-AI/Wan2.2-TI2V-5B"
    info = HfApi().model_info(model_id, revision=args.base_revision, files_metadata=True)
    if info.sha != args.base_revision:
        raise ValueError("Base revision did not resolve exactly")
    official = {row.rfilename: row for row in info.siblings}
    base = root / "wan_models/Wan2.2-TI2V-5B"
    for required in ["config.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"]:
        if not (base / required).is_file():
            raise ValueError(f"Missing base file: {required}")
    if not (base / "google/umt5-xxl/tokenizer.json").is_file():
        raise ValueError("Missing tokenizer")
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        name = path.relative_to(base).as_posix()
        entry = official[name]
        sha256 = digest(path)
        if entry.lfs:
            matched = sha256 == entry.lfs.sha256 and path.stat().st_size == entry.lfs.size
        else:
            matched = digest(path, git_blob=True) == entry.blob_id
        if not matched:
            raise ValueError(f"Official base file identity mismatch: {name}")
        files[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256,
        }
    extraction = json.loads(args.extraction.read_text())
    if extraction["status"] != "passed_exact_ema_extraction":
        raise ValueError("Missing exact EMA extraction evidence")
    checkpoint = root / "checkpoints/streamforce_v1_ckpt.pt"
    if (
        checkpoint.stat().st_size != extraction["output_bytes"]
        or digest(checkpoint) != extraction["output_sha256"]
    ):
        raise ValueError("Staged EMA archive differs from the verified local archive")
    files[checkpoint.relative_to(root).as_posix()] = {
        "bytes": checkpoint.stat().st_size,
        "sha256": extraction["output_sha256"],
    }
    record = {
        "status": "passed",
        "source_revision": source["revision"],
        "base_model": model_id,
        "base_revision": info.sha,
        "checkpoint_packaging": extraction,
        "original_checkpoint_identity_scope": (
            "user download from official Drive; no published digest available"
        ),
        "files": files,
        "scope": "artifact identity only; native inference and quality remain separate",
    }
    (root / "ARTIFACTS.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "passed", "files": len(files), "base_revision": info.sha}))


if __name__ == "__main__":
    main()
