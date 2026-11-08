"""Verify downloaded fallback weights against the official pinned Hub objects."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(repo, revisions):
    from huggingface_hub import HfApi

    roots = {
        "THUDM/CogVideoX-5b-I2V": repo / "weights/CogVideoX-5b-I2V",
        "brown-palm/force-prompting": repo / "checkpoints",
    }
    inventory = {}
    for model, root in roots.items():
        info = HfApi().model_info(model, revision=revisions[model], files_metadata=True)
        files = {row.rfilename: row for row in info.siblings}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            name = str(path.relative_to(root)).replace("\\", "/")
            if name not in files:
                raise ValueError(f"Unlisted official artifact: {name}")
            expected = files[name]
            actual = digest(path)
            lfs = expected.lfs
            if lfs:
                reference = lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]
                if actual != reference:
                    raise ValueError(f"Official LFS hash mismatch: {name}")
            else:
                raw = path.read_bytes()
                git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                if git_hash != expected.blob_id:
                    raise ValueError(f"Official Git object mismatch: {name}")
            inventory[str(path.relative_to(repo)).replace("\\", "/")] = {
                "sha256": actual,
                "bytes": path.stat().st_size,
            }
    source = json.loads((repo / "SOURCE.json").read_text())
    for name, expected in source["files_sha256"].items():
        if digest(repo / name) != expected:
            raise ValueError(f"External source was modified: {name}")
    manifest = {
        "status": "passed",
        "revisions": revisions,
        "files": inventory,
        "source_revision": source["revision"],
        "scope": "official pinned file identities and source; not generated-video quality",
    }
    (repo / "ARTIFACTS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    manifest = verify(args.repo.resolve(), probe["model_revisions"])
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
