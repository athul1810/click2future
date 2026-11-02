"""Build and inspect the Group 7 source-only code archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

URL = re.compile(r"(?:[a-z][a-z0-9+.-]*:[/]{2}|www[.]|git[@])", re.IGNORECASE)
ALLOWED_SUFFIXES = {".py", ".toml", ".yaml", ".json", ".md", ".txt", ".sbatch"}
DEMO_ASSETS = {
    "src/click2future/demo/static/index.html",
    "src/click2future/demo/static/style.css",
    "src/click2future/demo/static/app.js",
}
# TC2 jobs behind the final E0/E2/E3 runs and held-out results (D013-D015).
FINAL_RUN_SCRIPTS = {
    "cluster/cvae_study.sbatch",
    "cluster/action_horizon.sbatch",
    "cluster/diffusion_repeats.sbatch",
    "cluster/final_rehearsal.py",
    "cluster/final_rehearsal.sbatch",
    "cluster/final_evaluation.py",
    "cluster/final_evaluation.sbatch",
    "cluster/diffusion_long.sbatch",
}
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".aws",
    ".codex",
    ".agents",
    "datasets",
    "checkpoints",
    "outputs",
    "runs",
    "node_modules",
}


def _check(name, payload):
    path = PurePosixPath(name)
    allowed = (
        name
        in {"README.md", "pyproject.toml", "constraints.txt", ".python-version", "MANIFEST.json"}
        or (name.startswith("src/click2future/") and path.suffix == ".py")
        or name in DEMO_ASSETS
        or name in FINAL_RUN_SCRIPTS
        or (name.startswith("configs/") and path.suffix in {".yaml", ".toml", ".json"})
        or (
            path.parent == PurePosixPath("tests")
            and path.name.startswith("test_")
            and path.suffix == ".py"
        )
        or name == "tests/fixtures/tiny_scene.json"
        or name
        in {
            "submission/README.md",
            "submission/allowlist.json",
            "submission/smoke.py",
            "submission/dependency_versions.json",
            "submission/train_cvae.sbatch",
            "submission/train_diffusion.sbatch",
            "submission/sample_evaluation.sbatch",
            "submission/train_pilot.sbatch",
            "submission/evaluate_pilot.sbatch",
            "submission/train_trajectory.sbatch",
            "submission/train_contact.sbatch",
            "submission/wall_comparison.py",
            "submission/wall_comparison.sbatch",
            "submission/contact_risk_comparison.py",
            "submission/contact_risk_comparison.sbatch",
            "submission/world_comparison.py",
            "submission/world_comparison.sbatch",
            "submission/memory_comparison.py",
            "submission/memory_comparison.sbatch",
            "submission/coverage_comparison.py",
            "submission/coverage_comparison.sbatch",
            "submission/trajectory_response.py",
            "submission/evaluate_trajectory.sbatch",
            "submission/audit_trajectory.sbatch",
            "submission/sampler_study.py",
            "submission/study_sampler.sbatch",
            "submission/flow_evaluation.py",
            "submission/cvae_evaluation.py",
            "submission/cvae_force_probe.py",
            "submission/cvae_motion_timing.py",
            "submission/cvae_selection_drift.py",
            "submission/check_e4_collision.py",
            "submission/check_e4_collision.sbatch",
            "submission/pilot_image_metrics.py",
        }
    )
    if (
        not allowed
        or path.name == "AGENTS.md"
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or any(part in FORBIDDEN_PARTS or part.startswith(".env") for part in path.parts)
    ):
        raise ValueError(f"Forbidden archive path: {name}")
    if (
        path.suffix not in ALLOWED_SUFFIXES
        and name != ".python-version"
        and name not in DEMO_ASSETS
    ):
        raise ValueError(f"Forbidden archive file type: {name}")
    content = payload.decode("utf-8")
    if URL.search(content):
        raise ValueError(f"External link in {name}; remove it from the submission source")
    if re.search(r"-{5}BEGIN [A-Z ]*PRIVATE KEY-{5}|gh[pousr]_[A-Za-z0-9]{30,}", content):
        raise ValueError(f"Credential-like content in {name}")


def _dependency_files(root):
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = root / "uv.lock"
    if lock.is_file():
        versions = {}
        for item in tomllib.loads(lock.read_text())["package"]:
            if item["name"] != "click2future":
                versions.setdefault(item["name"], set()).add(item["version"].split("+")[0])
        # Platform alternatives with different public versions cannot share one pip constraint.
        pinned = {
            name: next(iter(values))
            for name, values in sorted(versions.items())
            if len(values) == 1
        }
    else:
        pinned = json.loads((root / "submission/dependency_versions.json").read_text())

    def pin(requirement):
        name = re.match(r"[A-Za-z0-9_.-]+", requirement)[0].lower().replace("_", "-")
        if name not in pinned:
            raise ValueError(f"No unambiguous dependency version for {name}")
        return f"{name}=={pinned[name]}"

    metadata = project["project"]
    lines = [
        "[build-system]",
        'requires = ["hatchling==1.27.0"]',
        'build-backend = "hatchling.build"',
        "",
        "[project]",
    ]
    for key in ("name", "version", "description", "requires-python"):
        lines.append(f"{key} = {json.dumps(metadata[key])}")
    lines += [
        'readme = "README.md"',
        "dependencies = " + json.dumps([pin(x) for x in metadata["dependencies"]]),
        "",
        "[project.optional-dependencies]",
        "cpu = " + json.dumps([pin(x) for x in metadata["optional-dependencies"]["cpu"]]),
        "image-metrics = "
        + json.dumps([pin(x) for x in metadata["optional-dependencies"]["image-metrics"]]),
        "dev = " + json.dumps([pin(x) for x in ("pytest", "ruff")]),
        "",
        "[tool.hatch.build.targets.wheel]",
        'packages = ["src/click2future"]',
        "",
        "[tool.pytest.ini_options]",
        'testpaths = ["tests"]',
        'pythonpath = ["src"]',
        'addopts = "-ra"',
        "",
    ]
    return {
        "pyproject.toml": "\n".join(lines).encode(),
        "constraints.txt": "".join(
            f"{name}=={version}\n" for name, version in pinned.items()
        ).encode(),
        "submission/dependency_versions.json": (
            json.dumps(pinned, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }


def build_archive(root: Path, output: Path):
    root = root.resolve()
    if output.name != "group_07_code.zip":
        raise ValueError("The code archive must be named group_07_code.zip")
    recipe_path = root / "submission/allowlist.json"
    recipe = json.loads(recipe_path.read_text())
    files = {}
    for pattern in recipe["include"]:
        matches = sorted(root.glob(pattern))
        if not matches:
            raise ValueError(f"Required source pattern matched no files: {pattern}")
        for path in matches:
            if (
                not path.is_file()
                or path.is_symlink()
                or any(p.is_symlink() for p in path.parents if p != root.parent)
            ):
                raise ValueError(f"Source must be a regular file without symlink parents: {path}")
            if not path.resolve().is_relative_to(root):
                raise ValueError("Source escapes the repository")
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    files.update(_dependency_files(root))
    files["README.md"] = (root / "submission/README.md").read_bytes()
    for name, payload in files.items():
        _check(name, payload)
    inventory = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(files.items())
    }
    files["MANIFEST.json"] = (
        json.dumps(
            {
                "format_version": 1,
                "group": 7,
                "status": "source_snapshot; training_and_final_submission_checks_still_required",
                "sha256": inventory,
            },
            indent=2,
        )
        + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate before replacing an existing archive; fixed metadata makes builds reproducible.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for name, payload in sorted(files.items()):
                info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compresslevel=9)
        result = inspect_archive(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    result["archive"] = str(output)
    return result


def inspect_archive(path: Path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or "MANIFEST.json" not in names:
            raise ValueError("Duplicate archive entries or missing inventory")
        files = {name: archive.read(name) for name in names}
        for name, payload in files.items():
            _check(name, payload)
        manifest = json.loads(files.pop("MANIFEST.json"))
        inventory = {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}
        if (
            manifest["format_version"] != 1
            or manifest["group"] != 7
            or inventory != manifest["sha256"]
        ):
            raise ValueError("Archive inventory or source hashes do not match")
        required = {
            "README.md",
            "pyproject.toml",
            "constraints.txt",
            "src/click2future/cli.py",
            "src/click2future/evaluation/aggregate.py",
            "submission/smoke.py",
        }
        if not required <= files.keys():
            raise ValueError("Archive is missing required source or local instructions")
    return {
        "status": "passed",
        "file_count": len(names),
        "bytes": path.stat().st_size,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks": [
            "allowed text files",
            "no external links",
            "source inventory hashes",
            "no datasets, weights, environments, or Git history",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path("dist/group_07_code.zip"))
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            build_archive(args.root, args.output)
            if args.command == "build"
            else inspect_archive(args.archive)
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Submission package failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
