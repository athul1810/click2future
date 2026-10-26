"""Paired training-only visible coverage, with identical forces and physical draws."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from click2future.data.action_supplement import parent_inventory, recipe_cases, source_rollout
from click2future.data.export import load_rollout, save_rollout
from click2future.data.generalization import _digest
from click2future.sim.generalization import make_generalization_scene
from click2future.sim.rollout import run_rollout

from .train_cvae import sha256
from .trajectory_data import encode_trajectory_rows
from .world_data import load_cache, world_geometry

ARMS = ("narrow", "broad")


def derived_vector_equal(actual, expected):
    """Bound platform trig roundoff in recipe reconstruction, never scoring.

    Stored bytes remain checksum-exact. Eight float64 ULPs allow only rounding
    in derived force/position vectors; physical labels and all other fields
    remain exact. The force actually used by training must also match in float32.
    """
    return (
        isinstance(actual, list)
        and len(actual) == len(expected) == 2
        and all(
            not isinstance(a, bool)
            and isinstance(a, (float, int))
            and math.isfinite(a)
            and math.isfinite(b)
            and abs(a - b) <= 8 * max(math.ulp(a), math.ulp(b))
            for a, b in zip(actual, expected, strict=True)
        )
    )


def scene_recipe_equal(actual, expected):
    expected = copy.deepcopy(expected)
    if not derived_vector_equal(actual["action"]["force_xy"], expected["action"]["force_xy"]):
        return False
    expected["action"]["force_xy"] = actual["action"]["force_xy"]
    if len(actual["objects"]) != len(expected["objects"]):
        return False
    for obj, reference in zip(actual["objects"], expected["objects"], strict=True):
        if not derived_vector_equal(obj["position"], reference["position"]):
            return False
        reference["position"] = obj["position"]
    return actual == expected


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def recipe_check(recipe):
    if (
        set(recipe) != {"seed", "pairs", "directions", "magnitudes"}
        or any(
            isinstance(recipe[k], bool) or not isinstance(recipe[k], int) for k in ("seed", "pairs")
        )
        or not 0 <= recipe["seed"] <= 10**9
        or not 1 <= recipe["pairs"] <= 256
    ):
        raise ValueError("Invalid visible-coverage recipe")
    # Reuse the original bounded action-grid validation, without reading rollouts.
    list(recipe_cases([{"physics_variant": 0}], recipe["directions"], recipe["magnitudes"]))


def templates(parent, recipe):
    recipe_check(recipe)
    manifest, sources = parent_inventory(parent)
    sources = [row for row in sources if row["physics_variant"] == 0]
    visible_seeds = {row["visible_seed"] for row in manifest["samples"]}
    if visible_seeds & set(range(recipe["seed"], recipe["seed"] + recipe["pairs"])):
        raise ValueError("New visible seeds overlap the parent corpus")
    return manifest, sources


def action_cases(recipe, pair):
    for profile in range(4):
        source = {"physics_variant": profile}
        for _, selected, direction, strength, force in recipe_cases(
            [source], recipe["directions"], recipe["magnitudes"]
        ):
            name = f"coverage-{pair:04d}-p{profile}-s{selected}-d{direction + 1}-m{strength + 1}"
            yield name, profile, selected, force


def support(obj, direction):
    if obj["shape"] == "circle":
        return obj["geometry"]["radius"]
    angle = obj["angle"]
    c, s = math.cos(angle), math.sin(angle)
    x, y = direction
    return (
        abs(x * c + y * s) * obj["geometry"]["width"]
        + abs(-x * s + y * c) * obj["geometry"]["height"]
    ) / 2


def paired_scene(base, recipe, arm, pair, profile, selected, force, name):
    """Only training appearances/layouts differ; physical labels match by object ID."""
    if arm not in ARMS:
        raise ValueError("Unknown coverage arm")
    visible_seed = recipe["seed"] + pair
    physical_seed = recipe["seed"] * 1_000_000 + pair * 1000 + profile + 1
    if arm == "narrow":
        scene = copy.deepcopy(base)
    else:
        scene = make_generalization_scene(
            visible_seed, split="train", action_variant="forward", physics_seed=physical_seed
        )
        rng = np.random.default_rng(np.random.SeedSequence([visible_seed, 0xC0A]))
        for obj in scene["objects"]:
            obj["angle"] = float(rng.uniform(-math.pi, math.pi))
        angle = rng.uniform(-math.pi, math.pi)
        direction = np.array([math.cos(angle), math.sin(angle)])
        distance = sum(support(obj, direction) for obj in scene["objects"]) + rng.uniform(
            0.02, 0.08
        )
        center = rng.uniform(0.35, 0.65, 2)
        for index, obj in enumerate(scene["objects"]):
            obj["position"] = (center + (index - 0.5) * distance * direction).tolist()
    hidden = np.random.default_rng(physical_seed)
    for obj in scene["objects"]:
        obj["hidden_physics"] = {
            "mass": float(hidden.uniform(0.5, 2.0)),
            "friction": float(hidden.uniform(0.1, 0.8)),
            "restitution": float(hidden.uniform(0.0, 0.4)),
        }
    scene["seeds"]["physics"] = physical_seed
    scene["condition_id"] = name
    scene["action_variant"] = "training_visible_coverage"
    scene["action"] = {"selected_object_id": selected, "force_xy": force, "force_duration": 0.25}
    return scene


def generate_pair(arguments):
    parent, output, source, recipe, pair = arguments
    base = source_rollout(parent, source)["scene_spec"]
    rows = []
    for name, profile, selected, force in action_cases(recipe, pair):
        row = {
            "sample_id": name,
            "pair": pair,
            "profile": profile,
            "selected_object_id": selected,
            "force_xy": force,
            "parent_sample_id": source["sample_id"],
            "split": "train",
            "cohorts": {},
        }
        for arm in ARMS:
            scene = paired_scene(base, recipe, arm, pair, profile, selected, force, name)
            clip = run_rollout(scene, frame_count=32, fps=8, physics_hz=240, resolution=64)
            paths = save_rollout(clip, Path(output) / arm, name)
            row["cohorts"][arm] = {
                "scene_group_id": scene["scene_group_id"],
                "visible_seed": scene["seeds"]["visible"],
                "contact": bool(clip["collision_onset"].any()),
                "first_frame_sha256": hashlib.sha256(clip["frames"][0].tobytes()).hexdigest(),
                **{key + "_sha256": sha256(path) for key, path in paths.items()},
            }
        # A common training-only stratum keeps sampled paired indices/noise equal.
        row["sampling_contact"] = any(c["contact"] for c in row["cohorts"].values())
        rows.append(row)
    return rows


def generate(parent, output, recipe, *, workers=1):
    parent, output = Path(parent), Path(output)
    manifest, sources = templates(parent, recipe)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Use one to eight bounded workers")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        (output / arm).mkdir()
    write(output / "recipe.json", recipe)  # Freeze before the first rendered sample.
    arguments = [
        (str(parent), str(output), sources[i % len(sources)], recipe, i)
        for i in range(recipe["pairs"])
    ]
    if workers == 1:
        groups = list(map(generate_pair, arguments))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            groups = list(pool.map(generate_pair, arguments))
    rows = [row for group in groups for row in group]
    result = {
        "schema_version": "paired-visible-coverage-0.1",
        "recipe": recipe,
        "parent_manifest_sha256": sha256(parent / "dataset_manifest.json"),
        "parent_split_sha256": sha256(parent / "splits.json"),
        "parent_manifest_hash": manifest["manifest_hash"],
        "samples": rows,
        "sample_count_per_arm": len(rows),
        "scope": "paired training only; no validation/test rollout read or modified",
        "sampling": "union of paired training contact labels; same strata in both arms",
        "generation_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "pillow", "pymunk")
            },
        },
        "generation_source_sha256": {
            path.relative_to(Path(__file__).parents[1]).as_posix(): sha256(path)
            for path in [
                Path(__file__),
                *sorted((Path(__file__).parents[1] / "sim").glob("*.py")),
                Path(__file__).parents[1] / "data/export.py",
                Path(__file__).parents[1] / "data/action_supplement.py",
                Path(__file__).parents[1] / "data/generalization.py",
            ]
        },
    }
    result["manifest_hash"] = _digest(result)
    write(output / "coverage.json", result)
    return result


def validate(parent, output):
    parent, output = Path(parent), Path(output)
    record = read(output / "coverage.json")
    recipe = record["recipe"]
    manifest, sources = templates(parent, recipe)
    if (
        record["schema_version"] != "paired-visible-coverage-0.1"
        or read(output / "recipe.json") != recipe
        or record["manifest_hash"]
        != _digest({k: v for k, v in record.items() if k != "manifest_hash"})
        or record["parent_manifest_sha256"] != sha256(parent / "dataset_manifest.json")
        or record["parent_split_sha256"] != sha256(parent / "splits.json")
        or record["parent_manifest_hash"] != manifest["manifest_hash"]
    ):
        raise ValueError("Coverage recipe or parent binding changed")
    expected = [(i, case) for i in range(recipe["pairs"]) for case in action_cases(recipe, i)]
    if len(record["samples"]) != len(expected) or record["sample_count_per_arm"] != len(expected):
        raise ValueError("Incomplete paired coverage")
    bases = {}
    for row, (pair, (name, profile, selected, force)) in zip(
        record["samples"], expected, strict=True
    ):
        source = sources[pair % len(sources)]
        if (
            any(
                row[key] != value
                for key, value in {
                    "sample_id": name,
                    "pair": pair,
                    "profile": profile,
                    "selected_object_id": selected,
                    "parent_sample_id": source["sample_id"],
                    "split": "train",
                }.items()
            )
            or not derived_vector_equal(row["force_xy"], force)
            or set(row["cohorts"]) != set(ARMS)
            or row["sampling_contact"] != any(c["contact"] for c in row["cohorts"].values())
        ):
            raise ValueError("Paired coverage membership or action changed")
        if pair not in bases:
            bases[pair] = source_rollout(parent, source)["scene_spec"]
        physical = []
        for arm in ARMS:
            folder = output / arm
            for kind, suffix in (("arrays", ".npz"), ("metadata", ".json")):
                if sha256(folder / (name + suffix)) != row["cohorts"][arm][kind + "_sha256"]:
                    raise ValueError("Coverage bytes changed")
            clip = load_rollout(folder, name)
            scene = paired_scene(bases[pair], recipe, arm, pair, profile, selected, force, name)
            if (
                not scene_recipe_equal(clip["scene_spec"], scene)
                or clip["scene_spec"]["action"]["force_xy"] != row["force_xy"]
                or not np.array_equal(clip["force_xy"], np.asarray(force, dtype=np.float32))
                or row["cohorts"][arm]["contact"] != bool(clip["collision_onset"].any())
                or row["cohorts"][arm]["first_frame_sha256"]
                != hashlib.sha256(clip["frames"][0].tobytes()).hexdigest()
                or row["cohorts"][arm]["scene_group_id"] != scene["scene_group_id"]
                or row["cohorts"][arm]["visible_seed"] != scene["seeds"]["visible"]
            ):
                raise ValueError("Coverage scene or observation changed")
            physical.append([obj["hidden_physics"] for obj in scene["objects"]])
        if physical[0] != physical[1]:
            raise ValueError("Unpaired hidden physics")
    return record


def append_cache(parent, base_cache, coverage, output, arm):
    if arm not in ARMS:
        raise ValueError("Unknown coverage arm")
    record = validate(parent, coverage)
    data, geometry, worlds, provenance = load_cache(base_cache)
    if (
        provenance["dataset_manifest_sha256"] != record["parent_manifest_sha256"]
        or provenance["split_manifest_sha256"] != record["parent_split_sha256"]
    ):
        raise ValueError("Base cache and coverage identify different parent data")
    extra, labels = encode_trajectory_rows(
        Path(coverage) / arm,
        {"train": [row["sample_id"] for row in record["samples"]]},
        augment=True,
    )
    by_id = {row["sample_id"]: row for row in record["samples"]}
    for example in extra["train"]["examples"]:
        example.update(
            root=str((Path(coverage) / arm).resolve()),
            contact=by_id[example["sample_id"]]["sampling_contact"],
        )
    extra_geometry, extra_worlds = world_geometry(parent, extra)
    for key in ("motion", "force", "features"):
        data["train"][key] = np.concatenate((data["train"][key], extra["train"][key]))
    data["train"]["examples"] += extra["train"]["examples"]
    for key in geometry["train"]:
        geometry["train"][key] = np.concatenate(
            (geometry["train"][key], extra_geometry["train"][key])
        )
    worlds["train"] = np.concatenate((worlds["train"], extra_worlds["train"]))
    labels.pop("preparation_seconds", None)
    provenance = {
        **provenance,
        "coverage_addition": {
            "arm": arm,
            "manifest_sha256": sha256(Path(coverage) / "coverage.json"),
            "base_cache_manifest_sha256": sha256(Path(base_cache) / "manifest.json"),
            "base_cache_arrays_sha256": sha256(Path(base_cache) / "arrays.npz"),
            "parent_provenance": provenance,
            "labels": labels,
            "examples_added": len(extra["train"]["examples"]),
            "scope": "new training only; original validation unchanged",
        },
        "geometry_sha256": {
            s: {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in g.items()}
            for s, g in geometry.items()
        },
        "world_label_sha256": {
            s: hashlib.sha256(v.tobytes()).hexdigest() for s, v in worlds.items()
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for split in data:
        arrays.update(
            {split + "_" + key: data[split][key] for key in ("motion", "force", "features")}
        )
        arrays.update({split + "_" + key: value for key, value in geometry[split].items()})
        arrays[split + "_world"] = worlds[split]
    np.savez_compressed(output / "arrays.npz", **arrays)
    from click2future.models.world_motion import WORLD_CONTRACT

    write(
        output / "manifest.json",
        {
            "format_version": 1,
            "world_contract": WORLD_CONTRACT,
            "provenance": provenance,
            "examples": {split: row["examples"] for split, row in data.items()},
            "array_sha256": sha256(output / "arrays.npz"),
            "scope": "paired training coverage addition; original validation unchanged",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "output", "recipe"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = generate(args.parent, args.output, read(args.recipe), workers=args.workers)
    validate(args.parent, args.output)
    print(
        json.dumps(
            {
                "samples_per_arm": result["sample_count_per_arm"],
                "manifest_hash": result["manifest_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
