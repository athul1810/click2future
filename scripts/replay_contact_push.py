"""Replay an observed push with real checkpoints and separate physical references."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import load_rollout
from click2future.evaluation.tracking import TrackerConfig, track_video
from click2future.sim import run_rollout
from click2future.training.sample_trajectory import load_trajectory, sample_trajectory_videos
from click2future.training.train_cvae import dataset_index, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with np.load(args.condition, allow_pickle=False) as arrays:
        first, mask, force = (arrays[key] for key in ("first_frame", "selected_mask", "force_xy"))
    _, rows, identity = dataset_index(args.dataset)
    references, profiles, group = [], set(), None
    for row in rows.values():
        profile = row["physics_variant"]
        if profile in profiles or (group and row["scene_group_id"] != group):
            continue
        ref = load_rollout(args.dataset, row["sample_id"])
        if not np.array_equal(ref["frames"][0], first):
            continue
        selected = [
            i
            for i, candidate in enumerate(ref["instance_masks"][0])
            if np.array_equal(candidate, mask)
        ]
        if len(selected) != 1:
            raise ValueError("Supplied mask does not identify exactly one reference object")
        group = row["scene_group_id"]
        profiles.add(profile)
        scene = copy.deepcopy(ref["scene_spec"])
        scene["action"].update(selected_object_id=selected[0], force_xy=force.tolist())
        references.append(run_rollout(scene, frame_count=32, fps=8, physics_hz=240, resolution=64))
    if len(references) != 4:
        raise ValueError("Require four existing development physics profiles for this image")
    videos, records = {}, {}
    seeds = [0, 1, 2, 3]
    for item in args.checkpoint:
        name, separator, filename = item.partition("=")
        if not separator or not name or name in records:
            raise ValueError("Use distinct name=checkpoint arguments")
        path = Path(filename)
        model, state = load_trajectory(path, args.device)
        generated = sample_trajectory_videos(
            model, first, mask, force, seeds=seeds, sampling_steps=16
        )
        samples = []
        for seed, clip in zip(seeds, generated, strict=True):
            track = track_video(
                clip,
                references[0]["instance_masks"][0],
                references[0]["object_valid"],
                TrackerConfig(),
            )
            excursion = np.linalg.norm(track.positions - track.positions[:1], axis=-1) * 64
            samples.append(
                {
                    "seed": seed,
                    "complete": track.tracked.all(0).tolist(),
                    "maximum_excursion_pixels": [
                        float(excursion[:, i].max()) if track.tracked[:, i].all() else None
                        for i in range(2)
                    ],
                }
            )
        videos[name] = generated
        records[name] = {
            "checkpoint_sha256": sha256(path),
            "step": state["step"],
            "samples": samples,
        }
        del model
    np.savez_compressed(
        args.output / "videos.npz",
        **videos,
        reference=np.stack([ref["frames"] for ref in references]),
    )
    result = {
        "scope": "saved interactive push diagnostic; not frozen validation or final-test evidence",
        "force": force.tolist(),
        "seeds": seeds,
        "selected_object": selected[0],
        "condition_sha256": sha256(args.condition),
        "source_sha256": sha256(Path(__file__)),
        "dataset_identity": identity,
        "models": records,
        "device": args.device,
        "torch": str(torch.__version__),
        "sampling_steps": 16,
        "video_sha256": sha256(args.output / "videos.npz"),
        "reference": {
            "scope": "four known physics profiles, separate from neural inputs",
            "profiles": sorted(profiles),
            "scene_group": group,
            "realizations": [
                {
                    "collision_onsets": int(ref["collision_onset"].sum()),
                    "maximum_excursion_pixels": (
                        np.linalg.norm(ref["positions"] - ref["positions"][:1], axis=-1).max(0) * 64
                    ).tolist(),
                }
                for ref in references
            ],
        },
        "test_splits_evaluated": [],
    }
    (args.output / "checks.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"models": records, "reference": result["reference"]}, indent=2))


if __name__ == "__main__":
    main()
