"""Fixed, all-sample development comparison of external photo preview settings."""

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cluster"))
from force_prompting_runtime import ForcePromptingBackend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated GPU")
    repo, gallery, out = args.repo.resolve(), args.gallery.resolve(), args.output.resolve()
    if json.loads((repo / "ARTIFACTS.json").read_text())["status"] != "passed":
        raise ValueError("Verify the official model artifacts first")
    out.mkdir(parents=True, exist_ok=False)
    row = next(
        r
        for r in csv.DictReader((gallery / "assets/samples/point_force.csv").open())
        if r["image"] == "point_3.png"
    )
    source = gallery / "assets/samples/images" / row["image"]
    original = Image.open(source).convert("RGB")
    if original.size != (int(row["width"]), int(row["height"])):
        raise ValueError("The gallery coordinates use a different image size")
    # Both output sizes have aspect 3:2, so the normalized crop point is identical.
    crop_width = original.height * 1.5
    left = (original.width - crop_width) / 2
    x = (float(row["coordx"]) - left) / crop_width
    y = float(row["coordy"]) / original.height
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("The registered crop removes the force point")
    control = {
        "model": {
            "mode": "point",
            "force": float(row["force"]),
            "angle": float(row["angle"]),
            "x_pos": x,
            "y_pos": y,
        }
    }
    record = {
        "status": "running",
        "study": "PHOTO-P01",
        "scope": "development profile comparison; no physical-quality promotion",
        "job_id": os.environ["SLURM_JOB_ID"],
        "input": row,
        "source_photo_sha256": digest(source),
        "control": control,
        "seeds": [0, 1, 2, 3],
        "runs": [],
        "stages": [],
        "model_artifacts_sha256": digest(repo / "ARTIFACTS.json"),
        "runtime_sha256": digest(
            Path(__file__).resolve().parents[1] / "cluster/force_prompting_runtime.py"
        ),
    }
    since = time.monotonic()

    def progress(stage):
        record["stages"].append({"stage": stage, "seconds": time.monotonic() - since})
        (out / "profiles.json").write_text(json.dumps(record, indent=2) + "\n")
        print(stage, flush=True)

    try:
        backend = ForcePromptingBackend(repo, progress=progress)
        record["model"] = backend.metadata.copy()
        profiles = [
            ("preview20", (432, 288), 20),
            ("native50", (720, 480), 50),
            ("native20", (720, 480), 20),
        ]
        first_preview = None
        for name, size, steps in profiles + [("preview20-repeat", (432, 288), 20)]:
            backend.set_profile(size, steps)
            prepared = np.array(ImageOps.fit(original, size, method=Image.Resampling.LANCZOS))
            Image.fromarray(prepared).save(out / (name + "-input.png"))
            signal = backend.signal(control, size=size)
            for seed in [0] if name.endswith("repeat") else [0, 1, 2, 3]:
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                frames = backend.generate(
                    prepared,
                    row["caption"],
                    signal,
                    seed,
                    on_step=lambda n, total, current_name=name, current_seed=seed: progress(
                        f"{current_name}_seed{current_seed}_step{n}_of{total}"
                    ),
                )
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                if frames.shape != (49, size[1], size[0], 3):
                    raise ValueError("Unexpected output shape")
                folder = out / name
                folder.mkdir(exist_ok=True)
                raw = folder / f"seed{seed}.npy"
                video = raw.with_suffix(".mp4")
                np.save(raw, frames, allow_pickle=False)
                imageio.mimwrite(
                    video, frames, fps=8, codec="libx264", quality=8, macro_block_size=1
                )
                result = {
                    "profile": name,
                    "size": list(size),
                    "steps": steps,
                    "seed": seed,
                    "seconds": seconds,
                    "raw_sha256": digest(raw),
                    "video_sha256": digest(video),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "frame0_rgb_mae": float(np.abs(frames[0].astype(float) - prepared).mean()),
                }
                if first_preview is None:
                    first_preview = frames.copy()
                if name.endswith("repeat"):
                    delta = np.abs(frames.astype(np.int16) - first_preview.astype(np.int16))
                    result.update(
                        repeat_pixel_max=int(delta.max()),
                        repeat_pixel_mean=float(delta.mean()),
                        repeat_pixel_exact=bool(np.array_equal(frames, first_preview)),
                    )
                    record["preview_replay_gate_passed"] = result["repeat_pixel_exact"]
                record["runs"].append(result)
                progress(f"{name}_seed{seed}_complete")
        record["status"] = "completed_profile_study"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        (out / "profiles.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
