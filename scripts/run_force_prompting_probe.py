"""Download official fallback weights and retain every native probe sample."""

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
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated GPU job")
    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "scope": "native functional smoke; not quality evidence",
        "runs": [],
        "stages": [],
        "job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
    }
    start = time.monotonic()

    def progress(stage):
        row = {
            "stage": stage,
            "seconds": time.monotonic() - start,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        }
        record["stages"].append(row)
        (out / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    try:
        identities = {}
        api = HfApi()
        for model, dest, patterns in [
            (
                "THUDM/CogVideoX-5b-I2V",
                repo / "weights/CogVideoX-5b-I2V",
                [
                    "model_index.json",
                    "tokenizer/*",
                    "text_encoder/*",
                    "transformer/*",
                    "vae/*",
                    "scheduler/*",
                ],
            ),
            (
                "brown-palm/force-prompting",
                repo / "checkpoints",
                ["step-5000-checkpoint-point-force.pt"],
            ),
        ]:
            revision = api.model_info(model).sha
            identities[model] = revision
            snapshot_download(
                model, revision=revision, local_dir=dest, allow_patterns=patterns, max_workers=2
            )
        progress("official_weights_downloaded")
        checkpoint = repo / "checkpoints/step-5000-checkpoint-point-force.pt"
        record["checkpoint_sha256"] = digest(checkpoint)
        record["model_revisions"] = identities
        record["torch"] = torch.__version__
        record["cuda"] = torch.version.cuda
        progress("loading_model")
        backend = ForcePromptingBackend(repo, progress=progress)
        record["model"] = backend.metadata
        folder = repo / "datasets/point-force/test/benchmark/ornament"
        sample = next(csv.DictReader((folder / "_ornament1_obj1_prompt1.csv").open()))
        (out / "input-record.json").write_text(json.dumps(sample, indent=2) + "\n")
        input_path = folder / "images" / sample["image"]
        image = np.array(
            Image.open(input_path).convert("RGB").resize((720, 480), Image.Resampling.LANCZOS)
        )
        Image.fromarray(image).save(out / "input.png")
        record["input_sha256"] = digest(out / "input.png")
        force = {
            "model": {
                "mode": "point",
                "force": float(sample["force"]),
                "angle": float(sample["angle"]),
                "x_pos": float(sample["coordx"]) / float(sample["width"]),
                "y_pos": float(sample["coordy"]) / float(sample["height"]),
            }
        }
        signal = backend.signal(force)
        for index, seed in enumerate(args.seeds):
            since = time.monotonic()
            torch.cuda.reset_peak_memory_stats()

            def step(n, total, current_seed=seed):
                progress(f"seed_{current_seed}_step_{n}_of_{total}")

            frames = backend.generate(image, sample["caption"], signal, seed, on_step=step)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - since
            if frames.shape != (49, 480, 720, 3):
                raise ValueError(f"Unexpected video shape {frames.shape}")
            raw = out / f"sample-{index}-seed{seed}.npy"
            video = raw.with_suffix(".mp4")
            np.save(raw, frames, allow_pickle=False)
            imageio.mimwrite(video, frames, fps=8, codec="libx264", quality=8, macro_block_size=1)
            record["runs"].append(
                {
                    "seed": seed,
                    "seconds": elapsed,
                    "frames": len(frames),
                    "raw_sha256": digest(raw),
                    "video_sha256": digest(video),
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            progress(f"seed_{seed}_complete")
        record["status"] = "passed_functional_probe"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = type(error).__name__ + ": " + str(error)
        record["traceback"] = traceback.format_exc()
        raise
    finally:
        (out / "probe.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
