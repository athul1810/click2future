"""Native A40 probe of an external pretrained model; no training or quality claim."""

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
from streamforce_runtime import load_backend


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
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--repeat-first", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the GPU probe in a Slurm allocation")
    repo, out = args.repo.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "scope": "native development smoke; not a benchmark or quality result",
        "job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seeds": args.seeds,
        "stages": [],
        "runs": [],
    }
    started = time.monotonic()

    def progress(stage):
        torch.cuda.synchronize()
        row = {
            "stage": stage,
            "seconds": time.monotonic() - started,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "gpu_allocated_bytes": torch.cuda.memory_allocated(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(),
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        }
        record["stages"].append(row)
        (out / "probe.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    try:
        sample = next(csv.DictReader((repo / "assets/samples/point_force.csv").open()))
        (out / "input-record.json").write_text(json.dumps(sample, indent=2) + "\n")
        prompt = sample["caption"]
        backend, metadata = load_backend(repo, [prompt], progress=progress)
        record["model"] = metadata
        from interactive_demo_force_adapter import (
            ForceAdapterConfig,
            build_model_condition_signal,
        )

        image_name = sample.get("image_path", sample.get("image", sample.get("filename")))
        image_path = repo / "assets/samples/images" / image_name
        image = np.asarray(Image.open(image_path).convert("RGB"))
        record["input_sha256"] = digest(image_path)
        payload = {
            "model": {
                "mode": "point",
                "force": float(sample["force"]),
                "angle": float(sample["angle"]),
                "x_pos": float(sample["coordx"]) / float(sample["width"]),
                "y_pos": float(sample["coordy"]) / float(sample["height"]),
            }
        }
        signal = build_model_condition_signal(payload, 480, 832, 81, ForceAdapterConfig())
        seeds = args.seeds + ([args.seeds[0]] if args.repeat_first else [])
        first_frames = None
        for index, seed in enumerate(seeds):
            first_chunk = []
            run_start = time.monotonic()

            def chunk(frames, block, recorded=first_chunk, since=run_start):
                if not recorded:
                    recorded.append(time.monotonic() - since)

            torch.cuda.reset_peak_memory_stats()
            generated = backend.generate_segment_streaming(
                image,
                prompt,
                signal,
                seed=seed,
                on_chunk=chunk,
            )
            torch.cuda.synchronize()
            seconds = time.monotonic() - run_start
            frames = generated["frames"]
            if frames.shape != (81, 480, 832, 3) or frames.dtype != np.uint8:
                raise ValueError(f"Unexpected generated video shape: {frames.shape}")
            raw = out / f"sample-{index}-seed{seed}.npy"
            video = raw.with_suffix(".mp4")
            np.save(raw, frames, allow_pickle=False)
            imageio.mimwrite(video, frames, fps=16, codec="libx264", quality=8, macro_block_size=1)
            row = {
                "seed": seed,
                "seconds": seconds,
                "time_to_first_chunk_seconds": first_chunk[0] if first_chunk else None,
                "frames": len(frames),
                "shape": list(frames.shape),
                "raw_sha256": digest(raw),
                "video_sha256": digest(video),
                "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                "stream_metrics": generated["stream_metrics"],
            }
            if first_frames is None:
                first_frames = frames.copy()
            elif index == len(args.seeds) and args.repeat_first:
                delta = np.abs(frames.astype(np.int16) - first_frames.astype(np.int16))
                row["repeat_pixel_max_absolute_difference"] = int(delta.max())
                row["repeat_pixel_mean_absolute_difference"] = float(delta.mean())
                row["repeat_pixel_exact"] = bool(np.array_equal(frames, first_frames))
            record["runs"].append(row)
            progress(f"sample_{index}_complete")
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
