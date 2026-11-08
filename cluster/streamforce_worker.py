"""Gated StreamForce photo worker. Each action owns a disposable GPU process.

Cancelled native asynchronous decoders cannot retain a CUDA context between
requests. The complete native four-seed probe is required before serving.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "src"),
]
from photo_worker import Cancelled, PhotoWorker, runtime_versions  # noqa: E402

from click2future.demo.photo_service import (  # noqa: E402
    PhotoService,
    model_force,
    read_json,
    sha256,
    validate_action,
    write_json,
)


def verify_native_probe(repo, probe_path):
    repo = Path(repo).resolve()
    manifest = read_json(repo / "ARTIFACTS.json", {})
    probe = read_json(probe_path, {})
    if (
        manifest.get("status") != "passed"
        or manifest.get("source_revision") != "57a15e95eca25bbbb32d6ac6322ba3e12fd45083"
        or probe.get("status") != "passed_functional_probe"
        or probe.get("seeds") != [0, 1, 2, 3]
        or probe.get("artifacts_manifest_sha256") != sha256(repo / "ARTIFACTS.json")
        or probe.get("loader_sha256") != sha256(Path(__file__).with_name("streamforce_runtime.py"))
    ):
        raise ValueError("A matching passed native four-seed StreamForce probe is required")
    runs = probe.get("runs", [])
    required = {
        "checkpoints/streamforce_v1_ckpt.pt",
        "wan_models/Wan2.2-TI2V-5B/config.json",
        "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
        "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
        "wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json",
    }
    if not required <= manifest.get("files", {}).keys():
        raise ValueError("Incomplete StreamForce artifact inventory")
    if len(runs) != 5 or [r.get("seed") for r in runs] != [0, 1, 2, 3, 0]:
        raise ValueError("Retain all four native seeds and the declared repeat")
    if any(
        r.get("shape") != [81, 480, 832, 3] or not r.get("native_float_output_finite") for r in runs
    ):
        raise ValueError("Native probe outputs do not satisfy the video contract")
    for name, expected in manifest.get("files", {}).items():
        path = (repo / name).resolve()
        if (
            not path.is_relative_to(repo)
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"StreamForce artifact changed: {name}")
    return probe


def validate_request(root, path, model):
    request = read_json(path)
    validate_action({k: request[k] for k in ("photo", "prompt", "anchor", "force", "seeds")})
    service = PhotoService(root, backend="streamforce")
    if (
        request.get("model", {}).get("identity") != model["identity"]
        or request.get("layout") != service.layout
    ):
        raise ValueError("Request belongs to a different model/runtime or native profile")
    image = service.root / "images" / (request["photo"] + ".png")
    if sha256(image) != request["input_sha256"]:
        raise ValueError("Input image changed")
    return request, image


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=8)


class StreamForceWorker(PhotoWorker):
    def __init__(self, repo, spool, probe, *, raw_root, idle_seconds=1800):
        self.repo = Path(repo).resolve()
        self.service = PhotoService(spool, backend="streamforce")
        self.root = self.service.root
        self.probe = Path(probe).resolve()
        self.raw_root = Path(raw_root).resolve()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.idle_seconds = idle_seconds
        self.stop, self.lock = threading.Event(), threading.Lock()
        self.stage, self.ready, self.active = "checking photo model", False, None

    def serve(self):
        def beat():
            while not self.stop.wait(5):
                self.heartbeat()

        self.heartbeat()
        heartbeat_thread = threading.Thread(target=beat, daemon=True)
        heartbeat_thread.start()
        try:
            probe = verify_native_probe(self.repo, self.probe)
            extension = importlib.util.find_spec("flash_attn_2_cuda")
            if (
                extension is None
                or sha256(Path(extension.origin)) != probe["flash_attention_binary_sha256"]
            ):
                raise ValueError("The native attention binary differs from the passed probe")
            identity = {
                "artifacts_manifest_sha256": sha256(self.repo / "ARTIFACTS.json"),
                "native_probe_sha256": sha256(self.probe),
                "loader_sha256": sha256(Path(__file__).with_name("streamforce_runtime.py")),
                "worker_sha256": sha256(Path(__file__)),
                "common_worker_sha256": sha256(Path(__file__).with_name("photo_worker.py")),
                "request_contract_sha256": sha256(
                    Path(__file__).resolve().parents[1] / "src/click2future/demo/photo_service.py"
                ),
                "software": runtime_versions(),
                "flash_attention_binary_sha256": probe["flash_attention_binary_sha256"],
            }
            model = {
                **probe["model"],
                **identity,
                **self.service.layout,
                "backend_id": "streamforce",
                "raw_root": str(self.raw_root),
                "identity": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
                "worker_host": platform.node(),
                "allocation_job_id": os.environ["SLURM_JOB_ID"],
                "process_policy": (
                    "one cold model load per action; four native seeds in one process"
                ),
                "zero_semantics": (
                    "zero control scalar; physical rest and uncertainty remain uncalibrated"
                ),
            }
            for name, version in probe["versions"].items():
                if importlib.metadata.version(name) != version:
                    raise ValueError(f"Inference library differs from the passed probe: {name}")
            write_json(self.root / "MODEL.json", model)
            for path in (self.root / "jobs").glob("*/status.json"):
                if read_json(path, {}).get("status") == "running":
                    write_json(
                        path,
                        {
                            "status": "error",
                            "stage": "previous worker interrupted; submit a new attempt",
                        },
                    )
            self.ready = True
            self.heartbeat("ready to generate")
            last_activity = time.monotonic()
            while not self.stop.is_set():
                pending = [
                    p
                    for p in (self.root / "jobs").glob("*/request.json")
                    if read_json(p.parent / "status.json", {}).get("status")
                    not in {"complete", "error", "cancelled"}
                ]
                if not pending:
                    if time.monotonic() - last_activity >= self.idle_seconds:
                        break
                    self.stop.wait(1)
                    continue
                path = min(pending, key=lambda p: p.stat().st_mtime)
                self.active = path.parent
                child = None
                try:
                    validate_request(self.root, path, model)
                    self.update("preparing four futures")
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child-request",
                        str(path),
                        "--repo",
                        str(self.repo),
                        "--spool",
                        str(self.root),
                        "--raw-root",
                        str(self.raw_root),
                    ]
                    with (
                        (self.active / "child.stdout.log").open("ab") as stdout,
                        (self.active / "child.stderr.log").open("ab") as stderr,
                    ):
                        child = subprocess.Popen(
                            command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
                        )
                        while child.poll() is None:
                            state = read_json(self.active / "status.json", {})
                            self.heartbeat(state.get("stage", "generating four futures"))
                            if (self.active / "cancel.json").exists():
                                raise Cancelled("cancelled by the user")
                            if self.stop.is_set():
                                raise InterruptedError("GPU allocation is stopping")
                            self.stop.wait(0.5)
                    if child.returncode != 0 or not (self.active / "result.json").is_file():
                        raise RuntimeError(
                            f"Native generation process exited {child.returncode}; logs retained"
                        )
                    write_json(
                        self.active / "status.json",
                        {"status": "complete", "stage": "four futures ready"},
                    )
                    if self.service.job(self.active.name)["status"] != "complete":
                        raise ValueError("Native result failed the four-sample integrity contract")
                except Cancelled:
                    if child is not None:
                        stop_child(child)
                    write_json(
                        self.active / "status.json",
                        {"status": "cancelled", "stage": "generation cancelled"},
                    )
                except Exception as error:
                    if child is not None:
                        stop_child(child)
                    write_json(
                        self.active / "status.json",
                        {"status": "error", "stage": "generation failed", "error": str(error)},
                    )
                    write_json(
                        self.active / "failure.json",
                        {"error": str(error), "traceback": traceback.format_exc()},
                    )
                finally:
                    if child is not None:
                        stop_child(child)
                    self.active = None
                    last_activity = time.monotonic()
                    self.heartbeat("ready to generate")
        finally:
            self.ready = False
            self.stop.set()
            self.heartbeat("worker stopped; recordings remain available")
            heartbeat_thread.join(timeout=6)


def child_request(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from streamforce_runtime import load_backend

    root, path = args.spool.resolve(), args.child_request.resolve()
    if path.parent.parent != root / "jobs" or path.name != "request.json":
        raise ValueError("Child request must be in the queue")
    model = read_json(root / "MODEL.json")
    request, image_path = validate_request(root, path, model)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if image.shape != (480, 832, 3):
        raise ValueError("Unexpected input image shape")

    def progress(stage):
        write_json(
            path.parent / "status.json",
            {"status": "running", "stage": stage.replace("_", " "), "updated_at": time.time()},
        )

    backend, _ = load_backend(args.repo, [request["prompt"]], progress=progress)
    original = backend.pipeline.inference_rolling_forcing_stream

    def checked(*values, **keywords):
        result = original(*values, **keywords)
        if not bool(torch.isfinite(result[0]).all()):
            raise FloatingPointError("Nonfinite native float video")
        return result

    backend.pipeline.inference_rolling_forcing_stream = checked
    from interactive_demo_force_adapter import ForceAdapterConfig, build_model_condition_signal

    control = build_model_condition_signal(model_force(request), 480, 832, 81, ForceAdapterConfig())
    raw_folder = args.raw_root.resolve() / path.parent.name
    raw_folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seed in enumerate(request["seeds"]):
        progress(f"generating future {index + 1} of 4")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        generated = backend.generate_segment_streaming(image, request["prompt"], control, seed=seed)
        torch.cuda.synchronize()
        frames = generated["frames"]
        if frames.shape != (81, 480, 832, 3) or frames.dtype != np.uint8:
            raise ValueError("Native output shape or dtype changed")
        raw = raw_folder / f"{index}.npy"
        temporary = raw.with_suffix(".npy.partial")
        with temporary.open("wb") as stream:
            np.save(stream, frames, allow_pickle=False)
        temporary.replace(raw)
        temporary = path.parent / f"{index}.partial.mp4"
        imageio.mimwrite(temporary, frames, fps=16, codec="libx264", quality=8, macro_block_size=1)
        video = path.parent / f"{index}.mp4"
        temporary.replace(video)
        rows.append(
            {
                "file": video.name,
                "seed": seed,
                "sha256": sha256(video),
                "bytes": video.stat().st_size,
                "raw_file": raw.name,
                "raw_location": str(raw),
                "raw_sha256": sha256(raw),
                "seconds": time.monotonic() - started,
                "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                "stream_metrics": generated["stream_metrics"],
            }
        )
        write_json(
            path.parent / "attempt.json",
            {
                "request": request,
                "videos": rows,
                **request["layout"],
                "model": model,
                "complete": False,
                "scope": "finished samples from an unfinished action",
            },
        )
    write_json(
        path.parent / "result.json",
        {
            "request": request,
            "videos": rows,
            **request["layout"],
            "model": model,
            "unique_raw_videos": len({r["raw_sha256"] for r in rows}),
            "scope": "external photo futures; physics and uncertainty remain uncalibrated",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "spool", "raw-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--child-request", type=Path)
    parser.add_argument("--idle-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the native photo worker in an allocated GPU job")
    if args.child_request:
        child_request(args)
    else:
        import fcntl

        if not args.probe:
            parser.error("--probe is required before serving")
        worker = StreamForceWorker(
            args.repo,
            args.spool,
            args.probe,
            raw_root=args.raw_root,
            idle_seconds=args.idle_seconds,
        )
        for name in (signal.SIGTERM, signal.SIGINT):
            signal.signal(name, lambda *_: worker.stop.set())
        with (worker.root / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker.serve()


if __name__ == "__main__":
    main()
