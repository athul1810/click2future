"""One allocated GPU processes durable photo jobs; disconnecting keeps records."""

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "src"),
]
from click2future.demo.photo_service import (  # noqa: E402 - external environment runs this checkout
    PhotoService,
    model_force,
    read_json,
    sha256,
    validate_action,
    write_json,
)


class Cancelled(Exception):
    pass


class PhotoWorker:
    def __init__(self, repo, spool, *, idle_seconds=1800, raw_root=None):
        self.repo = Path(repo).resolve()
        self.service = PhotoService(spool, backend="force-prompting")
        self.root = self.service.root
        self.idle_seconds = idle_seconds
        self.raw_root = Path(raw_root or self.repo.parent / "raw").resolve()
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.stage = "loading the photo model"
        self.ready = False
        self.active = None

    def heartbeat(self, stage=None):
        with self.lock:
            if stage is not None:
                self.stage = stage
            write_json(
                self.root / "WORKER.json",
                {
                    "ready": self.ready,
                    "updated_at": time.time(),
                    "stage": self.stage,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )

    def update(self, stage):
        self.heartbeat(stage.replace("_", " "))
        if self.active:
            if self.stop.is_set():
                raise InterruptedError(
                    "GPU allocation is stopping; retained outputs are incomplete"
                )
            if (self.active / "cancel.json").exists():
                raise Cancelled("cancelled by the user")
            write_json(
                self.active / "status.json",
                {
                    "status": "running",
                    "stage": self.stage,
                    "updated_at": time.time(),
                },
            )

    def loop(self):
        import imageio.v2 as imageio
        import numpy as np
        import torch
        from force_prompting_runtime import ForcePromptingBackend
        from PIL import Image

        def heartbeat_loop():
            while not self.stop.wait(5):
                self.heartbeat()

        self.heartbeat()
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        try:
            backend = ForcePromptingBackend(self.repo, progress=self.update)
            checkpoint = self.repo / "checkpoints/step-5000-checkpoint-point-force.pt"
            runtime = Path(__file__).with_name("force_prompting_runtime.py")
            identity = {
                "checkpoint_sha256": sha256(checkpoint),
                "runtime_sha256": sha256(runtime),
                "worker_sha256": sha256(Path(__file__)),
                "request_contract_sha256": sha256(
                    Path(__file__).resolve().parents[1] / "src/click2future/demo/photo_service.py"
                ),
                "artifacts_manifest_sha256": sha256(self.repo / "ARTIFACTS.json"),
                "source_revision": backend.metadata["source_revision"],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "size": [720, 480],
                "frames": 49,
                "fps": 8,
            }
            model = {
                **backend.metadata,
                **identity,
                "backend_id": "force-prompting",
                "identity": hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
                "worker_host": os.uname().nodename,
                "raw_root": str(self.raw_root),
                "allocation_job_id": os.environ["SLURM_JOB_ID"],
            }
            write_json(self.root / "MODEL.json", model)
            for path in (self.root / "jobs").glob("*/status.json"):
                previous = read_json(path, {})
                if previous.get("status") == "running":
                    write_json(
                        path,
                        {
                            "status": "error",
                            "stage": "previous worker interrupted; submit a new attempt",
                        },
                    )
            self.ready = True
            self.heartbeat("gpu worker ready")
            last_activity = time.monotonic()
            while not self.stop.is_set():
                pending = []
                for path in (self.root / "jobs").glob("*/request.json"):
                    state = read_json(path.parent / "status.json", {})
                    if state.get("status") not in {"complete", "error", "cancelled"}:
                        pending.append(path)
                if not pending:
                    if time.monotonic() - last_activity > self.idle_seconds:
                        break
                    self.stop.wait(1)
                    continue
                path = min(pending, key=lambda p: p.stat().st_mtime)
                self.active = path.parent
                try:
                    request = read_json(path)
                    validate_action(
                        {k: request[k] for k in ["photo", "prompt", "anchor", "force", "seeds"]}
                    )
                    if request["model"]["identity"] != model["identity"]:
                        raise ValueError("The request belongs to another model checkpoint/runtime")
                    if request["layout"] != self.service.layout:
                        raise ValueError("The request has a different native video contract")
                    photo_path = self.root / "images" / (request["photo"] + ".png")
                    if sha256(photo_path) != request["input_sha256"]:
                        raise ValueError("Input photograph hash mismatch")
                    image = np.array(Image.open(photo_path).convert("RGB"))
                    if image.shape != (480, 720, 3):
                        raise ValueError("The prepared photo has the wrong size")
                    self.update("preparing_caption")
                    backend.prepare_prompt(request["prompt"])
                    control = backend.signal(model_force(request))
                    rows = []
                    for index, seed in enumerate(request["seeds"]):
                        since = time.monotonic()
                        self.update(f"generating future {index + 1} of 4")
                        torch.cuda.reset_peak_memory_stats()

                        def step(n, total, future=index + 1):
                            self.update(f"future {future} of 4 · step {n} of {total}")

                        frames = backend.generate(
                            image, request["prompt"], control, seed, on_step=step
                        )
                        torch.cuda.synchronize()
                        if frames.shape != (49, 480, 720, 3) or frames.dtype != np.uint8:
                            raise ValueError("Unexpected generated frame contract")
                        raw_folder = self.raw_root / self.active.name
                        raw_folder.mkdir(exist_ok=True)
                        raw = raw_folder / f"{index}.npy"
                        np.save(raw, frames, allow_pickle=False)
                        temporary = self.active / f"{index}.partial.mp4"
                        imageio.mimwrite(
                            temporary, frames, fps=8, codec="libx264", quality=8, macro_block_size=1
                        )
                        video = self.active / f"{index}.mp4"
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
                                "seconds": time.monotonic() - since,
                                "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
                            }
                        )
                    result = {
                        "request": request,
                        "videos": rows,
                        **self.service.layout,
                        "model": model,
                        "unique_raw_videos": len({r["raw_sha256"] for r in rows}),
                        "scope": "photo futures; physics and uncertainty remain unevaluated",
                    }
                    write_json(self.active / "result.json", result)
                    write_json(
                        self.active / "status.json",
                        {"status": "complete", "stage": "four futures ready"},
                    )
                except Cancelled:
                    write_json(
                        self.active / "status.json",
                        {"status": "cancelled", "stage": "generation cancelled"},
                    )
                except Exception as error:
                    write_json(
                        self.active / "status.json",
                        {
                            "status": "error",
                            "stage": "generation failed",
                            "error": str(error),
                        },
                    )
                    write_json(
                        self.active / "failure.json",
                        {"error": str(error), "traceback": traceback.format_exc()},
                    )
                finally:
                    self.active = None
                    last_activity = time.monotonic()
                    self.heartbeat("gpu worker ready")
        finally:
            self.ready = False
            self.stop.set()
            self.heartbeat("gpu worker stopped; cached outputs remain available")
            thread.join(timeout=6)


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=int, default=1800)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Start the worker in a Slurm allocation")
    worker = PhotoWorker(
        args.repo, args.spool, idle_seconds=args.idle_seconds, raw_root=args.raw_root
    )
    with (worker.root / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: worker.stop.set())
        worker.loop()


if __name__ == "__main__":
    main()
