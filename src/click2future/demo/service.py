"""Bounded inference queue, observable-only boundary and verified replay cache."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from click2future.data import load_rollout
from click2future.sim import run_rollout

from .diagnostics import pixel_diagnostics
from .images import uploaded_scene


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def filmstrip(videos):
    # [K,T,H,W,3] -> [K*H,T*W,3], lossless pixels for synchronized canvas playback.
    return png(np.concatenate([np.concatenate(clip, axis=1) for clip in videos], axis=0))


def frame_count(scene):
    return scene.get("frame_count", len(scene["frames"]))


def validate_videos(videos, first, seeds, frames):
    videos = np.asarray(videos)
    if (
        videos.dtype != np.uint8
        or videos.shape != (len(seeds), frames, *first.shape)
        or not np.all(videos[:, 0] == first)
    ):
        raise ValueError("Model output must preserve frame 0 and match the native video contract")
    return videos


class DemoService:
    def __init__(self, dataset, sample_ids, models, *, cache, force_limit=0.5):
        self.models = dict(models)
        if not self.models or len(self.models) > 4:
            raise ValueError("Load one to four named checkpoints")
        if not np.isfinite(force_limit) or not 0 < force_limit <= 1.5:
            raise ValueError("Use a finite force limit in (0,1.5]")
        self.force_limit = min(force_limit, *(model.force_limit for model in self.models.values()))
        if not 1 <= len(sample_ids) <= 16 or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Choose one to sixteen distinct development presets")
        manifest = Path(dataset) / "dataset_manifest.json"
        self.splits = {}
        rows = []
        if manifest.is_file():
            rows = json.loads(manifest.read_text())["samples"]
            self.splits = {row["sample_id"]: row["split"] for row in rows}
        self.presets = {}
        self.reference_worlds = {}
        for sample_id in sample_ids:
            if self.splits and self.splits.get(sample_id) not in {"train", "validation"}:
                raise ValueError("Final-test or unsupported splits cannot be demo presets")
            if self.splits and sample_id not in self.splits:
                raise ValueError("Preset is absent from the dataset manifest")
            ref = load_rollout(dataset, sample_id)
            if (
                len(ref["frames"]) < 2
                or not np.allclose(np.diff(ref["timestamps"]), 1 / 8, rtol=0, atol=1e-8)
                or not np.isclose(ref["force_duration"], 0.25, rtol=0, atol=1e-7)
            ):
                raise ValueError("Demo scenes require 8 FPS and a fixed 0.25-second push")
            self.presets[sample_id] = ref
            condition = next(
                (row.get("condition_id") for row in rows if row["sample_id"] == sample_id), None
            )
            worlds = {sample_id: ref}
            if condition is not None:
                for row in rows:
                    if (
                        row.get("condition_id") == condition
                        and row["split"] == self.splits[sample_id]
                        and row["sample_id"] != sample_id
                    ):
                        other = load_rollout(dataset, row["sample_id"])
                        if not np.array_equal(other["frames"][0], ref["frames"][0]):
                            raise ValueError("Reference worlds must share the exact input image")
                        worlds[row["sample_id"]] = other
            self.reference_worlds[sample_id] = worlds
        if not self.presets:
            raise ValueError("Provide at least one preset")
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.jobs, self.lock = {}, threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="c2f-inference")

    def state(self):
        scenes = []
        with self.lock:
            inputs = [
                (name, ref, self.splits.get(name), list(self.reference_worlds[name]))
                for name, ref in self.presets.items()
            ]
        for name, ref, split, worlds in inputs:
            masks = ref["instance_masks"][0]
            ids = np.flatnonzero(ref["object_valid"])
            hit_map = np.full(masks.shape[1:], -1, dtype=int)
            objects = []
            for obj in ids:
                y, x = np.nonzero(masks[obj])
                if not len(x):
                    continue
                hit_map[masks[obj]] = int(obj)
                objects.append(
                    {"id": int(obj), "center": [float(x.mean() + 0.5), float(y.mean() + 0.5)]}
                )
            scenes.append(
                {
                    "id": name,
                    "image": png(ref["frames"][0]),
                    "hit_map": hit_map.tolist(),
                    "objects": objects,
                    "size": list(ref["frames"].shape[1:3]),
                    "frames": frame_count(ref),
                    "default_object": ref["selected_object_id"],
                    "default_force": ref["force_xy"].tolist(),
                    "split": split or "operator-selected example; split unverified",
                    "input_source": ref.get("input_source", "simulator preset"),
                    "reference_worlds": worlds,
                }
            )
        return {
            "scenes": scenes,
            "models": [m.metadata for m in self.models.values()],
            "force_limit": self.force_limit,
            "fps": 8,
            "notice": "research preview · checkpoint quality still needs validation",
        }

    def add_image(self, data):
        if not isinstance(data, dict) or set(data) != {"image"}:
            raise ValueError(
                "Supply PNG pixels only; no masks, scene recipes or physical properties"
            )
        ref = uploaded_scene(data["image"])
        identifier = "upload-" + hashlib.sha256(ref["frames"][0].tobytes()).hexdigest()
        with self.lock:
            if any(job["status"] in {"queued", "running"} for job in self.jobs.values()):
                raise ValueError("Wait for generation or cancel it before opening an image")
            if identifier not in self.presets:
                if sum(name.startswith("upload-") for name in self.presets) >= 8:
                    raise ValueError("Eight images are open; restart the local demo to clear them")
                self.reference_worlds[identifier] = {}
                self.splits[identifier] = "user upload; not benchmark-scored"
                self.presets[identifier] = ref
        return {"scene": next(row for row in self.state()["scenes"] if row["id"] == identifier)}

    def request(self, data):
        if not isinstance(data, dict) or set(data) != {
            "scene",
            "object",
            "force",
            "seeds",
            "models",
        }:
            raise ValueError("Provide scene, object, force, seeds and models only")
        if not isinstance(data["scene"], str) or data["scene"] not in self.presets:
            raise ValueError("Unknown scene")
        ref = self.presets[data["scene"]]
        obj = data["object"]
        if (
            isinstance(obj, bool)
            or not isinstance(obj, int)
            or not 0 <= obj < len(ref["object_valid"])
            or not ref["object_valid"][obj]
            or not ref["instance_masks"][0, obj].any()
        ):
            raise ValueError("Select a visible object")
        if (
            not isinstance(data["force"], list)
            or len(data["force"]) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in data["force"])
        ):
            raise ValueError("Force must contain two numbers")
        force = np.asarray(data["force"], dtype=np.float32)
        if not np.isfinite(force).all() or np.linalg.norm(force) > self.force_limit + 1e-6:
            raise ValueError("Push is outside the declared model force range")
        seeds, names = data["seeds"], data["models"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 4
            or any(
                isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**31 for s in seeds
            )
            or len(set(seeds)) != 4
        ):
            raise ValueError("Choose four distinct nonnegative integer seeds")
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) or n not in self.models for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Choose distinct loaded models")
        return ref, obj, force, seeds, names

    def reference(self, data):
        """Change only the labelled simulator panel, without invoking a model/cache."""
        if not isinstance(data, dict) or set(data) != {"scene", "object", "force", "world"}:
            raise ValueError("Provide scene, object, force and reference world only")
        _, obj, force, _, _ = self.request(
            {key: data[key] for key in ("scene", "object", "force")}
            | {"seeds": [0, 1, 2, 3], "models": [next(iter(self.models))]}
        )
        worlds = self.reference_worlds[data["scene"]]
        if not worlds:
            raise ValueError("An uploaded image has no simulator ground truth or known physics")
        if not isinstance(data["world"], str) or data["world"] not in worlds:
            raise ValueError("Unknown reference world")
        ref = worlds[data["world"]]
        scene = copy.deepcopy(ref["scene_spec"])
        scene["action"].update(selected_object_id=obj, force_xy=force.tolist())
        rollout = run_rollout(
            scene,
            **{
                key: ref["metadata"][key]
                for key in ("frame_count", "fps", "physics_hz", "resolution")
            },
        )
        if not np.array_equal(rollout["frames"][0], ref["frames"][0]):
            raise ValueError("Simulator reference changed the supplied image")
        return {
            "source": "simulator reference, one hidden-physics world",
            "world": data["world"],
            "strip": filmstrip(rollout["frames"][None]),
            "physical_properties": [o["hidden_physics"] for o in scene["objects"]],
            "notice": "reference only; model inputs and generated videos are unchanged",
        }

    def submit(self, data):
        self.request(data)
        with self.lock:
            if sum(job["status"] in {"queued", "running"} for job in self.jobs.values()) >= 2:
                raise ValueError("Generation queue is full; wait or cancel the current request")
            for old in list(self.jobs):
                if len(self.jobs) >= 16 and self.jobs[old]["status"] not in {"queued", "running"}:
                    del self.jobs[old]
            job_id = secrets.token_hex(12)
            job = {
                "status": "queued",
                "progress": "waiting for the model",
                "cancel": threading.Event(),
            }
            self.jobs[job_id] = job
        future = self.executor.submit(self._run, job, copy.deepcopy(data))
        with self.lock:
            job["future"] = future
        return job_id

    def job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError("Unknown generation request")
            return {key: value for key, value in job.items() if key not in {"cancel", "future"}}

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError("Unknown generation request")
            if job["status"] in {"queued", "running"}:
                job["cancel"].set()
                job.update(status="cancelled", progress="generation cancelled")
                if "future" in job:
                    job["future"].cancel()

    def _run(self, job, data):
        event = job["cancel"]
        try:
            with self.lock:
                if event.is_set():
                    return
                job.update(status="running", progress="generating futures")
            ref, obj, force, seeds, names = self.request(data)
            first, mask = ref["frames"][0].copy(), ref["instance_masks"][0, obj].copy()
            results = []
            for name in names:
                if event.is_set():
                    return
                model = self.models[name]
                identity = {
                    "format": 1,
                    "model": model.metadata,
                    "first": hashlib.sha256(first.tobytes()).hexdigest(),
                    "mask": hashlib.sha256(mask.tobytes()).hexdigest(),
                    "force": force.tolist(),
                    "seeds": seeds,
                    "frames": frame_count(ref),
                    "shape": list(first.shape),
                }
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                archive, meta = self.cache / (key + ".npz"), self.cache / (key + ".json")
                started, source = time.perf_counter(), "live model"
                with self.lock:
                    job["progress"] = f"generating with {name}"
                if archive.exists() and meta.exists():
                    saved = json.loads(meta.read_text())
                    if (
                        saved["identity"] != identity
                        or saved["sha256"] != hashlib.sha256(archive.read_bytes()).hexdigest()
                    ):
                        raise ValueError("Replay cache changed; use a fresh cache directory")
                    with np.load(archive, allow_pickle=False) as saved:
                        videos = saved["videos"]
                    source = "cached replay"
                else:
                    # This call cannot access reference state, future labels or hidden physics.
                    videos = model.predict(
                        first,
                        mask,
                        force,
                        seeds=seeds,
                        future_frames=frame_count(ref) - 1,
                        cancelled=event,
                    )
                    videos = validate_videos(videos, first, seeds, frame_count(ref))
                    if event.is_set():
                        return
                    temporary = archive.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, videos=videos)
                    temporary.replace(archive)
                    temporary_meta = meta.with_suffix(".tmp.json")
                    temporary_meta.write_text(
                        json.dumps(
                            {
                                "identity": identity,
                                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                            }
                        )
                    )
                    temporary_meta.replace(meta)
                validate_videos(videos, first, seeds, frame_count(ref))
                details = dict(model.metadata)
                if details.get("world_condition") is True:
                    from click2future.models.world_motion import WORLD_CONTRACT, sample_world

                    details["sampled_world_prior"] = {
                        "contract": WORLD_CONTRACT,
                        "properties": ["mass", "friction", "restitution"],
                        "object_order": "sampler canonical observed slots",
                        "draws": {str(seed): sample_world(seed)[0].tolist() for seed in seeds},
                        "scope": "sampled prior used by this model; not the scene's true physics",
                    }
                results.append(
                    {
                        "name": name,
                        "source": source,
                        "seconds": time.perf_counter() - started,
                        "strip": filmstrip(videos),
                        "details": details,
                        "pixel_tracks": pixel_diagnostics(
                            videos, ref["instance_masks"][0], ref["object_valid"]
                        ),
                    }
                )
            if event.is_set():
                return
            reference = (
                self.reference(
                    {
                        "scene": data["scene"],
                        "object": obj,
                        "force": force.tolist(),
                        "world": data["scene"],
                    }
                )
                if self.reference_worlds[data["scene"]]
                else None
            )
            result = {
                "models": results,
                "seeds": seeds,
                "force": force.tolist(),
                "object": obj,
                "frames": frame_count(ref),
                "fps": 8,
                "size": list(first.shape[:2]),
                "reference": reference,
                "notice": "Different hidden physical properties can give different valid outcomes.",
                "scene": data["scene"],
                "split": self.splits.get(data["scene"], "split unverified"),
                "input_source": ref.get("input_source", "simulator preset"),
            }
            with self.lock:
                if not event.is_set():
                    job.update(status="complete", progress="ready", result=result)
        except InterruptedError:
            with self.lock:
                job.update(status="cancelled", progress="generation cancelled")
        except Exception:
            import traceback

            traceback.print_exc()
            with self.lock:
                if not event.is_set():
                    job.update(
                        status="error", progress="generation failed; check the local server log"
                    )

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"].set()
        self.executor.shutdown(wait=True, cancel_futures=True)
