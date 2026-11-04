"""Durable four-sample photo requests shared with a separately allocated worker."""

import base64
import hashlib
import io
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

SIZE = (832, 480)
BACKENDS = {
    "streamforce": {
        "size": [832, 480],
        "frames": 81,
        "fps": 16,
        "minimum_label": "zero",
        "steps": 4,
    },
    "force-prompting": {
        "size": [720, 480],
        "frames": 49,
        "fps": 8,
        "minimum_label": "minimum",
        "steps": 50,
    },
    "force-prompting-20": {
        "size": [720, 480],
        "frames": 49,
        "fps": 8,
        "minimum_label": "minimum",
        "steps": 20,
    },
}
IMAGE_LIMIT = 12 * 1024 * 1024
JOB_ID = re.compile(r"[a-f0-9]{32}\Z")
PHOTO_ID = re.compile(r"photo-[a-f0-9]{64}\Z")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def prepare_photo(value, size=SIZE):
    if not isinstance(value, str) or "," not in value or len(value) > IMAGE_LIMIT * 4 // 3 + 100:
        raise ValueError("Choose a JPEG, PNG or WebP photo up to 12 MiB")
    prefix, encoded = value.split(",", 1)
    if prefix not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}:
        raise ValueError("Choose a JPEG, PNG or WebP photo")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > IMAGE_LIMIT:
            raise ValueError("Photo exceeds 12 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Use one still photograph")
            if min(image.size) < 64 or image.width * image.height > 32_000_000:
                raise ValueError("Use a photo between 64 pixels and 32 megapixels")
            image = ImageOps.exif_transpose(image).convert("RGB")
            original_size = list(image.size)
            image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
            pixels = image.tobytes()
            stream = io.BytesIO()
            image.save(stream, format="PNG")
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("Could not decode the photograph") from error
    content = stream.getvalue()
    return {
        "id": "photo-" + hashlib.sha256(pixels).hexdigest(),
        "original_size": original_size,
        "size": list(size),
        "preparation": "EXIF orientation, RGB conversion, centre crop to model aspect ratio",
        "input_sha256": hashlib.sha256(content).hexdigest(),
    }, content


def validate_action(data):
    if not isinstance(data, dict) or set(data) != {"photo", "prompt", "anchor", "force", "seeds"}:
        raise ValueError("Supply a photo, caption, push point, force and four seeds")
    if not isinstance(data["photo"], str) or not PHOTO_ID.fullmatch(data["photo"]):
        raise ValueError("Unknown photo")
    if not isinstance(data["prompt"], str) or not 1 <= len(data["prompt"].strip()) <= 2000:
        raise ValueError("Describe the scene in 1 to 2000 characters")
    result = {"photo": data["photo"], "prompt": data["prompt"].strip()}
    for name in ("anchor", "force"):
        values = data[name]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in values)
        ):
            raise ValueError("The push point and vector must contain two finite numbers")
        # JavaScript JSON normalizes -0 to 0; both encode the same action.
        result[name] = [0.0 if v == 0 else float(v) for v in values]
    if any(not 0 <= v <= 1 for v in result["anchor"]):
        raise ValueError("Choose a push point inside the photograph")
    if math.hypot(*result["force"]) > 1 + 1e-12:
        raise ValueError("Push strength must be between zero and one")
    seeds = data["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 4
        or any(type(v) is not int or not 0 <= v < 2**31 for v in seeds)
        or len(set(seeds)) != 4
    ):
        raise ValueError("Use four distinct integer seeds between 0 and 2147483647")
    result["seeds"] = seeds.copy()
    return result


def model_force(request):
    dx, dy = request["force"]
    return {
        "model": {
            "mode": "point",
            "force": math.hypot(dx, dy),
            "angle": math.degrees(math.atan2(-dy, dx)) % 360 if dx or dy else 0.0,
            "x_pos": request["anchor"][0],
            "y_pos": 1 - request["anchor"][1],
        }
    }


class PhotoService:
    def __init__(self, root, *, backend=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ("images", "jobs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        config = read_json(self.root / "CONFIG.json")
        if backend is None:
            backend = config["backend"] if config else "streamforce"
        if backend not in BACKENDS:
            raise ValueError("Unknown photo backend")
        if config and config["backend"] != backend:
            raise ValueError("Use a separate photo session for a different backend")
        self.backend = backend
        self.layout = BACKENDS[backend]
        if not config:
            write_json(self.root / "CONFIG.json", {"backend": backend})
        self.lock = threading.Lock()

    def worker(self):
        model = read_json(self.root / "MODEL.json", {})
        heartbeat = read_json(self.root / "WORKER.json", {})
        ready = bool(model.get("identity") and heartbeat.get("ready"))
        ready = ready and model.get("backend_id", self.backend) == self.backend
        fresh = 0 <= time.time() - heartbeat.get("updated_at", 0) < 90
        ready = ready and fresh
        stage = heartbeat.get("stage", "waiting for GPU worker")
        if heartbeat and not fresh:
            stage = "worker heartbeat expired; recorded runs remain available"
        return {
            "model": model,
            "ready": ready,
            "stage": stage,
        }

    def state(self):
        recordings = []
        for path in (self.root / "jobs").glob("*/result.json"):
            status = read_json(path.parent / "status.json", {})
            result = read_json(path)
            request = result.get("request", {})
            if status.get("status") == "complete" and len(result.get("videos", [])) == 4:
                recordings.append(
                    {
                        "id": path.parent.name,
                        "prompt": request.get("prompt", ""),
                        "force": request.get("force", []),
                        "created_at": request.get("created_at", 0),
                    }
                )
        return {
            **self.worker(),
            **self.layout,
            "backend": self.backend,
            "photos": [
                read_json(p)
                for p in sorted((self.root / "images").glob("*.json"))
                if p.with_suffix(".png").is_file()
            ],
            "recordings": sorted(recordings, key=lambda r: r["created_at"], reverse=True)[:20],
        }

    def add_image(self, data):
        if not isinstance(data, dict) or set(data) != {"image"}:
            raise ValueError("Supply image pixels only")
        metadata, content = prepare_photo(data["image"], tuple(self.layout["size"]))
        with self.lock:
            path = self.root / "images" / metadata["id"]
            if not path.with_suffix(".json").exists() or not path.with_suffix(".png").exists():
                if (
                    not path.with_suffix(".json").exists()
                    and len(list((self.root / "images").glob("*.json"))) >= 12
                ):
                    raise ValueError("This session already contains twelve photos")
                temporary = path.with_suffix(".png.tmp")
                temporary.write_bytes(content)
                write_json(path.with_suffix(".json"), metadata)
                temporary.replace(path.with_suffix(".png"))
        return {**metadata, "image": "data:image/png;base64," + base64.b64encode(content).decode()}

    def image(self, photo_id):
        if not PHOTO_ID.fullmatch(photo_id):
            raise KeyError(photo_id)
        path = self.root / "images" / photo_id
        metadata = read_json(path.with_suffix(".json"))
        if metadata is None or not path.with_suffix(".png").is_file():
            raise KeyError(photo_id)
        return {
            **metadata,
            "image": "data:image/png;base64,"
            + base64.b64encode(path.with_suffix(".png").read_bytes()).decode(),
        }

    def directory(self, job_id):
        if not JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        path = self.root / "jobs" / job_id
        if not (path / "request.json").is_file():
            raise KeyError(job_id)
        return path

    def job(self, job_id):
        root = self.directory(job_id)
        state = read_json(
            root / "status.json", {"status": "queued", "stage": "waiting for GPU worker"}
        )
        result = read_json(root / "result.json")
        if state["status"] == "complete":
            if result is None:
                return {"id": job_id, "status": "running", "stage": "receiving the four videos"}
            request = read_json(root / "request.json")
            videos = result.get("videos", [])
            valid = (
                len(videos) == 4
                and all(
                    result.get(key) == request["layout"][key]
                    for key in ("size", "frames", "fps", "steps")
                )
                and result.get("request") == request
                and all(
                    row.get("file") == f"{i}.mp4"
                    and row.get("seed") == request["seeds"][i]
                    and re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))
                    for i, row in enumerate(videos)
                )
            )
            if not valid:
                return {"id": job_id, "status": "error", "stage": "invalid four-sample result"}
            if any(not (root / row["file"]).is_file() for row in videos):
                return {"id": job_id, "status": "running", "stage": "receiving the four videos"}
            if any(sha256(root / row["file"]) != row["sha256"] for row in videos):
                return {"id": job_id, "status": "error", "stage": "video integrity check failed"}
        return {"id": job_id, **state, "result": result}

    def submit(self, data):
        request = validate_action(data)
        photo = self.image(request["photo"])
        worker = self.worker()
        request["model"] = worker["model"]
        request["layout"] = self.layout
        if photo["size"] != self.layout["size"]:
            raise ValueError("Open this photo again for the selected backend")
        request["input_sha256"] = photo["input_sha256"]
        key = hashlib.sha256(
            json.dumps(
                {**request, "model": worker["model"].get("identity")},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.lock:
            pending = 0
            for path in (self.root / "jobs").glob("*/request.json"):
                previous = read_json(path)
                state = self.job(path.parent.name)
                pending += state["status"] in {"queued", "running"}
                if previous.get("cache_key") == key and state["status"] == "complete":
                    if all(
                        sha256(path.parent / row["file"]) == row["sha256"]
                        for row in state["result"]["videos"]
                    ):
                        return {"id": path.parent.name, "source": "cached replay"}
                if (
                    previous.get("cache_key") == key
                    and state["status"] in {"queued", "running"}
                    and not (path.parent / "cancel.json").exists()
                ):
                    return {"id": path.parent.name, "source": "existing request"}
            if not worker["ready"]:
                raise ValueError("The GPU worker is unavailable; reconnect it before generating")
            if pending >= 2:
                raise ValueError("Two requests are already queued; wait or cancel one")
            job_id = uuid.uuid4().hex
            root = self.root / "jobs" / job_id
            root.mkdir()
            write_json(
                root / "request.json", {**request, "cache_key": key, "created_at": time.time()}
            )
            return {"id": job_id, "source": "live model request"}

    def cancel(self, job_id):
        root = self.directory(job_id)
        write_json(root / "cancel.json", {"requested_at": time.time()})
        return {"id": job_id, "status": "cancellation requested"}

    def media(self, job_id, name):
        root = self.directory(job_id)
        if not re.fullmatch(r"[0-3]\.mp4", name):
            raise KeyError(name)
        result = read_json(root / "result.json", {})
        if name not in {row["file"] for row in result.get("videos", [])}:
            raise KeyError(name)
        path = root / name
        if not path.is_file():
            raise KeyError(name)
        row = next(row for row in result["videos"] if row["file"] == name)
        if sha256(path) != row["sha256"]:
            raise KeyError(name)
        return path
