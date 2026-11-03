"""Loopback-only local inference with bounded synthetic PNG input."""

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .bundle import load_bundle
from .models import CheckpointModel
from .service import DemoService


def make_server(service, *, port=8765):
    token = secrets.token_urlsafe(32)
    assets = {
        "/": ("index.html", "text/html"),
        "/assets/style.css": ("style.css", "text/css"),
        "/assets/app.js": ("app.js", "text/javascript"),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, payload, mime="application/json"):
            body = json.dumps(payload).encode() if mime == "application/json" else payload
            self.send_response(code)
            self.send_header("Content-Type", mime + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; script-src 'self'; "
                "style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            actual = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{actual}", f"localhost:{actual}"}

        def do_GET(self):
            if not self.host_ok():
                self.send(403, {"error": "Local host required"})
                return
            path = urlsplit(self.path).path
            if path in assets:
                name, mime = assets[path]
                self.send(
                    200, files("click2future.demo").joinpath("static", name).read_bytes(), mime
                )
            elif path == "/api/state":
                self.send(200, {**service.state(), "token": token})
            elif path.startswith("/api/jobs/"):
                try:
                    self.send(200, service.job(path.removeprefix("/api/jobs/")))
                except KeyError:
                    self.send(404, {"error": "Unknown generation request"})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected = "http" + "://" + self.headers.get("Host", "")
            if (
                not self.host_ok()
                or self.headers.get("X-Demo-Token") != token
                or (origin and origin != expected)
            ):
                self.send(403, {"error": "Local demo session required"})
                return
            try:
                path = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= (90000 if path == "/api/images" else 4096):
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if path == "/api/images":
                    self.send(200, service.add_image(data))
                elif path == "/api/generate":
                    self.send(202, {"id": service.submit(data)})
                elif path == "/api/reference":
                    self.send(200, service.reference(data))
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    service.cancel(path[len("/api/jobs/") : -len("/cancel")])
                    self.send(200, {"status": "cancelled"})
                else:
                    self.send(404, {"error": "Not found"})
            except KeyError:
                self.send(404, {"error": "Unknown generation request"})
            except (ValueError, TypeError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--bundle", type=Path, help="Verified local checkpoint handoff directory")
    parser.add_argument("--models", nargs="+", help="One to four names in the bundle manifest")
    parser.add_argument("--samples", nargs="+", help="Explicit development/validation sample IDs")
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--checkpoint", action="append", help="DisplayName=local_checkpoint.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sampling-steps", type=int, default=32)
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=16,
        help="EDM Heun steps for supervised trajectory checkpoints",
    )
    parser.add_argument("--force-limit", type=float, default=0.5)
    parser.add_argument("--cache", type=Path, default=Path("outputs/demo-cache"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    bundled = {}
    if args.bundle:
        if args.dataset or args.checkpoint or not args.models:
            parser.error("Use --bundle with --models, without --dataset/--checkpoint")
        args.dataset, rows = load_bundle(args.bundle, args.models)
        args.checkpoint = [row["name"] + "=" + str(row["path"]) for row in rows]
        bundled = {row["name"]: row for row in rows}
    elif not args.dataset or not args.checkpoint or args.models:
        parser.error("Provide --dataset and --checkpoint, or --bundle and --models")
    if not 0 <= args.port <= 65535:
        parser.error("Invalid local port")
    import os
    import socket

    if socket.gethostname().lower() == "ccds-tc2" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("TC2 inference requires an allocated Slurm compute job")
    import torch

    torch.set_num_threads(4)
    samples = args.samples
    if samples is None:
        manifest = json.loads((args.dataset / "dataset_manifest.json").read_text())
        groups = {}
        for row in manifest["samples"]:
            if row["split"] == args.split and len(groups) < 8:
                groups.setdefault(row["scene_group_id"], row["sample_id"])
        samples = list(groups.values())
    models = {}
    for item in args.checkpoint:
        name, separator, path = item.partition("=")
        if not separator or not name or len(name) > 40 or name in models:
            parser.error("Use distinct DisplayName=checkpoint.pt arguments")
        models[name] = CheckpointModel(
            name,
            path,
            device=args.device,
            force_limit=args.force_limit,
            sampling_steps=args.sampling_steps,
            trajectory_steps=args.trajectory_steps,
        )
        if name in bundled:
            row = bundled[name]
            if models[name].metadata["training_steps"] != row["evaluated_checkpoint_step"]:
                parser.error("Checkpoint update count disagrees with bundle manifest")
            models[name].metadata.update(
                validation_quality=row["quality"], variant_notes=row["notes"]
            )
    service = DemoService(
        args.dataset, samples, models, cache=args.cache, force_limit=args.force_limit
    )
    server = make_server(service, port=args.port)
    print("demo ready: " + "http" + "://127.0.0.1:" + str(server.server_address[1]), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
