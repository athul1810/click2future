"""Loopback photo interface; inference happens in a separately allocated worker."""

import argparse
import json
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from .photo_service import BACKENDS, IMAGE_LIMIT, PhotoService


def make_server(service, *, port=8770):
    token = secrets.token_urlsafe(32)
    assets = {
        "/": ("photo.html", "text/html"),
        "/assets/photo.css": ("photo.css", "text/css"),
        "/assets/photo.js": ("photo.js", "text/javascript"),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def allowed_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def headers_for(self, code, size, mime):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; media-src 'self'; "
                "script-src 'self'; style-src 'self'; frame-ancestors 'none'",
            )

        def send(self, code, payload, mime="application/json"):
            body = json.dumps(payload).encode() if mime == "application/json" else payload
            self.headers_for(code, len(body), mime)
            self.end_headers()
            self.wfile.write(body)

        def video(self, path):
            size = path.stat().st_size
            header = self.headers.get("Range")
            start, end, code = 0, size - 1, 200
            if header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
                if not match or not any(match.groups()):
                    self.send(416, {"error": "Unsupported video range"})
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(0, size - int(right))
                if start >= size or end < start:
                    self.headers_for(416, 0, "video/mp4")
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            self.headers_for(code, end - start + 1, "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_GET(self):
            if not self.allowed_host():
                self.send(403, {"error": "Local host required"})
                return
            path = urlsplit(self.path).path
            try:
                if path in assets:
                    name, mime = assets[path]
                    self.send(
                        200, files("click2future.demo").joinpath("static", name).read_bytes(), mime
                    )
                elif path == "/api/state":
                    self.send(200, {**service.state(), "token": token})
                elif path.startswith("/api/photos/"):
                    self.send(200, service.image(path.removeprefix("/api/photos/")))
                elif path.startswith("/api/jobs/"):
                    self.send(200, service.job(path.removeprefix("/api/jobs/")))
                elif path.startswith("/media/"):
                    _, _, job_id, name = path.split("/")
                    self.video(service.media(job_id, name))
                else:
                    self.send(404, {"error": "Not found"})
            except (KeyError, ValueError, FileNotFoundError):
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            expected = "http" + "://" + self.headers.get("Host", "")
            if (
                not self.allowed_host()
                or self.headers.get("X-Demo-Token") != token
                or self.headers.get("Origin", expected) != expected
            ):
                self.send(403, {"error": "Local demo session required"})
                return
            try:
                path = urlsplit(self.path).path
                length = int(self.headers.get("Content-Length", "0"))
                limit = IMAGE_LIMIT * 4 // 3 + 512 if path == "/api/photos" else 4096
                if not 0 < length <= limit:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if path == "/api/photos":
                    self.send(200, service.add_image(data))
                elif path == "/api/generate":
                    self.send(202, service.submit(data))
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    if data:
                        raise ValueError("Cancellation takes no additional data")
                    self.send(200, service.cancel(path.split("/")[3]))
                else:
                    self.send(404, {"error": "Not found"})
            except KeyError:
                self.send(404, {"error": "Unknown request"})
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--backend", choices=tuple(BACKENDS))
    args = parser.parse_args()
    server = make_server(PhotoService(args.spool, backend=args.backend), port=args.port)
    address = "http" + f"://127.0.0.1:{server.server_address[1]}"
    print(f"photo interface: {address}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
