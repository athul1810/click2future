"""Image input preserves observable pixels and cannot invent simulator ground truth."""

import base64
import io
import json
import threading
import time
from http.client import HTTPConnection

import numpy as np
import pytest
from PIL import Image, ImageDraw

from click2future.data import generate_dataset
from click2future.demo.images import uploaded_scene
from click2future.demo.server import make_server
from click2future.demo.service import DemoService, png


def picture(*, objects=2):
    image = Image.new("RGB", (64, 64), (211, 215, 221))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 63, 63), outline=(45, 50, 49))
    if objects >= 1:
        draw.ellipse((12, 15, 22, 25), fill=(101, 190, 40))
    if objects >= 2:
        draw.rectangle((38, 37, 47, 48), fill=(143, 91, 190))
    if objects >= 3:
        draw.rectangle((40, 12, 46, 19), fill=(201, 110, 42))
    return np.asarray(image)


def encoded(image, format="PNG"):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def test_pixels_define_masks_and_only_one_observed_frame():
    first = picture()
    ref = uploaded_scene(png(first))
    assert np.array_equal(ref["frames"], first[None]) and ref["frame_count"] == 32
    assert ref["instance_masks"].shape == (1, 2, 64, 64)
    assert not np.any(ref["instance_masks"][0, 0] & ref["instance_masks"][0, 1])
    assert set(ref) == {
        "frames",
        "frame_count",
        "instance_masks",
        "object_valid",
        "selected_object_id",
        "force_xy",
        "input_source",
    }
    assert ref["instance_masks"][0, 0, 20, 17]
    assert ref["instance_masks"][0, 1, 42, 42]


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_opaque_png_modes_keep_the_supplied_visible_pixels(mode):
    image = Image.fromarray(picture())
    image = (
        image.convert("P", palette=Image.Palette.ADAPTIVE, colors=8, dither=Image.Dither.NONE)
        if mode == "P"
        else image.convert(mode)
    )
    ref = uploaded_scene(encoded(image))
    assert np.array_equal(ref["frames"][0], np.asarray(image.convert("RGB")))


@pytest.mark.parametrize(
    "invalid", ["size", "photo", "alpha", "jpeg", "one", "three", "data", "large"]
)
def test_unsupported_images_are_rejected_without_silent_conversion(invalid):
    image = Image.fromarray(picture())
    if invalid == "size":
        image = image.resize((128, 128))
    elif invalid == "photo":
        image = Image.fromarray(
            np.random.default_rng(4).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        )
    elif invalid == "alpha":
        image = image.convert("RGBA")
        image.putalpha(100)
    elif invalid in {"one", "three"}:
        image = Image.fromarray(picture(objects=1 if invalid == "one" else 3))
    value = encoded(image, "JPEG" if invalid == "jpeg" else "PNG")
    if invalid == "data":
        value = "data:image/png;base64,bad!"
    elif invalid == "large":
        value += "A" * 90000
    with pytest.raises(ValueError):
        uploaded_scene(value)


class PixelSpy:
    force_limit = 0.5
    metadata = {"id": "fixture", "checkpoint_sha256": "test-only"}

    def __init__(self):
        self.calls = []

    def predict(self, first, mask, force, *, seeds, future_frames, cancelled):
        self.calls.append((first.copy(), mask.copy(), force.copy(), seeds))
        return np.repeat(first[None, None], len(seeds) * (future_frames + 1), axis=0).reshape(
            len(seeds), future_frames + 1, *first.shape
        )


@pytest.fixture
def service(tmp_path):
    dataset = tmp_path / "data"
    generated = generate_dataset(
        dataset,
        group_count=1,
        variants_per_group=1,
        seed=850000,
        config={"frame_count": 4, "resolution": 64, "fps": 8},
    )
    identifier = generated["manifest"]["samples"][0]["sample_id"]
    app = DemoService(dataset, [identifier], {"fixture": PixelSpy()}, cache=tmp_path / "cache")
    yield app
    app.close()


def test_image_generation_uses_no_reference_or_hidden_labels(service, monkeypatch):
    first = picture()
    result = service.add_image({"image": png(first)})
    scene = result["scene"]
    assert scene["reference_worlds"] == [] and "user upload" in scene["split"]
    assert service.add_image({"image": png(first)})["scene"]["id"] == scene["id"]
    assert len(service.presets) == 2
    with pytest.raises(ValueError, match="PNG pixels only"):
        service.add_image({"image": png(first), "mass": 1})

    def no_physics(*args, **kwargs):
        pytest.fail("An image-only input cannot call the reference simulator")

    monkeypatch.setattr("click2future.demo.service.run_rollout", no_physics)
    payload = {
        "scene": scene["id"],
        "object": 1,
        "force": [0.0, 0.2],
        "seeds": [0, 1, 2, 3],
        "models": ["fixture"],
    }
    identifier = service.submit(payload)
    deadline = time.monotonic() + 10
    while (
        service.job(identifier)["status"] in {"queued", "running"} and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    job = service.job(identifier)
    assert job["status"] == "complete" and job["result"]["reference"] is None
    assert job["result"]["frames"] == 32 and job["result"]["seeds"] == [0, 1, 2, 3]
    actual_first, mask, force, seeds = service.models["fixture"].calls[0]
    assert np.array_equal(actual_first, first)
    assert np.array_equal(mask, service.presets[scene["id"]]["instance_masks"][0, 1])
    assert np.allclose(force, [0.0, 0.2])
    with pytest.raises(ValueError, match="no simulator ground truth"):
        service.reference(
            {"scene": scene["id"], "object": 1, "force": [0, 0.2], "world": scene["id"]}
        )


def test_image_http_requires_session_and_keeps_normal_request_limit(service):
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def call(method, path, data=None, headers=None):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(method, path, body=data, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read()
        connection.close()
        return result

    try:
        token = json.loads(call("GET", "/api/state")[1])["token"]
        headers = {"X-Demo-Token": token, "Content-Type": "application/json"}
        body = json.dumps({"image": png(picture())})
        assert call("POST", "/api/images", body)[0] == 403
        assert call("POST", "/api/images", body, headers)[0] == 200
        assert call("POST", "/api/images", "A" * 90001, headers)[0] == 400
        assert call("POST", "/api/generate", "A" * 4097, headers)[0] == 400
        assert len(service.presets) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_upload_inventory_is_bounded_and_duplicate_images_use_no_extra_slot(service):
    values = []
    for index in range(8):
        first = picture().copy()
        first[np.all(first == [101, 190, 40], axis=-1), 0] += index
        value = png(first)
        values.append(value)
        service.add_image({"image": value})
    assert len(service.presets) == 9
    service.add_image({"image": values[0]})
    first = picture().copy()
    first[np.all(first == [101, 190, 40], axis=-1), 0] += 9
    with pytest.raises(ValueError, match="Eight images"):
        service.add_image({"image": png(first)})
    assert len(service.presets) == 9
