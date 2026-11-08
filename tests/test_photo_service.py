"""Photo input, force coordinates, durable four-seed jobs and media identity."""

import base64
import hashlib
import io
import json
import threading
import time
from http.client import HTTPConnection

import pytest
from PIL import Image

from click2future.demo.photo_server import make_server
from click2future.demo.photo_service import (
    PhotoService,
    model_force,
    prepare_photo,
    read_json,
    validate_action,
    write_json,
)


def image_data(size=(160, 120), format="JPEG"):
    image = Image.new("RGB", size, (50, 90, 140))
    stream = io.BytesIO()
    image.save(stream, format=format)
    mime = "jpeg" if format == "JPEG" else format.lower()
    return "data:image/" + mime + ";base64," + base64.b64encode(stream.getvalue()).decode()


@pytest.fixture
def service(tmp_path):
    value = PhotoService(tmp_path)
    write_json(
        tmp_path / "MODEL.json", {"identity": "fixture-only", "backend": "transport fixture"}
    )
    write_json(tmp_path / "WORKER.json", {"ready": True, "updated_at": time.time()})
    return value


def action(service):
    photo = service.add_image({"image": image_data()})
    return {
        "photo": photo["id"],
        "prompt": "a vase on a table",
        "anchor": [0.2, 0.7],
        "force": [0.3, 0],
        "seeds": [0, 1, 2, 3],
    }


def complete(service, receipt):
    root = service.directory(receipt["id"])
    request = read_json(root / "request.json")
    rows = []
    for i, seed in enumerate(request["seeds"]):
        # These bytes exercise transfer identity; they are not generated videos.
        raw = ("transport fixture " + str(seed)).encode()
        (root / f"{i}.mp4").write_bytes(raw)
        rows.append({"file": f"{i}.mp4", "seed": seed, "sha256": hashlib.sha256(raw).hexdigest()})
    write_json(root / "result.json", {"request": request, "videos": rows, **request["layout"]})
    write_json(root / "status.json", {"status": "complete", "stage": "fixture complete"})
    return root


def test_normal_jpeg_is_oriented_cropped_and_recorded_without_physics(service):
    metadata, raw = prepare_photo(image_data())
    assert metadata["original_size"] == [160, 120]
    assert metadata["size"] == [832, 480]
    assert Image.open(io.BytesIO(raw)).size == (832, 480)
    assert hashlib.sha256(raw).hexdigest() == metadata["input_sha256"]
    result = service.add_image({"image": image_data()})
    assert service.image(result["id"])["image"] == result["image"]
    assert not {"mass", "scene_spec", "instance_masks", "reference"} & result.keys()
    assert len(service.state()["photos"]) == 1
    service.add_image({"image": image_data()})
    assert len(service.state()["photos"]) == 1


def test_incomplete_photo_publication_is_hidden_and_can_be_repaired(service):
    photo = service.add_image({"image": image_data()})
    path = service.root / "images" / (photo["id"] + ".png")
    expected = path.read_bytes()
    path.unlink()
    assert service.state()["photos"] == []
    with pytest.raises(KeyError):
        service.image(photo["id"])
    service.add_image({"image": image_data()})
    assert path.read_bytes() == expected


def test_stale_heartbeat_cannot_still_say_that_the_worker_is_ready(service):
    write_json(
        service.root / "WORKER.json",
        {"ready": True, "updated_at": time.time() - 100, "stage": "gpu worker ready"},
    )
    state = service.state()
    assert not state["ready"]
    assert "expired" in state["stage"]


@pytest.mark.parametrize("heartbeat_age", [0, 100])
def test_frozen_replay_stays_offline_without_an_expired_worker_warning(service, heartbeat_age):
    request = action(service)
    receipt = service.submit(request)
    complete(service, receipt)
    write_json(service.root / "OFFLINE_REPLAY.json", {"scope": "presentation snapshot"})
    write_json(
        service.root / "WORKER.json",
        {"ready": True, "updated_at": time.time() - heartbeat_age},
    )
    state = service.state()
    assert not state["ready"]
    assert state["stage"] == "recorded replay only"
    assert service.submit(request) == {"id": receipt["id"], "source": "cached replay"}
    request["force"] = [-0.3, 0]
    with pytest.raises(ValueError, match="unavailable"):
        service.submit(request)


@pytest.mark.parametrize(
    "value",
    ["not an image", "data:image/png;base64,??", image_data((32, 32)), image_data(format="GIF")],
)
def test_unsupported_photo_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        prepare_photo(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("force", [float("nan"), 0]),
        ("force", [2, 0]),
        ("force", [True, 0]),
        ("anchor", [-1, 0.5]),
        ("seeds", [0, 0, 1, 2]),
        ("seeds", [True, 1, 2, 3]),
        ("seeds", [0, 1, 2]),
        ("prompt", ""),
        ("photo", "../../somewhere"),
    ],
)
def test_invalid_requests_fail_before_queueing(service, field, value):
    request = action(service)
    request[field] = value
    with pytest.raises(ValueError):
        service.submit(request)
    assert not list((service.root / "jobs").iterdir())


def test_force_convention_maps_image_down_to_negative_model_angle(service):
    request = action(service)
    request["force"] = [0, 0.5]
    signal = model_force(validate_action(request))["model"]
    assert signal == {
        "mode": "point",
        "force": 0.5,
        "angle": 270.0,
        "x_pos": 0.2,
        "y_pos": pytest.approx(0.3),
    }
    request["force"] = [0, 0]
    assert model_force(request)["model"]["force"] == 0


def test_fallback_uses_its_native_crop_length_and_minimum_control_label(tmp_path):
    service = PhotoService(tmp_path, backend="force-prompting")
    photo = service.add_image({"image": image_data()})
    assert photo["size"] == [720, 480]
    assert service.state()["frames"] == 49
    assert service.state()["fps"] == 8
    assert service.state()["minimum_label"] == "minimum"
    with pytest.raises(ValueError, match="separate"):
        PhotoService(tmp_path, backend="streamforce")


def test_jobs_deduplicate_and_cache_is_identified_even_offline(service):
    request = action(service)
    receipt = service.submit(request)
    assert service.submit(request) == {"id": receipt["id"], "source": "existing request"}
    complete(service, receipt)
    assert service.job(receipt["id"])["status"] == "complete"
    write_json(service.root / "WORKER.json", {"ready": False, "updated_at": time.time()})
    assert service.submit(request) == {"id": receipt["id"], "source": "cached replay"}
    request["force"] = [-0.3, 0]
    with pytest.raises(ValueError, match="unavailable"):
        service.submit(request)


def test_signed_zero_survives_the_browser_json_round_trip(service):
    request = action(service)
    request["force"] = [0.3, -0.0]
    receipt = service.submit(request)
    complete(service, receipt)
    request["force"] = [0.3, 0]
    assert service.submit(request) == {"id": receipt["id"], "source": "cached replay"}


def test_recorded_run_picker_excludes_pending_and_incomplete_results(service):
    receipt = service.submit(action(service))
    assert service.state()["recordings"] == []
    root = complete(service, receipt)
    assert [row["id"] for row in service.state()["recordings"]] == [receipt["id"]]
    result = read_json(root / "result.json")
    result["videos"].pop()
    write_json(root / "result.json", result)
    assert service.state()["recordings"] == []


def test_restarting_the_same_model_does_not_invalidate_a_verified_cached_result(service):
    request = action(service)
    receipt = service.submit(request)
    complete(service, receipt)
    write_json(
        service.root / "MODEL.json",
        {
            "identity": "fixture-only",
            "allocation_job_id": "new-allocation",
            "worker_host": "another-worker",
            "raw_root": "/another/temporary/path",
        },
    )
    assert service.submit(request) == {"id": receipt["id"], "source": "cached replay"}


def test_every_seed_and_file_hash_is_required_for_completion(service):
    receipt = service.submit(action(service))
    root = complete(service, receipt)
    result = read_json(root / "result.json")
    result["videos"].pop()
    write_json(root / "result.json", result)
    assert service.job(receipt["id"])["status"] == "error"
    complete(service, receipt)
    (root / "0.mp4").write_bytes(b"tampered")
    assert service.job(receipt["id"])["status"] == "error"
    with pytest.raises(KeyError):
        service.media(receipt["id"], "0.mp4")


def test_output_cannot_relabel_its_action_or_sampler(service):
    receipt = service.submit(action(service))
    root = complete(service, receipt)
    result = read_json(root / "result.json")
    result["request"]["force"] = [-0.3, 0.0]
    write_json(root / "result.json", result)
    assert service.job(receipt["id"])["status"] == "error"
    complete(service, receipt)
    result = read_json(root / "result.json")
    result["steps"] += 1
    write_json(root / "result.json", result)
    assert service.job(receipt["id"])["status"] == "error"


def test_cancelled_and_failed_attempts_are_preserved(service):
    request = action(service)
    first = service.submit(request)
    service.cancel(first["id"])
    second = service.submit(request)
    assert second["id"] != first["id"]
    assert (service.directory(first["id"]) / "cancel.json").exists()
    assert service.job(second["id"])["status"] == "queued"
    request["force"] = [-0.3, 0]
    with pytest.raises(ValueError, match="Two requests"):
        service.submit(request)


def test_photo_http_session_ranges_and_path_boundaries(service):
    receipt = service.submit(action(service))
    root = complete(service, receipt)
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    try:
        conn.request("GET", "/api/state")
        response = conn.getresponse()
        state = json.loads(response.read())
        conn.request(
            "POST",
            "/api/generate",
            json.dumps(action(service)),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        path = f"/media/{receipt['id']}/0.mp4"
        conn.request("GET", path, headers={"Range": "bytes=2-7"})
        response = conn.getresponse()
        assert response.status == 206
        assert response.read() == (root / "0.mp4").read_bytes()[2:8]
        conn.request("GET", path, headers={"Range": "bytes=9999-"})
        response = conn.getresponse()
        assert response.status == 416
        response.read()
        conn.request("GET", f"/media/{receipt['id']}/../request.json")
        response = conn.getresponse()
        assert response.status == 404
        response.read()
        conn.request(
            "POST",
            f"/api/jobs/{receipt['id']}/cancel",
            "{}",
            {"X-Demo-Token": state["token"], "Origin": "http" + "://elsewhere"},
        )
        response = conn.getresponse()
        assert response.status == 403
        response.read()
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
