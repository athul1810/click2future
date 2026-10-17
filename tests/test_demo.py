"""Observable inference, real sampler parity, cancellation and local HTTP boundaries."""

import base64
import copy
import hashlib
import io
import json
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from click2future.data import generate_dataset, load_rollout
from click2future.demo.models import CheckpointModel
from click2future.demo.server import make_server
from click2future.demo.service import DemoService
from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.cvae_loss import cvae_loss
from click2future.training.sample_cvae import sample_checkpoint
from click2future.training.sample_diffusion import sample_videos
from click2future.training.train_cvae import (
    dataset_index,
    model_batch,
    save_checkpoint,
)
from click2future.training.train_diffusion import TrainConfig, Trainer


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo-data")
    generate_dataset(
        root,
        group_count=10,
        variants_per_group=1,
        seed=842000,
        config={"frame_count": 4, "resolution": 16, "fps": 8},
    )
    return root


def first_id(dataset, split="train"):
    return next(
        row["sample_id"]
        for row in json.loads((dataset / "dataset_manifest.json").read_text())["samples"]
        if row["split"] == split
    )


class ObservableSpy:
    """Test-only adapter; deliberately has no access to a simulator reference."""

    force_limit = 0.5

    def __init__(self, *, block=False):
        self.metadata = {"id": "fixture", "checkpoint_sha256": "test-only"}
        self.calls = []
        self.started, self.release = threading.Event(), threading.Event()
        if not block:
            self.release.set()

    def predict(self, first, mask, force, *, seeds, future_frames, cancelled):
        self.calls.append((first.copy(), mask.copy(), force.copy(), list(seeds)))
        self.started.set()
        while not self.release.wait(0.01):
            if cancelled.is_set():
                raise InterruptedError
        if cancelled.is_set():
            raise InterruptedError
        videos = np.broadcast_to(first, (len(seeds), future_frames + 1, *first.shape)).copy()
        for index, seed in enumerate(seeds):
            videos[index, 1:, 0, 0] = seed % 256
        return videos


@pytest.fixture
def service(dataset, tmp_path):
    model = ObservableSpy()
    instance = DemoService(
        dataset, [first_id(dataset)], {"fixture": model}, cache=tmp_path / "cache"
    )
    yield instance
    instance.close()


def request(service):
    scene = service.state()["scenes"][0]
    return {
        "scene": scene["id"],
        "object": scene["default_object"],
        "force": [0.0, 0.25],
        "seeds": [11, 12, 13, 14],
        "models": ["fixture"],
    }


def finish(service, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = service.job(identifier)
        if job["status"] not in {"queued", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Test inference did not finish")


def decode(strip):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(strip.split(",", 1)[1]))))


def test_observable_boundary_lossless_frames_and_explicit_reference(service):
    data = request(service)
    state = service.state()
    assert not {"scene_spec", "positions", "mass", "friction"} & state["scenes"][0].keys()
    assert state["scenes"][0]["split"] == "train"
    job = finish(service, service.submit(data))
    assert job["status"] == "complete"
    result = job["result"]
    first, mask, force, seeds = service.models["fixture"].calls[0]
    ref = service.presets[data["scene"]]
    np.testing.assert_array_equal(first, ref["frames"][0])
    np.testing.assert_array_equal(mask, ref["instance_masks"][0, data["object"]])
    np.testing.assert_array_equal(force, [0, 0.25])
    assert seeds == data["seeds"] and result["seeds"] == seeds
    strip = decode(result["models"][0]["strip"])
    assert strip.shape == (4 * 16, 4 * 16, 3)
    for row, seed in enumerate(seeds):
        np.testing.assert_array_equal(strip[row * 16 : (row + 1) * 16, :16], first)
        np.testing.assert_array_equal(strip[row * 16, 16], [seed, seed, seed])
    assert "simulator reference" in result["reference"]["source"]
    np.testing.assert_array_equal(decode(result["reference"]["strip"])[:16, :16], first)


@pytest.mark.parametrize("changed", ["force", "seeds", "object", "checkpoint", "source"])
def test_replay_identity_includes_every_sampling_input(service, changed):
    data = request(service)
    first = finish(service, service.submit(data))["result"]
    replay = finish(service, service.submit(data))["result"]
    assert first["models"][0]["source"] == "live model"
    assert replay["models"][0]["source"] == "cached replay"
    assert first["models"][0]["strip"] == replay["models"][0]["strip"]
    assert len(service.models["fixture"].calls) == 1
    altered = copy.deepcopy(data)
    if changed == "force":
        altered["force"] = [0, -0.25]
    elif changed == "seeds":
        altered["seeds"][0] += 100
    elif changed == "object":
        altered["object"] = next(
            obj["id"]
            for obj in service.state()["scenes"][0]["objects"]
            if obj["id"] != altered["object"]
        )
    else:
        service.models["fixture"].metadata[changed] = "changed"
    assert finish(service, service.submit(altered))["result"]["models"][0]["source"] == "live model"
    assert len(service.models["fixture"].calls) == 2


def test_hidden_reference_physics_does_not_change_model_cache(service):
    data = request(service)
    first = finish(service, service.submit(data))["result"]
    scene = service.presets[data["scene"]]["scene_spec"]
    scene["objects"][data["object"]]["hidden_physics"]["mass"] *= 5
    changed = finish(service, service.submit(data))["result"]
    assert changed["models"][0]["source"] == "cached replay"
    assert first["models"][0]["strip"] == changed["models"][0]["strip"]
    assert first["reference"]["strip"] != changed["reference"]["strip"]


def test_tampered_cache_fails_without_serving_it(service):
    finish(service, service.submit(request(service)))
    archive = next(service.cache.glob("*.npz"))
    archive.write_bytes(archive.read_bytes() + b"changed")
    job = finish(service, service.submit(request(service)))
    assert job["status"] == "error" and "result" not in job


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object", True),
        ("object", -1),
        ("object", 1000),
        ("force", [0, float("nan")]),
        ("force", [True, 0]),
        ("force", [0, 0.51]),
        ("seeds", [[], 1, 2, 3]),
        ("seeds", [1, 1, 2, 3]),
        ("seeds", [True, 1, 2, 3]),
        ("models", [["fixture"]]),
        ("models", ["fixture", "fixture"]),
        ("scene", "../outside"),
        ("extra", "hidden physics"),
    ],
)
def test_malformed_or_undeclared_requests_rejected(service, field, value):
    data = request(service)
    data[field] = value
    with pytest.raises(ValueError):
        service.submit(data)
    assert not service.jobs and not service.models["fixture"].calls


@pytest.mark.parametrize(
    "split", ["test", "id_test", "shape_ood", "texture_ood", "force_ood", "physics_test", "unknown"]
)
def test_final_test_presets_are_refused_before_loading_any_arrays(tmp_path, monkeypatch, split):
    sample = "reserved-example"
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"samples": [{"sample_id": sample, "split": split}]})
    )

    def forbidden_load(*_args, **_kwargs):
        pytest.fail("A reserved split was loaded before the demo rejected it")

    monkeypatch.setattr("click2future.demo.service.load_rollout", forbidden_load)
    with pytest.raises(ValueError, match="Final-test"):
        DemoService(tmp_path, [sample], {"fixture": ObservableSpy()}, cache=tmp_path / "cache")


def test_running_and_queued_cancellation_are_bounded_and_publish_nothing(dataset, tmp_path):
    model = ObservableSpy(block=True)
    service = DemoService(dataset, [first_id(dataset)], {"fixture": model}, cache=tmp_path)
    try:
        first = service.submit(request(service))
        assert model.started.wait(3)
        second = service.submit(request(service))
        with pytest.raises(ValueError, match="queue is full"):
            service.submit(request(service))
        service.cancel(second)
        service.cancel(first)
        assert finish(service, first)["status"] == "cancelled"
        assert finish(service, second)["status"] == "cancelled"
        model.release.set()
    finally:
        service.close()
    assert len(model.calls) == 1 and not list(tmp_path.glob("*.npz"))
    assert "result" not in service.job(first) and "result" not in service.job(second)


def test_cancellation_at_worker_start_cannot_leave_a_running_job(service):
    reached, release = threading.Event(), threading.Event()
    original_lock = service.lock

    class StartGate:
        def __enter__(self):
            if threading.current_thread().name.startswith("c2f-inference") and not reached.is_set():
                reached.set()
                assert release.wait(3)
            original_lock.acquire()

        def __exit__(self, *_):
            original_lock.release()

    service.lock = StartGate()
    identifier = service.submit(request(service))
    try:
        assert reached.wait(3)
        service.cancel(identifier)
    finally:
        release.set()
    service.executor.submit(lambda: None).result(timeout=3)
    assert service.job(identifier)["status"] == "cancelled"
    assert not service.models["fixture"].calls


def test_local_http_assets_session_origin_and_job_flow(service):
    server = make_server(service, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def call(method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    try:
        code, headers, body = call("GET", "/")
        assert code == 200 and b"generate four futures" in body
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        for path in ("/assets/app.js", "/assets/style.css"):
            assert call("GET", path)[0] == 200
        assert call("GET", "/assets/../models.py")[0] == 404
        assert call("GET", "/api/state", headers={"Host": "outside.invalid"})[0] == 403
        token = json.loads(call("GET", "/api/state")[2])["token"]
        headers = {"X-Demo-Token": token, "Content-Type": "application/json"}
        payload = json.dumps(request(service))
        assert call("POST", "/api/generate", payload)[0] == 403
        assert call("POST", "/api/generate", payload, {**headers, "Origin": "null"})[0] == 403
        assert call("POST", "/api/generate", "x" * 4097, headers)[0] == 400
        assert call("POST", "/api/generate", "[]", headers)[0] == 400
        code, _, body = call("POST", "/api/generate", payload, headers)
        identifier = json.loads(body)["id"]
        assert code == 202 and finish(service, identifier)["status"] == "complete"
        result = json.loads(call("GET", "/api/jobs/" + identifier)[2])
        assert result["status"] == "complete" and "cancel" not in result
        assert call("POST", "/api/jobs/unknown/cancel", "{}", headers)[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.fixture(scope="module")
def checkpoint_fixtures(dataset, tmp_path_factory):
    # One unit-test optimizer update exercises real checkpoint transport. These
    # tiny synthetic fixtures are never used as project results or demo assets.
    root = tmp_path_factory.mktemp("demo-checkpoints")
    config = TrainConfig(steps=2, base_channels=8, diffusion_steps=10, limit_train=1)
    trainer = Trainer(dataset, config)
    trainer.train_step()
    diffusion = root / "diffusion.pt"
    trainer.save(diffusion)
    settings = yaml.safe_load(Path("configs/training/cvae.yaml").read_text())
    settings["data"].update(total_frames=4, resolution=16)
    settings["model"].update(future_frames=3, base_channels=8, latent_dim=4)
    selected, rows, provenance = dataset_index(dataset)
    source = Path(__file__).parents[1] / "src/click2future/models/cvae.py"
    provenance["source_sha256"] = {
        "cvae.py": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    }
    model = ConditionalVideoVAE(**settings["model"])
    optimizer = torch.optim.AdamW(model.parameters())
    batch = model_batch(dataset, selected["train"][:1], rows, settings, device="cpu")
    output = model(batch.first_frame, batch.selected_mask, batch.force_xy, batch.future)
    cvae_loss(output, batch.future, beta=0.001).total.backward()
    optimizer.step()
    cvae = root / "cvae.pt"
    save_checkpoint(
        cvae,
        model=model,
        optimizer=optimizer,
        state={"step": 1},
        config=settings,
        provenance=provenance,
        train_generator=torch.Generator().manual_seed(4),
        shuffle_generator=torch.Generator().manual_seed(5),
    )
    return {"diffusion": diffusion, "cvae": cvae}


@pytest.mark.parametrize("kind", ["diffusion", "cvae"])
def test_checkpoint_adapter_equals_offline_sampler_and_preserves_rng(
    dataset, checkpoint_fixtures, tmp_path, kind
):
    rng = torch.get_rng_state().clone()
    checkpoint = checkpoint_fixtures[kind]
    model = CheckpointModel(kind, checkpoint, sampling_steps=3)
    assert torch.equal(rng, torch.get_rng_state())
    sample = first_id(dataset)
    ref = load_rollout(dataset, sample)
    seeds = [11, 12, 13, 14]
    actual = model.predict(
        ref["frames"][0],
        ref["selected_mask"],
        ref["force_xy"],
        seeds=seeds,
        future_frames=3,
        cancelled=threading.Event(),
    )
    assert torch.equal(rng, torch.get_rng_state())
    if kind == "diffusion":
        expected = sample_videos(
            model.model,
            model.diffusion,
            ref["frames"][0],
            ref["selected_mask"],
            ref["force_xy"],
            seeds=seeds,
            future_frames=3,
            sampling_steps=3,
        )
    else:
        sample_checkpoint(
            checkpoint,
            dataset,
            [sample],
            tmp_path / "offline",
            seeds=seeds,
            run_id="unit-test-only",
        )
        with np.load(tmp_path / "offline" / (sample + ".npz"), allow_pickle=False) as archive:
            expected = archive["videos"]
    np.testing.assert_array_equal(actual, expected)
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        model.predict(
            ref["frames"][0],
            ref["selected_mask"],
            ref["force_xy"],
            seeds=seeds,
            future_frames=3,
            cancelled=event,
        )


@pytest.mark.parametrize("kind", ["diffusion", "cvae"])
def test_zero_update_checkpoint_is_not_a_demo_result(checkpoint_fixtures, tmp_path, kind):
    state = torch.load(checkpoint_fixtures[kind], weights_only=True)
    if kind == "diffusion":
        state["step"] = 0
    else:
        state["state"]["step"] = 0
    path = tmp_path / "zero.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="trained checkpoint"):
        CheckpointModel(kind, path, sampling_steps=3)
