import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from axp_client.rag import model_manager as manager_module
from axp_client.rag.model_catalog import MODELS, ModelProfile
from axp_client.rag.model_manager import DownloadJob, ModelManager, ModelManagerError


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data = data
        self.status = status
        self.headers = headers or {}
        self.offset = 0

    def read(self, size=-1):
        if self.offset >= len(self.data):
            return b""
        end = len(self.data) if size < 0 else min(len(self.data), self.offset + size)
        value, self.offset = self.data[self.offset:end], end
        return value

    def getcode(self):
        return self.status

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class Opener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return next(self.responses)


@pytest.fixture
def fixture_model(monkeypatch):
    data = b"GGUF" + bytes(range(64))
    model = ModelProfile("fixture", "Fixture", "fast", "approved/repo", "a" * 40,
                         "fixture.gguf", hashlib.sha256(data).hexdigest(), len(data), "68 B")
    monkeypatch.setattr(manager_module, "MODELS", (model,))
    monkeypatch.setattr(manager_module, "catalog_model", lambda model_id: model if model_id == model.id else None)
    return model, data


def wait_for_job(job):
    for _ in range(200):
        if job.state not in manager_module.ACTIVE_DOWNLOAD_STATES:
            return
        time.sleep(.01)
    raise AssertionError("download did not finish")


def test_successful_download_publishes_atomically_and_writes_manifest(tmp_path, fixture_model):
    model, data = fixture_model
    manager = ModelManager(tmp_path, opener=Opener([Response(data)]))
    manager.start_download(model.id); wait_for_job(manager._job)
    assert manager._job.state == "ready"
    assert manager.model_path(model.id).read_bytes() == data
    manifest = json.loads(manager.manifest_path(model.id).read_text(encoding="utf-8"))
    assert manifest["actual_sha256"] == model.sha256
    assert not (manager.downloads_dir / "fixture.gguf.part").exists()


def test_partial_download_resumes_with_range_and_existing_hash(tmp_path, fixture_model):
    model, data = fixture_model; partial = data[:23]
    opener = Opener([Response(data[23:], 206, {"Content-Range": f"bytes 23-{len(data)-1}/{len(data)}"})])
    manager = ModelManager(tmp_path, opener=opener)
    part = manager.downloads_dir / "fixture.gguf.part"; part.write_bytes(partial)
    manager.start_download(model.id); wait_for_job(manager._job)
    assert opener.requests[0][0].get_header("Range") == "bytes=23-"
    assert manager.model_path(model.id).read_bytes() == data


def test_range_refusal_restarts_without_appending(tmp_path, fixture_model):
    model, data = fixture_model
    opener = Opener([Response(data), Response(data)])
    manager = ModelManager(tmp_path, opener=opener)
    (manager.downloads_dir / "fixture.gguf.part").write_bytes(data[:12])
    manager.start_download(model.id); wait_for_job(manager._job)
    assert len(opener.requests) == 2
    assert opener.requests[1][0].get_header("Range") is None
    assert manager.model_path(model.id).read_bytes() == data


@pytest.mark.parametrize(("payload", "error"), [
    (b"GGUFbad identity", "integrity_mismatch"),
    (b"NOPE" + bytes(range(64)), "invalid_gguf"),
])
def test_untrusted_download_is_not_published(tmp_path, fixture_model, monkeypatch, payload, error):
    model, _data = fixture_model
    if error == "invalid_gguf":
        model = ModelProfile(**{**model.__dict__, "sha256": hashlib.sha256(payload).hexdigest(),
                                "size_bytes": len(payload)})
        monkeypatch.setattr(manager_module, "catalog_model", lambda _model_id: model)
    manager = ModelManager(tmp_path, opener=Opener([Response(payload)]))
    manager.start_download(model.id); wait_for_job(manager._job)
    assert manager._job.state == "failed" and manager._job.error == error
    assert not manager.model_path(model.id).exists()
    assert not (manager.downloads_dir / "fixture.gguf.part").exists()


def test_cancellation_retains_resumable_partial(tmp_path, fixture_model):
    model, data = fixture_model
    response = Response(data[8:], 206, {"Content-Range": f"bytes 8-{len(data)-1}/{len(data)}"})
    manager = ModelManager(tmp_path, opener=Opener([response]))
    part = manager.downloads_dir / "fixture.gguf.part"; part.write_bytes(data[:8])
    manager._cancel.set(); job = DownloadJob("job", model.id, bytes_total=len(data))
    manager._download(model, job)
    assert job.state == "cancelled" and part.read_bytes() == data[:8]


def test_insufficient_disk_has_deterministic_failure(tmp_path, fixture_model, monkeypatch):
    model, data = fixture_model
    monkeypatch.setattr(manager_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    manager = ModelManager(tmp_path, opener=Opener([Response(data)]))
    manager.start_download(model.id); wait_for_job(manager._job)
    assert manager._job.state == "failed" and manager._job.error == "insufficient_disk"


def test_activation_failure_restores_settings_and_keeps_runtime(tmp_path, fixture_model, monkeypatch):
    model, data = fixture_model
    old = {"chat_active_model_id": "old", "chat_model_path": str(tmp_path / "old.gguf"),
           "chat_inference_device": "auto"}
    saved = []
    monkeypatch.setattr(manager_module, "load_settings", lambda: dict(old))
    monkeypatch.setattr(manager_module, "save_settings", lambda settings: saved.append(dict(settings)))
    class Runtime:
        busy = False
        def activate(self, _settings, _profile): raise RuntimeError("activation failed")
        def health(self): return {}
    manager = ModelManager(tmp_path, runtime=Runtime())
    manager.model_path(model.id).parent.mkdir(parents=True); manager.model_path(model.id).write_bytes(data)
    manager.manifest_path(model.id).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="activation failed"):
        manager.activate(model.id)
    assert saved[-1] == old


def test_active_model_cannot_be_removed(tmp_path, fixture_model, monkeypatch):
    model, _data = fixture_model
    monkeypatch.setattr(manager_module, "load_settings", lambda: {"chat_active_model_id": model.id})
    manager = ModelManager(tmp_path)
    with pytest.raises(ModelManagerError, match="model_active"):
        manager.remove(model.id)


def test_release_catalog_is_immutable_and_consistent():
    assert {model.id for model in MODELS} == {"qwen3-1.7b-q4km", "smollm3-3b-q4km"}
    for model in MODELS:
        assert len(model.revision) == 40 and all(char in "0123456789abcdef" for char in model.revision)
        assert len(model.sha256) == 64 and model.filename.endswith(".gguf")
        assert model.size_bytes > 0 and model.url.startswith("https://huggingface.co/")
