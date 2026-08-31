import json
import shutil
from types import SimpleNamespace

import pytest
from axp_client import server
from axp_client.rag import accelerator_manager as accelerator_module
from axp_client.rag import model_manager as model_module
from axp_client.rag.accelerator_manager import AcceleratorError, AcceleratorManager
from axp_client.rag.model_manager import DownloadJob, ModelManager, ModelManagerError
from axp_core import runtime


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "::1", "127.0.0.2"))
def test_loopback_bind_hosts_are_accepted(host):
    assert runtime.validate_loopback_host(host) == host


@pytest.mark.parametrize("host", ("0.0.0.0", "::", "192.168.1.20", "example.test"))
def test_non_loopback_bind_hosts_are_rejected(host):
    with pytest.raises(ValueError, match="web_host_must_be_loopback"):
        runtime.validate_loopback_host(host)


def _settings_paths(monkeypatch, tmp_path):
    paths = {"data": tmp_path, "runtime": tmp_path / "runtime", "logs": tmp_path / "logs",
             "settings": tmp_path / "settings.json"}
    paths["runtime"].mkdir(); paths["logs"].mkdir()
    monkeypatch.setattr(runtime, "runtime_paths", lambda: paths)
    monkeypatch.setattr(runtime, "installation_root", lambda: tmp_path)
    return paths


def test_settings_bootstrap_and_single_value_validation(monkeypatch, tmp_path):
    paths = _settings_paths(monkeypatch, tmp_path)
    settings = runtime.load_settings()
    assert paths["settings"].is_file() and settings["web_port"] == 8765
    paths["settings"].write_text(json.dumps({"web_port": True, "scan_interval_s": 42}), encoding="utf-8")
    settings = runtime.load_settings()
    assert settings["web_port"] == 8765 and settings["scan_interval_s"] == 42


def test_corrupt_settings_are_preserved_and_backup_recovered(monkeypatch, tmp_path):
    paths = _settings_paths(monkeypatch, tmp_path)
    paths["settings"].write_text("{broken", encoding="utf-8")
    backup = paths["settings"].with_suffix(".json.bak")
    backup.write_text(json.dumps({"web_port": 9123}), encoding="utf-8")
    assert runtime.load_settings()["web_port"] == 9123
    assert list(tmp_path.glob("settings.json.corrupt-*"))
    assert json.loads(paths["settings"].read_text(encoding="utf-8"))["web_port"] == 9123


def test_settings_save_rotates_valid_backup_and_bounds_recovery(monkeypatch, tmp_path):
    paths = _settings_paths(monkeypatch, tmp_path)
    old = {**runtime.DEFAULT_SETTINGS, "web_port": 9001}
    runtime.atomic_write_json(paths["settings"], old)
    runtime.save_settings({**old, "web_port": 9002})
    assert json.loads(paths["settings"].with_suffix(".json.bak").read_text())["web_port"] == 9001
    for index in range(5):
        paths["settings"].write_text("{broken", encoding="utf-8")
        runtime.load_settings()
    assert len(list(tmp_path.glob("settings.json.corrupt-*"))) == runtime.MAX_SETTINGS_RECOVERY_FILES


def _prepared_manager(monkeypatch, tmp_path):
    manager = AcceleratorManager(tmp_path / "runtime")
    monkeypatch.setattr(accelerator_module, "validate_archive", lambda _archive: None)
    def extract(_archive, staging):
        (staging / "bin").mkdir()
        (staging / "bin" / "server.exe").write_bytes(b"new")
    monkeypatch.setattr(accelerator_module, "safe_extract", extract)
    monkeypatch.setattr(accelerator_module, "discover_binaries",
                        lambda _root: {"server_path": "bin/server.exe"})
    return manager


def test_accelerator_install_replaces_existing_runtime(monkeypatch, tmp_path):
    manager = _prepared_manager(monkeypatch, tmp_path)
    manager.install_archive(tmp_path / "archive.zip")
    assert manager.server_path().read_bytes() == b"new"
    shutil.rmtree(manager.runtime_root)
    manager.runtime_root.mkdir(parents=True); (manager.runtime_root / "old.txt").write_text("old")
    manager.install_archive(tmp_path / "archive.zip")
    assert manager.server_path().read_bytes() == b"new" and not (manager.runtime_root / "old.txt").exists()


def test_accelerator_failed_publish_restores_old_runtime(monkeypatch, tmp_path):
    manager = _prepared_manager(monkeypatch, tmp_path)
    manager.runtime_root.mkdir(parents=True); (manager.runtime_root / "old.txt").write_text("old")
    real_replace = manager._replace_with_retry
    calls = 0
    def fail_staging(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("locked")
        return real_replace(source, destination)
    monkeypatch.setattr(manager, "_replace_with_retry", fail_staging)
    with pytest.raises(PermissionError):
        manager.install_archive(tmp_path / "archive.zip")
    assert (manager.runtime_root / "old.txt").read_text() == "old"
    assert not list(tmp_path.glob(".b10516-*"))


def test_removal_failures_are_observable(monkeypatch, tmp_path):
    manager = AcceleratorManager(tmp_path / "runtime")
    manager.runtime_root.mkdir(parents=True)
    monkeypatch.setattr(accelerator_module.shutil, "rmtree", lambda _path: (_ for _ in ()).throw(PermissionError()))
    with pytest.raises(AcceleratorError, match="accelerator_remove_failed"):
        manager.remove()

    profile = SimpleNamespace(id="fixture")
    monkeypatch.setattr(model_module, "catalog_model", lambda _model_id: profile)
    monkeypatch.setattr(model_module, "load_settings", lambda: {"chat_active_model_id": None})
    models = ModelManager(tmp_path / "models")
    models.model_path("fixture").parent.mkdir(parents=True)
    with pytest.raises(ModelManagerError, match="model_remove_failed"):
        models.remove("fixture")


def test_unexpected_model_worker_failure_is_terminal(monkeypatch, tmp_path):
    manager = ModelManager(tmp_path)
    model = SimpleNamespace(id="fixture", size_bytes=1)
    monkeypatch.setattr(model_module.shutil, "disk_usage", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))
    job = DownloadJob("id", "fixture")
    manager._download(model, job)
    assert (job.state, job.error) == ("failed", "model_download_failed")


def test_security_headers_and_admin_body_limit(tmp_path):
    class Models:
        def catalog(self): return {"models": []}
        def start_benchmark(self, _profile): raise AssertionError("oversized body was parsed")

    from tests.test_client_server import running_server, post_json
    with running_server(tmp_path / "unused.db", lambda _: None, model_manager=Models()) as httpd:
        import http.client
        for path in ("/", "/api/models", "/health"):
            connection = http.client.HTTPConnection("127.0.0.1", httpd.server_port)
            connection.request("GET", path)
            response = connection.getresponse(); response.read()
            assert response.getheader("X-Frame-Options") == "DENY"
            assert response.getheader("Referrer-Policy") == "no-referrer"
            assert "default-src 'self'" in response.getheader("Content-Security-Policy")
            connection.close()
        status, body = post_json(httpd, "/api/models/benchmark", b"x" * (server.MAX_ADMIN_BODY + 1))
    assert status == 413 and json.loads(body)["error"] == "request_too_large"
