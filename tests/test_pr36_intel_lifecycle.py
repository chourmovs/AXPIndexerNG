import os
from types import SimpleNamespace

from axp_client.rag.intel_sycl_backend import (
    INTEL_LOAD_HARD_TIMEOUT_S,
    INTEL_LOAD_STALL_WARN_AFTER_S,
    INTEL_LOAD_WARN_AFTER_S,
    LOOPBACK,
    IntelSyclBackend,
)


def make_backend(tmp_path, **kwargs):
    runtime = tmp_path / "runtime"; runtime.mkdir()
    server = runtime / "llama-server.exe"; server.write_bytes(b"MZ")
    model = tmp_path / "model.gguf"; model.write_bytes(b"GGUF")
    config = SimpleNamespace(context_size=1024, max_answer_tokens=8, temperature=.2,
                             top_p=.8, top_k=20, repeat_penalty=1.0)
    return IntelSyclBackend(model, config, runtime, server, **kwargs)


def test_watchdog_thresholds_and_loopback_are_fixed():
    assert (INTEL_LOAD_WARN_AFTER_S, INTEL_LOAD_STALL_WARN_AFTER_S, INTEL_LOAD_HARD_TIMEOUT_S) == (120, 240, 600)
    assert LOOPBACK == "127.0.0.1"


def test_ephemeral_auth_is_unique_private_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path / "data"))
    item = make_backend(tmp_path)
    item._create_auth(); first, artifact = item._api_key, item._auth_file
    assert artifact.read_text() == first
    # Windows does not implement POSIX permission bits: its stat mode normally
    # reports the writable/readable DOS attributes even after os.open(..., 0o600).
    # The file instead inherits the current user's private temporary-data ACL.
    if os.name != "nt":
        assert artifact.stat().st_mode & 0o077 == 0
    item.close(); assert not artifact.exists()
    item._create_auth(); assert item._api_key != first
    assert "api_key" not in str(item.health())
    item.close()


def test_snapshot_uses_fake_clock_without_real_waiting(tmp_path):
    clock = [0.0]; item = make_backend(tmp_path, monotonic=lambda: clock[0])
    item._load_progress.update(active=True, phase="runtime_initializing", started_monotonic=0,
                               last_native_activity_monotonic=0)
    clock[0] = 121
    assert item.model_load_progress()["active"] and item.model_load_progress()["elapsed_s"] == 121
    clock[0] = 300
    assert item.model_load_progress()["last_native_activity_age_s"] == 300
