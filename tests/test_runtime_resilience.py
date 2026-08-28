import json
import threading

import pytest
from axp_core.locking import FileLock, daemon_instance_running, daemon_lock_path
from axp_core.runtime import atomic_write_json
from axp_daemon.service import StatePublisher
from axp_tray import process
from axp_tray.state import read_daemon_state, should_auto_restart


def test_atomic_json_retries_transient_replace_failure(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_replace = __import__("os").replace
    attempts = []

    def flaky_replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError("antivirus contention")
        return real_replace(source, destination)

    monkeypatch.setattr("axp_core.runtime.os.replace", flaky_replace)
    monkeypatch.setattr("axp_core.runtime.time.sleep", lambda _: None)
    atomic_write_json(target, {"answer": 42})

    assert json.loads(target.read_text(encoding="utf-8")) == {"answer": 42}
    assert len(attempts) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_json_permanent_failure_is_bounded_and_cleans_up(tmp_path, monkeypatch):
    attempts = 0

    def failed_replace(*_):
        nonlocal attempts
        attempts += 1
        raise OSError("disk unavailable")

    monkeypatch.setattr("axp_core.runtime.os.replace", failed_replace)
    monkeypatch.setattr("axp_core.runtime.time.sleep", lambda _: None)
    with pytest.raises(OSError, match="disk unavailable"):
        atomic_write_json(tmp_path / "state.json", {"answer": 42})

    assert attempts == 5
    assert not list(tmp_path.glob("*.tmp"))


def test_heartbeat_thread_recovers_after_publication_failure(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    publisher = StatePublisher(interval_s=0.001)
    real_write = publisher._write
    recovered = threading.Event()
    calls = 0

    def flaky_write():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary contention")
        real_write()
        recovered.set()

    monkeypatch.setattr(publisher, "_write", flaky_write)
    publisher.start()
    assert recovered.wait(1)
    assert publisher.thread.is_alive()
    publisher.close()
    assert "publication failed" in caplog.text
    assert "publication recovered" in caplog.text


def test_stale_threshold_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    state_path = tmp_path / "runtime" / "daemon_state.json"
    atomic_write_json(state_path, {"state": "idle", "heartbeat_ms": 11_000})
    assert not read_daemon_state(now_ms=100_000)["stale"]
    assert read_daemon_state(now_ms=102_000)["stale"]


def test_lock_probe_and_watchdog_restart_gate(tmp_path):
    db = tmp_path / "catalog.db"
    runtime = tmp_path / "runtime"
    stale = {"state": "error", "stale": True}
    lock = FileLock(daemon_lock_path(db, runtime)).acquire()
    try:
        assert daemon_instance_running(db, runtime)
        assert not should_auto_restart(stale, "running", True, 0, 100, daemon_instance_present=True)
    finally:
        lock.release()

    assert not daemon_instance_running(db, runtime)
    assert should_auto_restart(stale, "running", True, 0, 100, daemon_instance_present=False)
    assert not should_auto_restart(stale, "stopped", True, 0, 100, daemon_instance_present=False)


def test_startup_race_lock_held_suppresses_spawn(tmp_path):
    db = tmp_path / "catalog.db"
    lock = FileLock(daemon_lock_path(db, tmp_path / "runtime")).acquire()
    try:
        missing_state = {"state": "stopped", "stale": True}
        assert not should_auto_restart(
            missing_state, "running", True, 0, 100,
            daemon_instance_present=daemon_instance_running(db, tmp_path / "runtime"),
        )
    finally:
        lock.release()


def test_manual_restart_waits_for_lock_release(monkeypatch):
    calls = []
    probes = iter((True, True, False))
    monkeypatch.setattr(process, "send_control", lambda command: calls.append(("stop", command)))
    monkeypatch.setattr(process, "daemon_instance_running", lambda _: next(probes))
    monkeypatch.setattr(process.time, "sleep", lambda _: None)
    monkeypatch.setattr(process, "start_daemon", lambda settings: calls.append(("start", settings)) or "new")
    assert process.restart_daemon({"db_path": "catalog.db"}) == "new"
    assert [kind for kind, _ in calls] == ["stop", "start"]


def test_manual_restart_refuses_spawn_while_lock_remains_held(monkeypatch):
    calls = []
    clock = iter((0, 0, 2))
    monkeypatch.setattr(process, "send_control", lambda command: None)
    monkeypatch.setattr(process, "daemon_instance_running", lambda _: True)
    monkeypatch.setattr(process.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(process.time, "sleep", lambda _: None)
    monkeypatch.setattr(process, "start_daemon", lambda settings: calls.append(settings))
    with pytest.raises(RuntimeError, match="lock is still held"):
        process.restart_daemon({"db_path": "catalog.db"}, timeout=1)
    assert not calls
