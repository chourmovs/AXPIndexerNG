import time

from axp_core.runtime import atomic_write_json, read_json, runtime_paths
from axp_daemon.service import DaemonControl, StatePublisher, send_control
from axp_tray.state import read_daemon_state, should_auto_restart


class Publisher:
    def __init__(self):
        self.value = {}

    def update(self, **values):
        self.value.update(values)


def test_atomic_state_write_and_stale_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    paths = runtime_paths()
    atomic_write_json(paths["state"], {"state": "idle", "heartbeat_ms": int(time.time() * 1000)})
    assert read_daemon_state()["state"] == "idle"
    atomic_write_json(paths["state"], {"state": "scanning", "heartbeat_ms": 1})
    stale = read_daemon_state(stale_after_s=1)
    assert stale["state"] == "error" and stale["stale"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_control_scan_pause_resume_and_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    publisher = Publisher()
    control = DaemonControl(publisher)
    send_control("scan"); control.poll(); assert control.scan_requested
    send_control("pause"); control.poll(); assert control.paused and publisher.value["state"] == "paused"
    send_control("resume"); control.poll(); assert not control.paused
    send_control("stop"); control.poll(); assert control.should_stop() and publisher.value["state"] == "stopping"


def test_heartbeat_json(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    publisher = StatePublisher()
    publisher._write()
    value = read_json(runtime_paths()["state"])
    assert value["pid"] and value["heartbeat_ms"] and value["state"] == "starting"


def test_watchdog_respects_intentional_stop_and_rate_limit():
    stale = {"stale": True}
    assert not should_auto_restart(stale, "stopped", True, 0, 100)
    assert not should_auto_restart(stale, "running", True, 50, 100)
    assert should_auto_restart(stale, "running", True, 40, 100)
    assert should_auto_restart({"state": "stopped", "stale": False}, "running", True, 0, 100)
