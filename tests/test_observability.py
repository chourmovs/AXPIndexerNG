from collections import namedtuple
from pathlib import Path

import pytest
from axp_daemon.service import DaemonControl, StatePublisher
from axp_tray.progress import RollingThroughput, estimate_eta_seconds, format_bytes, format_duration, progress_estimate


def test_progress_estimates_are_honest():
    assert progress_estimate(500, 1000).label == "~50%"
    assert progress_estimate(999, 1000).percent < 100
    assert progress_estimate(0).percent is None
    assert progress_estimate(1200, 1000).exceeded_baseline
    assert progress_estimate(1200, 1000).percent is None
    assert progress_estimate(3, scan_complete=True).percent == 100


def test_rolling_rates_eta_and_formatting():
    rates = RollingThroughput(window_s=30, minimum_span_s=2)
    assert rates.add(1, 0, 0, 0) == (None, None)
    assert rates.add(1, 10, 50, 300) == (5, 30)
    assert estimate_eta_seconds(50, 100, 5, elapsed_s=10) is None
    assert estimate_eta_seconds(50, 100, 5, elapsed_s=20) == 10
    assert estimate_eta_seconds(50, 100, 0, elapsed_s=10) is None
    assert rates.add(2, 11, 5, 5) == (None, None)
    assert format_duration(3670) == "1h 01m"
    assert format_bytes(1024**3) == "1.00 GiB"


def test_source_start_resets_live_counters(monkeypatch, tmp_path):
    monkeypatch.setattr("axp_daemon.service.runtime_paths", lambda: {"state": tmp_path / "state.json"})
    publisher = StatePublisher()
    control = DaemonControl(publisher)
    publisher.update(files_seen=9, files_completed=8, chunks_embedded=70)
    control.source_started({"id": 2, "path": "B"}, 123)
    assert publisher.value["files_seen"] == 0
    assert publisher.value["files_completed"] == 0
    assert publisher.value["chunks_embedded"] == 0
    assert publisher.value["source_scan_started_ms"] == 123
    assert publisher.value["progress_baseline_kind"] == "none"


def test_source_start_only_trusts_previous_complete_enumeration(monkeypatch, tmp_path):
    monkeypatch.setattr("axp_daemon.service.runtime_paths", lambda: {"state": tmp_path / "state.json"})
    publisher = StatePublisher()
    control = DaemonControl(publisher)
    control.source_started({"id": 2, "path": "B", "last_file_count": 25, "last_seen_count": 0}, 123)
    assert publisher.value["progress_baseline"] == 0
    assert publisher.value["progress_baseline_kind"] == "none"
    control.source_started({"id": 2, "path": "B", "last_file_count": 25, "last_seen_count": 1000}, 456)
    assert publisher.value["progress_baseline"] == 1000
    assert publisher.value["progress_baseline_kind"] == "previous_complete_scan"


def test_database_sizes_excludes_shm(tmp_path):
    pytest.importorskip("psutil")
    from axp_tray.resources import database_sizes

    db = tmp_path / "test.db"
    db.write_bytes(b"x" * 10)
    Path(f"{db}-wal").write_bytes(b"x" * 20)
    Path(f"{db}-shm").write_bytes(b"x" * 40)
    sizes = database_sizes(db)
    assert (sizes.database, sizes.wal, sizes.total) == (10, 20, 30)


class FakePsutil:
    class NoSuchProcess(Exception): pass
    class AccessDenied(Exception): pass
    class ZombieProcess(Exception): pass
    def __init__(self): self.io = 100
    def Process(self, _pid): return self
    def cpu_percent(self, _interval=None): return 42
    def memory_info(self): return namedtuple("Memory", "rss")(1024)
    def io_counters(self):
        self.io += 100
        return namedtuple("IO", "read_bytes write_bytes")(self.io, self.io * 2)
    def virtual_memory(self): return namedtuple("VM", "percent available")(67, 4096)
    def sensors_battery(self): return namedtuple("Battery", "power_plugged percent")(False, 58)


def test_resource_monitor_and_io_rates():
    pytest.importorskip("psutil")
    from axp_tray.resources import ResourceMonitor

    fake = FakePsutil(); times = iter((1, 3))
    monitor = ResourceMonitor(fake, lambda: next(times))
    first = monitor.sample(5); second = monitor.sample(5)
    assert first.process_cpu == 42 and first.process_rss == 1024
    assert second.read_bytes_s == 50 and second.write_bytes_s == 100
    assert second.system_ram_percent == 67 and second.power == "Battery 58%"


def test_resource_monitor_process_failure_is_harmless():
    pytest.importorskip("psutil")
    from axp_tray.resources import ResourceMonitor

    fake = FakePsutil()
    def missing(_pid): raise fake.NoSuchProcess()
    fake.Process = missing
    snapshot = ResourceMonitor(fake, lambda: 1).sample(5)
    assert snapshot.process_cpu is None and snapshot.system_cpu == 42


def test_coverage_and_last_success_presentation_states():
    from axp_tray.sources_window import coverage_display, last_success_display

    base = {"last_success_ms": None, "last_seen_count": 0, "last_content_count": 0,
            "last_metadata_count": 0, "status": "idle"}
    assert coverage_display(base) == "Pending first full scan"
    assert coverage_display(base, scanning=True) == "First scan in progress"
    assert last_success_display(base) == "Not completed yet"
    assert last_success_display(base, scanning=True) == "In progress"
    complete = {**base, "last_success_ms": 1000, "last_seen_count": 10, "last_content_count": 7,
                "last_metadata_count": 3}
    assert coverage_display(complete) == "100% absorbed / 70% content"
    assert "last complete" in coverage_display({**complete, "status": "offline"})
    assert "last complete" in coverage_display({**complete, "status": "error"})
    assert last_success_display(complete, scanning=True) != "In progress"
