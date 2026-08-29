"""Failure-safe resource sampling performed by the desktop tray."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass(frozen=True)
class ResourceSnapshot:
    process_cpu: float | None = None
    process_rss: int | None = None
    read_bytes_s: float | None = None
    write_bytes_s: float | None = None
    system_cpu: float | None = None
    system_ram_percent: float | None = None
    available_ram: int | None = None
    power: str | None = None


@dataclass(frozen=True)
class DatabaseSizes:
    database: int | None
    wal: int | None
    total: int | None


def database_sizes(path):
    path = Path(path)
    try:
        database = path.stat().st_size
    except OSError:
        database = None
    try:
        wal = Path(f"{path}-wal").stat().st_size
    except OSError:
        wal = 0 if database is not None else None
    return DatabaseSizes(database, wal, database + wal if database is not None and wal is not None else None)


class ResourceMonitor:
    def __init__(self, psutil_module=psutil, clock=time.monotonic):
        self.psutil = psutil_module
        self.clock = clock
        self.pid = None
        self.process = None
        self.previous_io = None

    def sample(self, pid):
        now = self.clock()
        process_cpu = process_rss = read_rate = write_rate = None
        try:
            if pid and pid != self.pid:
                self.pid, self.process, self.previous_io = pid, self.psutil.Process(pid), None
                self.process.cpu_percent(None)  # prime psutil's non-blocking measurement
            if self.process is not None:
                process_cpu = self.process.cpu_percent(None)
                process_rss = self.process.memory_info().rss
                io = self.process.io_counters()
                if self.previous_io:
                    old_time, old_read, old_write = self.previous_io
                    elapsed = now - old_time
                    if elapsed > 0 and io.read_bytes >= old_read and io.write_bytes >= old_write:
                        read_rate = (io.read_bytes - old_read) / elapsed
                        write_rate = (io.write_bytes - old_write) / elapsed
                self.previous_io = now, io.read_bytes, io.write_bytes
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, self.psutil.ZombieProcess, OSError):
            self.pid, self.process, self.previous_io = None, None, None
        try:
            system_cpu = self.psutil.cpu_percent(None)
            memory = self.psutil.virtual_memory()
            system_ram_percent, available_ram = memory.percent, memory.available
        except (self.psutil.Error, OSError, NotImplementedError):
            system_cpu = system_ram_percent = available_ram = None
        try:
            battery = self.psutil.sensors_battery()
            power = None if battery is None else ("AC power" if battery.power_plugged else f"Battery {battery.percent:.0f}%")
        except (self.psutil.Error, OSError, NotImplementedError):
            power = None
        return ResourceSnapshot(process_cpu, process_rss, read_rate, write_rate,
                                system_cpu, system_ram_percent, available_ram, power)
