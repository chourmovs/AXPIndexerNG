"""Pure presentation models for live indexing progress."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEstimate:
    percent: float | None
    baseline: int | None
    exceeded_baseline: bool = False
    complete: bool = False

    @property
    def label(self):
        if self.complete:
            return "100%"
        if self.exceeded_baseline:
            return "> previous scan size · estimating…"
        return f"~{self.percent:.0f}%" if self.percent is not None else "Scanning…"


def progress_estimate(completed, previous_seen=0, existing_documents=0, scan_complete=False):
    # existing_documents is retained for caller compatibility only. An indexed
    # document count is not proof of a complete filesystem enumeration.
    if scan_complete:
        return ProgressEstimate(100.0, previous_seen or None, complete=True)
    baseline = int(previous_seen or 0)
    if baseline <= 0:
        return ProgressEstimate(None, None)
    if completed > baseline:
        return ProgressEstimate(None, baseline, exceeded_baseline=True)
    return ProgressEstimate(min(max(completed, 0) / baseline * 100, 99.0), baseline)


class RollingThroughput:
    def __init__(self, window_s=30, minimum_span_s=2):
        self.window_s = window_s
        self.minimum_span_s = minimum_span_s
        self.source_id = None
        self.samples = deque()

    def add(self, source_id, timestamp, files, chunks):
        if source_id != self.source_id:
            self.source_id, self.samples = source_id, deque()
        self.samples.append((float(timestamp), int(files), int(chunks)))
        cutoff = float(timestamp) - self.window_s
        while len(self.samples) > 2 and self.samples[1][0] < cutoff:
            self.samples.popleft()
        return self.rates()

    def rates(self):
        if len(self.samples) < 2:
            return None, None
        first, last = self.samples[0], self.samples[-1]
        elapsed = last[0] - first[0]
        if elapsed < self.minimum_span_s:
            return None, None
        return max(0, last[1] - first[1]) / elapsed, max(0, last[2] - first[2]) / elapsed


def estimate_eta_seconds(completed, baseline, files_per_second, *, elapsed_s=0, minimum_elapsed_s=15):
    if not baseline or completed > baseline or elapsed_s < minimum_elapsed_s or not files_per_second:
        return None
    return max(0.0, baseline - completed) / files_per_second


def format_duration(seconds):
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_bytes(value):
    if value is None:
        return "—"
    value = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for index, unit in enumerate(units):
        if abs(value) < 1024 or index == len(units) - 1:
            if index == 0:
                return f"{value:.0f} {unit}"
            precision = 0 if value >= 100 else (1 if value >= 10 else 2)
            return f"{value:.{precision}f} {unit}"
        value /= 1024
