"""Bounded, request-local monotonic tracing; no SQL or filesystem diagnostics here."""
from __future__ import annotations

import contextvars
import functools
import threading
import time
import uuid
from contextlib import contextmanager

_current = contextvars.ContextVar("search_trace", default=None)
_parent = contextvars.ContextVar("search_span", default=None)


class SearchCancelled(Exception):
    pass


def current_trace():
    return _current.get()


def interval_union(intervals):
    """Length of the union, including overlapping concurrent child intervals."""
    end = None
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


class Trace:
    def __init__(self, request_id=None, *, max_spans=256, clock=time.perf_counter_ns):
        self.request_id = request_id or uuid.uuid4().hex
        self.clock, self.started = clock, clock()
        self.max_spans = max_spans
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.rows, self.active = {}, []
        self.truncated = False
        self.status, self.ended = "running", None

    def check(self):
        if self.cancelled.is_set():
            raise SearchCancelled("Search cancelled")

    @contextmanager
    def bind(self):
        a, b = _current.set(self), _parent.set(None)
        try:
            yield self
        finally:
            _parent.reset(b)
            _current.reset(a)

    def finish(self, status="completed"):
        with self.lock:
            self.status, self.ended = status, self.clock()

    def snapshot(self):
        with self.lock:
            now = self.ended or self.clock()
            rows = []
            for row in self.rows.values():
                item = {**row, "counters": dict(row["counters"]), "details": dict(row["details"])}
                running = [handle for handle in self.active if handle.row is row]
                item["inclusive_ms"] += sum((now - h.start) / 1e6 for h in running)
                item["self_ms"] += sum(h.own(now) / 1e6 for h in running)
                if running:
                    item["status"] = "running"
                    item["max_ms"] = max(item["max_ms"], max((now - h.start) / 1e6 for h in running))
                rows.append(item)
            return {"request_id": self.request_id, "clock": "server perf_counter_ns",
                    "elapsed_ms": (now - self.started) / 1e6, "status": self.status,
                    "active_stage": self.active[-1].row["name"] if self.active else None,
                    "active_stage_elapsed_ms": (now - self.active[-1].start) / 1e6 if self.active else None,
                    "truncated": self.truncated, "max_spans": self.max_spans, "spans": rows}


class _Handle:
    def __init__(self, trace=None, row=None, parent=None):
        self.trace, self.row, self.parent = trace, row, parent
        self.start = trace.clock() if trace else 0
        # Completed coverage before the earliest active child is finalized. All
        # later completed coverage is already covered by that still-active child,
        # whose full interval will be accounted for when it closes. This keeps
        # exact union accounting at constant memory, even for millions of calls
        # overlapping a single long-running child.
        self.covered_ns = 0

    def child_closed(self, start, end):
        frontier = min((h.start for h in self.trace.active if h.parent is self), default=end)
        self.covered_ns += max(0, min(end, frontier) - start)

    def own(self, now):
        intervals = [(h.start, now) for h in self.trace.active if h.parent is self]
        return max(0, now - self.start - self.covered_ns - interval_union(intervals))

    def count(self, **values):
        if self.row is not None:
            with self.trace.lock:
                for key, value in list(values.items())[:16]:
                    if key in self.row["counters"] or len(self.row["counters"]) < 16:
                        self.row["counters"][key] = self.row["counters"].get(key, 0) + value

    def detail(self, **values):
        if self.row is not None:
            with self.trace.lock:
                for key, value in list(values.items())[:16]:
                    if key in self.row["details"] or len(self.row["details"]) < 16:
                        self.row["details"][key] = value[:256] if isinstance(value, str) else value

    def skip(self, reason):
        if self.row is not None:
            self.row["status"] = "skipped"
            self.detail(reason=reason)


@contextmanager
def span(name, routine=None):
    trace = current_trace()
    if trace is None:
        yield _Handle()
        return
    trace.check()
    parent = _parent.get()
    key = (parent.row["span_id"] if parent and parent.row else None, routine or name, name)
    with trace.lock:
        row = trace.rows.get(key)
        if len(trace.active) >= 128:
            trace.truncated = True
            row = None
        elif row is None and len(trace.rows) < trace.max_spans:
            row = {"request_id": trace.request_id, "span_id": str(len(trace.rows) + 1),
                   "parent_span_id": key[0], "name": name, "routine": routine or name,
                   "start_ms": (trace.clock() - trace.started) / 1e6,
                   "inclusive_ms": 0.0, "self_ms": 0.0, "max_ms": 0.0,
                   "calls": 0, "status": "completed", "counters": {}, "details": {}}
            trace.rows[key] = row
        if row is None:
            trace.truncated = True
            yield_handle = None
        else:
            yield_handle = _Handle(trace, row, parent)
            row["calls"] += 1
            trace.active.append(yield_handle)
    if yield_handle is None:
        yield _Handle()
        return
    handle = yield_handle
    token = _parent.set(handle)
    try:
        yield handle
    except BaseException as exc:
        row["status"] = "cancelled" if trace.cancelled.is_set() or isinstance(exc, (SearchCancelled, KeyboardInterrupt)) else "error"
        raise
    finally:
        with trace.lock:
            end = trace.clock()
            duration = (end - handle.start) / 1e6
            row["inclusive_ms"] += duration
            row["self_ms"] += handle.own(end) / 1e6
            row["max_ms"] = max(row["max_ms"], duration)
            trace.active.remove(handle)
            if parent:
                parent.child_closed(handle.start, end)
        _parent.reset(token)


def traced(name):
    def decorate(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            with span(name, f"{fn.__module__}.{fn.__qualname__}"):
                return fn(*args, **kwargs)
        return call
    return decorate


def sql_rows(con, sql, values=(), *, name="SQL"):
    # SQLite may perform substantial work both in execute() and cursor stepping.
    with span(name, "sqlite.execute+fetchall"):
        with span("Exécution SQL", "sqlite.execute"):
            cursor = con.execute(sql, values)
        with span("Lecture des lignes", "sqlite.Cursor.fetchall") as measured:
            rows = cursor.fetchall()
            measured.count(rows_read=len(rows))
        return rows
