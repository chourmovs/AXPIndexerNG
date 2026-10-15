"""Thread-safe bounded progress registry for the threaded local HTTP server."""
import threading
import time
from collections import OrderedDict

from axp_core.tracing import Trace

SERVER_SCOPE = ("Search HTTP handler after correlation-ID parsing: query validation, reader wait/open, "
                "retrieval, presentation and result JSON encoding; "
                "excludes socket accept/backlog, final diagnostic envelope encoding and socket write")


class TraceRegistry:
    def __init__(self, *, max_entries=32, max_active=8, ttl=120):
        self.lock = threading.Lock()
        self.entries = OrderedDict()
        self.max_entries, self.max_active, self.ttl = max_entries, max_active, ttl

    def _prune(self):
        now = time.perf_counter_ns()
        for key, trace in list(self.entries.items()):
            if trace.ended and (now - trace.ended) / 1e9 > self.ttl:
                del self.entries[key]

    def start(self, request_id=None):
        with self.lock:
            self._prune()
            if request_id in self.entries:
                raise ValueError("request_id_already_used")
            if sum(t.ended is None for t in self.entries.values()) >= self.max_active:
                raise RuntimeError("search_busy")
            while len(self.entries) >= self.max_entries:
                key = next((k for k, t in self.entries.items() if t.ended is not None), None)
                if key is None:
                    raise RuntimeError("search_busy")
                del self.entries[key]
            trace = Trace(request_id)
            self.entries[trace.request_id] = trace
            return trace

    def get(self, request_id):
        with self.lock:
            self._prune()
            return self.entries.get(request_id)
