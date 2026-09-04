"""Thread-safe, read-only snapshots of client background initialization."""
import copy
import threading
import time


class ClientStartupState:
    """Publish capability readiness without exposing mutable worker state."""

    def __init__(self):
        self.started_at = time.perf_counter()
        self._lock = threading.Lock()
        self._value = {
            "phase": "starting",
            "server": {"state": "starting"},
            "search": {"state": "initializing", "phase": "loading_embeddings"},
            "local_ai": {"state": "initializing", "phase": "starting", "warmup": {"state": "pending"}},
            "timings": {},
        }

    def update(self, section=None, **values):
        with self._lock:
            (self._value if section is None else self._value[section]).update(values)
            search = self._value["search"]["state"]
            ai = self._value["local_ai"]["state"]
            healthy = {"ready", "ready_with_warmup_warning", "unconfigured"}
            terminal = healthy | {"failed"}
            if search == "ready" and ai in healthy:
                self._value["phase"] = "ready"
            elif search == "failed" or ai == "failed":
                self._value["phase"] = "degraded"
            elif search in terminal or ai in terminal:
                self._value["phase"] = "partially_ready"

    def timing(self, name, milliseconds=None):
        value = milliseconds if milliseconds is not None else (time.perf_counter() - self.started_at) * 1000
        with self._lock:
            self._value["timings"][name] = round(value, 1)

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._value)
