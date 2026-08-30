import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass


@dataclass
class InferenceOperation:
    operation_id: str
    future: object
    state: str
    started_at: float
    completed_at: float | None = None
    error: str | None = None


class NativeOperationSupervisor:
    """Own native work independently of short-lived HTTP observers."""
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="axp-chat-native")
        self._lock = threading.Lock()
        self._operation = None

    @property
    def busy(self):
        with self._lock:
            return self._operation is not None and not self._operation.future.done()

    def run(self, operation, heartbeat=lambda elapsed: None, *, interval=1.0):
        with self._lock:
            if self._operation is not None and not self._operation.future.done():
                raise RuntimeError("chat_busy")
            future = self._executor.submit(operation)
            current = InferenceOperation(uuid.uuid4().hex, future, "running", time.time())
            self._operation = current

        def completed(done):
            with self._lock:
                current.completed_at = time.time()
                current.state = "failed" if done.exception() else "completed"
                current.error = type(done.exception()).__name__ if done.exception() else None
        future.add_done_callback(completed)
        started = time.perf_counter()
        while True:
            try:
                return future.result(timeout=interval)
            except TimeoutError:
                heartbeat(round(time.perf_counter() - started, 1))

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=False)
