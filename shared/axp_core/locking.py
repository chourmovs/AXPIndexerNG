from __future__ import annotations

import hashlib
import os
from pathlib import Path


class AlreadyLocked(RuntimeError):
    pass


class FileLock:
    """OS-backed single-instance lock; the file itself may safely outlive a crash."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            self.handle.seek(0)
            self.handle.write(b"0")
            self.handle.flush()
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise AlreadyLocked(str(self.path)) from exc
        return self

    def release(self):
        if self.handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


def daemon_lock_path(db_path, runtime_dir=None):
    """Return the stable catalog-specific daemon lock path."""
    catalog_key = hashlib.sha256(str(Path(db_path).resolve()).casefold().encode()).hexdigest()[:16]
    if runtime_dir is None:
        from .runtime import runtime_paths

        runtime_dir = runtime_paths()["runtime"]
    return Path(runtime_dir) / f"daemon-{catalog_key}.lock"


def daemon_instance_running(db_path, runtime_dir=None):
    """Non-destructively probe whether the catalog daemon owns its OS lock."""
    probe = FileLock(daemon_lock_path(db_path, runtime_dir))
    try:
        probe.acquire()
    except AlreadyLocked:
        return True
    else:
        probe.release()
        return False
