from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path


def installation_root():
    return Path(__file__).resolve().parents[2]


def data_dir():
    return Path(os.getenv("AXPINDEXER_DATA_DIR", installation_root() / "data")).resolve()


def runtime_paths():
    root = data_dir()
    paths = {
        "data": root,
        "runtime": root / "runtime",
        "logs": root / "logs",
        "settings": root / "settings.json",
        "db": root / "axpindex.db",
        "state": root / "runtime" / "daemon_state.json",
        "control": root / "runtime" / "control.json",
        "desired": root / "runtime" / "desired_state.json",
        "daemon_lock": root / "runtime" / "daemon.lock",
        "tray_lock": root / "runtime" / "tray.lock",
    }
    for key in ("data", "runtime", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


DEFAULT_SETTINGS = {
    "db_path": "data/axpindex.db",
    "model_cache": "model-cache",
    "embedding_profile": "balanced",
    "embedding_batch_size": 64,
    "download_missing_models": True,
    "model_download_retry_s": 60,
    "scan_interval_s": 300,
    "web_host": "127.0.0.1",
    "web_port": 8765,
    "auto_start_daemon": True,
    "auto_restart_daemon": True,
}


ATOMIC_WRITE_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4)


def atomic_write_json(path, value):
    """Atomically write JSON, tolerating brief filesystem/antivirus contention."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(len(ATOMIC_WRITE_RETRY_DELAYS_S) + 1):
        temporary = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=f"{target.name}.{os.getpid()}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            return
        except OSError:
            if attempt >= len(ATOMIC_WRITE_RETRY_DELAYS_S):
                raise
            time.sleep(ATOMIC_WRITE_RETRY_DELAYS_S[attempt])
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def read_json(path, default=None):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def load_settings():
    paths = runtime_paths()
    current = read_json(paths["settings"], {}) or {}
    settings = {**DEFAULT_SETTINGS, **current}
    if settings != current:
        atomic_write_json(paths["settings"], settings)
    root = installation_root()
    for key in ("db_path", "model_cache"):
        value = Path(settings[key])
        settings[key] = str(value if value.is_absolute() else (root / value).resolve())
    return settings


def save_settings(settings):
    serializable = dict(settings)
    root = installation_root()
    for key in ("db_path", "model_cache"):
        try:
            serializable[key] = str(Path(serializable[key]).resolve().relative_to(root))
        except ValueError:
            serializable[key] = str(Path(serializable[key]).resolve())
    atomic_write_json(runtime_paths()["settings"], serializable)


def configure_logging(name, filename):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(runtime_paths()["logs"] / filename, maxBytes=5 * 1024 * 1024,
                                  backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
