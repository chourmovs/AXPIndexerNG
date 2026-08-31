from __future__ import annotations

import json
import ipaddress
import logging
import os
import tempfile
import time
import uuid
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
        "processes": root / "runtime" / "processes.json",
    }
    for key in ("data", "runtime", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


DEFAULT_SETTINGS = {
    "db_path": "data/axpindex.db",
    "model_cache": "model-cache",
    "chat_backend": "llama_cpp",
    "chat_model_path": "model-cache/chat/model.gguf",
    "chat_active_model_id": None,
    "chat_inference_device": "auto",
    "embedding_profile": "balanced",
    "embedding_batch_size": 64,
    "download_missing_models": True,
    "model_download_retry_s": 60,
    "scan_interval_s": 300,
    "web_host": "127.0.0.1",
    "web_port": 8765,
    "auto_start_daemon": True,
    "auto_restart_daemon": True,
    "daemon_runtime_mode": "interactive",
    "background_drive_mappings": {},
}


ATOMIC_WRITE_RETRY_DELAYS_S = (0.05, 0.1, 0.2, 0.4)
MAX_SETTINGS_RECOVERY_FILES = 3


def validate_loopback_host(host):
    """Return a safe loopback bind host without performing DNS resolution."""
    if not isinstance(host, str) or not host.strip():
        raise ValueError("web_host_must_be_loopback")
    value = host.strip()
    if value.rstrip(".").lower() == "localhost":
        return value
    try:
        if ipaddress.ip_address(value).is_loopback:
            return value
    except ValueError:
        pass
    raise ValueError("web_host_must_be_loopback")


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


def _valid_settings_object(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _normalize_settings(current):
    settings = {**DEFAULT_SETTINGS, **current}
    validators = {
        "web_host": lambda value: validate_loopback_host(value),
        "web_port": lambda value: value if type(value) is int and 1 <= value <= 65535 else None,
        "scan_interval_s": lambda value: value if type(value) is int and 0 < value <= 31_536_000 else None,
        "embedding_batch_size": lambda value: value if type(value) is int and 0 < value <= 1_000_000 else None,
        "model_download_retry_s": lambda value: value if type(value) is int and 0 < value <= 86_400 else None,
        "download_missing_models": lambda value: value if type(value) is bool else None,
        "auto_start_daemon": lambda value: value if type(value) is bool else None,
        "auto_restart_daemon": lambda value: value if type(value) is bool else None,
        "daemon_runtime_mode": lambda value: value if value in ("interactive", "scheduled_task") else None,
        "chat_inference_device": lambda value: value if value in ("auto", "cpu", "intel_gpu") else None,
        "background_drive_mappings": lambda value: value if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()) else None,
        **{key: (lambda value: value if isinstance(value, (str, os.PathLike)) and str(value).strip() else None)
           for key in ("db_path", "model_cache", "chat_model_path")},
    }
    for key, validator in validators.items():
        try:
            valid = validator(settings.get(key))
        except (TypeError, ValueError):
            valid = None
        if valid is None:
            logging.getLogger(__name__).warning("Invalid persisted setting %s; using default", key)
            settings[key] = DEFAULT_SETTINGS[key]
        else:
            settings[key] = str(valid) if isinstance(valid, os.PathLike) else valid
    return settings


def _preserve_corrupt_settings(path):
    target = Path(path)
    recovery = target.with_name(f"{target.name}.corrupt-{time.time_ns()}")
    os.replace(target, recovery)
    files = sorted(target.parent.glob(f"{target.name}.corrupt-*"), key=lambda item: item.stat().st_mtime_ns,
                   reverse=True)
    for old in files[MAX_SETTINGS_RECOVERY_FILES:]:
        try:
            old.unlink()
        except OSError:
            logging.getLogger(__name__).warning("Unable to prune settings recovery file", exc_info=True)


def load_settings():
    paths = runtime_paths()
    settings_path = paths["settings"]
    exists = settings_path.exists()
    current = _valid_settings_object(settings_path)
    if exists and current is None:
        logging.getLogger(__name__).error("Malformed settings file preserved; recovering configuration")
        _preserve_corrupt_settings(settings_path)
        current = _valid_settings_object(settings_path.with_suffix(settings_path.suffix + ".bak")) or {}
    elif current is None:
        current = {}
    settings = _normalize_settings(current)
    if not settings.get("installation_id"):
        settings["installation_id"] = str(uuid.uuid4())
    if settings != current:
        save_settings(settings)
    root = installation_root()
    for key in ("db_path", "model_cache", "chat_model_path"):
        value = Path(settings[key])
        settings[key] = str(value if value.is_absolute() else (root / value).resolve())
    return settings


def save_settings(settings):
    serializable = dict(settings)
    root = installation_root()
    for key in ("db_path", "model_cache", "chat_model_path"):
        try:
            serializable[key] = str(Path(serializable[key]).resolve().relative_to(root))
        except ValueError:
            serializable[key] = str(Path(serializable[key]).resolve())
    target = runtime_paths()["settings"]
    previous = _valid_settings_object(target)
    if previous is not None:
        atomic_write_json(target.with_suffix(target.suffix + ".bak"), previous)
    atomic_write_json(target, serializable)


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
