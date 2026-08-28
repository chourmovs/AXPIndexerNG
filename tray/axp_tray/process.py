from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from axp_core.runtime import atomic_write_json, installation_root, runtime_paths
from axp_daemon.service import send_control


def pythonw():
    bundled = installation_root() / "python" / "pythonw.exe"
    return bundled if bundled.exists() else Path(sys.executable)


def process_environment(settings):
    root = installation_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(root / folder) for folder in ("shared", "daemon", "client", "tray"))
    env["FASTEMBED_CACHE_PATH"] = settings["model_cache"]
    env["AXPINDEXER_DATA_DIR"] = str(runtime_paths()["data"])
    return env


def spawn(module, arguments, settings):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen([str(pythonw()), "-m", module, *map(str, arguments)], cwd=installation_root(),
                            env=process_environment(settings), creationflags=flags,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_daemon(settings):
    atomic_write_json(runtime_paths()["desired"], {"state": "running", "updated_ms": int(time.time() * 1000)})
    return spawn("axp_daemon", ["run", "--db", settings["db_path"], "--model-cache", settings["model_cache"],
                                "--embedding-profile", settings["embedding_profile"], "--scan-interval",
                                settings["scan_interval_s"], "--embedding-batch-size",
                                settings["embedding_batch_size"]], settings)


def stop_daemon(intentional=True):
    if intentional:
        atomic_write_json(runtime_paths()["desired"], {"state": "stopped", "updated_ms": int(time.time() * 1000)})
    return send_control("stop")


def restart_daemon(settings, timeout=15):
    send_control("stop")
    deadline = time.monotonic() + timeout
    from .state import read_daemon_state

    while time.monotonic() < deadline:
        if read_daemon_state().get("state") == "stopped":
            break
        time.sleep(0.25)
    return start_daemon(settings)


def client_healthy(settings, timeout=0.5):
    url = f"http://{settings['web_host']}:{settings['web_port']}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def ensure_client(settings):
    if client_healthy(settings):
        return None
    return spawn("axp_client", ["serve", "--db", settings["db_path"], "--host", settings["web_host"],
                                "--port", settings["web_port"]], settings)
