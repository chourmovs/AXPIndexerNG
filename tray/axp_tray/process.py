from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from axp_core.locking import daemon_instance_running
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
    arguments = ["run", "--db", settings["db_path"], "--model-cache", settings["model_cache"],
                 "--embedding-profile", settings["embedding_profile"], "--scan-interval",
                 settings["scan_interval_s"], "--embedding-batch-size", settings["embedding_batch_size"],
                 "--model-download-retry", settings["model_download_retry_s"]]
    if settings.get("download_missing_models", True):
        arguments.append("--allow-download")
    return spawn("axp_daemon", arguments, settings)


def stop_daemon(intentional=True):
    if intentional:
        atomic_write_json(runtime_paths()["desired"], {"state": "stopped", "updated_ms": int(time.time() * 1000)})
    return send_control("stop")


def restart_daemon(settings, timeout=15):
    send_control("stop")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not daemon_instance_running(settings["db_path"]):
            return start_daemon(settings)
        time.sleep(0.25)
    raise RuntimeError("Manual daemon restart timed out: daemon instance lock is still held")


def client_healthy(settings, timeout=0.5):
    url = f"http://127.0.0.1:{settings['web_port']}/health"
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


def stop_client(settings, timeout=1.0):
    """Gracefully stop this installation's local web client, if it is running."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{settings['web_port']}/api/shutdown", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False
