from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from axp_core.locking import daemon_instance_running
from axp_core.runtime import atomic_write_json, installation_root, read_json, runtime_paths
from axp_daemon.service import send_control

LOGGER = logging.getLogger("axp_tray")
OWNED_ROLES = frozenset({"daemon", "client"})
ROLE_MODULES = {"daemon": "axp_daemon", "client": "axp_client"}
CREATE_TIME_TOLERANCE_S = 0.01


def pythonw():
    bundled = installation_root() / "python" / "pythonw.exe"
    return bundled if bundled.exists() else Path(sys.executable)


def process_environment(settings):
    root = installation_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(root / folder) for folder in ("shared", "daemon", "client", "tray"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["FASTEMBED_CACHE_PATH"] = settings["model_cache"]
    env["AXPINDEXER_DATA_DIR"] = str(runtime_paths()["data"])
    return env


def _registry_path():
    return runtime_paths()["processes"]


def owned_processes():
    value = read_json(_registry_path(), {}) or {}
    entries = value.get("processes", []) if isinstance(value, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_registry(entries):
    atomic_write_json(_registry_path(), {"version": 1, "processes": entries})


def register_process(process, role, module, executable, launch_mode="interactive"):
    if role not in OWNED_ROLES:
        raise ValueError(f"Unsupported owned process role: {role}")
    identity = {
        "pid": process.pid,
        "create_time": psutil.Process(process.pid).create_time(),
        "role": role,
        "module": module,
        "executable": str(Path(executable).resolve()),
        "installation_root": str(installation_root().resolve()),
        "launch_mode": launch_mode,
    }
    entries = [item for item in owned_processes() if item.get("pid") != process.pid]
    entries.append(identity)
    _write_registry(entries)
    return identity


def _same_path(left, right):
    try:
        return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(Path(right).resolve()))
    except (OSError, TypeError, ValueError):
        return False


def is_owned_process(process, identity, allowed_roles=OWNED_ROLES):
    """Conservatively authenticate a live process against persisted AXP identity."""
    try:
        role = identity.get("role")
        if role not in allowed_roles or role not in OWNED_ROLES or process.pid == os.getpid():
            return False
        if process.pid != int(identity["pid"]):
            return False
        if abs(process.create_time() - float(identity["create_time"])) > CREATE_TIME_TOLERANCE_S:
            return False
        root = installation_root().resolve()
        if not _same_path(identity.get("installation_root"), root):
            return False
        expected_executables = {root / "python" / "pythonw.exe", root / "python" / "python.exe"}
        recorded = Path(identity["executable"])
        bundled_executable = any(_same_path(recorded, item) for item in expected_executables)
        development_executable = _same_path(recorded, sys.executable) and _same_path(process.exe(), recorded)
        if not bundled_executable and not development_executable:
            return False
        if not _same_path(process.exe(), recorded):
            return False
        command = [str(item) for item in process.cmdline()]
        module = ROLE_MODULES[role]
        module_launch = any(command[index:index + 2] == ["-m", module] for index in range(len(command) - 1))
        launcher_launch = role == "daemon" and any(
            Path(item).name.lower() == "axpindexerdaemon.pyw" and _same_path(item, root / "AXPIndexerDaemon.pyw")
            for item in command
        )
        return module_launch or launcher_launch
    except (KeyError, TypeError, ValueError, psutil.AccessDenied, psutil.NoSuchProcess, OSError):
        return False


def verified_owned_processes(allowed_roles=OWNED_ROLES):
    verified, retained = [], []
    for identity in owned_processes():
        if identity.get("role") not in allowed_roles:
            retained.append(identity)
            continue
        try:
            process = psutil.Process(int(identity["pid"]))
        except psutil.AccessDenied:
            # An unreadable live PID is never considered safe to terminate. Keep its
            # entry so a later cleanup can authenticate it if access becomes available.
            if identity.get("role") in allowed_roles:
                retained.append(identity)
            continue
        except (KeyError, TypeError, ValueError, psutil.NoSuchProcess):
            continue
        if is_owned_process(process, identity, allowed_roles):
            verified.append((process, identity))
            retained.append(identity)
    _write_registry(retained)
    return verified


def cleanup_owned_processes(allowed_roles, graceful_timeout=5.0, terminate_timeout=2.0):
    """Wait, then terminate and kill only identities authenticated to this install."""
    allowed_roles = frozenset(allowed_roles) & OWNED_ROLES
    LOGGER.info("Waiting for owned processes")
    remaining = verified_owned_processes(allowed_roles)
    if remaining and graceful_timeout > 0:
        deadline = time.monotonic() + graceful_timeout
        while remaining and time.monotonic() < deadline:
            time.sleep(0.1)
            remaining = verified_owned_processes(allowed_roles)
    for process, identity in remaining:
        if is_owned_process(process, identity, allowed_roles):
            LOGGER.warning("Process role=%s pid=%s did not exit gracefully", identity["role"], identity["pid"])
            LOGGER.warning("Terminating owned process role=%s pid=%s", identity["role"], identity["pid"])
            try:
                process.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    if remaining:
        psutil.wait_procs([process for process, _ in remaining], timeout=terminate_timeout)
    remaining = verified_owned_processes(allowed_roles)
    for process, identity in remaining:
        if is_owned_process(process, identity, allowed_roles):
            LOGGER.error("Force killing owned process role=%s pid=%s", identity["role"], identity["pid"])
            try:
                process.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
    if remaining:
        psutil.wait_procs([process for process, _ in remaining], timeout=terminate_timeout)
    verified_owned_processes(allowed_roles)
    LOGGER.info("Owned child processes stopped")


def spawn(role, arguments, settings, launch_mode="interactive"):
    module = ROLE_MODULES.get(role)
    if module is None:
        raise ValueError(f"Unsupported owned process role: {role}")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    executable = pythonw()
    process = subprocess.Popen(
        [str(executable), "-B", "-m", module, *map(str, arguments)], cwd=installation_root(),
        env=process_environment(settings), creationflags=flags, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        register_process(process, role, module, executable, launch_mode)
    except Exception:
        process.terminate()
        raise
    return process


def start_daemon(settings):
    atomic_write_json(runtime_paths()["desired"], {"state": "running", "updated_ms": int(time.time() * 1000)})
    if settings.get("daemon_runtime_mode") == "scheduled_task":
        from .background_task import TaskSchedulerBackend

        backend = TaskSchedulerBackend(settings)
        status = backend.status()
        if status.state not in ("ready", "running"):
            raise RuntimeError(status.message or "Background daemon unavailable; task repair required")
        return backend.run()
    arguments = ["run", "--db", settings["db_path"], "--model-cache", settings["model_cache"],
                 "--embedding-profile", settings["embedding_profile"], "--scan-interval",
                 settings["scan_interval_s"], "--embedding-device", settings["embedding_device"],
                 "--embedding-batch-size", settings["embedding_batch_size"],
                 "--model-download-retry", settings["model_download_retry_s"], "--launch-mode", "interactive"]
    if settings.get("download_missing_models", True):
        arguments.append("--allow-download")
    return spawn("daemon", arguments, settings)


def stop_daemon(intentional=True):
    if intentional:
        atomic_write_json(runtime_paths()["desired"], {"state": "stopped", "updated_ms": int(time.time() * 1000)})
    return send_control("stop") if intentional else send_control("stop", intentional=False, reason="manual_restart")


def restart_daemon(settings, timeout=15):
    if settings.get("daemon_runtime_mode") == "scheduled_task":
        raise RuntimeError("Interactive restart cannot control the scheduled daemon lifecycle")
    stop_daemon(intentional=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not daemon_instance_running(settings["db_path"]):
            cleanup_owned_processes({"daemon"}, graceful_timeout=0)
            return start_daemon(settings)
        time.sleep(0.25)
    cleanup_owned_processes({"daemon"}, graceful_timeout=0)
    if daemon_instance_running(settings["db_path"]):
        raise RuntimeError("Manual daemon restart failed: daemon instance lock is still held")
    return start_daemon(settings)


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
    return spawn("client", ["serve", "--db", settings["db_path"], "--host", settings["web_host"],
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


def recover_unhealthy_daemon(settings, state, reason, timeout=5.0):
    """Challenge the exact instance before recovering an authenticated process.

    Runs off the Tk thread. A responsive worker, changed instance, intentional
    stop or an unauthenticated PID vetoes termination.
    """
    import uuid
    from .state import read_daemon_state

    paths = runtime_paths()
    instance = state.get("instance_id")
    if not instance:
        LOGGER.warning("Recovery refused: legacy state has no instance identity pid=%s", state.get("pid"))
        return False
    request_id = uuid.uuid4().hex
    atomic_write_json(paths["runtime"] / "daemon_health_probe.json",
                      {"instance_id": instance, "request_id": request_id})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_daemon_state()
        if current.get("instance_id") != instance:
            return False
        reply = read_json(paths["runtime"] / "daemon_worker_health_reply.json", {}) or {}
        worker_replied = (reply.get("request_id") == request_id and reply.get("instance_id") == instance
                          and reply.get("pid") == state.get("pid") and
                          reply.get("process_create_time") == state.get("process_create_time"))
        if current.get("worker_health_probe_id") == request_id or worker_replied:
            LOGGER.info("Recovery vetoed: worker healthy instance=%s pid=%s", instance, current.get("pid"))
            return False
        if reason == "heartbeat_stale" and current.get("health_probe_id") == request_id:
            LOGGER.info("Recovery vetoed: publisher healthy instance=%s pid=%s worker=%s",
                        instance, current.get("pid"), current.get("worker_state"))
            return False
        time.sleep(0.1)
    current = read_daemon_state()
    desired = read_json(paths["desired"], {}) or {}
    if (current.get("instance_id") != instance or desired.get("state") != "running" or
            current.get("state") == "stopping"):
        return False
    matches = [(proc, identity) for proc, identity in verified_owned_processes({"daemon"})
               if identity.get("pid") == current.get("pid") and
               abs(identity.get("create_time", -1) - current.get("process_create_time", -2)) <= CREATE_TIME_TOLERANCE_S]
    if not matches and settings.get("daemon_runtime_mode") == "scheduled_task":
        # Scheduler launches are not children of the tray. Authenticate their
        # published creation time, executable, module, cwd and catalog argument.
        try:
            proc = psutil.Process(int(current["pid"]))
            command = proc.cmdline()
            db_argument = command[command.index("--db") + 1]
            identity = {"pid": current["pid"], "create_time": current["process_create_time"],
                        "role": "daemon", "executable": proc.exe(),
                        "installation_root": str(installation_root().resolve())}
            if (_same_path(proc.cwd(), installation_root()) and
                    _same_path(db_argument, settings["db_path"]) and
                    is_owned_process(proc, identity, {"daemon"})):
                matches = [(proc, identity)]
        except (KeyError, ValueError, IndexError, OSError, psutil.Error):
            pass
    if len(matches) != 1:
        LOGGER.warning("Recovery refused: process identity unavailable instance=%s pid=%s mode=%s",
                       instance, current.get("pid"), settings.get("daemon_runtime_mode"))
        return False
    LOGGER.warning("Daemon recovery reason=%s instance=%s pid=%s heartbeat_age_ms=%s worker=%s stage=%s",
                   reason, instance, current.get("pid"), current.get("heartbeat_age_ms"),
                   current.get("worker_state"), current.get("current_stage"))
    send_control("stop", intentional=False, target_instance_id=instance, reason=reason)
    # Only the challenged identity may be terminated. Other daemons and reused
    # PIDs are never included, even if the registry changes during the wait.
    proc, identity = matches[0]
    try:
        proc.wait(timeout=2)
    except psutil.TimeoutExpired:
        if not is_owned_process(proc, identity, {"daemon"}):
            return False
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except psutil.TimeoutExpired:
            if not is_owned_process(proc, identity, {"daemon"}):
                return False
            proc.kill()
            proc.wait(timeout=2)
    except psutil.NoSuchProcess:
        pass
    if daemon_instance_running(settings["db_path"]):
        return False
    if (read_json(paths["desired"], {}) or {}).get("state") != "running":
        return False
    start_daemon(settings)
    return True
