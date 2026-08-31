"""The single native Intel SYCL/Level Zero device probe contract."""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

DEVICE_SELECTOR = "level_zero:gpu"
MAX_EXCERPT = 4000
INTEL_GPU_RE = re.compile(r"(?=.*\bintel\b)(?=.*\b(?:gpu|graphics|arc|iris|uhd|xe)\b).+", re.I)


def child_environment(server_dir, runtime_root=None, environ=None):
    env = dict(os.environ if environ is None else environ)
    for key in tuple(env):
        upper = key.upper()
        if upper.startswith(("ONEAPI_", "SYCL_")) or upper == "ZE_AFFINITY_MASK":
            env.pop(key, None)
    server_dir = Path(server_dir).resolve()
    runtime_root = Path(runtime_root or server_dir).resolve()
    prefixes = [str(server_dir)]
    if runtime_root != server_dir:
        prefixes.append(str(runtime_root))
    env["PATH"] = os.pathsep.join((*prefixes, env.get("PATH", "")))
    env["ONEAPI_DEVICE_SELECTOR"] = DEVICE_SELECTOR
    return env


def parse_device_list(output):
    """Return normalized Intel GPU descriptions, excluding CPU-only devices."""
    devices = []
    for raw in str(output or "").splitlines():
        line = raw.strip()
        if re.search(r"\bcpu(?:-only)?\b", line, re.I) or not INTEL_GPU_RE.search(line):
            continue
        normalized = re.sub(r"^(?:\[?SYCL\s*\d+\]?|\d+)\s*:\s*", "", line, flags=re.I).strip()
        if normalized and normalized not in devices:
            devices.append(normalized[:300])
    return devices


def _excerpt(value, server_path):
    text = str(value or "").replace(str(server_path), "<runtime>/llama-server.exe")
    text = text.replace(str(server_path.parent), "<runtime>")
    text = re.sub(r"(?i)\b(token|secret|password|authorization)\s*[=:]\s*\S+", r"\1=<redacted>", text)
    return text[:MAX_EXCERPT]


def probe_sycl(server_path, runtime_root=None, timeout=15, runner=subprocess.run):
    started = time.monotonic()
    server_path = Path(server_path) if server_path else None
    base = {"installed": bool(server_path and server_path.exists()), "ok": False,
            "command_supported": False, "returncode": None, "device_count": 0,
            "device_name": None, "error_code": None, "stdout_excerpt": "",
            "stderr_excerpt": "", "duration_ms": 0}
    if not server_path or not server_path.exists():
        return {**base, "error_code": "intel_sycl_runtime_missing"}
    if not server_path.is_file() or not server_path.stat().st_size:
        return {**base, "installed": True, "error_code": "intel_sycl_runtime_invalid"}
    try:
        result = runner([str(server_path), "--list-devices"], cwd=str(server_path.parent),
            env=child_environment(server_path.parent, runtime_root), capture_output=True, text=True,
            timeout=timeout, check=False, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        return {**base, "installed": True, "error_code": "intel_sycl_probe_timeout",
                "stdout_excerpt": _excerpt(exc.stdout, server_path), "stderr_excerpt": _excerpt(exc.stderr, server_path),
                "duration_ms": round((time.monotonic() - started) * 1000)}
    except OSError as exc:
        return {**base, "installed": True, "error_code": "intel_gpu_driver_or_level_zero_unavailable",
                "stderr_excerpt": _excerpt(exc, server_path),
                "duration_ms": round((time.monotonic() - started) * 1000)}
    stdout, stderr = result.stdout or "", result.stderr or ""
    devices = parse_device_list(stdout + "\n" + stderr)
    error = None if result.returncode == 0 and devices else (
        "intel_sycl_device_not_found" if result.returncode == 0 else "intel_sycl_probe_command_failed")
    return {**base, "installed": True, "ok": error is None, "command_supported": True,
            "returncode": result.returncode, "device_count": len(devices),
            "device_name": devices[0] if devices else None, "error_code": error,
            "stdout_excerpt": _excerpt(stdout, server_path), "stderr_excerpt": _excerpt(stderr, server_path),
            "duration_ms": round((time.monotonic() - started) * 1000)}
