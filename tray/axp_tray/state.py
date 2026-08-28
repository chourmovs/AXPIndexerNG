import time

from axp_core.runtime import read_json, runtime_paths


def read_daemon_state(stale_after_s=30):
    value = read_json(runtime_paths()["state"], {}) or {}
    heartbeat = value.get("heartbeat_ms", 0)
    age_ms = int(time.time() * 1000) - heartbeat if heartbeat else None
    value["heartbeat_age_ms"] = age_ms
    value["stale"] = age_ms is None or age_ms > stale_after_s * 1000
    if value["stale"] and value.get("state") not in {"stopped", "stopping"}:
        value["state"] = "error"
        value["last_error"] = "daemon heartbeat stale"
    return value


def tooltip(state):
    current = state.get("state", "stopped")
    if state.get("stale"):
        return "AXPIndexerNG — ERROR — daemon heartbeat stale"
    if current == "scanning":
        discovered, processed = state.get("files_discovered", 0), state.get("files_processed", 0)
        progress = f" — {processed}/{discovered}" if discovered else ""
        return (f"AXPIndexerNG — Indexing{progress}")[:127]
    if current == "paused":
        return "AXPIndexerNG — Paused"
    if current == "idle":
        return f"AXPIndexerNG — Idle — {state.get('documents_total', 0):,} documents"[:127]
    return f"AXPIndexerNG — {current.upper()}"[:127]


def should_auto_restart(state, desired_state, enabled, last_restart_monotonic, now_monotonic):
    return bool(
        enabled and desired_state == "running" and state.get("stale")
        and now_monotonic - last_restart_monotonic >= 60
    )
