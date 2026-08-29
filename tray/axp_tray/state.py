import time

from axp_core.runtime import read_json, runtime_paths

HEARTBEAT_STALE_AFTER_S = 90
AUTO_RESTART_COOLDOWN_S = 60


def read_daemon_state(stale_after_s=HEARTBEAT_STALE_AFTER_S, now_ms=None):
    value = read_json(runtime_paths()["state"], {}) or {}
    heartbeat = value.get("heartbeat_ms", 0)
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    age_ms = current_ms - heartbeat if heartbeat else None
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
        from .progress import progress_estimate
        completed = state.get("files_completed", state.get("files_processed", 0))
        estimate = progress_estimate(completed, state.get("progress_baseline", 0))
        progress = f" {estimate.label}" if estimate.percent is not None else ""
        return f"AXPIndexerNG — Scanning{progress} — {completed:,} files"[:127]
    if current == "paused":
        return "AXPIndexerNG — Paused"
    if current == "idle":
        return f"AXPIndexerNG — Idle — {state.get('documents_total', 0):,} documents"[:127]
    return f"AXPIndexerNG — {current.upper()}"[:127]


def should_auto_restart(state, desired_state, enabled, last_restart_monotonic, now_monotonic,
                        daemon_instance_present=False):
    """Decide whether to spawn; an owned daemon lock always vetoes a spawn."""
    if desired_state != "running" or not enabled or daemon_instance_present:
        return False
    needs_restart = state.get("stale") or state.get("state") == "stopped"
    return bool(
        needs_restart and now_monotonic - last_restart_monotonic >= AUTO_RESTART_COOLDOWN_S
    )
