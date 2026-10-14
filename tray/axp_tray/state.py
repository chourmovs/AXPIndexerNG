import time

from axp_core.runtime import read_json, runtime_paths

HEARTBEAT_STALE_AFTER_S = 90
AUTO_RESTART_COOLDOWN_S = 60


def read_daemon_state(stale_after_s=HEARTBEAT_STALE_AFTER_S, now_ms=None):
    value = read_json(runtime_paths()["state"], {}) or {}
    heartbeat = value.get("heartbeat_ms", 0)
    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    monotonic_heartbeat = value.get("heartbeat_monotonic")
    if now_ms is None and isinstance(monotonic_heartbeat, (int, float)):
        delta = time.monotonic() - monotonic_heartbeat
        age_ms = delta * 1000 if delta >= 0 else None
    else:
        age_ms = current_ms - heartbeat if heartbeat else None
    value["heartbeat_age_ms"] = age_ms
    value["stale"] = age_ms is None or age_ms > stale_after_s * 1000
    return value


def heartbeat_display(state, delayed_after_s=5):
    """Describe freshness without overwriting the daemon's authoritative state."""
    age_ms = state.get("heartbeat_age_ms")
    if age_ms is None:
        return "Daemon state unavailable"
    age_s = max(0.0, age_ms / 1000)
    if state.get("stale"):
        return f"Daemon state STALE · {age_s:.0f}s old"
    if age_s >= delayed_after_s:
        return f"Daemon state delayed · {age_s:.0f}s old"
    return f"Daemon PID {state.get('pid', '—')} · state updated {age_s:.1f}s ago"


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


STARTUP_GRACE_S = 180
WORKER_STALL_AFTER_S = 1800


class DaemonWatchdog:
    """Monotonic, bounded recovery policy, including live lock holders."""
    def __init__(self):
        self.instance = None
        self.first_seen = None
        self.attempts = []
        self.last_attempt = None

    def reason(self, state, desired, enabled, instance_present, now):
        if desired != "running" or not enabled or state.get("state") == "stopping":
            return None
        identity = state.get("instance_id")
        if identity != self.instance or self.first_seen is None:
            self.instance, self.first_seen = identity, now
        self.attempts = [tick for tick in self.attempts if now - tick < 900]
        if len(self.attempts) >= 3:
            return None
        cooldown = min(300, 60 * 2 ** max(0, len(self.attempts) - 1))
        if self.last_attempt is not None and now - self.last_attempt < cooldown:
            return None
        if instance_present and now - self.first_seen < STARTUP_GRACE_S:
            return None
        if not instance_present:
            return "instance_absent"
        if state.get("stale"):
            return "heartbeat_stale"
        progress = state.get("worker_progress_monotonic")
        if isinstance(progress, (int, float)) and now - progress > WORKER_STALL_AFTER_S:
            return "worker_stalled"
        return None

    def attempted(self, now):
        self.last_attempt = now
        self.attempts.append(now)
