from __future__ import annotations

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from axp_core.database import connect
from axp_core.locking import AlreadyLocked, FileLock, daemon_instance_running
from axp_core.runtime import atomic_write_json, configure_logging, load_settings, read_json, runtime_paths
from axp_daemon.service import send_control

from .icons import make_icon
from .process import ensure_client, restart_daemon, start_daemon, stop_client, stop_daemon
from .sources_window import SourcesWindow
from .startup import is_enabled, repair_registration, set_enabled
from .state import read_daemon_state, should_auto_restart, tooltip

LOGGER = configure_logging("axp_tray", "tray.log")


def establish_startup_desired(settings, paths):
    """An explicit application launch supersedes a stopped state left by Exit."""
    if settings["auto_start_daemon"]:
        atomic_write_json(paths["desired"], {"state": "running", "updated_ms": int(time.time() * 1000)})


class TrayApplication:
    def __init__(self):
        import tkinter as tk

        import pystray

        self.pystray = pystray
        self.settings = load_settings()
        self.paths = runtime_paths()
        self.lock = FileLock(self.paths["tray_lock"]).acquire()
        self.root = tk.Tk()
        self.root.withdraw()
        self.state = read_daemon_state()
        self.icon_state = None
        self.last_restart = -60.0
        self.last_stale_lock_warning = None
        self.ui_queue = queue.Queue()
        self.shutting_down = False
        establish_startup_desired(self.settings, self.paths)
        repair_registration()
        self.icon = pystray.Icon("AXPIndexerNG", make_icon("starting"), "AXPIndexerNG", menu=self._menu())

    def _menu(self):
        p = self.pystray
        return p.Menu(
            p.MenuItem(lambda _: f"AXPIndexerNG — {self.state.get('state', 'stopped').title()}", None, enabled=False),
            p.MenuItem("Search...", self._search, default=True),
            p.MenuItem("Sources...", self._sources),
            p.Menu.SEPARATOR,
            p.MenuItem("Scan now", lambda *_: send_control("scan")),
            p.MenuItem(lambda _: "Resume indexing" if self.state.get("state") == "paused" else "Pause indexing",
                       self._toggle_pause),
            p.MenuItem("Restart daemon", self._restart),
            p.Menu.SEPARATOR,
            p.MenuItem(lambda _: f"Status: {self.state.get('documents_total', 0):,} documents", None, enabled=False),
            p.MenuItem(lambda _: f"Chunks: {self.state.get('chunks_total', 0):,}", None, enabled=False),
            p.MenuItem(lambda _: f"Current source: {self.state.get('current_source') or '—'}", None, enabled=False),
            p.MenuItem(lambda _: f"Current file: {Path(self.state.get('current_file')).name if self.state.get('current_file') else '—'}",
                       None, enabled=False),
            p.Menu.SEPARATOR,
            p.MenuItem("Open data folder", lambda *_: self._open(self.paths["data"])),
            p.MenuItem("Open logs", lambda *_: self._open(self.paths["logs"])),
            p.Menu.SEPARATOR,
            p.MenuItem("Start with Windows", self._toggle_startup, checked=lambda _: is_enabled()),
            p.Menu.SEPARATOR,
            p.MenuItem("Exit AXPIndexerNG", self._exit),
        )

    def _on_tk(self, callback):
        self.ui_queue.put(callback)

    def _drain_ui_queue(self):
        while True:
            try:
                self.ui_queue.get_nowait()()
            except queue.Empty:
                break
        self.root.after(100, self._drain_ui_queue)

    def _sources(self, *_):
        self._on_tk(lambda: SourcesWindow(self.root, self.settings["db_path"]))

    def _search(self, *_):
        threading.Thread(target=self._search_worker, name="axp-open-search", daemon=True).start()

    def _search_worker(self):
        try:
            ensure_client(self.settings)
            deadline = time.monotonic() + 8
            from .process import client_healthy

            while time.monotonic() < deadline and not client_healthy(self.settings):
                time.sleep(0.2)
            webbrowser.open(f"http://{self.settings['web_host']}:{self.settings['web_port']}/")
        except Exception:
            LOGGER.exception("Could not start/open search client")

    def _toggle_pause(self, *_):
        send_control("resume" if self.state.get("state") == "paused" else "pause")

    def _restart(self, *_):
        threading.Thread(target=self._restart_worker, name="axp-restart-daemon", daemon=True).start()

    def _restart_worker(self):
        try:
            restart_daemon(self.settings)
        except Exception:
            LOGGER.exception("Manual daemon restart failed")

    def _toggle_startup(self, *_):
        try:
            set_enabled(not is_enabled())
        except Exception:
            LOGGER.exception("Start-with-Windows update failed")

    def _open(self, path):
        try:
            os.startfile(str(path))
        except (AttributeError, OSError):
            LOGGER.exception("Could not open %s", path)

    def _exit(self, *_):
        if self.shutting_down:
            return
        self.shutting_down = True
        threading.Thread(target=self._shutdown_worker, name="axp-shutdown", daemon=True).start()

    def _shutdown_worker(self):
        try:
            LOGGER.info("Daemon shutdown requested from tray exit")
            stop_daemon(intentional=True)
        except Exception:
            LOGGER.exception("Daemon shutdown request failed")
        try:
            if stop_client(self.settings):
                LOGGER.info("Web client shutdown requested from tray exit")
        except Exception:
            LOGGER.exception("Web client shutdown request failed")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if read_daemon_state().get("state") == "stopped":
                LOGGER.info("Daemon stopped")
                break
            time.sleep(0.1)
        LOGGER.info("AXPIndexerNG tray exiting")
        try:
            self.icon.stop()
        except Exception:
            LOGGER.exception("Could not stop tray icon")
        self._on_tk(self.root.quit)

    def _poll(self):
        if self.shutting_down:
            return
        self.state = read_daemon_state()
        desired = read_json(self.paths["desired"], {}) or {}
        should_run = desired.get("state", "running" if self.settings["auto_start_daemon"] else "stopped") == "running"
        restart_candidate = should_run and (self.state.get("stale") or self.state.get("state") == "stopped")
        instance_present = daemon_instance_running(self.settings["db_path"]) if restart_candidate else False
        self.state["daemon_instance_present"] = instance_present
        self.state["restart_suppressed"] = bool(restart_candidate and instance_present)
        now = time.monotonic()
        if self.state["restart_suppressed"] and (
                self.last_stale_lock_warning is None or now - self.last_stale_lock_warning >= 60):
            LOGGER.warning("Daemon heartbeat stale but daemon instance is still active; duplicate restart suppressed")
            self.last_stale_lock_warning = now
        elif not self.state["restart_suppressed"]:
            self.last_stale_lock_warning = None
        if should_auto_restart(self.state, "running" if should_run else "stopped",
                               self.settings["auto_restart_daemon"], self.last_restart, now, instance_present):
            LOGGER.warning("Auto-restart: stale heartbeat age=%s ms old_pid=%s",
                           self.state.get("heartbeat_age_ms"), self.state.get("pid"))
            try:
                process = start_daemon(self.settings)
                self.last_restart = time.monotonic()
                LOGGER.info("Auto-restart launched PID %s", process.pid)
            except Exception:
                LOGGER.exception("Auto-restart failed")
        state_name = self.state.get("state", "stopped")
        if state_name != self.icon_state:
            self.icon.icon = make_icon(state_name)
            self.icon_state = state_name
        self.icon.title = tooltip(self.state)
        self.icon.update_menu()
        self.root.after(2000, self._poll)

    def run(self):
        LOGGER.info("Tray started from %s", Path.cwd())
        self.icon.run_detached()
        self._drain_ui_queue()
        self._poll()
        self.root.mainloop()
        self.lock.release()


def self_test(db=None):
    import tkinter

    import pystray
    from PIL import Image

    del tkinter, pystray
    icon = make_icon("idle")
    if not isinstance(icon, Image.Image) or icon.size != (32, 32):
        raise RuntimeError("Tray icon generation failed")
    paths = runtime_paths()
    temporary = None
    if db is None:
        temporary = tempfile.TemporaryDirectory()
        db = Path(temporary.name) / "self-test.db"
    with connect(db) as con:
        version = con.execute("SELECT version FROM schema_version").fetchone()[0]
        con.execute("SELECT count(*) FROM sources").fetchone()
    bundled = Path(__file__).resolve().parents[2] / "python" / "pythonw.exe"
    value = {
        "status": "ok", "schema_version": version, "db": str(Path(db).resolve()),
        "runtime_dir": str(paths["runtime"]), "pythonw": str(bundled), "pythonw_exists": bundled.exists(),
        "tkinter": True, "pystray": True, "pillow": True, "icon": list(icon.size),
    }
    if temporary:
        temporary.cleanup()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    test = sub.add_parser("self-test")
    test.add_argument("--db")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        import json

        print(json.dumps(self_test(args.db)))
        return
    try:
        TrayApplication().run()
    except AlreadyLocked:
        return


if __name__ == "__main__":
    main(sys.argv[1:])
