from __future__ import annotations

import os
import logging
import sqlite3
import time
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from axp_core.database import connect, open_catalog_reader
from axp_core.sources import (
    SourceError,
    add_source,
    coverage_percentages,
    disable_source,
    enable_source,
    extension_breakdown,
    list_sources,
)
from axp_daemon.service import send_control

from .drives import list_drives
from .progress import RollingThroughput, estimate_eta_seconds, format_bytes, format_duration, progress_estimate
from .state import heartbeat_display, read_daemon_state

LOGGER = logging.getLogger("axp_tray")


def add_gui_source(con, path):
    return add_source(con, path, recursive=True)


def coverage_display(row, scanning=False, estimate=None):
    if scanning:
        if not row["last_success_ms"]:
            return "First scan in progress"
        if estimate and estimate.exceeded_baseline:
            return "Previous scan size exceeded"
        return f"{estimate.label} scanning" if estimate and estimate.percent is not None else "Scanning…"
    if not row["last_success_ms"]:
        return "Pending first full scan"
    absorbed, content = coverage_percentages(row["last_seen_count"], row["last_content_count"],
                                             row["last_metadata_count"])
    value = f"{absorbed:.0f}% absorbed / {content:.0f}% content"
    return value if row["status"] == "idle" else f"{absorbed:.0f}% / {content:.0f}% · last complete"


def last_success_display(row, scanning=False):
    if row["last_success_ms"]:
        return datetime.fromtimestamp(row["last_success_ms"] / 1000, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
    return "In progress" if scanning else "Not completed yet"


class SourcesWindow:
    def __init__(self, root, db_path):
        self.db_path = db_path
        self.window = tk.Toplevel(root)
        self.window.title("AXPIndexerNG — Indexed locations")
        self.window.geometry("1100x700")
        self.window.minsize(850, 600)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(self.window, columns=("enabled", "label", "path", "status", "coverage", "files", "chunks", "last"),
                                 show="headings", selectmode="browse", height=8)
        for key, text, width in (("enabled", "Enabled", 60), ("label", "Label", 120), ("path", "Path", 250),
                                 ("status", "Status", 80), ("coverage", "Coverage", 185), ("files", "Indexed files", 90),
                                 ("chunks", "Chunks", 80), ("last", "Last successful scan", 145)):
            self.tree.heading(key, text=text); self.tree.column(key, width=width, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._show_details())

        activity = ttk.LabelFrame(self.window, text="Indexing activity", padding=8)
        activity.grid(row=1, column=0, sticky="ew", padx=10, pady=4); activity.columnconfigure(0, weight=1)
        self.activity_title = tk.StringVar(value="Daemon idle")
        ttk.Label(activity, textvariable=self.activity_title, font=("TkDefaultFont", 11, "bold")).grid(sticky="w")
        self.progress = ttk.Progressbar(activity, maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=4)
        self.progress_text = tk.StringVar(value="—")
        ttk.Label(activity, textvariable=self.progress_text).grid(row=1, column=1, padx=(8, 0))
        self.file_text = tk.StringVar(value="Current file: —")
        ttk.Label(activity, textvariable=self.file_text).grid(row=2, column=0, columnspan=2, sticky="w")
        self.counters_text = tk.StringVar(value="Completed 0 · Content 0 · Metadata 0 · Failed 0 · Chunks 0")
        ttk.Label(activity, textvariable=self.counters_text).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.speed_text = tk.StringVar(value="Speed — · Elapsed — · ETA —")
        ttk.Label(activity, textvariable=self.speed_text).grid(row=4, column=0, columnspan=2, sticky="w")
        self.catalog_text = tk.StringVar(value="Catalog: —")
        ttk.Label(activity, textvariable=self.catalog_text).grid(row=5, column=0, columnspan=2, sticky="w")
        self.heartbeat_text = tk.StringVar(value="Daemon state unavailable")
        ttk.Label(activity, textvariable=self.heartbeat_text).grid(row=6, column=0, columnspan=2, sticky="w")

        resources = ttk.LabelFrame(self.window, text="Resource usage", padding=8)
        resources.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        self.resource_text = tk.StringVar(value="AXP       CPU — · RAM —\nSystem    CPU — · RAM —\nStorage   DB — · WAL —")
        ttk.Label(resources, textvariable=self.resource_text).pack(anchor="w")
        self.monitor_health = tk.StringVar(value="Monitoring starting…")
        ttk.Label(resources, textvariable=self.monitor_health).pack(anchor="w")

        self.details = tk.StringVar(value="Select a source for completed coverage details.")
        ttk.Label(self.window, textvariable=self.details, justify="left").grid(row=3, column=0, sticky="ew", padx=13)
        buttons = ttk.Frame(self.window); buttons.grid(row=4, column=0, sticky="ew", padx=10, pady=8)
        for text, command in (("Explorer...", self.add_folder), ("Add drive...", self.add_drive), ("Add UNC...", self.add_unc),
                              ("Enable/Disable", self.toggle), ("Reindex", self.reindex), ("Remove", self.remove),
                              ("Open in Explorer", self.open_explorer)):
            ttk.Button(buttons, text=text, command=command).pack(side="left", padx=3)
        ttk.Button(buttons, text="Close", command=self._close).pack(side="right", padx=3)
        self.status = self.details
        from .resources import ResourceMonitor, database_sizes

        self.database_sizes = database_sizes
        self.monitor, self.rates = ResourceMonitor(), RollingThroughput()
        self.live_state = {}
        self._source_rows = []
        self._last_resource_snapshot = None
        self._last_resource_success_monotonic = None
        self._last_sizes = None
        self._refresh_job = self._live_job = None
        self.refresh(); self._refresh_live()

    def _close(self):
        if self._refresh_job: self.window.after_cancel(self._refresh_job)
        if self._live_job: self.window.after_cancel(self._live_job)
        self.window.destroy()

    def _connection(self): return connect(self.db_path)
    def _catalog_connection(self): return open_catalog_reader(self.db_path)
    def _selected_id(self):
        selection = self.tree.selection(); return int(selection[0]) if selection else None
    def _selected(self, con):
        source_id = self._selected_id()
        return con.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone() if source_id else None

    def refresh(self):
        try:
            with self._catalog_connection() as con:
                rows = list(list_sources(con))
            selected = self._selected_id()
            state = self.live_state
            # Only replace the widget after the complete query succeeds.
            self.tree.delete(*self.tree.get_children())
            self._source_rows = rows
            for row in rows:
                scanning = state.get("state") == "scanning" and state.get("current_source_id") == row["id"]
                estimate = progress_estimate(state.get("files_completed", 0), state.get("progress_baseline", 0)) if scanning else None
                status = "Scanning" if scanning else ("Waiting" if row["enabled"] and not row["last_success_ms"] and state.get("state") == "scanning" else row["status"].title())
                self.tree.insert("", "end", iid=str(row["id"]), values=("✓" if row["enabled"] else "○", row["label"], row["path"], status,
                                 coverage_display(row, scanning, estimate), f"{row['documents']:,}", f"{row['chunks']:,}",
                                 last_success_display(row, scanning)))
            if selected and self.tree.exists(str(selected)): self.tree.selection_set(str(selected))
            self._show_details()
        except (OSError, sqlite3.Error, ValueError, TypeError):
            self.monitor_health.set("Catalog refresh delayed · showing last update")
            LOGGER.warning("Catalog dashboard refresh delayed", exc_info=True)
        finally:
            try:
                if self.window.winfo_exists(): self._refresh_job = self.window.after(4000, self.refresh)
            except tk.TclError:
                pass

    def _refresh_live(self):
        try:
            try:
                self.live_state = read_daemon_state()
            except (OSError, ValueError, TypeError):
                LOGGER.warning("Daemon state refresh delayed", exc_info=True)
            try:
                self._render_live_state(self.live_state)
            except Exception:  # noqa: BLE001 -- isolate activity rendering from monitoring
                LOGGER.exception("Live dashboard activity render failed")
            try:
                self._refresh_resources(self.live_state.get("pid"))
            except Exception:  # noqa: BLE001 -- isolate monitoring from activity rendering
                LOGGER.exception("Live dashboard resource render failed")
        except Exception:  # noqa: BLE001 -- a Tk callback must never permanently die
            LOGGER.exception("Live dashboard refresh failed unexpectedly")
        finally:
            try:
                if self.window.winfo_exists(): self._live_job = self.window.after(1000, self._refresh_live)
            except tk.TclError:
                pass

    def _render_live_state(self, state):
            scanning = state.get("state") == "scanning"
            completed, chunks = state.get("files_completed", 0), state.get("chunks_embedded", 0)
            baseline = state.get("progress_baseline", 0) if state.get("progress_baseline_kind") == "previous_complete_scan" else 0
            estimate = progress_estimate(completed, baseline)
            self.activity_title.set(f"{Path(state.get('current_source') or '').name or 'Daemon'} — {state.get('state', 'stopped').upper()}")
            batch_docs = state.get("current_batch_documents", 0)
            current = (f"{state.get('current_stage', 'Working').title()} batch · {batch_docs:,} documents / "
                       f"{state.get('current_batch_chunks', 0):,} chunks") if batch_docs else (Path(state["current_file"]).name if state.get("current_file") else "—")
            self.file_text.set(f"Current: {current} · Stage: {state.get('current_stage', '—').title()}")
            self.counters_text.set(f"Completed {completed:,} / Seen {state.get('files_seen', 0):,} · Content {state.get('files_content', 0):,} · Metadata {state.get('files_metadata', 0):,} · Ignored {state.get('files_ignored', 0):,} · Failed {state.get('files_failed', 0):,} · Chunks {chunks:,}")
            if scanning and estimate.percent is None:
                self.progress.configure(mode="indeterminate"); self.progress.start(15)
            else:
                self.progress.stop(); self.progress.configure(mode="determinate"); self.progress["value"] = estimate.percent or 0
            if scanning and not baseline:
                self.progress_text.set(f"First complete scan · {completed:,} files completed · total being discovered")
            elif estimate.exceeded_baseline:
                self.progress_text.set(f"Previous scan size exceeded · {completed:,} files completed")
            else:
                self.progress_text.set(f"{estimate.label} · based on previous complete scan ({baseline:,} files)" if scanning else "—")
            monotonic_now = time.monotonic()
            file_rate, chunk_rate = self.rates.add(state.get("current_source_id"), monotonic_now, completed, chunks)
            started = state.get("source_scan_started_ms"); elapsed = max(0, time.time() - started / 1000) if started else 0
            eta = estimate_eta_seconds(completed, estimate.baseline, file_rate, elapsed_s=elapsed) if scanning and len(self.rates.samples) >= 3 else None
            speed = "—" if file_rate is None or not scanning else f"{file_rate:.1f} files/s · {chunk_rate:.1f} chunks/s"
            eta_text = ("available after first complete scan" if scanning and not baseline else
                        "recalculating…" if estimate.exceeded_baseline else
                        "~" + format_duration(eta) if eta is not None else "estimating…" if scanning else "—")
            self.speed_text.set(f"Speed {speed} · Elapsed {format_duration(elapsed) if scanning else '—'} · ETA {eta_text}")
            self.catalog_text.set(f"Catalog: {state.get('documents_total', 0):,} documents · {state.get('chunks_total', 0):,} chunks")
            self.heartbeat_text.set(heartbeat_display(state))

    def _refresh_resources(self, pid):
        now = time.monotonic()
        try:
            self._last_resource_snapshot = self.monitor.sample(pid)
            self._last_resource_success_monotonic = now
        except Exception:  # noqa: BLE001 -- retain the last successful sample
            LOGGER.warning("Resource monitoring sample delayed", exc_info=True)
        try:
            self._last_sizes = self.database_sizes(self.db_path)
        except (OSError, ValueError, TypeError):
            LOGGER.warning("Database size sample delayed", exc_info=True)
        sample, sizes = self._last_resource_snapshot, self._last_sizes
        if sample is not None:
            process_cpu = f"{sample.process_cpu:.0f}%" if sample.process_cpu is not None else "—"
            system_cpu = f"{sample.system_cpu:.0f}%" if sample.system_cpu is not None else "—"
            ram = f"{sample.system_ram_percent:.0f}%" if sample.system_ram_percent is not None else "—"
            self.resource_text.set(
                f"AXP       CPU {process_cpu} · RAM {format_bytes(sample.process_rss)} · Read {format_bytes(sample.read_bytes_s)}/s · Write {format_bytes(sample.write_bytes_s)}/s\n"
                f"System    CPU {system_cpu} · RAM {ram} · {format_bytes(sample.available_ram)} available · {sample.power or 'Power —'}\n"
                f"Storage   DB {format_bytes(sizes.database) if sizes else '—'} · WAL {format_bytes(sizes.wal) if sizes else '—'} · Total {format_bytes(sizes.total) if sizes else '—'}"
            )
        elif sizes is not None:
            self.resource_text.set(f"AXP       CPU — · RAM —\nSystem    CPU — · RAM —\nStorage   DB {format_bytes(sizes.database)} · WAL {format_bytes(sizes.wal)} · Total {format_bytes(sizes.total)}")
        age = None if self._last_resource_success_monotonic is None else now - self._last_resource_success_monotonic
        if age is None:
            self.monitor_health.set("Resources unavailable · monitoring has no valid sample yet")
        elif age < 2:
            self.monitor_health.set("Monitoring live")
        elif age <= 10:
            self.monitor_health.set(f"Monitoring delayed · last sample {age:.0f}s ago")
        else:
            self.monitor_health.set(f"Resources temporarily unavailable · last update {age:.0f}s ago")

    def _show_details(self):
        source_id = self._selected_id()
        if not source_id: return
        row = next((item for item in self._source_rows if item["id"] == source_id), None)
        if not row: return
        if not row["last_success_ms"]:
            self.details.set("Coverage: Pending first full scan. Live progress above is estimated when a baseline exists."); return
        absorbed, content = coverage_percentages(row["last_seen_count"], row["last_content_count"], row["last_metadata_count"])
        formats = sorted(((ext, values.get("metadata", 0)) for ext, values in extension_breakdown(row).items() if values.get("metadata", 0)), key=lambda item: (-item[1], item[0]))[:4]
        common = ", ".join(f"{ext} ({count})" for ext, count in formats) or "none"
        self.details.set(
            f"Last complete coverage: {row['last_seen_count']:,} seen · {row['last_content_count']:,} content · "
            f"{row['last_metadata_count']:,} metadata only · {row['last_ignored_count']:,} ignored · "
            f"{row['last_failed_count']:,} failed\nAbsorption {absorbed:.1f}% · Content coverage {content:.1f}% · "
            f"Top metadata-only formats: {common}"
        )

    def _add(self, path, *, allow_system_warning=True):
        if not path:
            return
        if allow_system_warning and path.casefold() == "c:\\" and not messagebox.askyesno(
            "System drive",
            "Indexing an entire system drive may include a very large number of irrelevant or protected files. "
            "Selecting specific document folders is recommended.\n\nContinue?",
            parent=self.window,
        ):
            return
        try:
            with self._connection() as con:
                add_gui_source(con, path)
            send_control("scan")
            self.refresh()
        except SourceError as exc:
            messagebox.showerror("Invalid source", str(exc), parent=self.window)

    def add_folder(self):
        self._add(filedialog.askdirectory(parent=self.window, mustexist=True, title="Choose indexed location"))

    def add_unc(self):
        value = simpledialog.askstring("Add UNC source", r"UNC path (\\server\share or subfolder):", parent=self.window)
        self._add(value, allow_system_warning=False)

    def add_drive(self):
        drives = list_drives()
        if not drives:
            messagebox.showerror("Drives", "No Windows drives were detected.", parent=self.window)
            return
        dialog = tk.Toplevel(self.window)
        dialog.title("Add drive")
        box = tk.Listbox(dialog, width=35, height=min(12, len(drives)))
        for drive in drives:
            box.insert("end", f"{drive.root[:2]}  {drive.kind}")
        box.pack(padx=12, pady=12)

        def choose():
            if box.curselection():
                value = drives[box.curselection()[0]].root
                dialog.destroy()
                self._add(value)

        ttk.Button(dialog, text="Add", command=choose).pack(pady=(0, 12))
        dialog.transient(self.window)
        dialog.grab_set()

    def toggle(self):
        with self._connection() as con:
            row = self._selected(con)
            if row:
                (disable_source if row["enabled"] else enable_source)(con, row["id"])
        self.refresh()

    def reindex(self):
        source_id = self._selected_id()
        if source_id and messagebox.askyesno("Reindex", "Reindex this location?\n\nExisting indexed data for this location will be rebuilt.",
                                             parent=self.window):
            send_control("reindex", source_id=source_id)
            self.status.set("Reindex request sent to the daemon")

    def remove(self):
        source_id = self._selected_id()
        if not source_id:
            return
        state = read_daemon_state()
        if state.get("state") == "scanning" and state.get("current_source_id") == source_id:
            messagebox.showerror("Source busy", "This location is currently being scanned. Pause indexing or wait before removing it.",
                                 parent=self.window)
            return
        if not messagebox.askyesno("Remove source", "Remove this indexed location?\n\nAll indexed documents and vectors belonging to this source will be removed from the catalog.\n\nFiles on disk will NOT be deleted.",
                                   parent=self.window):
            return
        send_control("remove", source_id=source_id)
        self.status.set("Removal request sent; indexed rows will be deleted at a safe daemon boundary")

    def open_explorer(self):
        with self._connection() as con:
            row = self._selected(con)
        if not row:
            return
        try:
            os.startfile(row["path"])
        except (AttributeError, OSError) as exc:
            messagebox.showerror("Open location", f"The location is unavailable.\n\n{exc}", parent=self.window)
