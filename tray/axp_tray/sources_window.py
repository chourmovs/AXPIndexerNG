from __future__ import annotations

import os
import tkinter as tk
from datetime import UTC, datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

from axp_core.database import connect
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
from .state import read_daemon_state


def add_gui_source(con, path):
    """Persist a source selected in the desktop UI with the desktop recursion policy."""
    return add_source(con, path, recursive=True)


class SourcesWindow:
    def __init__(self, root, db_path):
        self.db_path = db_path
        self.window = tk.Toplevel(root)
        self.window.title("AXPIndexerNG — Indexed locations")
        self.window.geometry("1040x550")
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        self.tree = ttk.Treeview(self.window, columns=("enabled", "label", "path", "kind", "status", "coverage", "files", "chunks", "last"),
                                 show="headings", selectmode="browse")
        headings = (("enabled", "Enabled", 65), ("label", "Label", 130), ("path", "Path", 260),
                    ("kind", "Type", 60), ("status", "Status", 75), ("coverage", "Coverage", 155), ("files", "Files", 60),
                    ("chunks", "Chunks", 85), ("last", "Last successful scan", 145))
        for key, text, width in headings:
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._show_details())
        add = ttk.Frame(self.window)
        add.pack(fill="x", padx=10)
        ttk.Button(add, text="Explorer...", command=self.add_folder).pack(side="left", padx=3)
        ttk.Button(add, text="Add drive...", command=self.add_drive).pack(side="left", padx=3)
        ttk.Button(add, text="Add UNC...", command=self.add_unc).pack(side="left", padx=3)
        actions = ttk.Frame(self.window)
        actions.pack(fill="x", padx=10, pady=8)
        ttk.Button(actions, text="Enable/Disable", command=self.toggle).pack(side="left", padx=3)
        ttk.Button(actions, text="Reindex", command=self.reindex).pack(side="left", padx=3)
        ttk.Button(actions, text="Remove", command=self.remove).pack(side="left", padx=3)
        ttk.Button(actions, text="Open in Explorer", command=self.open_explorer).pack(side="left", padx=3)
        ttk.Button(actions, text="Close", command=self.window.destroy).pack(side="right", padx=3)
        self.status = tk.StringVar(value="")
        self._refresh_job = None
        ttk.Label(self.window, textvariable=self.status).pack(fill="x", padx=13, pady=(0, 8))
        self.refresh()

    def _connection(self):
        return connect(self.db_path)

    def _selected_id(self):
        selection = self.tree.selection()
        return int(selection[0]) if selection else None

    def _selected(self, con):
        source_id = self._selected_id()
        return con.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone() if source_id else None

    def refresh(self):
        if self._refresh_job is not None:
            self.window.after_cancel(self._refresh_job)
            self._refresh_job = None
        selected = self._selected_id()
        self.tree.delete(*self.tree.get_children())
        with self._connection() as con:
            rows = list_sources(con)
        for row in rows:
            last = datetime.fromtimestamp(row["last_success_ms"] / 1000, UTC).astimezone().strftime("%Y-%m-%d %H:%M") if row["last_success_ms"] else "—"
            absorbed, content = coverage_percentages(row["last_seen_count"], row["last_content_count"],
                                                     row["last_metadata_count"])
            coverage = f"{absorbed:.0f}% absorbed / {content:.0f}% content" if row["last_success_ms"] else "—"
            self.tree.insert("", "end", iid=str(row["id"]), values=("✓" if row["enabled"] else "○", row["label"],
                             row["path"], row["kind"], row["status"].title(), coverage, f"{row['documents']:,}",
                             f"{row['chunks']:,}", last))
        if selected and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
        self.status.set("No indexed locations configured" if not rows else f"{len(rows)} indexed location(s)")
        self._show_details()
        if self.window.winfo_exists():
            self._refresh_job = self.window.after(3000, self.refresh)

    def _show_details(self):
        source_id = self._selected_id()
        if not source_id:
            return
        with self._connection() as con:
            row = con.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if not row:
            return
        absorbed, content = coverage_percentages(row["last_seen_count"], row["last_content_count"],
                                                 row["last_metadata_count"])
        formats = sorted(((ext, values.get("metadata", 0)) for ext, values in extension_breakdown(row).items()
                          if values.get("metadata", 0)), key=lambda item: (-item[1], item[0]))[:4]
        common = ", ".join(f"{ext} ({count})" for ext, count in formats) or "none"
        incomplete = " Scan status is incomplete; last complete coverage retained." if row["status"] != "idle" else ""
        self.status.set(
            f"Last complete scan — seen {row['last_seen_count']:,}; content {row['last_content_count']:,}; "
            f"metadata only {row['last_metadata_count']:,}; ignored {row['last_ignored_count']:,}; "
            f"failed {row['last_failed_count']:,}. Absorption {absorbed:.1f}%; content coverage {content:.1f}%. "
            f"Top metadata-only formats: {common}.{incomplete}"
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
