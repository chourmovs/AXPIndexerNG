"""Catalog source repository and Windows-aware path rules."""

from __future__ import annotations

import json
import ntpath
import os
import re
import time
from pathlib import Path

VALID_STATUSES = {"idle", "scanning", "paused", "offline", "error", "disabled"}
_DRIVE = re.compile(r"^[A-Za-z]:([\\/]|$)")


class SourceError(ValueError):
    pass


class SourceOverlapError(SourceError):
    def __init__(self, message: str, *, existing_id: int | None = None):
        super().__init__(message)
        self.existing_id = existing_id


def _is_windows_path(value: str) -> bool:
    return bool(_DRIVE.match(value)) or value.startswith(("\\\\", "//"))


def normalize_source_path(path: str | os.PathLike[str]) -> tuple[str, str]:
    """Return display path and deterministic key without requiring the path online."""
    value = os.fspath(path).strip()
    if not value:
        raise SourceError("Source path cannot be empty")
    if _is_windows_path(value):
        display = ntpath.normpath(value.replace("/", "\\"))
        drive, tail = ntpath.splitdrive(display)
        if drive and not tail:
            display = drive + "\\"
        if display.startswith("\\\\"):
            parts = [part for part in display[2:].split("\\") if part]
            if len(parts) < 2:
                raise SourceError("UNC path must include a server and share")
            display = "\\\\" + "\\".join(parts)
        return display, display.casefold()
    display = os.path.normpath(os.path.abspath(value))
    return display, os.path.normcase(display).casefold()


def detect_source_kind(path: str) -> str:
    display, _ = normalize_source_path(path)
    if display.startswith("\\\\"):
        return "unc"
    drive, tail = ntpath.splitdrive(display)
    return "drive" if drive and tail == "\\" else "folder"


def default_label(path: str) -> str:
    display, _ = normalize_source_path(path)
    if not _is_windows_path(display):
        return Path(display).name or display
    if display.startswith("\\\\"):
        return display.rstrip("\\").split("\\")[-1]
    drive, tail = ntpath.splitdrive(display)
    if drive and tail == "\\":
        return drive
    return ntpath.basename(display.rstrip("\\/")) or Path(display).name or display


def _parts(path_key: str) -> tuple[str, ...]:
    if _is_windows_path(path_key):
        drive, tail = ntpath.splitdrive(path_key)
        return (drive.casefold(), *(p for p in tail.replace("/", "\\").split("\\") if p))
    return tuple(Path(path_key).parts)


def is_parent_path(parent_key: str, child_key: str) -> bool:
    parent, child = _parts(parent_key), _parts(child_key)
    return len(parent) < len(child) and child[: len(parent)] == parent


def list_sources(con, *, enabled_only: bool = False):
    where = "WHERE s.enabled=1" if enabled_only else ""
    return con.execute(
        f"""SELECT s.*,
        (SELECT count(*) FROM documents d WHERE d.source_id=s.id) AS documents,
        (SELECT count(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.source_id=s.id) AS chunks
        FROM sources s {where} ORDER BY s.label COLLATE NOCASE, s.path_key"""
    ).fetchall()


def get_source(con, source_id: int):
    row = con.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    if row is None:
        raise SourceError(f"Unknown source id: {source_id}")
    return row


def find_overlap(con, path_key: str, *, exclude_id: int | None = None):
    for row in con.execute("SELECT id,path,path_key FROM sources").fetchall():
        if exclude_id is not None and row["id"] == exclude_id:
            continue
        if row["path_key"] == path_key:
            return row, "duplicate"
        if is_parent_path(row["path_key"], path_key):
            return row, "covered"
        if is_parent_path(path_key, row["path_key"]):
            return row, "subsumes"
    return None


def add_source(con, path, label=None, *, enabled=True, recursive=True, allow_subsume=False):
    display, key = normalize_source_path(path)
    overlap = find_overlap(con, key)
    if overlap:
        row, relation = overlap
        if relation in {"duplicate", "covered"}:
            raise SourceOverlapError(f"{display} is already covered by source {row['path']}", existing_id=row["id"])
        if not allow_subsume:
            raise SourceOverlapError(f"{display} would subsume existing source {row['path']}", existing_id=row["id"])
    now = int(time.time() * 1000)
    status = "idle" if enabled else "disabled"
    source_label = (label or "").strip() or default_label(display)
    cur = con.execute(
        """INSERT INTO sources(path,path_key,label,kind,enabled,recursive,status,created_ms,updated_ms)
        VALUES(?,?,?,?,?,?,?,?,?)""",
        (display, key, source_label, detect_source_kind(display), int(enabled),
         int(recursive), status, now, now),
    )
    con.commit()
    return get_source(con, cur.lastrowid)


def update_source(con, source_id, *, path=None, label=None, recursive=None, allow_subsume=False):
    old = get_source(con, source_id)
    display, key = normalize_source_path(path if path is not None else old["path"])
    overlap = find_overlap(con, key, exclude_id=source_id)
    if overlap and not (allow_subsume and overlap[1] == "subsumes"):
        row, relation = overlap
        verb = "is already covered by" if relation != "subsumes" else "would subsume"
        raise SourceOverlapError(f"{display} {verb} source {row['path']}", existing_id=row["id"])
    now = int(time.time() * 1000)
    con.execute(
        "UPDATE sources SET path=?,path_key=?,label=?,kind=?,recursive=?,updated_ms=? WHERE id=?",
        (display, key, (label if label is not None else old["label"]).strip(), detect_source_kind(display),
         int(recursive if recursive is not None else old["recursive"]), now, source_id),
    )
    con.commit()
    return get_source(con, source_id)


def enable_source(con, source_id, enabled=True):
    get_source(con, source_id)
    con.execute(
        "UPDATE sources SET enabled=?,status=?,updated_ms=? WHERE id=?",
        (int(enabled), "idle" if enabled else "disabled", int(time.time() * 1000), source_id),
    )
    con.commit()
    return get_source(con, source_id)


def disable_source(con, source_id):
    return enable_source(con, source_id, False)


def remove_source(con, source_id):
    source = get_source(con, source_id)
    con.execute("DELETE FROM sources WHERE id=?", (source_id,))
    con.commit()
    return source


def source_stats(con, source_id):
    get_source(con, source_id)
    row = con.execute(
        """SELECT count(DISTINCT d.id),count(c.id) FROM documents d
        LEFT JOIN chunks c ON c.document_id=d.id WHERE d.source_id=?""",
        (source_id,),
    ).fetchone()
    return {"documents": row[0], "chunks": row[1]}


def coverage_percentages(seen, content, metadata):
    """Return absorption and full-content percentages; ignored/failed remain in the denominator."""
    if not seen:
        return 0.0, 0.0
    return 100.0 * (content + metadata) / seen, 100.0 * content / seen


def extension_breakdown(source):
    try:
        return json.loads(source["last_extension_breakdown"] or "{}")
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def mark_source_status(con, source_id, status, *, error=None, **fields):
    if status not in VALID_STATUSES:
        raise SourceError(f"Invalid source status: {status}")
    allowed = {"last_scan_started_ms", "last_scan_completed_ms", "last_success_ms", "last_file_count",
               "last_chunk_count", "last_seen_count", "last_content_count", "last_metadata_count",
               "last_ignored_count", "last_failed_count", "last_extension_breakdown"}
    values = {k: v for k, v in fields.items() if k in allowed}
    values.update(status=status, last_error=error, updated_ms=int(time.time() * 1000))
    assignments = ",".join(f"{key}=?" for key in values)
    con.execute(f"UPDATE sources SET {assignments} WHERE id=?", (*values.values(), source_id))
    con.commit()
