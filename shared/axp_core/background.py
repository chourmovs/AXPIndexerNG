"""Windows background-indexing path and compatibility primitives.

This module contains configuration metadata only.  Authentication is deliberately
left to Windows and no credential is accepted by any API here.
"""
from __future__ import annotations

import ctypes
import ntpath
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResolvedSourcePath:
    logical_root: Path
    access_root: Path

    def logical_for(self, access_path) -> Path:
        relative = os.path.relpath(os.fspath(access_path), os.fspath(self.access_root))
        if relative == ".":
            return self.logical_root
        # pathlib on Windows gives the desired semantics; ntpath keeps tests portable.
        return Path(ntpath.join(os.fspath(self.logical_root), relative))


def drive_of(path: str) -> str:
    drive, _ = ntpath.splitdrive(os.fspath(path))
    return drive.upper()


def access_path_for(logical_path, mappings) -> Path:
    """Translate a trusted logical drive path using stored non-secret UNC metadata."""
    value = os.fspath(logical_path)
    drive = drive_of(value)
    unc = {key.upper(): root for key, root in (mappings or {}).items()}.get(drive)
    if not unc:
        return Path(value)
    tail = value[len(drive):].lstrip("\\/")
    return Path(ntpath.join(unc, tail))


def resolve_source_path(logical_root, mappings) -> ResolvedSourcePath:
    logical = Path(logical_root)
    return ResolvedSourcePath(logical, access_path_for(logical_root, mappings))


def windows_drive_mapping(drive: str) -> str | None:
    """Resolve a mapped remote drive with WNetGetConnectionW; never creates a mapping."""
    if os.name != "nt":
        return None
    drive = drive_of(drive)
    if not drive:
        return None
    if ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") != 4:  # DRIVE_REMOTE
        return None
    size = ctypes.c_ulong(32768)
    buffer = ctypes.create_unicode_buffer(size.value)
    result = ctypes.windll.mpr.WNetGetConnectionW(drive, buffer, ctypes.byref(size))
    return buffer.value if result == 0 else None


def background_preflight(source_paths, resolver=windows_drive_mapping):
    """Return compatibility results and mappings, blocking unresolved remote drives."""
    results, mappings = [], {}
    for source in source_paths:
        value = os.fspath(source)
        drive = drive_of(value)
        if value.startswith(("\\\\", "//")) or not drive:
            results.append({"path": value, "compatible": True, "access_path": value})
            continue
        mapping = resolver(drive)
        if mapping:
            mappings[drive] = mapping
            results.append({"path": value, "compatible": True,
                            "access_path": os.fspath(access_path_for(value, mappings))})
        else:
            # A non-remote/local drive is compatible. On Windows GetDriveType disambiguates it.
            remote = os.name == "nt" and ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") == 4
            results.append({"path": value, "compatible": not remote,
                            "error": "remote drive mapping could not be resolved" if remote else None})
    return {"compatible": all(item["compatible"] for item in results), "sources": results, "mappings": mappings}
