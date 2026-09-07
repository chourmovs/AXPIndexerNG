"""Canonical keys and safe SQL patterns for indexed document paths."""
from __future__ import annotations

import ntpath
import os
import re

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def canonical_path_key(path: str | os.PathLike[str]) -> str:
    """Return the platform path key used by ``documents.path_key``.

    Windows absolute and UNC paths are deliberately handled with ``ntpath`` even
    when diagnostics or tests run on another platform.  Native paths retain the
    scanner's historical ``normcase(abspath(...)).casefold()`` contract.
    """
    value = os.fspath(path)
    if _WINDOWS_ABSOLUTE.match(value) or value.startswith(("\\\\", "//")):
        return ntpath.normcase(ntpath.normpath(value)).casefold()
    return os.path.normcase(os.path.abspath(value)).casefold()


def escape_like(value: str) -> str:
    """Escape a literal for SQLite LIKE using backslash as ESCAPE."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def sql_path_prefix(path: str | os.PathLike[str]) -> str:
    """Return a parameter value matching an indexed path prefix literally."""
    return escape_like(canonical_path_key(path).rstrip("\\/")) + "%"
