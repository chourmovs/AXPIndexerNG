"""Defensive reader for immutable portable-build identity."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .runtime import installation_root

_VERSION = re.compile(r"(?:v[0-9][0-9A-Za-z.+-]*|dev-[0-9a-f]{7,40}|dev)")
_COMMIT = re.compile(r"[0-9a-fA-F]{7,40}")


def _validated(value):
    if not isinstance(value, dict):
        return None
    version, commit, release = value.get("version"), value.get("commit"), value.get("release")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        return None
    if commit not in (None, "") and (not isinstance(commit, str) or not _COMMIT.fullmatch(commit)):
        return None
    if type(release) is not bool:
        return None
    return {"version": version, "commit": commit[:7].lower() if commit else None, "release": release}


def build_info(path=None):
    """Return safe build metadata, falling back to a source-checkout identity."""
    source = path or os.getenv("AXP_BUILD_INFO") or installation_root() / "BUILD_INFO.json"
    try:
        source = Path(source)
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"version": "dev", "commit": None, "release": False}
    return _validated(value) or {"version": "dev", "commit": None, "release": False}
