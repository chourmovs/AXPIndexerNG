from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from axp_core.background import access_path_for
from axp_core.rich_document import (DOCX_RICH_EXTRACTOR_VERSION, RICH_DOCUMENT_SCHEMA_VERSION,
                                    RichDocument, RichDocumentError, extract_docx)
from axp_core.runtime import data_dir, load_settings

LOGGER = logging.getLogger(__name__)
ASSET_ID = re.compile(r"img-[0-9a-f]{32}\Z")
MAX_SHA_LOCKS = 256


class RichDocumentService:
    """Resolve indexed identities and lazily reconstruct immutable DOCX source content."""
    def __init__(self, db, *, cache_root=None, path_mapper=None):
        self.db = Path(db)
        self.cache_root = Path(cache_root) if cache_root is not None else data_dir() / "rich-cache"
        self.path_mapper = path_mapper
        self._locks_guard = threading.Lock()
        self._locks = OrderedDict()

    def _lock_for(self, sha):
        with self._locks_guard:
            lock = self._locks.setdefault(sha, threading.Lock())
            self._locks.move_to_end(sha)
            while len(self._locks) > MAX_SHA_LOCKS:
                key, candidate = next(iter(self._locks.items()))
                if candidate.locked():
                    self._locks.move_to_end(key); break
                self._locks.popitem(last=False)
            return lock

    def _row(self, document_id):
        try:
            uri = f"file:{self.db.resolve()}?mode=ro"
            con = sqlite3.connect(uri, uri=True)
            con.row_factory = sqlite3.Row
            try: row = con.execute("SELECT id,path,filename,extension,size_bytes,modified_unix_ms,sha256 FROM documents WHERE id=?", (document_id,)).fetchone()
            finally: con.close()
        except sqlite3.Error as exc:
            raise RichDocumentError("rich_document_unavailable") from exc
        if row is None: raise RichDocumentError("rich_document_not_found")
        return dict(row)

    def _source(self, row):
        logical = Path(row["path"])
        if self.path_mapper:
            mapped = Path(self.path_mapper(row["path"]))
        else:
            mapped = Path(access_path_for(row["path"], load_settings().get("background_drive_mappings", {})))
        return next((item for item in (logical, mapped) if item.is_file()), None)

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
        return digest.hexdigest()

    def _validate_source(self, source, row):
        stat = source.stat()
        same_metadata = stat.st_size == row["size_bytes"] and stat.st_mtime_ns // 1_000_000 == row["modified_unix_ms"]
        if not same_metadata and self._hash(source) != row["sha256"]:
            raise RichDocumentError("document_out_of_sync")

    def _read_cache(self, row):
        directory = self.cache_root / row["sha256"]
        try:
            value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if value.get("schema_version") != RICH_DOCUMENT_SCHEMA_VERSION or value.get("extractor_version") != DOCX_RICH_EXTRACTOR_VERSION:
                return None
            if value.get("document", {}).get("sha256") != row["sha256"]: return None
            for asset in value.get("assets", []):
                if not ASSET_ID.fullmatch(asset["id"]): return None
                target = directory / "assets" / asset["filename"]
                if not target.is_file() or target.stat().st_size != asset["byte_size"]: return None
            rich = RichDocument.from_manifest(value)
            diagnostics = {**dict(rich.diagnostics), "cache_hit": True}
            return replace(rich, document_id=row["id"], filename=row["filename"], diagnostics=diagnostics)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_cache(self, rich, blobs):
        self.cache_root.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_root / (".tmp-" + uuid.uuid4().hex)
        target = self.cache_root / rich.sha256
        backup = self.cache_root / (".old-" + uuid.uuid4().hex)
        try:
            assets = temporary / "assets"; assets.mkdir(parents=True)
            for asset in rich.assets:
                path = assets / asset.filename
                with path.open("wb") as handle:
                    handle.write(blobs[asset.id]); handle.flush(); os.fsync(handle.fileno())
            manifest = temporary / "manifest.json"
            with manifest.open("w", encoding="utf-8") as handle:
                json.dump(rich.to_manifest(), handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush(); os.fsync(handle.fileno())
            if target.exists(): os.replace(target, backup)
            os.replace(temporary, target)
            shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            if backup.exists() and not target.exists(): os.replace(backup, target)
            raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def get_document(self, document_id):
        started = time.perf_counter(); row = self._row(int(document_id))
        if str(row["extension"]).lower() != ".docx": raise RichDocumentError("rich_document_unsupported")
        lock = self._lock_for(row["sha256"])
        with lock:
            validation_started = time.perf_counter(); source = self._source(row)
            cached = self._read_cache(row)
            if source is None:
                if cached is not None: return self._logged(cached, started, 0, 0)
                raise RichDocumentError("rich_document_unavailable")
            try: self._validate_source(source, row)
            except OSError as exc:
                if cached is not None: return self._logged(cached, started, 0, 0)
                raise RichDocumentError("rich_document_unavailable") from exc
            validation_ms = (time.perf_counter() - validation_started) * 1000
            if cached is not None: return self._logged(cached, started, validation_ms, 0)
            extract_started = time.perf_counter()
            rich, blobs = extract_docx(source, document_id=row["id"], filename=row["filename"], sha256=row["sha256"])
            extract_ms = (time.perf_counter() - extract_started) * 1000
            write_started = time.perf_counter(); self._write_cache(rich, blobs)
            write_ms = (time.perf_counter() - write_started) * 1000
            rich = replace(rich, diagnostics={**dict(rich.diagnostics), "cache_hit": False})
            return self._logged(rich, started, validation_ms, extract_ms, write_ms)

    def _logged(self, rich, started, validation_ms, extract_ms, write_ms=0):
        elapsed = (time.perf_counter() - started) * 1000
        telemetry = {"rich_document_total_ms": round(elapsed, 3),
                     "rich_document_source_validation_ms": round(validation_ms, 3),
                     "rich_document_extract_ms": round(extract_ms, 3),
                     "rich_document_cache_write_ms": round(write_ms, 3)}
        rich = replace(rich, diagnostics={**dict(rich.diagnostics), **telemetry})
        LOGGER.info("Rich document request document_id=%s extension=.docx cache_hit=%s fidelity=%s blocks=%s tables=%s images=%s unsupported=%s duplicates_suppressed=%s elapsed_ms=%.1f",
                    rich.document_id, rich.diagnostics.get("cache_hit"), rich.fidelity, len(rich.blocks),
                    rich.diagnostics.get("tables"), rich.diagnostics.get("images"), rich.diagnostics.get("unsupported_objects"),
                    rich.diagnostics.get("duplicates_suppressed", 0), elapsed)
        return rich

    def get_asset(self, document_id, asset_id):
        if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id):
            raise RichDocumentError("rich_asset_invalid")
        rich = self.get_document(document_id)
        asset = next((item for item in rich.assets if item.id == asset_id), None)
        if asset is None: raise RichDocumentError("rich_asset_not_found")
        path = self.cache_root / rich.sha256 / "assets" / asset.filename
        if not path.is_file() or path.stat().st_size != asset.byte_size:
            raise RichDocumentError("rich_document_unavailable")
        return asset, path
