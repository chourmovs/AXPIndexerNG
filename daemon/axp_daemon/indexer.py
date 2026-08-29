from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from axp_core.identifiers import extract_identifiers
from axp_core.sources import (
    add_source,
    get_source,
    list_sources,
    mark_source_status,
    normalize_source_path,
    source_stats,
)
from axp_core.vectors import upsert

from .chunker import Chunk, chunk_text
from .extractors import extract
from .scanner import SourceUnavailable, discover, is_ignored_document, is_supported_document, path_key, sha256


@dataclass
class PreparedDocument:
    path: Path
    stat: object
    digest: str
    old: object
    title: str
    chunks: list
    ingestion_mode: str


def embedding_input(chunk, title, filename):
    context = [f"Document: {title or filename}"]
    if chunk.section_heading:
        context.append(f"Section: {chunk.section_heading}")
    return "\n".join(context) + "\n\n" + chunk.text


def _result():
    keys = (
        "files_discovered", "files_scanned", "files_seen", "files_content", "files_metadata", "files_ignored",
        "files_unchanged", "files_hashed", "files_extracted", "files_new",
        "files_modified", "files_deleted", "files_failed", "files_completed", "chunks_generated", "chunks_embedded",
    )
    value = {key: 0 for key in keys}
    value.update(new=0, modified=0, unchanged=0, deleted=0, failed=0, db_insert_ms=0.0, embedding_ms=0.0,
                 extension_breakdown={})
    return value


def _control_call(control, name, *args):
    function = getattr(control, name, None) if control is not None else None
    return function(*args) if function else None


def _record_document_failure(item, exc, result, control):
    result[f"files_{item.ingestion_mode}"] -= 1
    breakdown = result["extension_breakdown"][item.path.suffix.casefold() or "[no extension]"]
    breakdown[item.ingestion_mode] -= 1
    breakdown["failed"] += 1
    result["files_failed"] += 1
    result["files_completed"] += 1
    result["failed"] += 1
    _control_call(control, "file_error", item.path, exc)
    _control_call(control, "progress", result)


def _embed_group(items, embedder, result, control):
    """Embed a group, recursively isolating a bad document or an oversized backend batch."""
    if not items:
        return []
    inputs = [embedding_input(chunk, item.title, item.path.name) for item in items for chunk in item.chunks]
    if not inputs:
        return [(item, []) for item in items]
    _control_call(control, "stage", "embedding")
    started = time.perf_counter()
    try:
        vectors = list(embedder.embed_documents(inputs))
        if len(vectors) != len(inputs):
            raise RuntimeError(f"Embedding backend returned {len(vectors)} vectors for {len(inputs)} chunks")
    except Exception as exc:  # noqa: BLE001 -- split the batch before sacrificing a document
        result["embedding_ms"] += (time.perf_counter() - started) * 1000
        if len(items) > 1:
            middle = len(items) // 2
            return (_embed_group(items[:middle], embedder, result, control)
                    + _embed_group(items[middle:], embedder, result, control))
        _record_document_failure(items[0], RuntimeError(f"Embedding failed after batch isolation: {exc}"),
                                 result, control)
        return []
    result["embedding_ms"] += (time.perf_counter() - started) * 1000
    result["chunks_embedded"] += len(vectors)
    embedded, offset = [], 0
    for item in items:
        item_vectors = vectors[offset : offset + len(item.chunks)]
        offset += len(item.chunks)
        embedded.append((item, item_vectors))
    return embedded


def _flush(con, source_id, pending, embedder, result, control):
    if not pending:
        return
    embedded = _embed_group(list(pending), embedder, result, control)
    _control_call(control, "stage", "committing")
    db_started = time.perf_counter()
    for item, item_vectors in embedded:
        con.execute("SAVEPOINT document_write")
        try:
            now = int(time.time() * 1000)
            if item.old:
                doc_id = item.old["id"]
                con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (doc_id,))
                con.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                con.execute(
                    """UPDATE documents SET source_id=?,path=?,extension=?,size_bytes=?,modified_unix_ms=?,sha256=?,
                    indexed_unix_ms=?,title=?,filename=?,ingestion_mode=? WHERE id=?""",
                    (source_id, str(item.path), item.path.suffix.lower(), item.stat.st_size,
                     item.stat.st_mtime_ns // 1_000_000, item.digest, now, item.title, item.path.name,
                     item.ingestion_mode, doc_id),
                )
                result["files_modified"] += 1
                result["modified"] += 1
            else:
                cur = con.execute(
                    """INSERT INTO documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,
                    indexed_unix_ms,title,filename,ingestion_mode) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (source_id, str(item.path), path_key(item.path), item.path.suffix.lower(), item.stat.st_size,
                     item.stat.st_mtime_ns // 1_000_000, item.digest, now, item.title, item.path.name,
                     item.ingestion_mode),
                )
                doc_id = cur.lastrowid
                result["files_new"] += 1
                result["new"] += 1
            for no, (chunk, vector) in enumerate(zip(item.chunks, item_vectors)):
                identifiers = extract_identifiers(chunk.text, item.path.name)
                normalized = " ".join(value for value, _ in identifiers)
                cur = con.execute(
                    """INSERT INTO chunks(document_id,chunk_no,text,page_no,char_start,char_end,section_heading,identifiers)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (doc_id, no, chunk.text, chunk.page_no, chunk.char_start, chunk.char_end,
                     chunk.section_heading, normalized),
                )
                con.execute(
                    "INSERT OR REPLACE INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
                    (cur.lastrowid, chunk.text, item.title, item.path.name, chunk.section_heading, normalized),
                )
                upsert(con, cur.lastrowid, vector)
            con.execute("RELEASE document_write")
            result["files_completed"] += 1
        except Exception as exc:  # noqa: BLE001 -- isolate a failed document transaction
            con.execute("ROLLBACK TO document_write")
            con.execute("RELEASE document_write")
            _record_document_failure(item, RuntimeError(f"Database write failed: {exc}"), result, control)
    con.commit()
    result["db_insert_ms"] += (time.perf_counter() - db_started) * 1000
    _control_call(control, "progress", result)
    _control_call(control, "batch_committed", con, result)
    pending.clear()


def scan_source(con, source_id, embedder, *, embedding_batch_size=64, control=None):
    began = time.perf_counter()
    source = get_source(con, source_id)
    result = _result()
    started_ms = int(time.time() * 1000)
    mark_source_status(con, source_id, "scanning", last_scan_started_ms=started_ms)
    _control_call(control, "source_started", source, started_ms)
    seen, pending, pending_chunks = set(), [], 0
    disabled_during_scan = False
    traversal = discover(source["path"], recursive=bool(source["recursive"]), include_ignored=True)
    _control_call(control, "stage", "discovering")
    try:
        for path in traversal:
            if _control_call(control, "should_stop"):
                break
            _control_call(control, "wait_if_paused")
            if not get_source(con, source_id)["enabled"]:
                disabled_during_scan = True
                break
            result["files_discovered"] += 1
            result["files_seen"] += 1
            extension = path.suffix.casefold() or "[no extension]"
            breakdown = result["extension_breakdown"].setdefault(
                extension, {"seen": 0, "content": 0, "metadata": 0, "ignored": 0, "failed": 0})
            breakdown["seen"] += 1
            if is_ignored_document(path.name):
                result["files_ignored"] += 1
                result["files_completed"] += 1
                breakdown["ignored"] += 1
                _control_call(control, "progress", result)
                continue
            result["files_scanned"] += 1
            _control_call(control, "current_file", source, path, result)
            key = path_key(path)
            seen.add(key)
            mode = "content" if is_supported_document(path.name) else "metadata"
            result[f"files_{mode}"] += 1
            breakdown[mode] += 1
            try:
                _control_call(control, "stage", "checking")
                stat = path.stat()
                mtime = stat.st_mtime_ns // 1_000_000
                old = con.execute("SELECT * FROM documents WHERE path_key=?", (key,)).fetchone()
                if (old and old["size_bytes"] == stat.st_size and old["modified_unix_ms"] == mtime
                        and old["ingestion_mode"] == mode):
                    result["files_unchanged"] += 1
                    result["unchanged"] += 1
                    result["files_completed"] += 1
                    _control_call(control, "progress", result)
                    continue
                digest = ""
                if mode == "content":
                    _control_call(control, "stage", "hashing")
                    result["files_hashed"] += 1
                    digest = sha256(path)
                if mode == "content" and old and old["sha256"] == digest and old["ingestion_mode"] == mode:
                    con.execute(
                        "UPDATE documents SET source_id=?,path=?,size_bytes=?,modified_unix_ms=? WHERE id=?",
                        (source_id, str(path), stat.st_size, mtime, old["id"]),
                    )
                    result["files_unchanged"] += 1
                    result["unchanged"] += 1
                    result["files_completed"] += 1
                    _control_call(control, "progress", result)
                    continue
                if mode == "content":
                    _control_call(control, "stage", "extracting")
                    sections = extract(path)
                    result["files_extracted"] += 1
                    _control_call(control, "stage", "chunking")
                    chunks = [chunk for text, page in sections for chunk in chunk_text(text, page)]
                else:
                    _control_call(control, "stage", "metadata")
                    relative = path.parent.name
                    text = (f"Filename: {path.name}\nExtension: {extension}\nFolder: {relative}\nPath: {path}\n"
                            f"Size: {stat.st_size} bytes")
                    chunks = [Chunk(text, None, 0, len(text), "File metadata")]
                result["chunks_generated"] += len(chunks)
                title = path.stem.replace("_", " ").replace("-", " ").strip()
                pending.append(PreparedDocument(path, stat, digest, old, title, chunks, mode))
                pending_chunks += len(chunks)
                if pending_chunks >= embedding_batch_size:
                    _flush(con, source_id, pending, embedder, result, control)
                    pending_chunks = 0
            except Exception as exc:  # noqa: BLE001 -- isolate a bad document
                result[f"files_{mode}"] -= 1
                breakdown[mode] -= 1
                breakdown["failed"] += 1
                result["files_failed"] += 1
                result["files_completed"] += 1
                result["failed"] += 1
                _control_call(control, "file_error", path, exc)
                _control_call(control, "progress", result)
        _flush(con, source_id, pending, embedder, result, control)
    except (SourceUnavailable, OSError) as exc:
        con.rollback()
        mark_source_status(con, source_id, "offline", error=str(exc), last_scan_completed_ms=int(time.time() * 1000))
        result.update(scan_complete=False, status="offline", last_error=str(exc))
        result["total_indexing_ms"] = (time.perf_counter() - began) * 1000
        return result
    except Exception as exc:  # noqa: BLE001 -- isolate one failed source from the scan cycle
        con.rollback()
        mark_source_status(con, source_id, "error", error=str(exc), last_scan_completed_ms=int(time.time() * 1000))
        result.update(scan_complete=False, status="error", last_error=str(exc))
        result["total_indexing_ms"] = (time.perf_counter() - began) * 1000
        return result

    stopped = bool(_control_call(control, "should_stop"))
    scan_complete = traversal.complete and not stopped and not disabled_during_scan
    if scan_complete:
        _control_call(control, "stage", "reconciling")
        for row in con.execute("SELECT id,path_key FROM documents WHERE source_id=?", (source_id,)).fetchall():
            if row["path_key"] not in seen:
                con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (row["id"],))
                con.execute("DELETE FROM documents WHERE id=?", (row["id"],))
                result["files_deleted"] += 1
                result["deleted"] += 1
        con.commit()
    stats = source_stats(con, source_id)
    completed = int(time.time() * 1000)
    if not scan_complete:
        error = traversal.errors[-1] if traversal.errors else "Scan interrupted before complete enumeration"
        status = "disabled" if disabled_during_scan else ("error" if traversal.errors else "paused")
        mark_source_status(con, source_id, status, error=error, last_scan_completed_ms=completed,
                           last_file_count=stats["documents"], last_chunk_count=stats["chunks"])
    else:
        mark_source_status(con, source_id, "idle", last_scan_completed_ms=completed, last_success_ms=completed,
                           last_file_count=stats["documents"], last_chunk_count=stats["chunks"],
                           last_seen_count=result["files_seen"], last_content_count=result["files_content"],
                           last_metadata_count=result["files_metadata"], last_ignored_count=result["files_ignored"],
                           last_failed_count=result["files_failed"],
                           last_extension_breakdown=json.dumps(result["extension_breakdown"], sort_keys=True))
    result.update(scan_complete=scan_complete, status="idle" if scan_complete else status,
                  enumeration_errors=traversal.errors, documents=stats["documents"], chunks=stats["chunks"])
    result["total_indexing_ms"] = (time.perf_counter() - began) * 1000
    seconds = result["embedding_ms"] / 1000
    result["embedding_throughput_chunks_s"] = result["chunks_embedded"] / seconds if seconds else 0.0
    return result


def scan(con, root, embedder, *, embedding_batch_size=64, control=None):
    display, key = normalize_source_path(root)
    source = con.execute("SELECT * FROM sources WHERE path_key=?", (key,)).fetchone()
    if source is None:
        source = add_source(con, display)
    return scan_source(con, source["id"], embedder, embedding_batch_size=embedding_batch_size, control=control)


def scan_all(con, embedder, *, embedding_batch_size=64, control=None):
    return {
        str(source["id"]): scan_source(con, source["id"], embedder, embedding_batch_size=embedding_batch_size,
                                       control=control)
        for source in list_sources(con, enabled_only=True)
        if not _control_call(control, "should_stop")
    }
