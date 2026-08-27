import time
from pathlib import Path

from axp_core.identifiers import extract_identifiers
from axp_core.vectors import upsert

from .chunker import chunk_text
from .extractors import extract
from .scanner import discover, path_key, sha256


def embedding_input(chunk, title, filename):
    context = [f"Document: {title or filename}"]
    if chunk.section_heading:
        context.append(f"Section: {chunk.section_heading}")
    return "\n".join(context) + "\n\n" + chunk.text


def scan(con, root, embedder):
    began = time.perf_counter()
    root = str(Path(root).resolve())
    seen = set()
    result = {k: 0 for k in ("new", "modified", "unchanged", "deleted", "failed")}
    result.update(files_scanned=0, files_extracted=0, chunks_generated=0, chunks_embedded=0, db_insert_ms=0.0)
    for path in discover(root):
        result["files_scanned"] += 1
        key = path_key(path)
        seen.add(key)
        st = path.stat()
        mtime = st.st_mtime_ns // 1_000_000
        old = con.execute("SELECT * FROM documents WHERE path_key=?", (key,)).fetchone()
        if old and old["size_bytes"] == st.st_size and old["modified_unix_ms"] == mtime:
            result["unchanged"] += 1
            continue
        digest = sha256(path)
        if old and old["sha256"] == digest:
            con.execute("UPDATE documents SET size_bytes=?,modified_unix_ms=? WHERE id=?", (st.st_size, mtime, old["id"]))
            result["unchanged"] += 1
            continue
        try:
            sections = extract(path)
            result["files_extracted"] += 1
            chunks = [c for text, page in sections for c in chunk_text(text, page)]
            title = path.stem.replace("_", " ").replace("-", " ").strip()
            inputs = [embedding_input(c, title, path.name) for c in chunks]
            embed_started = time.perf_counter()
            vectors = embedder.embed_documents(inputs) if chunks else []
            result["embedding_ms"] = result.get("embedding_ms", 0.0) + (time.perf_counter() - embed_started) * 1000
            result["chunks_generated"] += len(chunks)
            result["chunks_embedded"] += len(vectors)
            now = int(time.time() * 1000)
            db_started = time.perf_counter()
            if old:
                doc_id = old["id"]
                con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (doc_id,))
                con.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                result["modified"] += 1
            else:
                cur = con.execute(
                    "INSERT INTO documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (root, str(path), key, path.suffix.lower(), st.st_size, mtime, digest, now, title, path.name),
                )
                doc_id = cur.lastrowid
                result["new"] += 1
            if old:
                con.execute(
                    "UPDATE documents SET path=?,extension=?,size_bytes=?,modified_unix_ms=?,sha256=?,indexed_unix_ms=?,title=?,filename=? WHERE id=?",
                    (str(path), path.suffix.lower(), st.st_size, mtime, digest, now, title, path.name, doc_id),
                )
            for no, (chunk, vector) in enumerate(zip(chunks, vectors)):
                identifiers = extract_identifiers(chunk.text, path.name)
                normalized = " ".join(x for x, _ in identifiers)
                cur = con.execute(
                    "INSERT INTO chunks(document_id,chunk_no,text,page_no,char_start,char_end,section_heading,identifiers) VALUES(?,?,?,?,?,?,?,?)",
                    (doc_id, no, chunk.text, chunk.page_no, chunk.char_start, chunk.char_end, chunk.section_heading, normalized),
                )
                con.execute(
                    "INSERT OR REPLACE INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
                    (cur.lastrowid, chunk.text, title, path.name, chunk.section_heading, normalized),
                )
                upsert(con, cur.lastrowid, vector)
            con.commit()
            result["db_insert_ms"] += (time.perf_counter() - db_started) * 1000
        except Exception:  # noqa: BLE001 -- one bad document must not abort the source-root scan
            con.rollback()
            result["failed"] += 1
    for row in con.execute("SELECT id,path_key FROM documents WHERE source_root=?", (root,)).fetchall():
        if row["path_key"] not in seen:
            con.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id=?)", (row["id"],))
            con.execute("DELETE FROM documents WHERE id=?", (row["id"],))
            result["deleted"] += 1
    con.commit()
    result["total_indexing_ms"] = (time.perf_counter() - began) * 1000
    seconds = result.get("embedding_ms", 0) / 1000
    result["embedding_throughput_chunks_s"] = result["chunks_embedded"] / seconds if seconds else 0.0
    return result
