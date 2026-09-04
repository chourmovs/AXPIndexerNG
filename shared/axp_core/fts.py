import re

from .identifiers import extract_identifiers

# FTS declaration order: body, title, filename, heading, identifiers.
BM25_WEIGHTS = (1.0, 4.0, 3.0, 4.0, 8.0)
TOKEN_RE = re.compile(r"[^\W_]+(?:[-._/][^\W_]+)*", re.UNICODE)


def build_query(query):
    """Convert user text into quoted FTS atoms; no user MATCH syntax survives."""
    tokens = [m.group(0) for m in TOKEN_RE.finditer(query or "")]
    normalized_ids = {normalized for normalized, _ in extract_identifiers(query)}
    terms = []
    for token in tokens:
        safe = token.replace('"', '""')
        terms.append(f'"{safe}"')
        compact = re.sub(r"[-._/]", "", token).upper()
        if compact in normalized_ids and compact != token.upper():
            terms.append(f'"{compact}"')
    return " OR ".join(dict.fromkeys(terms))


def search(con, query, limit=20):
    match = build_query(query)
    if not match or limit <= 0:
        return []
    limit = min(int(limit), 500)
    rows = con.execute(
        """WITH ranked AS (
 SELECT rowid chunk_id,bm25(chunks_fts,?,?,?,?,?) bm25_score
 FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25_score,rowid LIMIT ?
 )
 SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,c.section_heading heading,
 d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,c.text snippet,c.identifiers,
 ranked.bm25_score
 FROM ranked JOIN chunks c ON c.id=ranked.chunk_id JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 ORDER BY ranked.bm25_score,ranked.chunk_id""",
        (*BM25_WEIGHTS, match, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_scoped(con, query, *, source_ids=None, path_prefixes=None, extensions=None,
                  modified_after_ms=None, modified_before_ms=None, limit=20):
    """Run FTS with document restrictions applied before the ranked LIMIT."""
    match = build_query(query)
    if not match or limit <= 0:
        return []
    clauses, values = ["chunks_fts MATCH ?"], [match]
    for column, supplied in (("d.source_id", source_ids), ("lower(d.extension)", extensions)):
        items = tuple(supplied or ())
        if items:
            clauses.append(f"{column} IN ({','.join('?' for _ in items)})")
            values.extend(item.casefold() if isinstance(item, str) else int(item) for item in items)
    prefixes = tuple(path_prefixes or ())
    if prefixes:
        clauses.append("(" + " OR ".join("lower(d.path_key) LIKE ? ESCAPE '\\'" for _ in prefixes) + ")")
        values.extend(_prefix_pattern(value) for value in prefixes)
    if modified_after_ms is not None:
        clauses.append("d.modified_unix_ms>=?"); values.append(int(modified_after_ms))
    if modified_before_ms is not None:
        clauses.append("d.modified_unix_ms<?"); values.append(int(modified_before_ms))
    rows = con.execute(
        f"""WITH ranked AS (
 SELECT chunks_fts.rowid chunk_id,bm25(chunks_fts,?,?,?,?,?) bm25_score
 FROM chunks_fts JOIN chunks c0 ON c0.id=chunks_fts.rowid
 JOIN documents d ON d.id=c0.document_id WHERE {' AND '.join(clauses)}
 ORDER BY bm25_score,chunks_fts.rowid LIMIT ?)
 SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,c.section_heading heading,
 d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,
 c.text snippet,c.identifiers,ranked.bm25_score
 FROM ranked JOIN chunks c ON c.id=ranked.chunk_id JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id ORDER BY ranked.bm25_score,ranked.chunk_id""",
        (*BM25_WEIGHTS, *values, min(int(limit), 2000)),
    ).fetchall()
    return [dict(row) for row in rows]


def _prefix_pattern(value):
    normalized = str(value).replace("\\", "/").casefold().rstrip("/")
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def search_documents(con, query, document_ids):
    """Return every lexical match in the selected documents (no global cap)."""
    ids = sorted({int(value) for value in document_ids})
    match = build_query(query)
    if not ids or not match:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"""SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,c.section_heading heading,
 d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,c.text snippet,c.identifiers,
 bm25(chunks_fts,?,?,?,?,?) bm25_score
 FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 WHERE chunks_fts MATCH ? AND c.document_id IN ({placeholders}) ORDER BY bm25_score,c.id""",
        (*BM25_WEIGHTS, match, *ids),
    ).fetchall()
    return [dict(row) for row in rows]
