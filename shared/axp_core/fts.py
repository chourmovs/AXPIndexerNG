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
        """SELECT c.id chunk_id,c.document_id,c.chunk_no,c.page_no,c.section_heading heading,
 d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,c.text snippet,c.identifiers,
 bm25(chunks_fts,?,?,?,?,?) bm25_score
 FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 WHERE chunks_fts MATCH ? ORDER BY bm25_score, c.id LIMIT ?""",
        (*BM25_WEIGHTS, match, limit),
    ).fetchall()
    return [dict(r) for r in rows]
