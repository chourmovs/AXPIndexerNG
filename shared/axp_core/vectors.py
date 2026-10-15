from .tracing import span, traced, sql_rows
import struct


@traced("Sérialisation du vecteur de requête")
def serialize(vector):
    return struct.pack(f"{len(vector)}f", *map(float, vector))


def upsert(con, chunk_id, vector):
    con.execute("DELETE FROM chunk_vectors WHERE rowid=?", (chunk_id,))
    con.execute("INSERT INTO chunk_vectors(rowid,embedding) VALUES (?,?)", (chunk_id, serialize(vector)))


@traced('Vecteurs : recherche native globale')
def search(con, vector, limit=20):
    limit = min(max(0, int(limit)), 500)
    if not limit:
        return []
    rows = sql_rows(con, """SELECT v.rowid chunk_id,v.distance vector_distance,c.document_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,
 c.text snippet,c.identifiers
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 WHERE embedding MATCH ? AND k=? ORDER BY distance""", (serialize(vector), limit),
        name="KNN sqlite-vec, top-k et jointures métadonnées")
    with span("Conversion des lignes", "rows_to_dict") as measured:
        measured.count(candidates=len(rows))
        return [dict(r) for r in rows]


@traced('Vecteurs : distance native par document')
def search_documents(con, vector, document_ids):
    """Scan existing vectors for selected documents in one bounded SQL query."""
    ids = sorted({int(value) for value in document_ids})
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = sql_rows(con, f"""SELECT v.rowid chunk_id,coalesce(vec_distance_cosine(v.embedding,?),1.0) vector_distance,
 c.document_id,c.chunk_no,c.page_no,c.section_heading heading,d.path,d.filename,d.title,
 d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,c.text snippet,c.identifiers
 FROM chunks c CROSS JOIN chunk_vectors v ON v.rowid=c.id
 JOIN documents d ON d.id=c.document_id JOIN sources s ON s.id=d.source_id
 WHERE c.document_id IN ({placeholders})""", (serialize(vector), *ids),
        name="Distances sqlite-vec et métadonnées par document")
    # Distance is evaluated by sqlite-vec in native code; no vector blobs cross
    # into Python and no repeated unpack/norm/dot conversion is needed.
    with span("Conversion et classement local", "vectors.local_sort") as measured:
        measured.count(vectors_visited=len(rows), candidates=len(rows))
        return sorted((dict(row) for row in rows), key=lambda row: (row["vector_distance"], row["chunk_id"]))
