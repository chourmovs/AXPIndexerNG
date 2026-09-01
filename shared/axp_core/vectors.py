import math
import struct


def serialize(vector):
    return struct.pack(f"{len(vector)}f", *map(float, vector))


def upsert(con, chunk_id, vector):
    con.execute("DELETE FROM chunk_vectors WHERE rowid=?", (chunk_id,))
    con.execute("INSERT INTO chunk_vectors(rowid,embedding) VALUES (?,?)", (chunk_id, serialize(vector)))


def search(con, vector, limit=20):
    limit = min(max(0, int(limit)), 500)
    if not limit:
        return []
    rows = con.execute(
        """SELECT v.rowid chunk_id,v.distance vector_distance,c.document_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,d.source_id,s.label source_label,s.path source_path,
 c.text snippet,c.identifiers
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id
 WHERE embedding MATCH ? AND k=? ORDER BY distance""",
        (serialize(vector), limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_documents(con, vector, document_ids):
    """Scan existing vectors for selected documents in one bounded SQL query."""
    ids = sorted({int(value) for value in document_ids})
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    rows = con.execute(
        f"""SELECT v.rowid chunk_id,v.embedding,c.document_id,c.chunk_no,c.page_no,
 c.section_heading heading,d.path,d.filename,d.title,d.ingestion_mode,d.source_id,
 s.label source_label,s.path source_path,c.text snippet,c.identifiers
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 JOIN sources s ON s.id=d.source_id WHERE c.document_id IN ({placeholders})""", ids).fetchall()
    query = [float(value) for value in vector]
    query_norm = math.sqrt(sum(value * value for value in query))
    result = []
    for raw in rows:
        item = dict(raw)
        blob = item.pop("embedding")
        values = struct.unpack(f"{len(blob) // 4}f", blob)
        norm = math.sqrt(sum(value * value for value in values))
        similarity = (sum(a * b for a, b in zip(query, values)) / (query_norm * norm)
                      if query_norm and norm else 0.0)
        item["vector_distance"] = 1.0 - max(-1.0, min(1.0, similarity))
        result.append(item)
    return sorted(result, key=lambda row: (row["vector_distance"], row["chunk_id"]))
