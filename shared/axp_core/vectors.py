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
 c.section_heading heading,d.path,d.filename,d.title,c.text snippet,c.identifiers
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 WHERE embedding MATCH ? AND k=? ORDER BY distance""",
        (serialize(vector), limit),
    ).fetchall()
    return [dict(r) for r in rows]
