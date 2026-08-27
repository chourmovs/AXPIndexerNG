def search(con, query, limit=20):
    rows=con.execute('''SELECT c.id chunk_id,c.document_id,c.chunk_no,d.path,
 snippet(chunks_fts,0,'<mark>','</mark>',' … ',24) snippet,bm25(chunks_fts) score
 FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
 WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?''',(query,limit)).fetchall()
    return [dict(r) for r in rows]
