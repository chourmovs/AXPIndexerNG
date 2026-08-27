import struct

def serialize(vector): return struct.pack(f'{len(vector)}f', *map(float,vector))
def upsert(con, chunk_id, vector):
    con.execute('DELETE FROM chunk_vectors WHERE rowid=?',(chunk_id,)); con.execute('INSERT INTO chunk_vectors(rowid,embedding) VALUES (?,?)',(chunk_id,serialize(vector)))
def search(con, vector, limit=20):
    rows=con.execute('''SELECT v.rowid chunk_id,v.distance,c.document_id,c.chunk_no,d.path,c.text snippet
 FROM chunk_vectors v JOIN chunks c ON c.id=v.rowid JOIN documents d ON d.id=c.document_id
 WHERE embedding MATCH ? AND k=? ORDER BY distance''',(serialize(vector),limit)).fetchall()
    return [dict(r) for r in rows]
