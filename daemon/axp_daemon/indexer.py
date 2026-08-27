import time
from pathlib import Path
from axp_core.vectors import upsert
from .chunker import chunk_text
from .extractors import extract
from .scanner import discover,path_key,sha256

def scan(con,root,embedder):
    root=str(Path(root).resolve()); seen=set(); result={k:0 for k in ('new','modified','unchanged','deleted','failed')}
    for path in discover(root):
        key=path_key(path); seen.add(key); st=path.stat(); mtime=st.st_mtime_ns//1_000_000
        old=con.execute('SELECT * FROM documents WHERE path_key=?',(key,)).fetchone()
        if old and old['size_bytes']==st.st_size and old['modified_unix_ms']==mtime: result['unchanged']+=1; continue
        digest=sha256(path)
        if old and old['sha256']==digest:
            con.execute('UPDATE documents SET size_bytes=?,modified_unix_ms=? WHERE id=?',(st.st_size,mtime,old['id'])); result['unchanged']+=1; continue
        try:
            sections=extract(path); chunks=[c for text,page in sections for c in chunk_text(text,page)]
            vectors=embedder.embed_documents([c.text for c in chunks]) if chunks else []
            now=int(time.time()*1000)
            if old: doc_id=old['id']; con.execute('DELETE FROM chunks WHERE document_id=?',(doc_id,)); result['modified']+=1
            else:
                cur=con.execute('INSERT INTO documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) VALUES(?,?,?,?,?,?,?,?)',(root,str(path),key,path.suffix.lower(),st.st_size,mtime,digest,now)); doc_id=cur.lastrowid; result['new']+=1
            if old: con.execute('UPDATE documents SET path=?,extension=?,size_bytes=?,modified_unix_ms=?,sha256=?,indexed_unix_ms=? WHERE id=?',(str(path),path.suffix.lower(),st.st_size,mtime,digest,now,doc_id))
            for no,(chunk,vector) in enumerate(zip(chunks,vectors)):
                cur=con.execute('INSERT INTO chunks(document_id,chunk_no,text,page_no,char_start,char_end) VALUES(?,?,?,?,?,?)',(doc_id,no,chunk.text,chunk.page_no,chunk.char_start,chunk.char_end)); upsert(con,cur.lastrowid,vector)
            con.commit()
        except Exception: con.rollback(); result['failed']+=1
    for row in con.execute('SELECT id,path_key FROM documents WHERE source_root=?',(root,)).fetchall():
        if row['path_key'] not in seen: con.execute('DELETE FROM documents WHERE id=?',(row['id'],)); result['deleted']+=1
    con.commit(); return result
