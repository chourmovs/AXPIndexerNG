from axp_core.database import connect
from axp_core.vectors import search,upsert
def test_vector(tmp_path):
 c=connect(tmp_path/'x.db',dimension=3);d=c.execute("insert into documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values('r','p','k','.txt',1,1,'x',1)").lastrowid;x=c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,'x')",(d,)).lastrowid;upsert(c,x,[1,0,0]);c.commit();assert search(c,[1,0,0])[0]['chunk_id']==x;c.execute('delete from documents');c.commit();assert not search(c,[1,0,0])
