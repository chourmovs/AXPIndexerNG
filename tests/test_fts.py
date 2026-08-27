from axp_core.database import connect
from axp_core.fts import search


def test_fts(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    d = c.execute(
        "insert into documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values('r','p','k','.txt',1,1,'x',1)"
    ).lastrowid
    c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,'equipment EQ-42 batch B19')", (d,))
    c.commit()
    assert search(c, "EQ")
