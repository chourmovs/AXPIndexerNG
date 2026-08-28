from axp_core.database import connect
from axp_core.fts import search
from axp_core.sources import add_source


def test_fts(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    source_id = add_source(c, tmp_path / "root")["id"]
    d = c.execute(
        "insert into documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values(?,'p','k','.txt',1,1,'x',1)",
        (source_id,),
    ).lastrowid
    c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,'equipment EQ-42 batch B19')", (d,))
    c.commit()
    result = search(c, "EQ")[0]
    assert result["source_id"] == source_id and result["source_label"] == "root"
