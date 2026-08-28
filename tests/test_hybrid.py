from axp_core.database import connect
from axp_core.hybrid import search
from axp_core.sources import add_source
from axp_core.vectors import upsert


def test_hybrid(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    source_id = add_source(c, tmp_path / "root")["id"]
    d = c.execute(
        "insert into documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values(?,'p','k','.txt',1,1,'x',1)",
        (source_id,),
    ).lastrowid
    x = c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,'reactor pressure')", (d,)).lastrowid
    upsert(c, x, [1, 0, 0])
    c.commit()
    assert search(c, "reactor", [1, 0, 0])[0]["lexical_rank"] == 1
