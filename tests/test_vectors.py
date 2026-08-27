import pytest

from axp_core.database import connect
from axp_core.vectors import search, upsert


def test_vector(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    d = c.execute(
        "insert into documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values('r','p','k','.txt',1,1,'x',1)"
    ).lastrowid
    x = c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,'x')", (d,)).lastrowid
    upsert(c, x, [1, 0, 0])
    c.commit()
    assert search(c, [1, 0, 0])[0]["chunk_id"] == x
    c.execute("delete from documents")
    c.commit()
    assert not search(c, [1, 0, 0])


def test_cosine_distance_ordering(tmp_path):
    c = connect(tmp_path / "cosine.db", dimension=3)
    d = c.execute(
        "insert into documents(source_root,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) values('r','p','k','.txt',1,1,'x',1)"
    ).lastrowid
    chunk_ids = []
    for chunk_no, (text, vector) in enumerate(
        (("identical", [1, 0, 0]), ("closer", [1, 1, 0]), ("unrelated", [0, 1, 0]))
    ):
        chunk_id = c.execute(
            "insert into chunks(document_id,chunk_no,text) values(?,?,?)", (d, chunk_no, text)
        ).lastrowid
        upsert(c, chunk_id, vector)
        chunk_ids.append(chunk_id)
    c.commit()

    rows = search(c, [1, 0, 0])

    assert [row["chunk_id"] for row in rows] == chunk_ids
    assert rows[0]["vector_distance"] == pytest.approx(0.0, abs=1e-6)
    assert [row["vector_distance"] for row in rows] == sorted(row["vector_distance"] for row in rows)
