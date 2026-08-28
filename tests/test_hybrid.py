import pytest
from axp_core.database import connect
from axp_core.hybrid import SearchConfig, search
from axp_core.sources import add_source
from axp_core.vectors import upsert


def _add(c, source_id, key, text, vector):
    document_id = c.execute(
        "insert into documents(source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms) "
        "values(?,?,?,'.txt',1,1,'x',1)",
        (source_id, key, key),
    ).lastrowid
    chunk_id = c.execute("insert into chunks(document_id,chunk_no,text) values(?,0,?)", (document_id, text)).lastrowid
    c.execute(
        "insert into chunks_fts(rowid,text,title,filename,heading,identifiers) values(?,?,?,?,?,?)",
        (chunk_id, text, key, key, "", ""),
    )
    upsert(c, chunk_id, vector)
    return chunk_id


def test_hybrid(tmp_path):
    c = connect(tmp_path / "x.db", dimension=3)
    source_id = add_source(c, tmp_path / "root")["id"]
    _add(c, source_id, "reactor.txt", "reactor pressure", [1, 0, 0])
    c.commit()
    assert search(c, "reactor", [1, 0, 0])[0]["lexical_rank"] == 1


def test_irrelevant_vector_tail_is_filtered_but_lexical_topic_survives(tmp_path):
    c = connect(tmp_path / "relevance.db", dimension=3)
    source_id = add_source(c, tmp_path / "root")["id"]
    relevant = _add(c, source_id, "reactor.txt", "reactor pressure control", [1, 0, 0])
    lexical = _add(c, source_id, "bleach.txt", "bleach dosing procedure", [-1, 0, 0])
    _add(c, source_id, "canteen.txt", "weekly canteen menu", [0, 1, 0])
    c.commit()

    rows = search(c, "reactor pressure", [1, 0, 0])
    assert [row["chunk_id"] for row in rows] == [relevant]
    assert rows[0]["vector_similarity"] == pytest.approx(1.0)
    assert {"relevance_score", "vector_similarity", "lexical_coverage"} <= rows[0].keys()

    rows = search(c, "bleach dosing", [1, 0, 0], config=SearchConfig(min_vector_similarity=0.9))
    assert lexical in {row["chunk_id"] for row in rows}
