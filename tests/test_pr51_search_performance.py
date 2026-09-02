"""PR51 invariants: cheaper hydration and safe reuse, never cheaper retrieval."""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from axp_core.database import SearchReaderPool, configure_search_reader
from axp_core.fts import BM25_WEIGHTS, build_query, search


@pytest.fixture(autouse=True)
def _without_optional_vector_extension(monkeypatch):
    """These FTS/pool tests remain runnable when the optional native wheel is absent."""
    monkeypatch.setattr("axp_core.database.load_vectors", lambda con: "test")


def _fts_database(path, rows=600):
    from axp_core.database import connect
    from axp_core.sources import add_source

    con = connect(path)
    con.execute("DROP TRIGGER chunks_fts_insert")
    con.execute("DROP TRIGGER chunks_fts_delete")
    source_id = add_source(con, path.parent / "fixture")['id']
    for number in range(rows):
        filename = "MSDS MTBE SIMFEX.pdf" if number == 0 else f"document-{number}.pdf"
        document_id = con.execute(
            "INSERT INTO documents(source_id,path,path_key,filename,title,extension,size_bytes,"
            "modified_unix_ms,sha256,indexed_unix_ms) VALUES(?,?,?,?,?,'.pdf',1,1,?,1)",
            (source_id, filename, filename.casefold(), filename, f"Heptane étude {number}", str(number)),
        ).lastrowid
        text = ("density of MTBE 0.74 g/cm³" if number == 0 else
                f"n-Heptane ammoniaque SOP-{number:04d} density reference")
        chunk_id = con.execute(
            "INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,0,?)", (document_id, text)
        ).lastrowid
        con.execute(
            "INSERT INTO chunks_fts(rowid,text,title,filename,heading,identifiers) VALUES(?,?,?,?,?,?)",
            (chunk_id, text, f"Heptane étude {number}", filename, "density", f"SOP{number:04d}"),
        )
    con.commit()
    con.close()


def _legacy_search(con, query, limit):
    match = build_query(query)
    return con.execute(
        """SELECT c.id,bm25(chunks_fts,?,?,?,?,?) score FROM chunks_fts
        JOIN chunks c ON c.id=chunks_fts.rowid JOIN documents d ON d.id=c.document_id
        JOIN sources s ON s.id=d.source_id WHERE chunks_fts MATCH ?
        ORDER BY score,c.id LIMIT ?""",
        (*BM25_WEIGHTS, match, limit),
    ).fetchall()


@pytest.mark.parametrize("query", [
    "Heptane", "density OR MTBE", "SOP-0042", "MSDS MTBE SIMFEX", "étude",
])
@pytest.mark.parametrize("limit", [20, 100, 500])
def test_ranked_fts_hydration_has_exact_legacy_parity(tmp_path, query, limit):
    path = tmp_path / "parity.db"
    _fts_database(path)
    pool = SearchReaderPool(path)
    with pool.acquire() as (con, _):
        legacy = _legacy_search(con, query, limit)
        optimized = search(con, query, limit)
    pool.close()
    assert [row[0] for row in legacy] == [row["chunk_id"] for row in optimized]
    assert [row[1] for row in legacy] == pytest.approx([row["bm25_score"] for row in optimized], abs=1e-12)


def test_reader_is_query_only_and_reused(tmp_path):
    path = tmp_path / "pool.db"
    _fts_database(path, 2)
    pool = SearchReaderPool(path)
    with pool.acquire() as (first, reused):
        assert reused is False
        identity = id(first)
        assert first.execute("PRAGMA query_only").fetchone()[0] == 1
        assert first.execute("PRAGMA cache_size").fetchone()[0] == -65536
        assert first.execute("PRAGMA temp_store").fetchone()[0] == 2
        with pytest.raises(sqlite3.OperationalError):
            first.execute("DELETE FROM chunks")
    with pool.acquire() as (second, reused):
        assert reused is True and id(second) == identity
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        with pool.acquire():
            pass


def test_pool_is_bounded_and_never_lends_a_reader_twice(tmp_path):
    path = tmp_path / "bounded.db"
    _fts_database(path, 2)
    pool = SearchReaderPool(path, size=2)
    active = set()
    peak = 0
    lock = threading.Lock()

    def borrow():
        nonlocal peak
        with pool.acquire() as (con, _):
            with lock:
                assert id(con) not in active
                active.add(id(con))
                peak = max(peak, len(active))
            time.sleep(0.02)
            with lock:
                active.remove(id(con))

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda _: borrow(), range(6)))
    pool.close()
    assert peak == 2


def test_reused_reader_observes_next_wal_commit(tmp_path):
    from axp_core.database import connect

    path = tmp_path / "wal.db"
    _fts_database(path, 1)
    pool = SearchReaderPool(path, size=1)
    with pool.acquire() as (reader, _):
        assert reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    writer = connect(path)
    document_id = writer.execute("SELECT id FROM documents LIMIT 1").fetchone()[0]
    writer.execute("INSERT INTO chunks(document_id,chunk_no,text) VALUES(?,1,'new commit')", (document_id,))
    writer.commit()
    writer.close()
    with pool.acquire() as (reader, reused):
        assert reused and reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 2
    pool.close()


def test_mmap_failure_is_optional():
    class MmapUnavailable:
        def execute(self, sql):
            if "mmap_size=" in sql:
                raise sqlite3.OperationalError("unsupported")
            if sql.startswith("SELECT count"):
                raise sqlite3.OperationalError("fixture has no tables")
            return self

        def fetchone(self):
            return (0,)

    diagnostics = configure_search_reader(MmapUnavailable())
    assert diagnostics["sqlite_mmap_size"] == 0
