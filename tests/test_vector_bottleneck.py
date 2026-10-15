"""Exact KNN/hydration parity, transactional behavior and request-local timings."""
import pytest

from axp_core.database import connect
from axp_core.sources import add_source
from axp_core import vectors


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / 'vectors.db'
    con = connect(path, dimension=3)
    source = add_source(con, tmp_path / 'source')['id']
    con.execute("UPDATE sources SET enabled=0,status='disabled' WHERE id=?", (source,))
    for i, vector in enumerate(([1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]), 1):
        con.execute("INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,"
                    "sha256,indexed_unix_ms,title) VALUES(?,?,?,? ,'.txt',1,1,'x',1,?)",
                    (i, source, f'/doc/{i}', f'/doc/{i}', f'Title {i}'))
        con.execute("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(?,?,0,?)", (i, i, f'body {i}'))
        vectors.upsert(con, i, vector)
    con.commit()
    yield path, con
    con.close()


@pytest.mark.parametrize('limit', [0, -1, 1, 2, 3, 100, 501])
def test_split_preserves_complete_rows_native_ties_and_bounds(catalog, limit):
    _, con = catalog
    # Disabled sources are historically included in the global search.
    assert vectors.search_split(con, [1, 0, 0], limit) == vectors.search(con, [1, 0, 0], limit)


def test_native_top_k_precedes_inner_join_and_does_not_refill(catalog):
    _, con = catalog
    # Old/corrupt indexes can have orphan vectors. Their distance still consumes
    # one native KNN slot; removing them before KNN would change the semantics.
    vectors.upsert(con, 999, [1, 0, 0])
    con.commit()
    for k in (1, 2, 3, 4, 5):
        assert vectors.search_split(con, [1, 0, 0], k) == vectors.search(con, [1, 0, 0], k)
    con.execute('DELETE FROM documents WHERE id=1')
    con.execute("UPDATE documents SET title='modified' WHERE id=2")
    con.commit()
    assert vectors.search_split(con, [1, 0, 0]) == vectors.search(con, [1, 0, 0])


def test_enrichment_is_one_query_and_diagnostics_are_opt_in(catalog):
    _, con = catalog
    sql = []
    con.set_trace_callback(sql.append)
    first = {}
    vectors.search_split(con, [1, 0, 0], timings=first)
    assert len([s for s in sql if s.startswith('SELECT')]) == 2
    assert len([s for s in sql if 'WHERE c.id IN' in s]) == 1
    assert not any('EXPLAIN' in s or 'count(' in s for s in sql)
    second = {}
    vectors.search(con, [0, 1, 0], timings=second)
    assert set(second) == {'vector_combined_execute_ms', 'vector_combined_fetchall_ms'}
    assert set(first) == {'vector_knn_execute_ms', 'vector_knn_fetchall_ms',
                          'vector_enrichment_execute_ms', 'vector_enrichment_fetchall_ms'}
    report = vectors.developer_diagnostics(con, [1, 0, 0], 2)
    assert report['dimension'] == 3 and report['useful_vector_bytes'] == 4 * 3 * 4
    assert report['vectors_visited'] is None
    assert all(value['plan'] for value in report['queries'].values())


def test_split_uses_same_snapshot_during_concurrent_delete_and_update(catalog, monkeypatch):
    path, reader = catalog
    writer = connect(path)
    original = vectors.knn
    expected = vectors.search(reader, [1, 0, 0])

    def commit_between_stages(*args, **kwargs):
        result = original(*args, **kwargs)
        writer.execute('DELETE FROM documents WHERE id=1')
        writer.execute("UPDATE documents SET title='new version' WHERE id=2")
        writer.commit()
        return result

    monkeypatch.setattr(vectors, 'knn', commit_between_stages)
    assert vectors.search_split(reader, [1, 0, 0]) == expected
    assert not reader.in_transaction
    current = vectors.search(reader, [1, 0, 0])
    assert 1 not in [row['document_id'] for row in current]
    assert next(row for row in current if row['document_id'] == 2)['title'] == 'new version'
    writer.close()


def test_split_preserves_caller_transaction_and_rolls_back_on_failure(catalog, monkeypatch):
    _, con = catalog
    con.execute('BEGIN')
    vectors.search_split(con, [1, 0, 0])
    assert con.in_transaction
    def fail(*args, **kwargs):
        raise RuntimeError('hydration failure')
    monkeypatch.setattr(vectors, 'enrich', fail)
    with pytest.raises(RuntimeError):
        vectors.search_split(con, [1, 0, 0])
    assert con.in_transaction
    con.rollback()
