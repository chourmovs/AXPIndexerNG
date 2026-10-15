"""Counting is optional and must never delay retrieval or share request readers."""
import threading

from axp_core import database
from axp_core.sources import add_source
from axp_core.vectors import search, upsert


def catalog(path):
    con = database.connect(path, dimension=3)
    from axp_core.metadata import ensure_index_signature
    ensure_index_signature(con, 'diagnostic-test', 3)
    source = add_source(con, path.parent / 'source')['id']
    con.execute("INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,"
                "sha256,indexed_unix_ms) VALUES(1,?,'p','p','.txt',1,1,'x',1)", (source,))
    con.execute("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(1,1,0,'searchable')")
    upsert(con, 1, [1, 0, 0])
    con.commit()
    return con


def test_no_counts_blocks_reader_open_or_search_and_refresh_is_single_flight(tmp_path, monkeypatch):
    path = tmp_path / 'test.db'
    writer = catalog(path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = database._refresh_diagnostics
    calls = []
    def blocked_worker(path, key):
        calls.append(key)
        entered.set()
        try:
            assert release.wait(5)
            original(path, key)
        finally:
            finished.set()
    monkeypatch.setattr(database, '_refresh_diagnostics', blocked_worker)
    pool = database.SearchReaderPool(path, size=2)
    try:
        with pool.acquire() as (reader, _):
            assert entered.wait(2)
            statements = []
            reader.set_trace_callback(statements.append)
            first = database.search_reader_diagnostics(reader)
            assert first['catalog_snapshot_status'] == 'unavailable'
            assert first['catalog_snapshot_refreshing']
            assert search(reader, [1, 0, 0])[0]['chunk_id'] == 1
            # Neither a second reader nor an explicit concurrent request duplicates it.
            with pool.acquire() as (second, _):
                assert database.search_reader_diagnostics(second, refresh=True)['total_chunks'] is None
            assert len(calls) == 1
            assert not any('count(' in sql.lower() for sql in statements)
        release.set()
        assert finished.wait(5)
        with pool.acquire() as (reader, _):
            sample = database.search_reader_diagnostics(reader)
            assert sample['total_chunks'] == sample['total_vectors'] == 1
            assert sample['catalog_snapshot_age_ms'] >= 0
            assert sample['catalog_snapshot_scope'].startswith('whole catalog')
        with pool.acquire() as (reader, _):
            assert database.search_reader_diagnostics(reader)['catalog_snapshot_sampled_ms'] == sample['catalog_snapshot_sampled_ms']
        writer.execute("UPDATE sources SET last_error='unrelated status write'")
        writer.commit()
        with pool.acquire() as (reader, _):
            assert database.search_reader_diagnostics(reader)['catalog_snapshot_status'] == 'sampled'
        assert len(calls) == 1
    finally:
        release.set()
        finished.wait(5)
        pool.close()
        writer.close()


def test_age_based_refresh_does_not_mix_catalogs(tmp_path, monkeypatch):
    first = catalog(tmp_path / 'first.db')
    second = catalog(tmp_path / 'second.db')
    try:
        a = database.search_reader_diagnostics(first, refresh=True)
        second.execute('DELETE FROM documents')
        second.commit()
        b = database.search_reader_diagnostics(second, refresh=True)
        assert a['total_chunks'] == 1 and b['total_chunks'] == 0
        key, _ = database._diagnostic_key(first)
        with database._DIAGNOSTIC_LOCK:
            database._DIAGNOSTIC_SNAPSHOTS[key]['snapshot']['catalog_snapshot_sampled_ms'] -= 400_000
        monkeypatch.setattr(database, '_refresh_diagnostics', lambda *args: None)
        aged = database.search_reader_diagnostics(first)
        assert aged['catalog_snapshot_status'] == 'aged' and aged['total_chunks'] == 1
    finally:
        first.close()
        second.close()


def test_counts_share_one_wal_snapshot(tmp_path):
    path = tmp_path / 'coherent.db'
    writer = catalog(path)
    reader = database.connect(path, readonly=True)
    class CommitBetweenCounts:
        def execute(self, sql):
            cursor = reader.execute(sql)
            if sql == 'SELECT count(*) FROM chunks':
                writer.execute('DELETE FROM documents')
                writer.commit()
            return cursor
    try:
        result = database._collect_diagnostic_snapshot(CommitBetweenCounts(), ('test', str(path)))
        assert result['total_chunks'] == result['total_vectors'] == 1
        assert reader.execute('SELECT count(*) FROM chunks').fetchone()[0] == 0
    finally:
        reader.close()
        writer.close()


def test_diagnostic_worker_failure_never_disables_search(tmp_path, monkeypatch):
    con = catalog(tmp_path / 'failure.db')
    def unavailable(*args, **kwargs):
        raise RuntimeError('no worker threads')
    monkeypatch.setattr(threading.Thread, 'start', unavailable)
    try:
        sample = database.search_reader_diagnostics(con)
        assert sample['catalog_snapshot_status'] == 'unavailable'
        assert sample['catalog_snapshot_error'] == 'RuntimeError'
        assert search(con, [1, 0, 0])
    finally:
        con.close()


def test_explicit_refresh_never_publishes_uncommitted_counts(tmp_path):
    import pytest
    con = catalog(tmp_path / 'uncommitted.db')
    try:
        con.execute('DELETE FROM documents')
        with pytest.raises(ValueError, match='caller transaction'):
            database.search_reader_diagnostics(con, refresh=True)
        con.rollback()
    finally:
        con.close()


def test_http_search_reports_independent_availability_without_counts(tmp_path, monkeypatch):
    import http.client
    import json
    from http.server import ThreadingHTTPServer
    from axp_client.server import make_handler
    from axp_core.metadata import ensure_index_signature

    path = tmp_path / 'http.db'
    con = catalog(path)
    class Embedder:
        model_id = 'diagnostic-test'
        dimension = 3
        def embed_query(self, query):
            return [1, 0, 0]
    ensure_index_signature(con, Embedder.model_id, 3)
    con.commit()
    con.close()
    # Leave counts deliberately absent throughout the request.
    release, finished = threading.Event(), threading.Event()
    def delayed(*args):
        try:
            release.wait(5)
        finally:
            finished.set()
    monkeypatch.setattr(database, '_refresh_diagnostics', delayed)
    pool = database.SearchReaderPool(path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(path, Embedder(), search_readers=pool))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        request = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        request.request('GET', '/api/search?q=searchable&explain=1')
        response = request.getresponse()
        body = json.loads(response.read())
        request.close()
        assert response.status == 200 and body['results']
        diagnostics = body['diagnostics']
        assert diagnostics['catalog_snapshot_status'] == 'unavailable'
        assert diagnostics['search_data']['status'] == 'available'
        assert diagnostics['identity_cache']['status'] in ('hit', 'miss')
        assert diagnostics['identity_cache']['age_ms'] is not None
        assert diagnostics['identity_cache']['generation']
        request = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        request.request('GET', '/api/search?q=searchable&explain=1')
        warm = json.loads(request.getresponse().read())
        request.close()
        assert warm['diagnostics']['identity_cache']['status'] == 'hit'
        assert warm['diagnostics']['identity_cache']['age_ms'] >= 0
        assert warm['diagnostics']['identity_cache']['generation'] == diagnostics['identity_cache']['generation']
        assert warm['diagnostics']['catalog_snapshot_status'] == 'unavailable'
        assert warm['request_trace']['request_id'] != body['request_trace']['request_id']
        assert all(row['request_id'] == warm['request_trace']['request_id'] for row in warm['request_trace']['spans'])
        assert 'http_search_total_ms' in body['timings']
        assert 'vector_combined_execute_ms' in body['timings']
    finally:
        release.set()
        finished.wait(5)
        server.shutdown()
        server.server_close()
        thread.join()
        pool.close()
