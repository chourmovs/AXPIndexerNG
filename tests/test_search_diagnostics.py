"""Hierarchical timing, identity freshness, and progress during blocking retrieval."""
import contextvars
import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from axp_core.database import SearchReaderPool, connect
from axp_core.tracing import SearchCancelled, Trace, current_trace, interval_union, span
from axp_client.search_diagnostics import TraceRegistry
from tests.test_search_lifecycle_recovery import catalog


def test_overlapping_children_use_union_and_aggregate_loops():
    now = [0]
    trace = Trace(clock=lambda: now[0] * 1_000_000)
    with trace.bind(), span('parent'):
        first_context, second_context = contextvars.copy_context(), contextvars.copy_context()
        now[0] = 10
        first = span('first'); first_context.run(first.__enter__)
        now[0] = 20
        second = span('second'); second_context.run(second.__enter__)
        now[0] = 60
        first_context.run(first.__exit__, None, None, None)
        now[0] = 80
        second_context.run(second.__exit__, None, None, None)
        now[0] = 100
    trace.finish()
    parent = trace.snapshot()['spans'][0]
    assert parent['inclusive_ms'] == 100
    assert parent['self_ms'] == 30
    assert interval_union([(0, 8), (3, 5), (4, 10)]) == 10
    now[0] = 110
    with trace.bind(), span('loops'):
        for _ in range(2000):
            with span('iteration') as measured:
                measured.count(rows_read=1)
                now[0] += 1
    rows = trace.snapshot()['spans']
    repeated = next(row for row in rows if row['name'] == 'iteration')
    assert repeated['calls'] == 2000 and repeated['inclusive_ms'] == 2000
    assert repeated['max_ms'] == 1 and repeated['counters']['rows_read'] == 2000
    assert len(rows) == 5


def test_concurrent_request_isolation_errors_cancellation_and_bounds():
    barrier = threading.Barrier(2)
    def run(index):
        trace = Trace(str(index), max_spans=3)
        with trace.bind(), span('request'):
            barrier.wait(timeout=2)
            assert current_trace() is trace
            for n in range(5):
                with span(str(n)): pass
        trace.finish()
        return trace.snapshot()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(run, range(2)))
    assert current_trace() is None
    assert all(r['truncated'] and len(r['spans']) == 3 for r in results)
    assert all(all(s['request_id'] == r['request_id'] for s in r['spans']) for r in results)
    for error, status in ((ValueError('failure'), 'error'), (SearchCancelled(), 'cancelled')):
        trace = Trace()
        with pytest.raises(type(error)), trace.bind(), span('parent'), span('child'):
            raise error
        trace.finish(status)
        assert all(row['status'] == status for row in trace.snapshot()['spans'])
        assert not trace.active


def resolve_with_trace(con, query='alpha'):
    from axp_client.rag.spiral import resolve_identity_documents
    trace = Trace()
    with trace.bind():
        ids = resolve_identity_documents(con, query)
    return ids, trace.snapshot()


def cache_detail(trace):
    return next(row['details'] for row in trace['spans'] if row['routine'] == 'identity_cache.identity_resolution')


def test_identity_invalidation_relevant_changes_only_and_rollback(tmp_path):
    writer = catalog(tmp_path)
    reader = connect(tmp_path / 'audit.db', readonly=True)
    assert resolve_with_trace(reader)[0] == [1]
    _, warm = resolve_with_trace(reader)
    assert cache_detail(warm)['cache'] == 'hit'
    generation = cache_detail(warm)['generation']
    writer.execute("UPDATE sources SET status='scanning',last_scan_started_ms=123 WHERE id=1")
    writer.execute("UPDATE documents SET indexed_unix_ms=100 WHERE id=1")
    writer.commit()
    _, still_warm = resolve_with_trace(reader)
    assert cache_detail(still_warm)['generation'] == generation
    assert cache_detail(still_warm)['cache'] == 'hit'
    writer.execute("UPDATE documents SET title='BETA',filename='BETA.txt',path='/beta',path_key='/beta' WHERE id=1")
    writer.commit()
    ids, changed = resolve_with_trace(reader)
    assert ids == [] and cache_detail(changed)['cache'] == 'miss'
    assert cache_detail(changed)['invalidation_reason'] == 'identity_metadata_changed'
    assert resolve_with_trace(reader, 'beta')[0] == [1]
    writer.execute("UPDATE sources SET label='NEWCOLLECTION' WHERE id=1")
    writer.commit()
    _, changed = resolve_with_trace(reader, 'beta')
    assert cache_detail(changed)['cache'] == 'miss'
    writer.execute("DELETE FROM documents WHERE id=1")
    writer.rollback()
    assert resolve_with_trace(reader, 'beta')[0] == [1]
    writer.execute("DELETE FROM documents WHERE id=1"); writer.commit()
    assert resolve_with_trace(reader, 'beta')[0] == []
    reader.close(); writer.close()


def test_identity_single_flight_and_snapshot_publication(tmp_path, monkeypatch):
    from axp_client.rag import spiral
    writer = catalog(tmp_path); writer.close()
    original = spiral._resolve_identity_documents
    entered, release = threading.Event(), threading.Event()
    calls = []
    def blocking(*args, **kwargs):
        calls.append(1); entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(spiral, '_resolve_identity_documents', blocking)
    def run():
        con = connect(tmp_path / 'audit.db', readonly=True)
        try: return resolve_with_trace(con)
        finally: con.close()
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(run)
        assert entered.wait(2)
        second = executor.submit(run)
        release.set()
        a, b = first.result(3), second.result(3)
    assert len(calls) == 1 and a[0] == b[0] == [1]
    assert sorted([cache_detail(a[1])['cache'], cache_detail(b[1])['cache']]) == ['hit', 'miss']


def test_snapshot_stays_coherent_when_writer_commits_during_rebuild(tmp_path, monkeypatch):
    from axp_client.rag import spiral
    writer = catalog(tmp_path)
    original = spiral._resolve_identity_documents
    def mutate(con, *args, **kwargs):
        writer.execute("UPDATE documents SET title='BETA',filename='BETA',path='/beta',path_key='/beta' WHERE id=1")
        writer.commit()
        return original(con, *args, **kwargs)
    monkeypatch.setattr(spiral, '_resolve_identity_documents', mutate)
    reader = connect(tmp_path / 'audit.db', readonly=True)
    assert resolve_with_trace(reader)[0] == [1]  # Entirely pre-commit snapshot.
    monkeypatch.setattr(spiral, '_resolve_identity_documents', original)
    assert resolve_with_trace(reader)[0] == []  # Next request sees new revision.
    assert resolve_with_trace(reader, 'beta')[0] == [1]
    reader.close(); writer.close()


def test_registry_bounds_reject_duplicate_and_expire():
    registry = TraceRegistry(max_entries=2, max_active=1, ttl=0)
    a = registry.start('a')
    with pytest.raises(ValueError): registry.start('a')
    with pytest.raises(RuntimeError): registry.start('b')
    a.finish(); assert registry.get('a') is None
    b = registry.start('b'); b.finish()
    assert registry.start('c').request_id == 'c'


@contextmanager
def running(db, embedder):
    from axp_client.server import make_handler
    pool = SearchReaderPool(db)
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(db, embedder, search_readers=pool))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True); thread.start()
    try: yield httpd
    finally:
        httpd.shutdown(); httpd.server_close(); thread.join(3); pool.close()


def request(httpd, path, method='GET', headers=None):
    con = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=3)
    try:
        con.request(method, path, headers=headers or {})
        response = con.getresponse()
        return response.status, json.loads(response.read())
    finally: con.close()


def test_progress_visible_before_blocking_phase_finishes_and_cancel(tmp_path, monkeypatch):
    from axp_client import server
    entered, release = threading.Event(), threading.Event()
    c = catalog(tmp_path); c.close()
    def blocking(*args, **kwargs):
        with span('Blocking measured search'):
            entered.set(); assert release.wait(3)
        return {'results':[], 'timings':{}}
    monkeypatch.setattr(server, 'search', blocking)
    with running(tmp_path / 'audit.db', object()) as httpd, ThreadPoolExecutor(1) as executor:
        future = executor.submit(request, httpd, '/api/search?q=alpha&explain=1&request_id=abcdef0123456789')
        try:
            assert entered.wait(2)
            code, progress = request(httpd, '/api/search/progress?request_id=abcdef0123456789')
            assert code == 200 and progress['status'] == 'running'
            assert progress['active_stage'] == 'Blocking measured search'
            assert not future.done()
            assert request(httpd, '/api/search/cancel?request_id=abcdef0123456789', 'POST')[0] == 200
        finally: release.set()
        code, final = future.result(3)
        assert code == 499 and final['request_trace']['status'] == 'cancelled'
        assert not any(s['status'] == 'running' for s in final['request_trace']['spans'])


def test_real_search_trace_envelope_and_legacy_response(tmp_path):
    from conftest import FakeEmbedder
    from axp_core.metadata import index_signature
    c = catalog(tmp_path)
    c.execute("INSERT INTO metadata VALUES('index_signature',?)", (json.dumps(index_signature('test',3), sort_keys=True, separators=(',',':')),))
    c.commit(); c.close()
    with running(tmp_path / 'audit.db', FakeEmbedder()) as httpd:
        code, result = request(httpd, '/api/search?q=alpha&explain=1')
        assert code == 200 and result['results']
        trace = result['request_trace']
        assert trace['status'] == 'completed'
        assert 'socket' in trace['server_scope']
        assert any(s['routine'] == 'sqlite.Cursor.fetchall' for s in trace['spans'])
        assert all(0 <= s['self_ms'] <= s['inclusive_ms'] + 1e-6 for s in trace['spans'])
        code, legacy = request(httpd, '/api/search?q=alpha')
        assert code == 200 and isinstance(legacy, list)
        assert request(httpd, '/api/search?q=alpha&profile=unknown')[0] == 400
        assert request(httpd, '/api/search/progress', headers={'Origin':'https://example.com'})[0] == 403


def test_unicode_prefilter_preserves_exhaustive_ranking_without_full_catalog(tmp_path):
    from axp_client.rag.spiral import _resolve_identity_documents
    from axp_core.identity_index import identity_candidate_ids
    c = catalog(tmp_path)
    for number, name in enumerate(('procédures', 'ﬁnalization', 'ＳＡＦＥＴＹ', 'Straße', 'PROJECT-X', 'AB-123'), 2):
        c.execute("INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(?,1,?,? ,'.txt',0,100,'',0,?,?)",
                  (number, name, name, name, name+'.txt'))
    c.commit()
    assert identity_candidate_ids(c, 'unobtainium', set()) == set()
    queries = ('alpha', 'procedure', 'finalization', 'safety', 'strasse', 'PROJECT X', 'AB123', 'inconnu')
    indexed = [_resolve_identity_documents(c, q) for q in queries]
    # Force the authoritative legacy exhaustive route, independently of both indexes/cache.
    c.execute('ALTER TABLE document_identity_fts RENAME TO hidden_identity')
    exhaustive = [_resolve_identity_documents(c, q) for q in queries]
    assert indexed == exhaustive
    c.execute('ALTER TABLE hidden_identity RENAME TO document_identity_fts')
    statements = []; c.set_trace_callback(statements.append)
    assert _resolve_identity_documents(c, 'strasse') == [5]
    assert not any('FROM chunks' in statement for statement in statements)
    c.set_trace_callback(None)
    c.execute("UPDATE documents SET filename='NEWNAME.txt',title='NEWNAME',path='/new',path_key='/new' WHERE id=5")
    c.commit()
    assert _resolve_identity_documents(c, 'strasse') == []
    assert _resolve_identity_documents(c, 'newname') == [5]
    c.close()


def test_cache_addition_scope_and_failed_rebuild_are_not_stale(tmp_path, monkeypatch):
    from axp_client.rag import spiral
    from axp_client.rag.spiral import RetrievalScope
    writer = catalog(tmp_path); reader = connect(tmp_path / 'audit.db', readonly=True)
    assert resolve_with_trace(reader, 'newname')[0] == []
    writer.execute("INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(2,1,'/new','/new','.txt',0,100,'',0,'NEWNAME','NEWNAME.txt')")
    writer.commit()
    assert resolve_with_trace(reader, 'newname')[0] == [2]
    original = spiral._resolve_identity_documents
    def fail(*args, **kwargs): raise ValueError('synthetic rebuild failure')
    monkeypatch.setattr(spiral, '_resolve_identity_documents', fail)
    with pytest.raises(ValueError): resolve_with_trace(reader, 'NEWNAME')
    monkeypatch.setattr(spiral, '_resolve_identity_documents', original)
    assert resolve_with_trace(reader, 'NEWNAME')[0] == [2]
    assert spiral.resolve_identity_documents(reader, 'newname', scope=RetrievalScope(modified_after_ms=101)) == []
    writer.execute('UPDATE documents SET modified_unix_ms=102 WHERE id=2'); writer.commit()
    assert spiral.resolve_identity_documents(reader, 'newname', scope=RetrievalScope(modified_after_ms=101)) == [2]
    reader.close(); writer.close()


def test_sql_cancellation_interrupts_worker_and_trace_error_finalization(tmp_path, monkeypatch):
    from axp_client import server
    from axp_core.tracing import sql_rows
    entered = threading.Event()
    c = catalog(tmp_path); c.close()
    def long_sql(con, *args, **kwargs):
        entered.set()
        sql_rows(con, 'WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n')
        return {'results':[], 'timings':{}}
    monkeypatch.setattr(server, 'search', long_sql)
    with running(tmp_path / 'audit.db', object()) as httpd, ThreadPoolExecutor(1) as executor:
        future = executor.submit(request, httpd, '/api/search?q=test&explain=1&request_id=aaaabbbbccccdddd')
        assert entered.wait(2)
        assert request(httpd, '/api/search/cancel?request_id=aaaabbbbccccdddd', 'POST')[0] == 200
        status, body = future.result(3)
        assert status == 499 and body['request_trace']['status'] == 'cancelled'
        assert all(row['status'] != 'running' for row in body['request_trace']['spans'])
        def fail(*args, **kwargs): raise RuntimeError('synthetic')
        monkeypatch.setattr(server, 'search', fail)
        status, body = request(httpd, '/api/search?q=test&explain=1')
        assert status == 500 and body['request_trace']['status'] == 'error'


def test_unicode_backfill_is_transactional_and_shared_by_writer_connections(tmp_path):
    c = catalog(tmp_path)
    for action in ('insert','update','delete'):
        c.execute('DROP TRIGGER document_identity_unicode_'+action)
    c.execute('DROP TABLE document_identity_unicode_fts'); c.commit(); c.close()
    barrier = threading.Barrier(2)
    def open_writer(_):
        barrier.wait(timeout=2)
        con = connect(tmp_path / 'audit.db')
        try:
            return resolve_with_trace(con)[0], con.execute('SELECT count(*) FROM document_identity_unicode_fts').fetchone()[0]
        finally: con.close()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(open_writer, range(2)))
    assert results == [([1],1), ([1],1)]


def test_rag_handles_document_deleted_between_routing_and_hydration(tmp_path, monkeypatch):
    from axp_client.rag import coordinator
    c = catalog(tmp_path)
    def routed_then_deleted(con, *args, **kwargs):
        con.execute('DELETE FROM documents WHERE id=1'); con.commit()
        return [1]
    monkeypatch.setattr(coordinator, 'resolve_identity_documents', routed_then_deleted)
    retriever = coordinator.RetrievalCoordinator(search_fn=lambda *a, **kw: [])
    assert retriever._identity(c, 'alpha', coordinator.RetrievalPolicy()) == ([], None)
    c.close()
