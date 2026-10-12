"""Regressions for the October latency and lifecycle audit."""

import time
from types import SimpleNamespace

import pytest

from axp_core.database import connect, search_reader_diagnostics
from axp_core.identity_index import ensure_identity_index
from axp_core.runtime import read_json, runtime_paths, atomic_write_json
from axp_core.sources import add_source
from axp_core.vectors import search_documents, upsert
from axp_client.rag.spiral import resolve_identity_documents, RetrievalScope
from axp_daemon.service import StatePublisher, DaemonControl, send_control
from axp_tray.state import DaemonWatchdog, read_daemon_state
from axp_tray.process import recover_unhealthy_daemon


def catalog(tmp_path):
    c = connect(tmp_path / "audit.db", dimension=3)
    source = add_source(c, tmp_path / "fixture")["id"]
    c.execute(
        "INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(1,?,'/fixture/alpha.txt','/fixture/alpha.txt','.txt',0,100,'',0,'Étude ALPHA','ALPHA.txt')",
        (source,),
    )
    c.execute("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(1,1,0,'Pressure 12 bar')")
    upsert(c, 1, [1, 0, 0])
    c.commit()
    return c


def test_additive_index_backfill_idempotence_and_crud(tmp_path):
    c = catalog(tmp_path)
    # Simulate a preexisting v4 index with chunks/vectors already populated.
    for trigger in ("insert", "update", "delete"):
        c.execute("DROP TRIGGER document_identity_" + trigger)
    c.execute("DROP TABLE document_identity_fts")
    c.commit()
    ensure_identity_index(c)
    ensure_identity_index(c)
    assert resolve_identity_documents(c, "etude alpha") == [1]
    c.execute(
        "UPDATE documents SET filename='BETA.txt',title='BETA',path='/fixture/beta.txt',path_key='/fixture/beta.txt' WHERE id=1"
    )
    c.commit()
    assert not resolve_identity_documents(c, "alpha")
    assert resolve_identity_documents(c, "beta") == [1]
    assert c.execute("SELECT embedding FROM chunk_vectors WHERE rowid=1").fetchone()
    assert c.execute("SELECT version FROM schema_version").fetchone()[0] == 4
    c.execute("DELETE FROM documents WHERE id=1")
    c.commit()
    assert not resolve_identity_documents(c, "beta")
    c.close()


def test_identity_recall_parity_no_candidate_cap_unicode_and_source_changes(tmp_path):
    c = catalog(tmp_path)
    cases = [
        ("PROJECT-X.txt", "PROJECT X"),
        ("AB-123.txt", "AB123"),
        ("procédures.txt", "procedure"),
        ("ﬁnalization.txt", "finalization"),
    ]
    for number, (filename, query) in enumerate(cases, 2):
        c.execute(
            "INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(?,1,?,? ,'.txt',0,100,'',0,'',?)",
            (number, filename, filename, filename),
        )
    c.commit()
    queries = [q for _, q in cases] + ["alpha", "fixture alpha", "unknown"]
    indexed = [resolve_identity_documents(c, q) for q in queries]
    c.execute("ALTER TABLE document_identity_fts RENAME TO hidden_identity_fts")
    legacy = [resolve_identity_documents(c, q) for q in queries]
    assert indexed == legacy
    c.execute("ALTER TABLE hidden_identity_fts RENAME TO document_identity_fts")
    c.execute("UPDATE sources SET label='COLLECTION' WHERE id=1")
    c.commit()
    assert resolve_identity_documents(c, "COLLECTION alpha") == [1]
    assert not resolve_identity_documents(c, "alpha", scope=RetrievalScope(modified_after_ms=101))
    c.close()


def test_native_scoped_vector_distance_and_query_plan(tmp_path):
    c = catalog(tmp_path)
    c.execute("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(2,1,1,'Orthogonal')")
    upsert(c, 2, [0, 1, 0])
    c.commit()
    rows = search_documents(c, [1, 0, 0], [1])
    assert [r["vector_distance"] for r in rows] == pytest.approx([0, 1], abs=1e-6)
    assert all(row["vector_distance"] == 1 for row in search_documents(c, [0, 0, 0], [1]))
    plan = c.execute(
        "EXPLAIN QUERY PLAN SELECT v.embedding FROM chunks c CROSS JOIN chunk_vectors v ON v.rowid=c.id WHERE c.document_id=1"
    ).fetchall()
    assert any("document_id=?" in row[3] for row in plan)
    c.close()


def test_interactive_diagnostics_reuse_snapshot_and_label_changes(tmp_path):
    c = catalog(tmp_path)
    before = search_reader_diagnostics(c)
    statements = []
    c.set_trace_callback(statements.append)
    c.execute("INSERT INTO chunks(document_id,chunk_no,text) VALUES(1,1,'new')")
    c.commit()
    current = search_reader_diagnostics(c)
    assert current["catalog_snapshot_stale"] and current["total_chunks"] == before["total_chunks"]
    assert not any("count(" in sql.lower() for sql in statements)
    c.close()


def test_heartbeat_continues_during_long_worker_and_health_challenge(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    p = StatePublisher(interval_s=0.01)
    p.start()
    first = read_daemon_state()["heartbeat_monotonic"]
    atomic_write_json(
        runtime_paths()["runtime"] / "daemon_health_probe.json",
        {"instance_id": p.instance_id, "request_id": "challenge"},
    )
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        state = read_daemon_state()
        if state.get("health_probe_id") == "challenge" and state["heartbeat_monotonic"] > first:
            break
        time.sleep(0.01)
    assert state["health_probe_id"] == "challenge"
    assert state["heartbeat_monotonic"] > first
    assert state["worker_progress_monotonic"] == p.value["worker_progress_monotonic"]
    DaemonControl(p).poll()
    reply = read_json(runtime_paths()["runtime"] / "daemon_worker_health_reply.json")
    assert reply["instance_id"] == p.instance_id and reply["request_id"] == "challenge"
    p._write()
    assert read_daemon_state()["worker_health_probe_id"] == "challenge"
    p.close()


def test_obsolete_stop_is_not_replayed_and_restart_preserves_intent(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    old = StatePublisher()
    old._write()
    send_control("stop")
    new = StatePublisher()
    control = DaemonControl(new)
    control.poll()
    assert not control.stop
    new._write()
    atomic_write_json(runtime_paths()["desired"], {"state": "running"})
    send_control("stop", intentional=False, reason="watchdog")
    control.poll()
    assert control.stop
    assert read_json(runtime_paths()["desired"])["state"] == "running"


def test_watchdog_recovers_stalled_worker_and_limits_restart_storms():
    w = DaemonWatchdog()
    state = {"instance_id": "i", "state": "scanning", "stale": False, "worker_progress_monotonic": 0}
    assert w.reason(state, "running", True, True, 0) is None
    assert w.reason(state, "running", True, True, 100) is None
    assert w.reason(state, "running", True, True, 1801) == "worker_stalled"
    w.attempted(1801)
    assert w.reason(state, "running", True, True, 1820) is None
    assert w.reason(state, "stopped", True, True, 1861) is None
    for now in (1861, 1981):
        assert w.reason(state, "running", True, True, now) == "worker_stalled"
        w.attempted(now)
    assert w.reason(state, "running", True, False, 2500) is None
    assert w.reason(state, "running", True, False, 2900) == "instance_absent"


def test_monotonic_heartbeat_ignores_wall_clock_jump(tmp_path, monkeypatch):
    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    p = StatePublisher()
    p._write()
    monkeypatch.setattr("axp_tray.state.time.time", lambda: 99999999999)
    assert not read_daemon_state()["stale"]


def test_live_stalled_process_is_authenticated_before_recovery(tmp_path, monkeypatch):
    from axp_tray import process

    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    state = {"instance_id": "i", "pid": 234, "process_create_time": 10, "state": "scanning"}
    atomic_write_json(runtime_paths()["state"], state)
    atomic_write_json(runtime_paths()["desired"], {"state": "running"})
    calls = []

    class Process:
        def wait(self, timeout):
            if not calls:
                raise process.psutil.TimeoutExpired(timeout, 234)

        def terminate(self):
            calls.append("terminate")

    proc = Process()
    monkeypatch.setattr(process, "verified_owned_processes", lambda roles: [(proc, {"pid": 234, "create_time": 10})])
    monkeypatch.setattr(process, "is_owned_process", lambda *args: True)
    monkeypatch.setattr(process, "daemon_instance_running", lambda db: False)
    monkeypatch.setattr(process, "start_daemon", lambda settings: calls.append("start"))
    assert recover_unhealthy_daemon({"db_path": "test"}, state, "worker_stalled", timeout=0)
    assert calls == ["terminate", "start"]
    calls.clear()
    monkeypatch.setattr(process, "verified_owned_processes", lambda roles: [])
    assert not recover_unhealthy_daemon({"db_path": "test"}, state, "worker_stalled", timeout=0)
    assert calls == []


def test_rag_weak_identity_rescores_document_then_stops_with_evidence(tmp_path):
    from axp_client.rag.coordinator import RetrievalCoordinator

    c = catalog(tmp_path)
    embedded = []
    searched = []

    class Embedder:
        def embed_query(self, query):
            embedded.append(query)
            return [1, 0, 0] if query == "pressure" else [0, 1, 0]

    def search(*args, **kwargs):
        searched.append(1)
        return {"results": []}

    result = RetrievalCoordinator(search_fn=search).retrieve(c, Embedder(), "ALPHA pressure")
    assert result.decision.answerable
    assert embedded == ["ALPHA pressure", "pressure"]
    assert not searched
    assert result.trace[0]["initial_route"] == "identity_fast_path"
    assert result.trace[0]["final_route"] == "identity_expanded"
    assert result.trace[0]["evidence_retained"] > 0
    assert len({row["chunk_id"] for row in result.retrieval.content_evidence}) == len(result.retrieval.content_evidence)
    c.close()


@pytest.mark.parametrize("depth", [0, 1])
def test_rag_expansion_is_bounded_and_abstains_without_content_proof(tmp_path, depth):
    from axp_client.rag.coordinator import RetrievalCoordinator

    c = catalog(tmp_path)
    embedded = []
    searched = []

    class Embedder:
        def embed_query(self, query):
            embedded.append(query)
            return [0, 1, 0]

    def search(*args, **kwargs):
        searched.append(1)
        return {"results": []}

    result = RetrievalCoordinator(search_fn=search).retrieve(c, Embedder(), "ALPHA unobtainium", search_depth=depth)
    assert not result.decision.answerable
    assert len(searched) == 1
    assert len(embedded) <= 2 and len(set(embedded)) == len(embedded)
    assert result.trace[0]["global_hybrid_calls"] == 1
    assert result.trace[0]["expansion_reason"] == ("search_depth" if depth else "weak_retrieval")
    c.close()


def test_search_depth_explores_beyond_sufficient_identity(tmp_path):
    from axp_client.rag.coordinator import RetrievalCoordinator

    c = catalog(tmp_path)
    searched = []
    embedder = SimpleNamespace(embed_query=lambda q: [1, 0, 0])

    def search(*args, **kwargs):
        searched.append(1)
        return {"results": []}

    shallow = RetrievalCoordinator(search_fn=search).retrieve(c, embedder, "ALPHA pressure")
    assert shallow.decision.answerable and not searched
    deep = RetrievalCoordinator(search_fn=search).retrieve(c, embedder, "ALPHA pressure", search_depth=1)
    assert deep.decision.answerable and len(searched) == 1
    assert deep.trace[0]["expansion_reason"] == "search_depth"
    c.close()


def test_health_reply_and_instance_change_veto_termination(tmp_path, monkeypatch):
    from axp_tray import process

    monkeypatch.setenv("AXPINDEXER_DATA_DIR", str(tmp_path))
    state = {"instance_id": "i", "pid": 234}

    def snapshot():
        probe = read_json(runtime_paths()["runtime"] / "daemon_health_probe.json")
        return {**state, "worker_health_probe_id": probe["request_id"]}

    monkeypatch.setattr("axp_tray.state.read_daemon_state", snapshot)
    monkeypatch.setattr(process, "verified_owned_processes", lambda _: pytest.fail("Must not terminate healthy worker"))
    assert not recover_unhealthy_daemon({"db_path": "test"}, state, "worker_stalled", timeout=0.1)
    monkeypatch.setattr("axp_tray.state.read_daemon_state", lambda: {"instance_id": "new"})
    assert not recover_unhealthy_daemon({"db_path": "test"}, state, "worker_stalled", timeout=0.1)


def test_ttft_ignores_role_progress_empty_delta_and_distinguishes_reasoning(tmp_path, monkeypatch):
    from tests.test_pr48_lfm_reasoning import reasoning_backend

    events = [
        '{"choices":[{"delta":{"role":"assistant"}}]}',
        '{"prompt_progress":{"processed":1,"total":10}}',
        '{"choices":[{"delta":{"content":""}}]}',
        '{"choices":[{"delta":{"reasoning_content":"hidden work"}}]}',
        '{"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"timings":{"prompt_ms":12500}}',
    ]
    backend, _ = reasoning_backend(tmp_path, monkeypatch, events)
    assert backend.generate(system_prompt="system", user_prompt="user") == "answer"
    timing = backend.last_telemetry
    assert timing["first_generated_token_ms"] == timing["reasoning_ttft_ms"]
    assert timing["visible_ttft_ms"] > timing["first_generated_token_ms"]
    assert timing["generated_fragments"] == 1
    assert timing["ttft_prefill_inconsistent"]


def test_metadata_only_stream_has_no_first_generated_token(tmp_path, monkeypatch):
    from tests.test_pr48_lfm_reasoning import reasoning_backend

    backend, _ = reasoning_backend(
        tmp_path, monkeypatch, ['{"choices":[{"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}']
    )
    backend.generate(system_prompt="system", user_prompt="user")
    assert backend.last_telemetry["first_generated_token_ms"] is None
    assert backend.last_telemetry["time_to_first_token_ms"] is None


def test_weak_identity_hybrid_expansion_adds_another_document(tmp_path):
    from axp_client.rag.coordinator import RetrievalCoordinator

    c = catalog(tmp_path)
    c.execute(
        "INSERT INTO documents(id,source_id,path,path_key,extension,size_bytes,modified_unix_ms,sha256,indexed_unix_ms,title,filename) VALUES(2,1,'/fixture/reference.txt','/fixture/reference.txt','.txt',0,100,'',0,'Reference','Reference.txt')"
    )
    c.execute("INSERT INTO chunks(id,document_id,chunk_no,text) VALUES(2,2,0,'Pressure 12 bar')")
    upsert(c, 2, [0, 1, 0])
    c.commit()
    embedder = SimpleNamespace(embed_query=lambda q: [0, 1, 0])
    calls = []

    def search(*args, **kwargs):
        calls.append(1)
        return {
            "results": [
                {
                    "chunk_id": 2,
                    "document_id": 2,
                    "filename": "Reference.txt",
                    "snippet": "Pressure 12 bar",
                    "vector_similarity": 1.0,
                    "content_lexical_coverage": 1.0,
                    "passage_score": 0.9,
                }
            ]
        }

    result = RetrievalCoordinator(search_fn=search).retrieve(c, embedder, "ALPHA pressure")
    assert calls == [1]
    assert result.decision.answerable
    assert {row["document_id"] for row in result.retrieval.content_evidence} == {1, 2}
    assert result.trace[0]["drilldown_documents"] == 2
    c.close()
