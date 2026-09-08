import sqlite3

from axp_client import search as search_module
from axp_client.rag.retrieval import DocumentDrilldownResult


def test_search_identity_channel_does_not_require_body_occurrence(monkeypatch):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE sources(id INTEGER PRIMARY KEY,label TEXT,path TEXT);
      CREATE TABLE documents(id INTEGER PRIMARY KEY,source_id INTEGER,title TEXT,filename TEXT,path TEXT,
        path_key TEXT,extension TEXT,modified_unix_ms INTEGER,ingestion_mode TEXT);
      CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,page_no INTEGER,
        section_heading TEXT,text TEXT,identifiers TEXT);
      INSERT INTO sources VALUES(1,'SEQUENCES','K:\\R&D\\SEQUENCES');
      INSERT INTO documents VALUES(7,1,'SPECTRE','SPECTRE.docx','K:\\R&D\\SEQUENCES\\SPECTRE.docx',
        'k:\\r&d\\sequences\\spectre.docx','.docx',0,'content');
      INSERT INTO chunks VALUES(70,7,0,NULL,'','Reaction temperature 42 °C','');
    """)
    raw = {"results": [], "timings": {"total_ms": 0}, "candidate_counts": {}}
    monkeypatch.setattr(search_module, "raw_search", lambda *_args, **_kwargs: raw)

    def drilldown(connection, _embedder, _query, ids, **_kwargs):
        rows = search_module._metadata_identity_rows(connection, ids)
        return DocumentDrilldownResult(rows, {"drilldown_total_ms": 0}, [])

    monkeypatch.setattr(search_module, "retrieve_document_passages", drilldown)
    result = search_module.search(con, None, "montre-moi la séquence SPECTRE", explain=True)
    assert result["results"][0]["filename"] == "SPECTRE.docx"
    assert result["results"][0]["metadata_candidate"] is True
    assert result["results"][0]["filename_identity_match"] is True
    assert result["results"][0]["metadata_identity_coverage"] == 1.0
    assert result["timings"]["metadata_identity_ms"] >= 0
    assert result["metadata_identity_candidates"] == 1


def test_failed_metadata_document_is_materialized_without_fake_evidence():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE sources(id INTEGER PRIMARY KEY,label TEXT,path TEXT);
      CREATE TABLE documents(id INTEGER PRIMARY KEY,source_id INTEGER,title TEXT,filename TEXT,path TEXT,
        ingestion_mode TEXT);
      CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,page_no INTEGER,
        section_heading TEXT,text TEXT,identifiers TEXT);
      INSERT INTO sources VALUES(1,'SEQUENCES','K:\\R&D\\SEQUENCES');
      INSERT INTO documents VALUES(8,1,'FOOBAR','FOOBAR.doc','K:\\R&D\\SEQUENCES\\FOOBAR.doc','metadata');
    """)
    row = search_module._metadata_identity_rows(con, [8])[0]
    assert row["snippet"] == "Content extraction unavailable"
    assert row["ingestion_mode"] == "metadata"
    assert row["metadata_candidate"] is True
