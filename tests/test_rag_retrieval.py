import sqlite3

from axp_client.rag.answerability import decide_answerability
from axp_client.rag.retrieval import retrieve_rag_candidates


def test_authoritative_retrieval_filters_metadata_and_uses_content_identifiers(tmp_path):
    path = tmp_path / "rag.db"
    with sqlite3.connect(path) as con:
        con.executescript("""
            CREATE TABLE documents(id INTEGER PRIMARY KEY, filename TEXT, ingestion_mode TEXT);
            CREATE TABLE chunks(id INTEGER PRIMARY KEY, document_id INTEGER, text TEXT);
            INSERT INTO documents VALUES(1, 'R042500-report.pdf', 'content');
            INSERT INTO documents VALUES(2, 'metadata.dwg', 'metadata');
            INSERT INTO documents VALUES(3, 'content.pdf', 'content');
            INSERT INTO chunks VALUES(10, 1, 'This document contains general validation information.');
            INSERT INTO chunks VALUES(30, 3, 'R042500 was changed to reduce mixing time.');
        """)
        con.row_factory = sqlite3.Row
        rows = [
            {"document_id": 2, "chunk_id": 20, "vector_similarity": .99, "lexical_coverage": 1.0},
            {"document_id": 1, "chunk_id": 10, "vector_similarity": .46, "lexical_coverage": .3,
             "exact_identifier_match": True, "exact_filename_match": True},
            {"document_id": 3, "chunk_id": 30, "vector_similarity": .46, "lexical_coverage": .3},
        ]

        def search(*args, **kwargs):
            return {"results": rows}

        retrieval = retrieve_rag_candidates(con, None, "Why was R042500 changed?", search_fn=search)

    assert [row["document_id"] for row in retrieval.content_evidence] == [1, 3]
    assert retrieval.metadata_related == [{"document_id": 2, "filename": "metadata.dwg"}]
    assert retrieval.content_evidence[0]["exact_identifier_match"] is True
    assert retrieval.content_evidence[0]["exact_content_identifier_match"] is False
    assert retrieval.content_evidence[1]["exact_content_identifier_match"] is True
    assert decide_answerability([retrieval.content_evidence[0]]).answerable is False
    assert decide_answerability([retrieval.content_evidence[1]]).reason == "exact_supported"


def test_metadata_only_is_no_content_evidence(tmp_path):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE documents(id INTEGER PRIMARY KEY, filename TEXT, ingestion_mode TEXT);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, document_id INTEGER, text TEXT);
        INSERT INTO documents VALUES(2, 'R042500.dwg', 'metadata');
    """)
    def search(*args, **kwargs):
        return {"results": [
            {"document_id": 2, "chunk_id": 20, "vector_similarity": .99, "lexical_coverage": 1.0}
        ]}
    retrieval = retrieve_rag_candidates(con, None, "R042500", search_fn=search)
    decision = decide_answerability(retrieval.content_evidence)
    assert not decision.answerable and decision.reason == "no_content_evidence"
