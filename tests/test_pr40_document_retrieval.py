import sqlite3

import pytest

from axp_client.rag.context import ContextConfig, build_context, select_distinct_seeds
from axp_client.rag.depth import depth_policy, validate_search_depth
from axp_client.rag.retrieval import rank_documents
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance, diversify


def row(document_id, score, chunk_id=1, chunk_no=0):
    return {"document_id": document_id, "evidence_score": score, "relevance_score": score,
            "chunk_id": chunk_id, "chunk_no": chunk_no, "filename": f"{document_id}.pdf", "title": document_id}


def test_score_order_and_stable_document_cap():
    rows = [row("A", .88, 1), row("A", .84, 2), row("B", .82, 3), row("C", .79, 4)]
    assert [x["relevance_score"] for x in diversify(rows, 4, 3)] == [.88, .84, .82, .79]
    crowded = [row("A", 1-i/100, i) for i in range(20)] + [row("B", .7, 30)]
    assert [x["document_id"] for x in diversify(crowded, 10, 3)] == ["A", "A", "A", "B"]


def scored(*, vector, content, title="", filename="", heading=""):
    item = {"snippet": content, "identifiers": "", "title": title, "filename": filename,
            "heading": heading, "vector_distance": 1-vector, "lexical_rank": 1,
            "exact_identifier_match": False, "exact_phrase_match": False,
            "exact_filename_match": False, "exact_priority": 0}
    _relevance(item, _meaningful_terms("DOMINO bleach mixing"), SearchConfig(min_lexical_coverage=.1))
    return item


def test_title_signal_convergence_and_metadata_false_positive():
    matching = scored(vector=.72, content="DOMINO bleach mixing", title="DOMINO bleach mixing study")
    unrelated = scored(vector=.72, content="DOMINO bleach mixing", title="Unrelated report")
    assert matching["evidence_score"] > unrelated["evidence_score"]
    converged = scored(vector=.72, content="DOMINO bleach mixing")
    semantic_only = scored(vector=.80, content="unrelated prose")
    assert converged["evidence_score"] > semantic_only["evidence_score"]
    misleading = scored(vector=.05, content="unrelated prose", title="DOMINO")
    assert misleading["title_coverage"] > 0
    assert not _relevance(misleading, _meaningful_terms("DOMINO bleach mixing"), SearchConfig())


def test_document_density_and_distinct_seeds():
    dense = [row(1, score, index, index) for index, score in enumerate((.82, .78, .74), 1)]
    isolated = [row(2, score, index+10, index) for index, score in enumerate((.86, .39, .32), 1)]
    assert rank_documents(dense + isolated)[0]["document_id"] == 1
    hits = [row("A", .88, 12, 12), row("A", .86, 13, 13), row("A", .81, 27, 27)]
    assert [x["chunk_no"] for x in select_distinct_seeds(hits, 2)] == [12, 27]


def test_fair_context_and_existing_schema_compatibility():
    con = sqlite3.connect(":memory:"); con.row_factory = sqlite3.Row
    con.executescript("""
      CREATE TABLE documents(id INTEGER PRIMARY KEY,title TEXT,filename TEXT,path TEXT);
      CREATE TABLE chunks(id INTEGER PRIMARY KEY,document_id INTEGER,chunk_no INTEGER,text TEXT,page_no INTEGER,section_heading TEXT);
      INSERT INTO documents VALUES(1,'One','one.pdf','one.pdf'),(2,'Two','two.pdf','two.pdf');
      INSERT INTO chunks VALUES(1,1,0,'alpha evidence',NULL,''),(2,1,10,'secondary alpha',NULL,''),(3,2,0,'beta evidence',NULL,'');
    """)
    hits = [row(1,.9,1,0), row(1,.8,2,10), row(2,.85,3,0)]
    result = build_context(con, hits, ContextConfig(max_documents=2,max_seeds_per_document=2,neighbor_radius=0,
        max_blocks=2, character_budget=10000))
    assert [block.document_id for block in result.blocks] == [1, 2]


@pytest.mark.parametrize("value", [-1, 2, 1.5, "1", True, None])
def test_search_depth_validation(value):
    with pytest.raises(ValueError): validate_search_depth(value)


def test_depth_scaling_targets():
    assert validate_search_depth(0) == 0 and validate_search_depth(1) == 1
    policy = depth_policy(1, evidence_tokens=3072, answer_tokens=256)
    assert (policy.target_evidence_tokens, policy.target_answer_tokens, policy.retrieval_limit,
            policy.candidate_depth) == (4608, 384, 36, 150)
