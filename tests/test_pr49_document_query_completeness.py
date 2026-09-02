from axp_client import search as search_module
from axp_client.rag.retrieval import (
    DocumentDrilldownResult,
    classify_query_evidence_intent,
    rank_documents,
)
from axp_core.hybrid import SearchConfig, _meaningful_terms, _relevance, fold_search_text
from axp_core.identifiers import extract_identifiers


def scored(document_id, chunk_id, content, title, score):
    row = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "snippet": content,
        "identifiers": "",
        "title": title,
        "filename": f"{title}.docx",
        "heading": "",
        "vector_distance": 1 - score,
        "lexical_rank": 1,
        "exact_identifier_match": False,
        "exact_phrase_match": False,
        "exact_filename_match": False,
    }
    _relevance(row, _meaningful_terms("sequence domino"), SearchConfig(min_lexical_coverage=.1))
    return row


def ranked_fixture(query="sequence domino", complete_title="DOMINO",
                   complete_content="Séquence réactionnelle DEP-1448-DOMINO"):
    partial = [scored(1, number, "SEQUENCE REACTIONNELLE", "ASSYS SéquenceReactionelle 2019", .74)
               for number in range(1, 4)]
    complete = scored(2, 4, complete_content, complete_title, .60)
    intent = classify_query_evidence_intent(query)
    # Re-score when the generic fixture uses terms other than sequence/domino.
    for row in [*partial, complete]:
        _relevance(row, _meaningful_terms(query), SearchConfig(min_lexical_coverage=.1))
    return rank_documents([*partial, complete], intent=intent), partial, complete


def test_search_folding_is_case_and_accent_insensitive():
    assert {fold_search_text(value) for value in ("SEQUENCE", "Séquence", "séquence", "sequence")} == {
        "sequence"
    }
    assert _meaningful_terms("Séquence réactionnelle densité") == {
        "sequence", "reactionnelle", "densite",
    }


def test_complete_short_query_document_outranks_repeated_partial_hits():
    ranked, _, _ = ranked_fixture()
    assert ranked[0]["title"] == "DOMINO"
    assert ranked[0]["document_query_coverage"] == 1.0
    assert ranked[0]["document_metadata_coverage"] > 0
    assert ranked[0]["complete_query_match"] is True
    assert ranked[1]["complete_query_match"] is False


def test_completeness_priority_is_generic():
    partial = [scored(1, number, "project planning", "Repeated project notes", .74)
               for number in range(1, 4)]
    complete = scored(2, 4, "Project Phoenix", "PHOENIX", .60)
    intent = classify_query_evidence_intent("project phoenix")
    for row in [*partial, complete]:
        _relevance(row, _meaningful_terms("project phoenix"), SearchConfig(min_lexical_coverage=.1))
    assert rank_documents([*partial, complete], intent=intent)[0]["title"] == "PHOENIX"


def test_semantic_fallback_keeps_score_order_without_complete_match():
    first = scored(1, 1, "sequence only", "Alpha", .80)
    second = scored(2, 2, "domino only", "Beta", .60)
    intent = classify_query_evidence_intent("sequence domino")
    ranked = rank_documents([first, second], intent=intent)
    assert not any(document["complete_query_match"] for document in ranked)
    assert ranked[0]["document_id"] == 1


def test_standalone_search_returns_one_diagnostic_card_per_document(monkeypatch):
    ranked, partial, complete = ranked_fixture()
    raw = {"results": [*partial, complete], "timings": {"total_ms": 0}, "candidate_counts": {}}
    monkeypatch.setattr(search_module, "raw_search", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(search_module, "retrieve_document_passages", lambda *_args, **_kwargs:
                        DocumentDrilldownResult([complete, *partial], {"drilldown_total_ms": 0}, []))
    monkeypatch.setattr(search_module, "rank_documents", lambda *_args, **_kwargs: ranked)

    result = search_module.search(None, None, "sequence domino", explain=True)
    assert [row["document_id"] for row in result["results"]] == [2, 1]
    assert len({row["document_id"] for row in result["results"]}) == len(result["results"])
    for key in ("document_score", "document_query_coverage", "document_metadata_coverage",
                "complete_query_match", "passage_score", "vector_similarity",
                "content_lexical_coverage"):
        assert key in result["results"][0]


def test_rag_document_rank_retains_multiple_passages():
    ranked, partial, _ = ranked_fixture()
    assert len(ranked[1]["ranked_hits"]) == len(partial) == 3


def test_identifier_extraction_is_unchanged_by_human_text_folding():
    value = "DEP-1448 CZL-0181-106 R12100"
    assert [identifier for identifier, _ in extract_identifiers(value)] == [
        "DEP1448", "CZL0181106", "R12100",
    ]
    assert fold_search_text(value) == "dep-1448 czl-0181-106 r12100"
